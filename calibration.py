"""Post-hoc few-shot affine calibration for cross-dataset predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TAG_TO_VARIANT = {
    "full": "full",
    "concat": "concat",
    "nophysics": "no-physics",
    "noclip": "no-clip",
    "noattribute": "no-attribute",
    "regressiononly": "regression-only",
    "hybridfusion": "hybrid-fusion",
    "aquaclipfinal": "aquaclip-final",
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {"srcc": math.nan, "plcc": math.nan, "krcc": math.nan, "rmse": math.nan, "nrmse": math.nan}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    value_range = float(y_true.max() - y_true.min())
    return {
        "srcc": float(stats.spearmanr(y_true, y_pred).correlation),
        "plcc": float(stats.pearsonr(y_true, y_pred).statistic),
        "krcc": float(stats.kendalltau(y_true, y_pred).correlation),
        "rmse": rmse,
        "nrmse": rmse / value_range if value_range > 0 else math.nan,
    }


def parse_name(path: Path):
    stem = path.stem.removesuffix("_predictions")
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "crossablation":
        return None
    tag, train_dataset, test_dataset = parts[1], parts[2], parts[3]
    return TAG_TO_VARIANT.get(tag, tag), train_dataset, test_dataset


def affine_fit(x: np.ndarray, y: np.ndarray, positive_slope: bool = False) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) < 1e-8:
        return 1.0, float(y.mean() - x.mean())
    a, b = np.polyfit(x, y, deg=1)
    if positive_slope and a <= 0:
        a = 1e-6
        b = float(y.mean() - a * x.mean())
    return float(a), float(b)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="v1/outputs/crossablation")
    parser.add_argument("--output", default="v1/outputs/v1_fewshot_affine_calibration.csv")
    parser.add_argument("--ratios", default="0.01,0.05,0.1,0.2")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--variants", default="full,concat,regression-only,no-attribute,no-physics,no-clip,hybrid-fusion")
    parser.add_argument("--positive-slope", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ratios = [float(item) for item in args.ratios.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    rows = []

    for path in Path(args.prediction_dir).glob("crossablation_*_predictions.csv"):
        parsed = parse_name(path)
        if parsed is None:
            continue
        variant, train_dataset, test_dataset = parsed
        if variant not in variants:
            continue
        df = pd.read_csv(path).dropna(subset=["mos_norm", "pred"]).reset_index(drop=True)
        if len(df) < 5:
            continue
        base = regression_metrics(df["mos_norm"], df["pred"])
        rows.append(
            {
                "variant": variant,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "ratio": 0.0,
                "seed": 0,
                "n_cal": 0,
                "n_eval": int(len(df)),
                "a": 1.0,
                "b": 0.0,
                **base,
            }
        )
        for ratio in ratios:
            n_cal = max(2, int(round(len(df) * ratio)))
            n_cal = min(n_cal, len(df) - 2)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                cal_idx = rng.choice(len(df), size=n_cal, replace=False)
                eval_mask = np.ones(len(df), dtype=bool)
                eval_mask[cal_idx] = False
                cal = df.iloc[cal_idx]
                eva = df.loc[eval_mask]
                a, b = affine_fit(
                    cal["pred"].to_numpy(),
                    cal["mos_norm"].to_numpy(),
                    positive_slope=args.positive_slope,
                )
                pred_cal = a * eva["pred"].to_numpy() + b
                metrics = regression_metrics(eva["mos_norm"], pred_cal)
                rows.append(
                    {
                        "variant": variant,
                        "train_dataset": train_dataset,
                        "test_dataset": test_dataset,
                        "ratio": ratio,
                        "seed": seed,
                        "n_cal": int(n_cal),
                        "n_eval": int(len(eva)),
                        "a": a,
                        "b": b,
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    summary = (
        out.groupby(["variant", "ratio"])
        .agg(
            srcc_mean=("srcc", "mean"),
            plcc_mean=("plcc", "mean"),
            nrmse_mean=("nrmse", "mean"),
            nrmse_median=("nrmse", "median"),
        )
        .reset_index()
        .sort_values(["ratio", "nrmse_mean", "srcc_mean"], ascending=[True, True, False])
    )
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {output}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
