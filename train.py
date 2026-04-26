"""Train and evaluate the final AquaCLIP-QA model on cached tabular features."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from aquaclip.data import (  # noqa: E402
    BranchStandardizer,
    add_dataset_normalized_mos,
    default_split,
    feature_columns,
    load_feature_table,
    normalize_by_range,
    parse_dataset_spec,
    sample_few_shot,
)
from aquaclip.model import ReliabilityAquaClip, pairwise_ranking_loss  # noqa: E402


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {"srcc": math.nan, "plcc": math.nan, "krcc": math.nan, "rmse": math.nan, "nrmse": math.nan}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    value_range = float(np.max(y_true) - np.min(y_true))
    return {
        "srcc": float(stats.spearmanr(y_true, y_pred).correlation),
        "plcc": float(stats.pearsonr(y_true, y_pred).statistic),
        "krcc": float(stats.kendalltau(y_true, y_pred).correlation),
        "rmse": rmse,
        "nrmse": rmse / value_range if value_range > 0 else math.nan,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True, help="name:clip_scores_csv:physics_csv")
    parser.add_argument("--test", action="append", default=[], help="name:clip_scores_csv:physics_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--target-mode", choices=["train-range", "per-dataset"], default="train-range")
    parser.add_argument("--few-shot-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.05)
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "concat",
            "no-physics",
            "no-clip",
            "no-attribute",
            "no-attribute-gate",
            "regression-only",
            "hybrid-fusion",
            "aquaclip-final",
        ],
        default="full",
    )
    parser.add_argument("--reliability-aux-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def variant_config(variant: str) -> dict[str, bool | float]:
    config: dict[str, bool | float] = {
        "use_clip": True,
        "use_attr": True,
        "use_phys": True,
        "use_reliability": True,
        "reliability_for_prediction": True,
        "use_attr_gate": True,
        "hybrid_fusion": False,
    }
    if variant == "concat":
        config["use_reliability"] = False
        config["reliability_for_prediction"] = False
    elif variant == "no-physics":
        config["use_phys"] = False
        config["use_attr_gate"] = False
    elif variant == "no-clip":
        config["use_clip"] = False
    elif variant == "no-attribute":
        config["use_attr"] = False
        config["use_attr_gate"] = False
    elif variant == "no-attribute-gate":
        config["use_attr_gate"] = False
    elif variant == "regression-only":
        pass
    elif variant == "hybrid-fusion":
        config["hybrid_fusion"] = True
    elif variant == "aquaclip-final":
        config["reliability_for_prediction"] = False
    elif variant != "full":
        raise ValueError(f"Unknown variant: {variant}")
    return config


def uses_regression_only_loss(variant: str) -> bool:
    return variant == "regression-only"


def build_frames(args):
    train_specs = [parse_dataset_spec(item) for item in args.train]
    test_specs = [parse_dataset_spec(item) for item in args.test]

    if not test_specs:
        if len(train_specs) != 1:
            raise ValueError("In-domain mode expects exactly one --train dataset.")
        df = default_split(load_feature_table(train_specs[0]), seed=args.seed)
        raw_train = df[df["split"].astype(str).str.lower() == "train"].copy()
        raw_test = df[df["split"].astype(str).str.lower() == "test"].copy()
        raw_train = sample_few_shot(raw_train, args.few_shot_ratio, args.seed)
        y_min = float(raw_train["mos"].min())
        y_max = float(raw_train["mos"].max())
        train = normalize_by_range(raw_train, y_min, y_max)
        test = normalize_by_range(raw_test, y_min, y_max)
        protocol = "in-domain"
    else:
        train_frames = [add_dataset_normalized_mos(load_feature_table(spec)) for spec in train_specs]
        test_frames = [add_dataset_normalized_mos(load_feature_table(spec)) for spec in test_specs]
        train = pd.concat(train_frames, ignore_index=True)
        train = sample_few_shot(train, args.few_shot_ratio, args.seed)
        test = pd.concat(test_frames, ignore_index=True)
        protocol = "cross-dataset"
    return train.reset_index(drop=True), test.reset_index(drop=True), protocol


def make_loader(arrays: dict[str, np.ndarray], indices: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(
        torch.from_numpy(arrays["clip"][indices]),
        torch.from_numpy(arrays["attr"][indices]),
        torch.from_numpy(arrays["phys"][indices]),
        torch.from_numpy(arrays["target"][indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def predict(model, arrays: dict[str, np.ndarray], batch_size: int, device: torch.device):
    model.eval()
    preds, rels, gates = [], [], []
    idx = np.arange(len(arrays["target"]))
    loader = make_loader(arrays, idx, batch_size, shuffle=False)
    with torch.no_grad():
        for clip, attr, phys, _ in loader:
            clip = clip.to(device)
            attr = attr.to(device)
            phys = phys.to(device)
            pred, rel, gate = model(clip, attr, phys)
            preds.append(pred.cpu().numpy())
            rels.append(rel.cpu().numpy())
            gates.append(gate.cpu().numpy())
    return np.concatenate(preds), np.concatenate(rels), np.concatenate(gates)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_df, test_df, protocol = build_frames(args)
    cols = feature_columns(train_df, test_df)
    needed = ["mos_norm"] + cols["clip"] + cols["attr"] + cols["phys"]
    train_df = train_df.dropna(subset=needed).reset_index(drop=True)
    test_df = test_df.dropna(subset=needed).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    all_indices = rng.permutation(len(train_df))
    val_size = max(1, int(round(len(train_df) * 0.15))) if len(train_df) > 8 else max(1, len(train_df) // 4)
    val_idx = all_indices[:val_size]
    fit_idx = all_indices[val_size:] if len(all_indices) > val_size else all_indices

    standardizer = BranchStandardizer()
    standardizer.fit(train_df.iloc[fit_idx], cols)
    train_arrays = standardizer.transform(train_df, cols)
    test_arrays = standardizer.transform(test_df, cols)

    model = ReliabilityAquaClip(
        clip_dim=len(cols["clip"]),
        attr_dim=len(cols["attr"]),
        phys_dim=len(cols["phys"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        **variant_config(args.variant),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    reg_loss = nn.SmoothL1Loss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    train_loader = make_loader(train_arrays, fit_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(train_arrays, val_idx, args.batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for clip, attr, phys, target in train_loader:
            clip = clip.to(device)
            attr = attr.to(device)
            phys = phys.to(device)
            target = target.to(device)
            pred, _, _, aux_pred = model(clip, attr, phys, return_aux=True)
            rank_weight = 0.0 if uses_regression_only_loss(args.variant) else args.rank_weight
            loss = reg_loss(pred, target) + rank_weight * pairwise_ranking_loss(pred, target)
            if aux_pred is not None and args.reliability_aux_weight > 0:
                aux_loss = reg_loss(aux_pred, target) + rank_weight * pairwise_ranking_loss(aux_pred, target)
                loss = loss + args.reliability_aux_weight * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(target)
            n_seen += len(target)

        model.eval()
        val_total = 0.0
        val_seen = 0
        with torch.no_grad():
            for clip, attr, phys, target in val_loader:
                clip = clip.to(device)
                attr = attr.to(device)
                phys = phys.to(device)
                target = target.to(device)
                pred, _, _ = model(clip, attr, phys)
                loss = reg_loss(pred, target)
                val_total += float(loss.cpu()) * len(target)
                val_seen += len(target)
        val_loss = val_total / max(1, val_seen)
        train_loss = total / max(1, n_seen)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred, train_rel, train_gate = predict(model, train_arrays, args.batch_size, device)
    test_pred, test_rel, test_gate = predict(model, test_arrays, args.batch_size, device)
    train_metrics = regression_metrics(train_arrays["target"], train_pred)
    test_metrics = regression_metrics(test_arrays["target"], test_pred)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.experiment_name

    result = {
        "experiment": stem,
        "protocol": protocol,
        "variant": args.variant,
        "device": str(device),
        "epochs_ran": len(history),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_dims": {k: len(v) for k, v in cols.items()},
        "few_shot_ratio": args.few_shot_ratio,
        "rank_weight": 0.0 if uses_regression_only_loss(args.variant) else args.rank_weight,
        "reliability_aux_weight": args.reliability_aux_weight if args.variant == "aquaclip-final" else 0.0,
        "model_role": "final-main-concat-with-reliability-explainer" if args.variant == "aquaclip-final" else args.variant,
        "train": train_metrics,
        "test": test_metrics,
        "reliability_mean_test": {
            "r_clip": float(test_rel[:, 0].mean()),
            "r_attr": float(test_rel[:, 1].mean()),
            "r_phys": float(test_rel[:, 2].mean()),
        },
    }

    pred_df = test_df[["image_path", "dataset", "mos", "mos_norm"]].copy()
    if "split" in test_df.columns:
        pred_df["split"] = test_df["split"].values
    pred_df["pred"] = test_pred
    pred_df["r_clip"] = test_rel[:, 0]
    pred_df["r_attr"] = test_rel[:, 1]
    pred_df["r_phys"] = test_rel[:, 2]
    for idx, col in enumerate(cols["attr"]):
        pred_df[f"gate_{col}"] = test_gate[:, idx]

    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir / f"{stem}_history.csv", index=False)
    pred_df.to_csv(output_dir / f"{stem}_predictions.csv", index=False)
    torch.save({"model": model.state_dict(), "standardizer": standardizer.stats, "columns": cols, "result": result}, output_dir / f"{stem}_model.pt")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
