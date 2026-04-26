"""Data helpers for AquaCLIP-QA tabular feature experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = {"image_path", "mos", "dataset", "split", "scene", "method", "water_type", "std"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    clip_path: str
    physics_path: str


def parse_dataset_spec(value: str) -> DatasetSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("Dataset spec must be name:clip_scores_csv:physics_csv")
    return DatasetSpec(parts[0], parts[1], parts[2])


def default_split(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42) -> pd.DataFrame:
    if "split" in df.columns and df["split"].notna().any():
        return df.copy()
    out = df.copy()
    test_idx = out.sample(frac=test_ratio, random_state=seed).index
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    return out


def prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([col for col in df.columns if col.startswith(prefix)])


def load_feature_table(spec: DatasetSpec) -> pd.DataFrame:
    clip = pd.read_csv(spec.clip_path)
    physics = pd.read_csv(spec.physics_path)
    physics = physics.drop(columns=["mos", "dataset", "split"], errors="ignore")
    df = clip.merge(physics, on="image_path", how="inner")
    df["dataset"] = spec.name
    df["mos"] = pd.to_numeric(df["mos"], errors="coerce")
    return df.dropna(subset=["mos"]).reset_index(drop=True)


def add_dataset_normalized_mos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mos = out["mos"].astype(float)
    value_range = float(mos.max() - mos.min())
    if value_range <= 0:
        raise ValueError("Cannot normalize a constant MOS range.")
    out["mos_norm"] = (mos - float(mos.min())) / value_range
    return out


def normalize_by_range(df: pd.DataFrame, min_value: float, max_value: float) -> pd.DataFrame:
    out = df.copy()
    value_range = max_value - min_value
    if value_range <= 0:
        raise ValueError("Cannot normalize a constant MOS range.")
    out["mos_norm"] = (out["mos"].astype(float) - min_value) / value_range
    return out


def sample_few_shot(train: pd.DataFrame, ratio: float | None, seed: int) -> pd.DataFrame:
    if ratio is None or ratio >= 1.0:
        return train
    if ratio <= 0:
        raise ValueError("few-shot ratio must be positive.")
    n = max(2, int(round(len(train) * ratio)))
    return train.sample(n=min(n, len(train)), random_state=seed).reset_index(drop=True)


def feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str]]:
    attr = sorted((set(prefixed_columns(train, "attr_")) & set(prefixed_columns(test, "attr_"))) - {"attr_mean"})
    phys = sorted(set(prefixed_columns(train, "phy_")) & set(prefixed_columns(test, "phy_")))
    clip = sorted(set(prefixed_columns(train, "clip_emb_")) & set(prefixed_columns(test, "clip_emb_")))
    if not clip or not attr or not phys:
        raise ValueError("Need CLIP embedding, attribute, and physics feature columns.")
    return {"clip": clip, "attr": attr, "phys": phys}


class BranchStandardizer:
    def __init__(self) -> None:
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df: pd.DataFrame, cols: dict[str, list[str]]) -> None:
        for branch, branch_cols in cols.items():
            values = df[branch_cols].to_numpy(dtype=np.float32)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std[std < 1e-6] = 1.0
            self.stats[branch] = (mean, std)

    def transform(self, df: pd.DataFrame, cols: dict[str, list[str]]) -> dict[str, np.ndarray]:
        out = {}
        for branch, branch_cols in cols.items():
            mean, std = self.stats[branch]
            values = df[branch_cols].to_numpy(dtype=np.float32)
            out[branch] = (values - mean) / std
        out["target"] = df["mos_norm"].to_numpy(dtype=np.float32)
        return out
