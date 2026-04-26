"""Reliability-aware physics-calibrated fusion model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ReliabilityAquaClip(nn.Module):
    def __init__(
        self,
        clip_dim: int = 512,
        attr_dim: int = 7,
        phys_dim: int = 18,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_clip: bool = True,
        use_attr: bool = True,
        use_phys: bool = True,
        use_reliability: bool = True,
        reliability_for_prediction: bool | None = None,
        use_attr_gate: bool = True,
        hybrid_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.use_clip = use_clip
        self.use_attr = use_attr
        self.use_phys = use_phys
        self.use_reliability = use_reliability
        self.reliability_for_prediction = use_reliability if reliability_for_prediction is None else reliability_for_prediction
        self.use_attr_gate = use_attr_gate and use_attr and use_phys
        self.hybrid_fusion = hybrid_fusion
        self.attr_dim = attr_dim
        self.active_names = [
            name
            for name, enabled in [
                ("clip", use_clip),
                ("attr", use_attr),
                ("phys", use_phys),
            ]
            if enabled
        ]
        if not self.active_names:
            raise ValueError("At least one branch must be enabled.")

        self.attr_gate = (
            nn.Sequential(
                nn.Linear(phys_dim, 64),
                nn.GELU(),
                nn.Linear(64, attr_dim),
                nn.Sigmoid(),
            )
            if self.use_attr_gate
            else None
        )
        self.clip_proj = self._projector(clip_dim, hidden_dim, dropout) if use_clip else None
        self.attr_proj = self._projector(attr_dim, hidden_dim, dropout) if use_attr else None
        self.phys_proj = self._projector(phys_dim, hidden_dim, dropout) if use_phys else None
        if self.hybrid_fusion:
            self.reliability = nn.Sequential(
                nn.Linear(hidden_dim * len(self.active_names), hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.active_names)),
            )
            self.concat_fusion = nn.Sequential(
                nn.Linear(hidden_dim * len(self.active_names), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.hybrid_merge = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.reliability_aux_head = None
        elif self.use_reliability:
            self.reliability = nn.Sequential(
                nn.Linear(hidden_dim * len(self.active_names), hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.active_names)),
            )
            self.concat_fusion = (
                None
                if self.reliability_for_prediction
                else nn.Sequential(
                    nn.Linear(hidden_dim * len(self.active_names), hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.hybrid_merge = None
            self.reliability_aux_head = self._quality_head(hidden_dim, dropout) if not self.reliability_for_prediction else None
        else:
            self.reliability = None
            self.concat_fusion = nn.Sequential(
                nn.Linear(hidden_dim * len(self.active_names), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.hybrid_merge = None
            self.reliability_aux_head = None
        self.quality_head = self._quality_head(hidden_dim, dropout)

    @staticmethod
    def _quality_head(hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _projector(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, clip: torch.Tensor, attr: torch.Tensor, phys: torch.Tensor, return_aux: bool = False):
        branch_vectors = []
        if self.use_clip:
            branch_vectors.append(("clip", self.clip_proj(clip)))
        if self.use_attr:
            if self.attr_gate is not None:
                attr_gate = self.attr_gate(phys)
                attr = attr * attr_gate
            else:
                attr_gate = torch.ones_like(attr)
            branch_vectors.append(("attr", self.attr_proj(attr)))
        else:
            attr_gate = attr.new_zeros((attr.shape[0], self.attr_dim))
        if self.use_phys:
            branch_vectors.append(("phys", self.phys_proj(phys)))

        fused_input = torch.cat([z for _, z in branch_vectors], dim=-1)
        full_reliability = fused_input.new_zeros((fused_input.shape[0], 3))
        aux_pred = None
        if self.hybrid_fusion:
            active_reliability = torch.softmax(self.reliability(fused_input), dim=-1)
            z_rel = sum(active_reliability[:, idx : idx + 1] * branch_vectors[idx][1] for idx in range(len(branch_vectors)))
            z_concat = self.concat_fusion(fused_input)
            z = self.hybrid_merge(torch.cat([z_concat, z_rel], dim=-1))
            for idx, (name, _) in enumerate(branch_vectors):
                full_idx = {"clip": 0, "attr": 1, "phys": 2}[name]
                full_reliability[:, full_idx] = active_reliability[:, idx]
        elif self.reliability is not None:
            active_reliability = torch.softmax(self.reliability(fused_input), dim=-1)
            z_rel = sum(active_reliability[:, idx : idx + 1] * branch_vectors[idx][1] for idx in range(len(branch_vectors)))
            z = z_rel if self.reliability_for_prediction else self.concat_fusion(fused_input)
            if self.reliability_aux_head is not None:
                aux_pred = self.reliability_aux_head(z_rel).squeeze(-1)
            for idx, (name, _) in enumerate(branch_vectors):
                full_idx = {"clip": 0, "attr": 1, "phys": 2}[name]
                full_reliability[:, full_idx] = active_reliability[:, idx]
        else:
            z = self.concat_fusion(fused_input)
            for name, _ in branch_vectors:
                full_idx = {"clip": 0, "attr": 1, "phys": 2}[name]
                full_reliability[:, full_idx] = 1.0 / len(branch_vectors)

        pred = self.quality_head(z).squeeze(-1)
        if return_aux:
            return pred, full_reliability, attr_gate, aux_pred
        return pred, full_reliability, attr_gate


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_target = target[:, None] - target[None, :]
    sign = torch.sign(diff_target)
    mask = sign != 0
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    diff_pred = pred[:, None] - pred[None, :]
    return F.softplus(-sign[mask] * diff_pred[mask]).mean()
