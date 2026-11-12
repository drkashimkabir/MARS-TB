"""Multi-task evidential head, manuscript Eq. 12 (Sensoy et al., 2018 parameterisation).

    evidence e = softplus(logit),  alpha = e + 1,  S = sum_k alpha_k,
    p_k = alpha_k / S,             U = K / S.

Binary task: p_tb = alpha_pos / S, U = 2 / S.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mars_tb.config import HeadConfig

TASKS = ("tb", "pattern", "severity")


def evidential_outputs(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    evidence = F.softplus(logits.float())
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    k = alpha.shape[-1]
    return {
        "logits": logits,
        "alpha": alpha,
        "prob": alpha / strength,
        "uncertainty": k / strength.squeeze(-1),
    }


def softmax_outputs(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"logits": logits, "alpha": None, "prob": torch.softmax(logits.float(), -1), "uncertainty": None}  # type: ignore[dict-item]


class MultiTaskHead(nn.Module):
    """Three linear heads on z_d: TB (2), pattern (4), severity (4)."""

    def __init__(self, dim: int, cfg: HeadConfig):
        super().__init__()
        self.kind = cfg.type
        self.classes = {"tb": 2, "pattern": cfg.pattern_classes, "severity": cfg.severity_classes}
        self.heads = nn.ModuleDict({t: nn.Linear(dim, k) for t, k in self.classes.items()})

    def forward(self, z: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        fn = evidential_outputs if self.kind == "evidential" else softmax_outputs
        out = {t: fn(head(z)) for t, head in self.heads.items()}
        out["tb"]["p_tb"] = out["tb"]["prob"][:, 1]
        return out
