"""Evidential and multi-task losses (manuscript Eq. 12).

    L_t = NLL(y, alpha) + lambda_anneal * KL( Dir(alpha~) || Dir(1) )
    alpha~ = y + (1 - y) * alpha        (misleading evidence only; Sensoy et al. 2018)
    L = 1.0 L_TB + 0.3 L_pattern + 0.3 L_severity,   lambda_anneal = min(1, epoch / 10)

Auxiliary targets use -1 for "no verified label". Such samples contribute NOTHING to the
corresponding task loss, so unavailable pattern/severity annotations are never imputed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mars_tb.config import LossConfig

IGNORE = -1


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ), per sample. Closed form, shape (B,)."""
    k = alpha.shape[-1]
    s = alpha.sum(-1, keepdim=True)
    first = torch.lgamma(s.squeeze(-1)) - torch.lgamma(torch.tensor(float(k), device=alpha.device))
    first = first - torch.lgamma(alpha).sum(-1)
    second = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(s))).sum(-1)
    return first + second


def evidential_nll(alpha: torch.Tensor, onehot: torch.Tensor, kind: str = "log") -> torch.Tensor:
    s = alpha.sum(-1, keepdim=True)
    if kind == "log":
        return (onehot * (torch.log(s) - torch.log(alpha))).sum(-1)
    if kind == "digamma":
        return (onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(-1)
    if kind == "mse":
        p = alpha / s
        var = p * (1 - p) / (s + 1)
        return ((onehot - p) ** 2 + var).sum(-1)
    raise ValueError(f"unknown nll type: {kind}")


def evidential_loss(alpha: torch.Tensor, target: torch.Tensor, kl_weight: float, nll_type: str = "log") -> torch.Tensor:
    """Mean loss over samples whose target != IGNORE; zero (with grad path) if none are valid."""
    valid = target != IGNORE
    if not valid.any():
        return alpha.sum() * 0.0
    a, t = alpha[valid], target[valid]
    onehot = F.one_hot(t, num_classes=a.shape[-1]).float()
    alpha_tilde = onehot + (1.0 - onehot) * a
    per = evidential_nll(a, onehot, nll_type) + kl_weight * dirichlet_kl_to_uniform(alpha_tilde)
    return per.mean()


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid].float(), target[valid])


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: LossConfig, head_type: str = "evidential"):
        super().__init__()
        self.cfg = cfg
        self.head_type = head_type
        self.tasks = ("tb", "pattern", "severity")

    def kl_weight(self, epoch: int) -> float:
        return min(1.0, epoch / self.cfg.kl_anneal_epochs)

    def forward(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = outputs["tb"]["logits"].new_zeros(()).float()
        parts: dict[str, float] = {}
        lam = self.kl_weight(epoch)
        for w, task in zip(self.cfg.weights, self.tasks, strict=True):
            if task not in outputs:  # single-task baselines expose only the TB task
                continue
            tgt = targets[task]
            if self.head_type == "evidential":
                loss = evidential_loss(outputs[task]["alpha"], tgt, lam, self.cfg.nll_type)
            else:
                loss = masked_cross_entropy(outputs[task]["logits"], tgt)
            parts[task] = float(loss.detach())
            total = total + w * loss
        parts["total"] = float(total.detach())
        return total, parts
