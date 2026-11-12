"""AdamW with separate learning rates for LoRA / new modules / backbone, and the LR schedule."""

from __future__ import annotations

import math

import torch

from mars_tb.config import TrainConfig


def lr_factor_schedule(
    epoch_float: float, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float
) -> float:
    """Absolute LR at a (fractional) epoch: linear warmup 0 -> base, then cosine base -> min_lr."""
    if base_lr <= 0:
        return 0.0
    if epoch_float < warmup_epochs:
        return base_lr * epoch_float / warmup_epochs
    progress = (epoch_float - warmup_epochs) / max(1e-9, total_epochs - warmup_epochs)
    progress = min(1.0, progress)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer != "adamw":
        raise ValueError("only adamw is specified by the manuscript")
    groups = model.parameter_groups()  # type: ignore[operator]
    spec = [
        ("lora", cfg.lr_lora),
        ("new", cfg.lr_new),
        ("backbone", cfg.lr_backbone),
    ]
    param_groups = []
    for name, lr in spec:
        params = groups[name]
        if not params:
            continue
        if name == "backbone" and lr == 0.0:
            raise ValueError("trainable backbone parameters with lr_backbone=0 would never move")
        param_groups.append({"params": params, "lr": lr, "base_lr": lr, "name": name})
    if not param_groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(param_groups, betas=cfg.betas, weight_decay=cfg.weight_decay)


def set_lr(optimizer: torch.optim.Optimizer, cfg: TrainConfig, epoch_float: float) -> dict[str, float]:
    out = {}
    for g in optimizer.param_groups:
        g["lr"] = lr_factor_schedule(epoch_float, cfg.warmup_epochs, cfg.epochs, g["base_lr"], cfg.min_lr)
        out[g["name"]] = g["lr"]
    return out
