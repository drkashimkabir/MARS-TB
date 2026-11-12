"""Checkpoint save/load with an embedded manifest and SHA-256 sidecar.

Checkpoints hold tensors plus plain-Python metadata only, so they can be loaded with
``torch.load(weights_only=True)`` (no pickle code execution). Loading refuses a checkpoint whose
bytes do not match the recorded hash.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from mars_tb.config import ModelConfig, from_dict
from mars_tb.training.reproducibility import environment_record
from mars_tb.utils.io import sha256_file

FORMAT_VERSION = 1


class ChecksumError(RuntimeError):
    pass


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    *,
    epoch: int,
    seed: int,
    val_metrics: dict[str, float],
    threshold: float | None,
    preprocess: dict[str, Any],
    label_mapping: dict[str, Any],
    train_cfg: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write ``path`` and ``path.sha256``; return the checkpoint's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": dataclasses.asdict(model_cfg),
        "epoch": epoch,
        "seed": seed,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "threshold": threshold,
        "preprocess": preprocess,
        "label_mapping": label_mapping,
        "train_config": train_cfg,
        "environment": environment_record(),
        "extra": extra or {},
    }
    if optimizer is not None:
        blob["optimizer_state"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def load_checkpoint(path: str | Path, verify: bool = True) -> dict[str, Any]:
    path = Path(path)
    if verify:
        side = path.with_suffix(path.suffix + ".sha256")
        if not side.is_file():
            raise ChecksumError(f"missing checksum sidecar {side.name}; refusing to load")
        expected = side.read_text(encoding="utf-8").split()[0]
        if sha256_file(path) != expected:
            raise ChecksumError(f"{path.name} does not match its recorded SHA-256")
    return torch.load(path, map_location="cpu", weights_only=True)


def model_config_from_checkpoint(blob: dict[str, Any]) -> ModelConfig:
    return from_dict(ModelConfig, blob["model_config"])


def restore_model(path: str | Path, verify: bool = True) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Rebuild the architecture from the stored config and load weights. Returns (model, blob, sha)."""
    from mars_tb.models.baselines import build_baseline
    from mars_tb.models.mars_tb import MARSTB

    blob = load_checkpoint(path, verify=verify)
    baseline = blob.get("extra", {}).get("baseline")
    model: torch.nn.Module = build_baseline(baseline) if baseline else MARSTB(model_config_from_checkpoint(blob))
    model.load_state_dict(blob["model_state"], strict=True)
    return model, blob, sha256_file(path)
