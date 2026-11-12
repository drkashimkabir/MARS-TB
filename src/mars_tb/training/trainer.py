"""Training loop: gradient accumulation, AMP, clipping, validation-only model selection.

Model selection, early stopping and the operating threshold use the VALIDATION loader only.
The test and external loaders are never touched here.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mars_tb.config import ModelConfig, TrainConfig, effective_batch_size
from mars_tb.evaluation.metrics import roc_auc
from mars_tb.evaluation.threshold_selection import youden_threshold
from mars_tb.models.losses import MultiTaskLoss
from mars_tb.models.mars_tb import count_parameters
from mars_tb.training.checkpointing import save_checkpoint
from mars_tb.training.optimizer import build_optimizer, set_lr


class EarlyStopper:
    """Stops after ``patience`` epochs without a strict improvement in the monitored metric."""

    def __init__(self, patience: int):
        self.patience, self.best, self.bad_epochs = patience, -np.inf, 0

    def step(self, value: float) -> bool:
        """Record ``value``; return True if it is a new best."""
        if value > self.best:
            self.best, self.bad_epochs = value, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        out = model(batch["image"].to(device))["tb"]
        ps.append(out["p_tb"].float().cpu().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        out_dir: str | Path,
        device: torch.device | str = "cpu",
        preprocess: dict[str, Any] | None = None,
        world_size: int = 1,
        is_main: bool = True,
    ):
        self.model = model.to(device)
        self.model_cfg, self.cfg = model_cfg, train_cfg
        self.train_loader, self.val_loader = train_loader, val_loader
        self.out_dir = Path(out_dir)
        self.device = torch.device(device)
        self.preprocess = preprocess or {}
        self.world_size, self.is_main = world_size, is_main
        self.optimizer = build_optimizer(self.model, train_cfg)
        self.criterion = MultiTaskLoss(train_cfg.loss, model_cfg.head.type)
        self.use_amp = train_cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.stopper = EarlyStopper(train_cfg.early_stopping_patience)
        # Multi-GPU path (torchrun): NOT exercised in CI, see training/distributed.py.
        self.fwd: nn.Module = self.model
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ids = [self.device.index] if self.device.type == "cuda" else None
            self.fwd = DDP(self.model, device_ids=ids, find_unused_parameters=True)
        self.history: list[dict[str, Any]] = []
        self.extra = {"baseline": model.name} if hasattr(model, "net") and hasattr(model, "name") else {}
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        if self.is_main:
            with open(self.out_dir / "train_log.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=float) + "\n")

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accum = self.cfg.grad_accum_steps
        steps_per_epoch = max(1, len(self.train_loader) // accum)
        sums: dict[str, float] = {}
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(self.train_loader):
            batch = _to_device(batch, self.device)
            set_lr(self.optimizer, self.cfg, epoch + min(i // accum, steps_per_epoch) / steps_per_epoch)
            step_now = (i + 1) % accum == 0 or (i + 1) == len(self.train_loader)
            defer = self.world_size > 1 and not step_now
            skip_sync = cast(Any, self.fwd).no_sync() if defer else contextlib.nullcontext()
            with skip_sync:
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = self.fwd(batch["image"])
                targets = {"tb": batch["label"], "pattern": batch["pattern"], "severity": batch["severity"]}
                loss, parts = self.criterion(out, targets, epoch)
                self.scaler.scale(loss / accum).backward()
            if step_now:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1
        return {f"train_loss_{k}": v / max(n_batches, 1) for k, v in sums.items()}

    def fit(self) -> dict[str, Any]:
        """Train; returns a summary including the path/hash of the best-validation checkpoint."""
        seed = self.cfg.seed
        meta = {
            "effective_batch_size": effective_batch_size(self.cfg, self.world_size),
            "world_size": self.world_size,
            "parameters": count_parameters(self.model),
            "train_config": asdict(self.cfg),
        }
        if self.is_main:
            (self.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        best_path = self.out_dir / "best.ckpt"
        best_hash, best_epoch = "", -1
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            train_stats = self._train_epoch(epoch)
            y, p = validate(self.model, self.val_loader, self.device)
            auc = roc_auc(y, p)
            improved = self.stopper.step(auc if np.isfinite(auc) else -np.inf)
            self._log({"epoch": epoch, "val_auc": auc, "seconds": time.time() - t0, **train_stats})
            if improved and self.is_main:
                thr = youden_threshold(y, p)  # validation-only threshold
                best_hash = save_checkpoint(
                    best_path,
                    self.model,
                    self.model_cfg,
                    epoch=epoch,
                    seed=seed,
                    val_metrics={"auc": auc},
                    threshold=thr,
                    preprocess=self.preprocess,
                    label_mapping={"0": "TB-negative", "1": "TB-positive"},
                    train_cfg=asdict(self.cfg),
                    optimizer=self.optimizer,
                    extra=self.extra,
                )
                best_epoch = epoch
            if self.stopper.should_stop:
                break
        return {
            "best_val_auc": float(self.stopper.best),
            "best_epoch": best_epoch,
            "checkpoint": str(best_path),
            "checkpoint_sha256": best_hash,
            "epochs_run": len(self.history),
            **meta,
        }
