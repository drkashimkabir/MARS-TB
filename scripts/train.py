#!/usr/bin/env python
"""Train MARS-TB (or an ablation / baseline) on a split CSV. Test rows are never loaded.

Single process:   python scripts/train.py --split-csv ... --root ... --out-dir runs/s42 --seed 42
Multi-GPU:        torchrun --nproc_per_node=2 scripts/train.py ...   (NOT exercised in CI)

The global optimiser batch is per-GPU batch x accumulation x world size; it is printed at start.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from mars_tb.config import (
    ModelConfig,
    TrainConfig,
    deep_merge,
    effective_batch_size,
    from_dict,
    load_train_config,
    load_yaml,
    validate_model_config,
)
from mars_tb.data.split_manager import read_split, verify_split
from mars_tb.models.ablations import get_ablation
from mars_tb.models.baselines import build_baseline
from mars_tb.pipeline import make_loader, pick_device
from mars_tb.training import distributed
from mars_tb.training.reproducibility import environment_record
from mars_tb.training.trainer import Trainer
from mars_tb.utils.seed import seed_everything

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "configs" / "model" / "mars_tb.yaml"
DEFAULT_TRAIN = Path(__file__).resolve().parents[1] / "configs" / "training" / "default.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-csv", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-config", type=Path, nargs="+", default=[DEFAULT_MODEL])
    ap.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--ablation", default=None, help="name from mars_tb.models.ablations")
    ap.add_argument("--baseline", default=None, help="ResNet-50 | DenseNet-121 | EfficientNet-B4 | ViT-B/16 | Swin-B")
    ap.add_argument("--baseline-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override (e.g. for a smoke run)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-unverified-split", action="store_true")
    args = ap.parse_args(argv)

    split_df, meta = read_split(args.split_csv)
    report = verify_split(split_df, meta)
    if not report.ok and not args.allow_unverified_split:
        print("split failed verification:\n  " + "\n  ".join(report.errors), file=sys.stderr)
        return 1

    raw: dict = {}
    for path in args.model_config:
        raw = deep_merge(raw, load_yaml(path))
    if args.ablation:
        raw = deep_merge(raw, dict(get_ablation(args.ablation)))
    model_cfg = from_dict(ModelConfig, raw)
    validate_model_config(model_cfg)
    train_cfg: TrainConfig = load_train_config(args.train_config)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
        overrides["warmup_epochs"] = min(train_cfg.warmup_epochs, max(0, args.epochs - 1))
    if args.baseline:
        overrides["lr_backbone"] = args.baseline_lr
    train_cfg = dataclasses.replace(train_cfg, **overrides)

    rank, world, device = distributed.setup()
    if args.device != "auto":
        device = pick_device(args.device)
    seed_everything(train_cfg.seed)
    if rank == 0:
        print(
            f"effective global batch = {effective_batch_size(train_cfg, world)} "
            f"({train_cfg.batch_size_per_gpu} x {train_cfg.grad_accum_steps} x {world} process(es))"
        )
        print("split provenance:", (meta or {}).get("provenance", "UNKNOWN (no sidecar)"))

    import torch

    from mars_tb.data.preprocessing import PreprocessConfig
    from mars_tb.models.mars_tb import MARSTB

    img = model_cfg.backbone.img_size
    pp = PreprocessConfig(size=img)
    tr = split_df[split_df["split"] == "train"]
    va = split_df[split_df["split"] == "val"]
    train_loader = make_loader(
        tr,
        args.root,
        "train",
        pp,
        train_cfg.batch_size_per_gpu,
        train=True,
        sampler_scheme=train_cfg.sampler_weights,
        rank=rank,
        world_size=world,
        seed=train_cfg.seed,
        num_workers=train_cfg.num_workers,
    )
    val_loader = make_loader(va, args.root, "val", pp, train_cfg.batch_size_per_gpu, num_workers=train_cfg.num_workers)
    model: torch.nn.Module = build_baseline(args.baseline) if args.baseline else MARSTB(model_cfg)
    trainer = Trainer(
        model,
        model_cfg,
        train_cfg,
        train_loader,
        val_loader,
        args.out_dir,
        device,
        preprocess=dataclasses.asdict(pp),
        world_size=world,
        is_main=(rank == 0),
    )
    summary = trainer.fit()
    if rank == 0:
        (args.out_dir / "environment.json").write_text(json.dumps(environment_record(), indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "train_config"}, indent=2, default=str))
    distributed.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
