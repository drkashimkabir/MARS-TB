"""Glue used by the command-line scripts (kept in the package so it is unit-testable)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mars_tb.config import ModelConfig, TrainConfig
from mars_tb.data.augmentation import EVAL_SPLITS
from mars_tb.data.datasets import CXRDataset, DistributedWeightedSampler, make_weighted_sampler
from mars_tb.data.metadata import validate_manifest, validate_predictions
from mars_tb.data.preprocessing import PreprocessConfig, preprocess_config_from_dict
from mars_tb.evaluation.evaluator import predict
from mars_tb.models.mars_tb import MARSTB
from mars_tb.training.checkpointing import restore_model
from mars_tb.training.trainer import Trainer
from mars_tb.utils.seed import seed_everything, worker_init_fn


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(
    df: pd.DataFrame,
    root: str | Path,
    split: str,
    pp: PreprocessConfig,
    batch_size: int,
    *,
    train: bool = False,
    sampler_scheme: str = "neg_over_pos",
    seed: int = 0,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    validate: bool = True,
) -> DataLoader:
    ds = CXRDataset(df, root, split if train else (split if split in EVAL_SPLITS else "test"), pp, validate=validate)
    if not train:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = df["label_binary"].to_numpy()
    sampler = (
        DistributedWeightedSampler(labels, sampler_scheme, rank, world_size, seed)
        if world_size > 1
        else make_weighted_sampler(labels, sampler_scheme, seed)
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def train_from_split(
    split_df: pd.DataFrame,
    root: str | Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Train on rows with split=='train', select on split=='val'. 'test' rows are never loaded."""
    seed_everything(train_cfg.seed)
    pp = PreprocessConfig(size=model_cfg.backbone.img_size)
    tr = split_df[split_df["split"] == "train"]
    va = split_df[split_df["split"] == "val"]
    if tr.empty or va.empty:
        raise ValueError("split table needs non-empty 'train' and 'val' rows")
    dev = pick_device(device)
    train_loader = make_loader(
        tr,
        root,
        "train",
        pp,
        train_cfg.batch_size_per_gpu,
        train=True,
        sampler_scheme=train_cfg.sampler_weights,
        seed=train_cfg.seed,
        num_workers=train_cfg.num_workers,
    )
    val_loader = make_loader(va, root, "val", pp, train_cfg.batch_size_per_gpu, num_workers=train_cfg.num_workers)
    trainer = Trainer(
        MARSTB(model_cfg),
        model_cfg,
        train_cfg,
        train_loader,
        val_loader,
        out_dir,
        dev,
        preprocess=dataclasses.asdict(pp),
    )
    return trainer.fit()


def predict_from_checkpoint(
    ckpt: str | Path,
    manifest: pd.DataFrame,
    root: str | Path,
    *,
    split: str,
    dataset: str,
    model_name: str = "MARS-TB",
    threshold: float | None = None,
    batch_size: int = 8,
    device: str = "auto",
    num_workers: int = 0,
    verify_checksum: bool = True,
    labelled: bool = True,
) -> pd.DataFrame:
    """Run inference and return a schema-conformant prediction table.

    Threshold defaults to the validation-selected value stored in the checkpoint; it is never
    recomputed on the data being predicted.
    """
    if labelled:  # unlabelled inputs only need safe, existing paths (checked when each image is read)
        problems = validate_manifest(manifest, root)
        if problems:
            raise ValueError("invalid manifest:\n  " + "\n  ".join(problems))
    model, blob, sha = restore_model(ckpt, verify=verify_checksum)
    thr = blob["threshold"] if threshold is None else threshold
    if thr is None:
        raise ValueError("checkpoint carries no validation threshold; pass one explicitly")
    pp = preprocess_config_from_dict(blob["preprocess"])
    dev = pick_device(device)
    loader = make_loader(manifest, root, split, pp, batch_size, num_workers=num_workers, validate=labelled)
    model.to(dev)
    df = predict(
        model,
        loader,
        dev,
        dataset=dataset,
        split=split,
        model_name=model_name,
        seed=blob["seed"],
        threshold=float(thr),
        checkpoint_hash=sha,
    )
    bad = validate_predictions(df)
    if bad:
        raise RuntimeError("exported predictions failed schema validation:\n  " + "\n  ".join(bad))
    return df
