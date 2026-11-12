"""PyTorch dataset over a validated manifest, plus class-balancing sampler utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from mars_tb.data.augmentation import build_transform
from mars_tb.data.label_mapping import (
    MISSING_AUX,
    PATTERN_CLASSES,
    SEVERITY_CLASSES,
    encode_aux,
)
from mars_tb.data.metadata import validate_manifest
from mars_tb.data.preprocessing import PreprocessConfig, load_and_preprocess
from mars_tb.utils.io import safe_join


class CXRDataset(Dataset):
    """Returns the normalised image, binary label, auxiliary labels (-1 if unverified) and ids."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        root: str | Path,
        split: str,
        cfg: PreprocessConfig = PreprocessConfig(),
        validate: bool = True,
    ):
        if validate:
            problems = validate_manifest(manifest, root)
            if problems:
                raise ValueError("invalid manifest:\n  " + "\n  ".join(problems))
        self.df = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.cfg = cfg
        self.transform = build_transform(split, cfg)
        n = len(self.df)
        # -1 marks "no reference label" (inference on unlabelled images); never confused with a class
        self.labels = (
            [int(v) if pd.notna(v) else -1 for v in self.df["label_binary"]] if "label_binary" in self.df else [-1] * n
        )
        self.pattern = (
            encode_aux(self.df["pattern_label"], PATTERN_CLASSES) if "pattern_label" in self.df else [MISSING_AUX] * n
        )
        self.severity = (
            encode_aux(self.df["severity_label"], SEVERITY_CLASSES)
            if "severity_label" in self.df
            else [MISSING_AUX] * n
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.df.iloc[i]
        img = load_and_preprocess(safe_join(self.root, row["path"]), self.cfg)
        patient = row["patient_id"] if "patient_id" in self.df and pd.notna(row.get("patient_id")) else ""
        return {
            "image": self.transform(img),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "pattern": torch.tensor(self.pattern[i], dtype=torch.long),
            "severity": torch.tensor(self.severity[i], dtype=torch.long),
            "image_id": str(row["image_id"]),
            "patient_id": str(patient),
        }


def class_sampling_weights(labels: np.ndarray, scheme: str = "neg_over_pos") -> dict[int, float]:
    """Per-class sampling weights derived from the supplied (training) labels.

    ``neg_over_pos``: positives get n_neg / n_pos, negatives 1.0, mirroring the manuscript's
    "4.50 = 9085 / 2019" recipe applied to the labels actually in the training split. For the
    revised cohort (6,000 / 714) this is 8.40, NOT the manuscript's stale 4.50.
    ``fixed:<w>``: positives get ``w`` (use to run the stale 4.50 deliberately).
    """
    labels = np.asarray(labels)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training labels must contain both classes")
    if scheme == "neg_over_pos":
        return {0: 1.0, 1: n_neg / n_pos}
    if scheme.startswith("fixed:"):
        return {0: 1.0, 1: float(scheme.split(":", 1)[1])}
    raise ValueError(f"unknown sampler weight scheme {scheme!r}")


def make_weighted_sampler(labels: np.ndarray, scheme: str = "neg_over_pos", seed: int = 0) -> WeightedRandomSampler:
    w = class_sampling_weights(labels, scheme)
    weights = torch.tensor([w[int(y)] for y in labels], dtype=torch.double)
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(labels), replacement=True, generator=gen)


class DistributedWeightedSampler(torch.utils.data.Sampler[int]):
    """Weighted sampling that gives each rank a disjoint, equally sized slice of one global draw."""

    def __init__(self, labels: np.ndarray, scheme: str, rank: int, world_size: int, seed: int = 0):
        w = class_sampling_weights(labels, scheme)
        self.weights = torch.tensor([w[int(y)] for y in labels], dtype=torch.double)
        self.rank, self.world, self.seed, self.epoch = rank, world_size, seed, 0
        self.num_samples = len(labels) // world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        draw = torch.multinomial(self.weights, self.num_samples * self.world, True, generator=gen)
        return iter(draw[self.rank :: self.world].tolist())
