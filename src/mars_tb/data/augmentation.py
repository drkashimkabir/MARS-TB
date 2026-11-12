"""Training-only augmentation (manuscript Section 2.2, Table 2 steps 8a-8d).

Order: flip -> rotate -> colour jitter on the [0,1] tensor, then ImageNet normalisation, then
RandomErasing. Erasing happens after normalisation, so ``value=0`` is exactly the channel mean
("channel-mean fill value"). Validation and test inputs receive ONLY normalisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T

from mars_tb.data.preprocessing import PreprocessConfig, normalize

TRAIN_SPLITS = frozenset({"train"})
EVAL_SPLITS = frozenset({"val", "test", "external"})


def rotate_bicubic(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a (C,H,W) tensor about its centre with bicubic sampling and zero fill.

    torchvision's tensor rotate supports only nearest/bilinear, so the manuscript's "bicubic,
    fill 0" rotation is implemented directly with ``grid_sample``. Bicubic can overshoot, so
    callers clamp to [0, 1] afterwards.
    """
    c, h, w = x.shape
    theta = math.radians(angle_deg)
    mat = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0]],
        dtype=x.dtype,
    ).unsqueeze(0)
    grid = F.affine_grid(mat, [1, c, h, w], align_corners=False)
    out = F.grid_sample(x.unsqueeze(0), grid, mode="bicubic", padding_mode="zeros", align_corners=False)
    return out[0]


class RandomRotationBicubic:
    """Uniform random rotation in [-degrees, +degrees] (bicubic, fill 0)."""

    def __init__(self, degrees: float = 15.0):
        self.degrees = (-float(degrees), float(degrees))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        angle = float(torch.empty(1).uniform_(*self.degrees))
        return rotate_bicubic(x, angle)


class EvalTransform:
    """Deterministic: normalisation only."""

    def __init__(self, cfg: PreprocessConfig = PreprocessConfig()):
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return normalize(x, self.cfg)


class TrainTransform:
    """Stochastic augmentation; uses the global torch RNG (seeded per run / worker)."""

    def __init__(self, cfg: PreprocessConfig = PreprocessConfig()):
        self.cfg = cfg
        self.geometric = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                RandomRotationBicubic(15.0),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
            ]
        )
        self.erase = T.RandomErasing(p=0.1, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geometric(x).clamp_(0.0, 1.0)
        return self.erase(normalize(x, self.cfg))


def build_transform(split: str, cfg: PreprocessConfig = PreprocessConfig()):
    if split in TRAIN_SPLITS:
        return TrainTransform(cfg)
    if split in EVAL_SPLITS:
        return EvalTransform(cfg)
    raise ValueError(f"unknown split {split!r}; augmentation is only defined for {sorted(TRAIN_SPLITS | EVAL_SPLITS)}")
