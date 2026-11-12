"""Preprocessing pipeline (manuscript Section 2.2 / Table 2), in the manuscript's order.

    1 format standardisation   2 bit-depth -> float [0,1]   3 quality filter
    4 CLAHE on uint8 [0,255] (clip 2.0, 8x8 tiles)           5 bicubic resize to 512x512 (no pad)
    6 grey -> 3 channels        7 ImageNet normalisation

Steps 1-6 are :func:`load_and_preprocess` (returns a float tensor in [0,1], 3xHxW); step 7 is
:func:`normalize` and is applied by the transform pipelines in ``augmentation.py``.

DICOM handling deviates from a literal reading of Table 2 on purpose: "DICOM -> 16-bit PNG then
/65535" would leave a 12-bit detector image occupying only 6 % of the unit range and CLAHE on the
resulting uint8 would be near-black. Here DICOM values are scaled by ``2**BitsStored - 1`` and
MONOCHROME1 images are inverted so that bone/air polarity matches MONOCHROME2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageRejected(ValueError):
    """Image failed the quality filter (Table 2, step 3)."""


@dataclass(frozen=True)
class PreprocessConfig:
    size: int = 512
    clahe_clip: float = 2.0
    clahe_tiles: tuple[int, int] = (8, 8)
    min_edge: int = 256
    aspect_min: float = 0.5
    aspect_max: float = 2.0
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def preprocess_config_from_dict(d: dict[str, Any]) -> PreprocessConfig:
    """Rebuild a PreprocessConfig from a checkpoint dict (lists -> tuples)."""
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    return PreprocessConfig(**kwargs)  # type: ignore[arg-type]


def _scale_to_unit(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    raise ValueError(f"unsupported pixel dtype {arr.dtype}")


def read_dicom(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    import pydicom  # local import: optional at runtime for non-DICOM users

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    bits = int(getattr(ds, "BitsStored", 16))
    full = float(2**bits - 1)
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        arr = full - arr
    # Only pixel statistics are kept; no patient metadata leaves this function.
    meta = {"bits_stored": bits, "photometric": str(getattr(ds, "PhotometricInterpretation", ""))}
    return np.clip(arr / full, 0.0, 1.0), meta


def read_image(path: str | Path) -> np.ndarray:
    """Read PNG/JPEG/DICOM to a single-channel float32 array in [0, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return read_dicom(path)[0]
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"could not decode image: {path}")
    if raw.ndim == 3:  # BGR(A) -> luminance, preserving dtype
        raw = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY)
    return _scale_to_unit(raw)


def quality_check(height: int, width: int, cfg: PreprocessConfig = PreprocessConfig()) -> None:
    if min(height, width) < cfg.min_edge:
        raise ImageRejected(f"shorter edge {min(height, width)} < {cfg.min_edge}")
    ratio = width / height
    if ratio < cfg.aspect_min or ratio > cfg.aspect_max:
        raise ImageRejected(f"aspect ratio {ratio:.2f} outside [{cfg.aspect_min}, {cfg.aspect_max}]")


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def apply_clahe(img_u8: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise ValueError("CLAHE input must be uint8 in [0,255]")
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_tiles)
    return clahe.apply(img_u8)


def resize_bicubic(img_u8: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_CUBIC)


def preprocess_array(img: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> torch.Tensor:
    """Steps 3-6 on a float [0,1] 2-D array -> float32 tensor (3, size, size) in [0, 1]."""
    if img.ndim != 2:
        raise ValueError("expected a single-channel 2-D image")
    quality_check(img.shape[0], img.shape[1], cfg)
    u8 = resize_bicubic(apply_clahe(to_uint8(img), cfg), cfg.size)
    gray = torch.from_numpy(u8.astype(np.float32) / 255.0)
    return gray.unsqueeze(0).repeat(3, 1, 1)


def load_and_preprocess(path: str | Path, cfg: PreprocessConfig = PreprocessConfig()) -> torch.Tensor:
    return preprocess_array(read_image(path), cfg)


def normalize(x: torch.Tensor, cfg: PreprocessConfig = PreprocessConfig()) -> torch.Tensor:
    mean = torch.tensor(cfg.mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(cfg.std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std
