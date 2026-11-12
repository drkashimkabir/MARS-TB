"""Operating-threshold selection. Thresholds are chosen on VALIDATION data only and then frozen."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mars_tb.evaluation.metrics import roc_curve
from mars_tb.utils.io import read_json, write_json_atomic

SELECTABLE_SPLITS = frozenset({"val"})


class ThresholdLeakageError(RuntimeError):
    """Raised when a threshold would be tuned on test or external labels."""


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    finite = np.isfinite(thr)
    best = int(np.argmax(np.where(finite, j, -np.inf)))
    return float(thr[best])


def sensitivity_constrained_threshold(y: np.ndarray, p: np.ndarray, min_sensitivity: float) -> float:
    """Highest threshold (hence best specificity) that still reaches ``min_sensitivity``."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where((tpr >= min_sensitivity) & np.isfinite(thr))[0]
    if len(ok) == 0:
        raise ValueError(f"no threshold reaches sensitivity {min_sensitivity}")
    return float(thr[ok[0]])


@dataclass(frozen=True)
class FrozenThreshold:
    value: float
    method: str
    selected_on_dataset: str
    selected_on_split: str
    n: int
    image_id_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_threshold(
    y: np.ndarray,
    p: np.ndarray,
    image_ids: list[str],
    dataset: str,
    split: str,
    method: str = "youden",
    min_sensitivity: float = 0.9,
    allow_exploratory: bool = False,
) -> FrozenThreshold:
    if split not in SELECTABLE_SPLITS and not allow_exploratory:
        raise ThresholdLeakageError(
            f"refusing to select a threshold on split {split!r}; use the validation split "
            "(pass allow_exploratory=True only for clearly-labelled exploratory analyses)"
        )
    if method == "youden":
        value = youden_threshold(y, p)
    elif method == "sensitivity":
        value = sensitivity_constrained_threshold(y, p, min_sensitivity)
    else:
        raise ValueError(f"unknown method {method!r}")
    digest = hashlib.sha256("\n".join(sorted(image_ids)).encode()).hexdigest()
    return FrozenThreshold(value, method, dataset, split, len(y), digest)


def save_threshold(path: str | Path, thr: FrozenThreshold) -> None:
    write_json_atomic(path, thr.to_dict())


def load_threshold(path: str | Path) -> FrozenThreshold:
    return FrozenThreshold(**read_json(path))
