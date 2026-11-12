"""Synthetic radiograph-like fixtures for CI and smoke runs.

These images are NOT clinical data and results on them are NOT experiments. Every row is written
with ``dataset='synthetic_fixture'`` and ``label_provenance='synthetic'`` so they cannot be
mistaken for a real cohort. "Positive" images contain bright blobs in the upper zones; the signal
is deliberately learnable so a smoke training run can be seen to learn.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _thorax(rng: np.random.Generator, size: int) -> np.ndarray:
    """Randomised thorax-like scene: random lung geometry, global gradient and rib-like stripes."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    angle = rng.uniform(0, 2 * np.pi)
    gradient = 0.12 * (np.cos(angle) * xx + np.sin(angle) * yy)
    img = 0.15 + gradient + 0.05 * rng.standard_normal((size, size)).astype(np.float32)
    rx, ry = rng.uniform(0.13, 0.21), rng.uniform(0.26, 0.38)
    gap = rng.uniform(0.18, 0.26)
    cy = rng.uniform(0.44, 0.56)
    level = rng.uniform(0.45, 0.65)
    for cx in (0.5 - gap, 0.5 + gap):
        lung = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0
        img[lung] = level + 0.03 * rng.standard_normal(lung.sum()).astype(np.float32)
        freq, phase = rng.uniform(18, 34), rng.uniform(0, 6.28)
        img[lung] += 0.06 * np.sin(freq * yy[lung] + phase)
    return img


def make_synthetic_dataset(root: str | Path, n_per_class: int = 12, size: int = 320, seed: int = 0) -> pd.DataFrame:
    """Write PNGs under ``root/images`` and return a manifest DataFrame (no split column)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for label in (0, 1):
        for i in range(n_per_class):
            img = _thorax(rng, size)
            if label == 1:
                for _ in range(int(rng.integers(2, 5))):
                    cy, cx = rng.uniform(0.28, 0.45), rng.choice([0.28, 0.72]) + rng.uniform(-0.05, 0.05)
                    cv2.circle(img, (int(cx * size), int(cy * size)), int(0.04 * size), (0.95,), -1)
            img = cv2.GaussianBlur(img, (0, 0), 1.2)
            u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            name = f"{'tb' if label else 'h'}{i:04d}.png"
            cv2.imwrite(str(root / "images" / name), u8)
            rows.append(
                {
                    "dataset": "synthetic_fixture",
                    "image_id": name[:-4],
                    "path": f"images/{name}",
                    "patient_id": f"synthetic-{label}-{i:04d}",
                    "label_binary": label,
                    "diagnostic_category": "active_tb" if label else "healthy",
                    "label_provenance": "synthetic",
                }
            )
    return pd.DataFrame(rows)
