"""Calibration metrics for the positive-class probability, plus validation-fitted temperature."""

from __future__ import annotations

import numpy as np
from scipy import optimize

_EPS = 1e-12


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Equal-width bins over [0,1]: mean predicted p, observed positive rate and count per bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins)
    freq = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "count": count,
            "mean_pred": np.where(count > 0, conf / count, np.nan),
            "observed": np.where(count > 0, freq / count, np.nan),
        }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / N) |observed_b - mean_pred_b| for the positive-class probability."""
    r = reliability_curve(y, p, n_bins)
    mask = r["count"] > 0
    w = r["count"][mask] / r["count"].sum()
    return float(np.sum(w * np.abs(r["observed"][mask] - r["mean_pred"][mask])))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(np.asarray(p, dtype=float)) / temperature))


def fit_temperature(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Single temperature minimising validation NLL. NEVER fit on test or external labels."""
    y = np.asarray(y_val, dtype=float)

    def nll(t: float) -> float:
        return log_loss(y, apply_temperature(p_val, t))

    res = optimize.minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)
