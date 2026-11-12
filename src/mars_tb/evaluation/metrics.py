"""Binary classification metrics computed from actual predictions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

NAN = float("nan")


def _check(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y and scores must be 1-D arrays of equal length")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels must be 0/1")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return y.astype(int), s


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; NaN if only one class is present."""
    y, s = _check(y, s)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return NAN
    ranks = stats.rankdata(s, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds); thresholds descending, starting above the maximum score."""
    y, s = _check(y, s)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(y_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = 1 + idx - tps
    tpr = np.r_[0.0, tps / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps / max(fps[-1], 1)]
    thr = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thr


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the precision-recall step curve (the scikit-learn definition)."""
    y, s = _check(y, s)
    if y.sum() == 0:
        return NAN
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    idx = np.r_[np.where(np.diff(s_sorted))[0], len(y_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    prec = tps / (1 + idx)
    rec = tps / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def confusion(y: np.ndarray, yhat: np.ndarray) -> dict[str, int]:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    return {
        "tn": int(((y == 0) & (yhat == 0)).sum()),
        "fp": int(((y == 0) & (yhat == 1)).sum()),
        "fn": int(((y == 1) & (yhat == 0)).sum()),
        "tp": int(((y == 1) & (yhat == 1)).sum()),
    }


def _div(a: float, b: float) -> float:
    return a / b if b else NAN


def mcc(tp: float, tn: float, fp: float, fn: float) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else NAN


def mcc_from_rates(sensitivity: float, specificity: float, n_pos: int, n_neg: int) -> float:
    """MCC implied by rates and class counts (used to audit tables, not for model evaluation)."""
    tp, fn = sensitivity * n_pos, (1 - sensitivity) * n_pos
    tn, fp = specificity * n_neg, (1 - specificity) * n_neg
    return mcc(tp, tn, fp, fn)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return NAN, NAN
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def binary_report(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Threshold-dependent and threshold-free metrics with exact binomial intervals."""
    y, p = _check(y, p)
    yhat = (p >= threshold).astype(int)
    c = confusion(y, yhat)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n_pos, n_neg = tp + fn, tn + fp
    sens, spec = _div(tp, n_pos), _div(tn, n_neg)
    ppv, npv = _div(tp, tp + fp), _div(tn, tn + fn)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    return {
        **c,
        "n": len(y),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": _div(n_pos, len(y)),
        "threshold": float(threshold),
        "roc_auc": roc_auc(y, p),
        "pr_auc": average_precision(y, p),
        "accuracy": _div(tp + tn, len(y)),
        "sensitivity": sens,
        "sensitivity_ci": clopper_pearson(tp, n_pos),
        "specificity": spec,
        "specificity_ci": clopper_pearson(tn, n_neg),
        "ppv": ppv,
        "npv": npv,
        "precision": ppv,
        "recall": sens,
        "f1": f1,
        "mcc": mcc(tp, tn, fp, fn),
    }
