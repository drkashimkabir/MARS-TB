"""Statistical inference: DeLong, bootstrap (optionally patient-clustered), Holm, seed summaries.

Every function here operates on supplied predictions. None synthesises a plausible-looking number.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats

from mars_tb.evaluation.metrics import NAN, roc_auc


def _fast_delong(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014). ``scores`` is (k, N) with the positives in the first ``n_pos`` columns."""
    k, n_total = scores.shape
    n_neg = n_total - n_pos
    pos, neg = scores[:, :n_pos], scores[:, n_pos:]
    tx = np.vstack([stats.rankdata(r, method="average") for r in pos])
    ty = np.vstack([stats.rankdata(r, method="average") for r in neg])
    tz = np.vstack([stats.rankdata(r, method="average") for r in scores])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / n_pos + sy / n_neg


def _prepare(y: np.ndarray, *scores: np.ndarray) -> tuple[np.ndarray, int]:
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        raise ValueError("DeLong needs both classes")
    order = np.argsort(-y, kind="mergesort")
    mat = np.vstack([np.asarray(s, dtype=float)[order] for s in scores])
    return mat, int(y.sum())


def delong_auc_ci(y: np.ndarray, s: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    mat, n_pos = _prepare(y, s)
    aucs, cov = _fast_delong(mat, n_pos)
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc": float(aucs[0]),
        "se": se,
        "lo": max(0.0, float(aucs[0] - z * se)),
        "hi": min(1.0, float(aucs[0] + z * se)),
    }


def delong_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Paired two-sided DeLong test of AUC(s1) - AUC(s2) on the same cases."""
    mat, n_pos = _prepare(y, s1, s2)
    aucs, cov = _fast_delong(mat, n_pos)
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast)
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {
            "auc1": float(aucs[0]),
            "auc2": float(aucs[1]),
            "diff": diff,
            "se": 0.0,
            "z": NAN,
            "p": NAN,
            "lo": diff,
            "hi": diff,
        }
    se = float(np.sqrt(var))
    z = diff / se
    zc = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc1": float(aucs[0]),
        "auc2": float(aucs[1]),
        "diff": diff,
        "se": se,
        "z": float(z),
        "p": float(2 * stats.norm.sf(abs(z))),
        "lo": diff - zc * se,
        "hi": diff + zc * se,
    }


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values in the input order (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def _resample_indices(
    y: np.ndarray, rng: np.random.Generator, groups: np.ndarray | None, stratified: bool
) -> np.ndarray:
    if groups is not None:
        uniq, inverse = np.unique(groups, return_inverse=True)
        members = [np.where(inverse == i)[0] for i in range(len(uniq))]
        pick = rng.integers(0, len(uniq), len(uniq))
        return np.concatenate([members[i] for i in pick])
    if stratified:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    return rng.integers(0, len(y), len(y))


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Percentile bootstrap CI. With ``groups`` (e.g. patient ids) whole clusters are resampled,
    which respects within-patient correlation; without them images are treated as independent."""
    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(seed)
    point = metric(y, scores)
    stats_: list[float] = []
    for _ in range(n_boot):
        idx = _resample_indices(y, rng, None if groups is None else np.asarray(groups), groups is None)
        yb = y[idx]
        if len(set(yb.tolist())) < 2:
            continue
        stats_.append(metric(yb, scores[idx]))
    arr = np.array(stats_)
    lo, hi = np.nanpercentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(arr) else (NAN, NAN)
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(len(arr)),
        "clustered": groups is not None,
    }


def paired_bootstrap_diff(
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    y, s1, s2 = np.asarray(y), np.asarray(s1), np.asarray(s2)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_indices(y, rng, None if groups is None else np.asarray(groups), groups is None)
        if len(set(y[idx].tolist())) < 2:
            continue
        diffs.append(metric(y[idx], s1[idx]) - metric(y[idx], s2[idx]))
    arr = np.array(diffs)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
    return {
        "diff": float(metric(y, s1) - metric(y, s2)),
        "lo": float(lo),
        "hi": float(hi),
        "p_bootstrap": float(min(1.0, p_two)),
        "n_boot": int(len(arr)),
    }


def seed_summary(values: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
    """Mean, sample SD and t-interval over independent training seeds."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        raise ValueError("need at least two seeds")
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": float(v.mean()), "sd": sd, "lo": float(v.mean() - half), "hi": float(v.mean() + half)}
