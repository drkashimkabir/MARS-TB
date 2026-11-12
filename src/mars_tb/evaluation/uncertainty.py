"""Uncertainty analysis for evidential outputs: does U actually flag the errors?"""

from __future__ import annotations

import numpy as np

from mars_tb.evaluation.metrics import NAN, average_precision, roc_auc


def error_detection(y: np.ndarray, yhat: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """AUROC/AUPRC of ``u`` as a score for 'this prediction is wrong'."""
    err = (np.asarray(y) != np.asarray(yhat)).astype(int)
    u = np.asarray(u, dtype=float)
    if err.sum() == 0 or err.sum() == len(err):
        return {"n_errors": int(err.sum()), "auroc": NAN, "auprc": NAN}
    return {"n_errors": int(err.sum()), "auroc": roc_auc(err, u), "auprc": average_precision(err, u)}


def risk_coverage(y: np.ndarray, yhat: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray | float]:
    """Keep the k most-certain predictions (lowest U); risk = error rate among kept."""
    err = (np.asarray(y) != np.asarray(yhat)).astype(float)
    order = np.argsort(np.asarray(u, dtype=float), kind="mergesort")
    cum_err = np.cumsum(err[order])
    k = np.arange(1, len(err) + 1)
    coverage, risk = k / len(err), cum_err / k
    return {"coverage": coverage, "risk": risk, "aurc": float(np.mean(risk))}


def selective_accuracy(y: np.ndarray, yhat: np.ndarray, u: np.ndarray, coverage: float) -> float:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    n_keep = max(1, int(round(coverage * len(y))))
    keep = np.argsort(np.asarray(u, dtype=float), kind="mergesort")[:n_keep]
    return float(np.mean(np.asarray(y)[keep] == np.asarray(yhat)[keep]))


def uncertainty_summary(y: np.ndarray, yhat: np.ndarray, u: np.ndarray) -> dict[str, float]:
    ok = np.asarray(y) == np.asarray(yhat)
    u = np.asarray(u, dtype=float)

    def stat(x: np.ndarray, fn) -> float:
        return float(fn(x)) if len(x) else NAN

    return {
        "mean_u_correct": stat(u[ok], np.mean),
        "mean_u_incorrect": stat(u[~ok], np.mean),
        "median_u_correct": stat(u[ok], np.median),
        "median_u_incorrect": stat(u[~ok], np.median),
    }
