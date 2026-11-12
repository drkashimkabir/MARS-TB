#!/usr/bin/env python
"""ROC, reliability, risk-coverage and uncertainty figures from a prediction file (PNG)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mars_tb.evaluation.calibration import expected_calibration_error, reliability_curve  # noqa: E402
from mars_tb.evaluation.metrics import binary_report, roc_curve  # noqa: E402
from mars_tb.evaluation.statistical_tests import delong_auc_ci  # noqa: E402
from mars_tb.evaluation.uncertainty import risk_coverage  # noqa: E402


def roc_figure(df: pd.DataFrame, path: Path) -> None:
    y, p = df["actual_label"].astype(int).to_numpy(), df["p_tb"].to_numpy(float)
    fpr, tpr, _ = roc_curve(y, p)
    ci = delong_auc_ci(y, p)
    rep = binary_report(y, p, float(df["threshold"].iloc[0]))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC {ci['auc']:.3f} ({ci['lo']:.3f}-{ci['hi']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    ax.scatter(
        [1 - rep["specificity"]],
        [rep["sensitivity"]],
        zorder=3,
        label=f"threshold {rep['threshold']:.2f}: Se {rep['sensitivity']:.3f}, Sp {rep['specificity']:.3f}",
    )
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{df['dataset'].iloc[0]} / {df['split'].iloc[0]}",
    )
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def reliability_figure(df: pd.DataFrame, path: Path) -> None:
    y, p = df["actual_label"].astype(int).to_numpy(), df["p_tb"].to_numpy(float)
    r = reliability_curve(y, p, 10)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(r["mean_pred"], r["observed"], "o-", label=f"ECE {expected_calibration_error(y, p, 10):.3f}")
    ax.set(xlabel="Mean predicted P(TB)", ylabel="Observed TB frequency", title="Reliability")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def uncertainty_figures(df: pd.DataFrame, out: Path) -> None:
    y, yhat, u = (
        df["actual_label"].astype(int).to_numpy(),
        df["predicted_label"].to_numpy(),
        df["uncertainty"].to_numpy(float),
    )
    rc = risk_coverage(y, yhat, u)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(rc["coverage"], rc["risk"])
    ax.set(xlabel="Coverage", ylabel="Error rate on retained cases", title=f"Risk-coverage (AURC {rc['aurc']:.4f})")
    fig.savefig(out / "risk_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ok = y == yhat
    bins = np.linspace(0, u.max(), 25)
    ax.hist(u[ok], bins=bins, alpha=0.6, label="correct", density=True)
    if (~ok).any():
        ax.hist(u[~ok], bins=bins, alpha=0.6, label="incorrect", density=True)
    ax.set(xlabel="Evidential uncertainty U", ylabel="Density")
    ax.legend()
    fig.savefig(out / "uncertainty_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args(argv)
    df = pd.read_csv(args.predictions).dropna(subset=["actual_label"])
    if df["actual_label"].nunique() < 2:
        print("error: need both classes to draw ROC/reliability figures", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roc_figure(df, args.out_dir / "roc.png")
    reliability_figure(df, args.out_dir / "reliability.png")
    if df["uncertainty"].notna().all():
        uncertainty_figures(df, args.out_dir)
    print(f"wrote figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
