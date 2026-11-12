#!/usr/bin/env python
"""Does evidential uncertainty flag errors? Error-detection AUROC/AUPRC, risk-coverage, selective accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mars_tb.evaluation.uncertainty import error_detection, risk_coverage, selective_accuracy, uncertainty_summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-curve", type=Path, help="risk-coverage CSV")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.predictions).dropna(subset=["actual_label"])
    if df["uncertainty"].isna().any():
        print("error: uncertainty is missing (softmax models have none); nothing to analyse", file=sys.stderr)
        return 2
    y, yhat, u = (
        df["actual_label"].astype(int).to_numpy(),
        df["predicted_label"].to_numpy(),
        df["uncertainty"].to_numpy(float),
    )
    rc = risk_coverage(y, yhat, u)
    report = {
        "n": len(df),
        "error_rate": float((y != yhat).mean()),
        **error_detection(y, yhat, u),
        **uncertainty_summary(y, yhat, u),
        "aurc": rc["aurc"],
        "selective_accuracy": {str(c): selective_accuracy(y, yhat, u, c) for c in (1.0, 0.9, 0.8, 0.7)},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.out_curve:
        pd.DataFrame({"coverage": rc["coverage"], "risk": rc["risk"]}).to_csv(args.out_curve, index=False)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
