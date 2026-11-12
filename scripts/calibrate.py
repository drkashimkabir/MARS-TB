#!/usr/bin/env python
"""Fit a temperature on VALIDATION predictions and (optionally) apply it to another prediction file.

Post-hoc calibration changes probabilities, so a threshold chosen before calibration no longer
applies: re-select it on calibrated validation probabilities. The manuscript reports no
post-hoc calibration, so this is an additional analysis, not a reproduction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mars_tb.evaluation.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    log_loss,
)


def _scores(df: pd.DataFrame, p: pd.Series) -> dict:
    y = df["actual_label"].astype(int).to_numpy()
    return {
        "ece": expected_calibration_error(y, p.to_numpy(float)),
        "brier": brier_score(y, p.to_numpy(float)),
        "nll": log_loss(y, p.to_numpy(float)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-predictions", required=True, type=Path)
    ap.add_argument("--apply-predictions", type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-calibrated", type=Path)
    args = ap.parse_args(argv)
    val = pd.read_csv(args.val_predictions).dropna(subset=["actual_label"])
    if set(val["split"]) != {"val"}:
        print("error: temperature must be fitted on validation predictions only", file=sys.stderr)
        return 3
    t = fit_temperature(val["actual_label"].astype(int).to_numpy(), val["p_tb"].to_numpy(float))
    report = {
        "temperature": t,
        "fitted_on": "val",
        "n_val": len(val),
        "val_before": _scores(val, val["p_tb"]),
        "val_after": _scores(val, pd.Series(apply_temperature(val["p_tb"].to_numpy(float), t))),
    }
    if args.apply_predictions:
        tgt = pd.read_csv(args.apply_predictions)
        cal = apply_temperature(tgt["p_tb"].to_numpy(float), t)
        lab = tgt.dropna(subset=["actual_label"])
        report["target_before"] = _scores(lab, lab["p_tb"])
        report["target_after"] = _scores(
            lab, pd.Series(apply_temperature(lab["p_tb"].to_numpy(float), t), index=lab.index)
        )
        report["target_dataset_split"] = sorted(set(map(str, tgt["dataset"] + ":" + tgt["split"])))
        if args.out_calibrated:
            tgt = tgt.assign(p_tb_uncalibrated=tgt["p_tb"], p_tb=cal)
            tgt.drop(columns=["predicted_label"]).to_csv(args.out_calibrated, index=False, lineterminator="\n")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"temperature {t:.4f}; val ECE {report['val_before']['ece']:.4f} -> {report['val_after']['ece']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
