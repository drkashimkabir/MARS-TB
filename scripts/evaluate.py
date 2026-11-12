#!/usr/bin/env python
"""Evaluate a checkpoint on a labelled manifest (internal test or an external cohort).

The operating threshold is the validation-selected value stored in the checkpoint (or a frozen
threshold file from select_threshold.py); it is never re-estimated here. Writes the prediction
export and a metrics JSON containing DeLong and (patient-clustered if ids exist) bootstrap CIs,
calibration and, for evidential models, uncertainty analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mars_tb.evaluation.calibration import brier_score, expected_calibration_error, log_loss
from mars_tb.evaluation.evaluator import metrics_from_predictions, prediction_arrays
from mars_tb.evaluation.statistical_tests import bootstrap_ci, delong_auc_ci
from mars_tb.evaluation.threshold_selection import load_threshold
from mars_tb.evaluation.uncertainty import error_detection, risk_coverage, uncertainty_summary
from mars_tb.pipeline import predict_from_checkpoint


def summarise(preds: pd.DataFrame, n_boot: int, seed: int) -> dict:
    y, p = prediction_arrays(preds)
    out = metrics_from_predictions(preds)
    out["auc_delong"] = delong_auc_ci(y, p)
    groups = preds["patient_id"].to_numpy() if preds["patient_id"].notna().all() else None
    out["auc_bootstrap"] = bootstrap_ci(y, p, n_boot=n_boot, seed=seed, groups=groups)
    out["calibration"] = {
        "brier": brier_score(y, p),
        "nll": log_loss(y, p),
        "ece_15bins": expected_calibration_error(y, p),
    }
    out["patient_ids_available"] = groups is not None
    if preds["uncertainty"].notna().all():
        u = preds["uncertainty"].to_numpy(dtype=float)
        yhat = preds["predicted_label"].to_numpy()
        rc = risk_coverage(y, yhat, u)
        out["uncertainty"] = {**error_detection(y, yhat, u), **uncertainty_summary(y, yhat, u), "aurc": rc["aurc"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dataset", required=True, help="dataset name written to the export")
    ap.add_argument("--split", required=True, choices=["val", "test", "external"])
    ap.add_argument("--only-split", help="keep only manifest rows with this split value")
    ap.add_argument("--threshold-json", type=Path, help="frozen threshold file (default: checkpoint's)")
    ap.add_argument("--model-name", default="MARS-TB")
    ap.add_argument("--out-predictions", required=True, type=Path)
    ap.add_argument("--out-metrics", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    if args.only_split:
        manifest = manifest[manifest["split"] == args.only_split]
    if manifest.empty:
        print("error: no rows to evaluate", file=sys.stderr)
        return 2
    thr = load_threshold(args.threshold_json).value if args.threshold_json else None
    preds = predict_from_checkpoint(
        args.checkpoint,
        manifest,
        args.root,
        split=args.split,
        dataset=args.dataset,
        model_name=args.model_name,
        threshold=thr,
        batch_size=args.batch_size,
        device=args.device,
    )
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out_predictions, index=False, lineterminator="\n")
    metrics = summarise(preds, args.n_boot, args.seed)
    metrics["checkpoint_hash"] = preds["checkpoint_hash"].iloc[0]
    metrics["dataset"], metrics["split"] = args.dataset, args.split
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(
        json.dumps(metrics, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o)),
        encoding="utf-8",
    )
    print(f"wrote {args.out_predictions} ({len(preds)} rows) and {args.out_metrics}")
    print(
        f"AUC {metrics['roc_auc']:.4f}  sens {metrics['sensitivity']:.4f}  spec {metrics['specificity']:.4f}  "
        f"threshold {metrics['threshold']:.4f} (validation-selected)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
