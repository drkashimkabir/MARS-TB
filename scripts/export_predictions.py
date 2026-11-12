#!/usr/bin/env python
"""Export the full prediction table for a labelled split and validate it against the schema.

Thin wrapper over ``evaluate.py``'s prediction step: use it when only the CSV is wanted. It never
fills values that the model did not produce; rows that cannot be scored abort the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mars_tb.data.metadata import validate_predictions
from mars_tb.pipeline import predict_from_checkpoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test", "external"])
    ap.add_argument("--only-split", help="keep only manifest rows with this split value")
    ap.add_argument("--model-name", default="MARS-TB")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    if args.only_split:
        manifest = manifest[manifest["split"] == args.only_split]
    preds = predict_from_checkpoint(
        args.checkpoint,
        manifest,
        args.root,
        split=args.split,
        dataset=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    problems = validate_predictions(preds)
    if problems:
        print("schema validation failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out}: {len(preds)} rows, schema OK, checkpoint {preds['checkpoint_hash'].iloc[0][:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
