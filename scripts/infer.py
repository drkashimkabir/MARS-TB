#!/usr/bin/env python
"""Run inference on one image, a directory, or a CSV manifest, and write CSV or JSON.

Unlabelled inputs get ``actual_label = null`` (never a placeholder 0). The threshold is the
checkpoint's validation-selected value unless overridden. Research use only: no clinical claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mars_tb.pipeline import predict_from_checkpoint

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


def build_input(args: argparse.Namespace) -> tuple[pd.DataFrame, Path, bool]:
    if args.image:
        p = args.image.resolve()
        return pd.DataFrame([{"dataset": args.dataset, "image_id": p.stem, "path": p.name}]), p.parent, False
    if args.dir:
        root = args.dir.resolve()
        files = sorted(f for f in root.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
        rows = [
            {
                "dataset": args.dataset,
                "image_id": f.relative_to(root).with_suffix("").as_posix(),
                "path": f.relative_to(root).as_posix(),
            }
            for f in files
        ]
        return pd.DataFrame(rows), root, False
    df = pd.read_csv(args.manifest, dtype={"image_id": str})
    return df, args.root, "label_binary" in df.columns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--dir", type=Path)
    src.add_argument("--manifest", type=Path)
    ap.add_argument("--root", type=Path, help="dataset root for --manifest")
    ap.add_argument("--dataset", default="user_supplied")
    ap.add_argument("--split", default="test", choices=["val", "test", "external"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", required=True, type=Path, help="*.csv or *.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    if args.manifest and not args.root:
        ap.error("--manifest requires --root")
    df, root, labelled = build_input(args)
    if df.empty:
        print("error: no images found", file=sys.stderr)
        return 2
    preds = predict_from_checkpoint(
        args.checkpoint,
        df,
        root,
        split=args.split,
        dataset=args.dataset,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
        labelled=labelled,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".json":
        args.out.write_text(preds.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        preds.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out}: {len(preds)} predictions (threshold {preds['threshold'].iloc[0]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
