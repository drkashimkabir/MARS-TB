#!/usr/bin/env python
"""Generate a reproducible stratified (optionally patient-grouped) train/val/test split.

The output is labelled ``provenance: newly_generated``. It is NOT the manuscript's historical
partition, whose exact assignment files are unavailable. ``--strata label`` reproduces the
manuscript's Table 1 *counts* exactly (6,714/714/6,000 etc.) on an 11,190-image cohort, but counts
do not identify which image went where.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mars_tb.data.metadata import validate_manifest
from mars_tb.data.split_manager import generate_split, save_split, split_metadata, verify_split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", type=Path, help="dataset root (enables path-safety checks)")
    ap.add_argument("--name", required=True, help="split name, e.g. tbx11k_seed0")
    ap.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    ap.add_argument("--strata", choices=["category", "label"], default="category")
    ap.add_argument("--group-column", default=None, help="e.g. patient_id, if reliable ids exist")
    args = ap.parse_args(argv)

    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    problems = validate_manifest(manifest, args.root)
    if problems:
        print("manifest invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    strata = "diagnostic_category" if args.strata == "category" and "diagnostic_category" in manifest else None
    ratios = tuple(args.ratios)
    split = generate_split(manifest, ratios, args.seed, strata, args.group_column)  # type: ignore[arg-type]
    meta = split_metadata(split, args.name, args.seed, ratios, strata, args.group_column)  # type: ignore[arg-type]
    report = verify_split(split, meta)
    if not report.ok:
        print("generated split failed verification:\n  " + "\n  ".join(report.errors), file=sys.stderr)
        return 1
    csv, meta_path = save_split(split, meta, args.out_dir)
    print(f"wrote {csv} and {meta_path}")
    for s, c in meta["counts"].items():
        print(f"  {s:5s} n={c['n']:6d} positive={c['positive']:5d}")
    for w in report.warnings:
        print(f"  warning: {w}")
    print("provenance: newly_generated (NOT the manuscript's historical split)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
