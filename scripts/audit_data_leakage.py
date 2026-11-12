#!/usr/bin/env python
"""Audit a split for exact / near-duplicate images that cross partitions.

Exit code 0 = no overlap found (single-hash candidates are listed but do not fail), 1 = exact or
confirmed near-duplicate overlap, 2 = usage/IO error. A clean result is NOT proof of
patient-level separation unless ``patient_ids_available`` is true in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mars_tb.data.leakage_audit import FAILING_VERDICTS, audit_manifest
from mars_tb.data.split_manager import read_split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-csv", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--max-hamming", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("--report", type=Path, help="write the full JSON report here")
    ap.add_argument("--no-perceptual", action="store_true")
    args = ap.parse_args(argv)
    df, _ = read_split(args.split_csv)
    try:
        rep = audit_manifest(df, args.root, args.max_hamming, not args.no_perceptual)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()}
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return 1 if rep["verdict"] in FAILING_VERDICTS else 0


if __name__ == "__main__":
    sys.exit(main())
