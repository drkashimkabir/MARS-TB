#!/usr/bin/env python
"""Verify a split CSV (+ sidecar): labels, categories, patient disjointness, files, hashes, checksum."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mars_tb.data.split_manager import read_split, verify_split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-csv", required=True, type=Path)
    ap.add_argument("--root", type=Path, help="check that every file exists under this root")
    ap.add_argument("--check-hashes", action="store_true", help="also verify per-file sha256 if present")
    args = ap.parse_args(argv)
    df, meta = read_split(args.split_csv)
    rep = verify_split(df, meta, args.root, args.check_hashes)
    print(
        json.dumps(
            {
                "ok": rep.ok,
                "errors": rep.errors,
                "warnings": rep.warnings,
                "summary": rep.summary,
                "provenance": (meta or {}).get("provenance", "no sidecar"),
            },
            indent=2,
            default=str,
        )
    )
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
