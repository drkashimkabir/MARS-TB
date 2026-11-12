#!/usr/bin/env python
"""Write a small SYNTHETIC radiograph-like dataset for smoke tests and CI. Not clinical data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mars_tb.testing.synthetic import make_synthetic_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-per-class", type=int, default=12)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    df = make_synthetic_dataset(args.out_dir, args.n_per_class, args.size, args.seed)
    manifest = args.out_dir / "manifest.csv"
    df.to_csv(manifest, index=False, lineterminator="\n")
    print(f"wrote {len(df)} synthetic images + {manifest} (dataset='synthetic_fixture'; NOT clinical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
