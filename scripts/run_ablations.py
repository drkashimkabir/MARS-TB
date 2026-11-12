#!/usr/bin/env python
"""Train and evaluate every ablation x seed on a fixed split, writing one prediction file per run.

Each run uses scripts/train.py then scripts/export_predictions.py. Use --dry-run to print the
commands. The manuscript's ablation wiring is undefined; definitions live in mars_tb.models.mars_tb.
Aggregation over seeds (mean/SD/t-CI) is done by reproduce_tables.py from the prediction files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mars_tb.config import load_yaml
from mars_tb.models.ablations import ABLATIONS

HERE = Path(__file__).resolve().parent
SEEDS_FILE = HERE.parent / "configs" / "training" / "seeds.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-csv", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dataset", default="tbx11k")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--ablations", nargs="+", default=list(ABLATIONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    seeds = args.seeds or load_yaml(SEEDS_FILE)["seeds"]
    unknown = [a for a in args.ablations if a not in ABLATIONS]
    if unknown:
        print(f"unknown ablations: {unknown}", file=sys.stderr)
        return 2
    for name in args.ablations:
        for seed in seeds:
            run = args.out_dir / name / f"seed{seed}"
            train = [
                sys.executable,
                str(HERE / "train.py"),
                "--split-csv",
                str(args.split_csv),
                "--root",
                str(args.root),
                "--out-dir",
                str(run),
                "--ablation",
                name,
                "--seed",
                str(seed),
            ]
            if args.epochs:
                train += ["--epochs", str(args.epochs)]
            export = [
                sys.executable,
                str(HERE / "export_predictions.py"),
                "--checkpoint",
                str(run / "best.ckpt"),
                "--manifest",
                str(args.split_csv),
                "--root",
                str(args.root),
                "--dataset",
                args.dataset,
                "--split",
                "test",
                "--only-split",
                "test",
                "--model-name",
                name,
                "--out",
                str(run / "test_predictions.csv"),
            ]
            for cmd in (train, export):
                print(" ".join(cmd))
                if not args.dry_run and subprocess.run(cmd, check=False).returncode != 0:  # noqa: S603
                    print(f"FAILED: {name} seed {seed}", file=sys.stderr)
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
