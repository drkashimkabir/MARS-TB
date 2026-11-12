#!/usr/bin/env python
"""Select and freeze an operating threshold from VALIDATION predictions only.

Refuses test/external/train rows unless --exploratory is given (then the output is labelled so and
must not be used for confirmatory results).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mars_tb.evaluation.threshold_selection import ThresholdLeakageError, save_threshold, select_threshold


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--method", choices=["youden", "sensitivity"], default="youden")
    ap.add_argument("--min-sensitivity", type=float, default=0.9)
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.predictions, dtype={"image_id": str}).dropna(subset=["actual_label"])
    splits, datasets = df["split"].unique(), df["dataset"].unique()
    if len(splits) != 1 or len(datasets) != 1:
        print(f"error: expected one dataset/split, found {list(datasets)} / {list(splits)}", file=sys.stderr)
        return 2
    try:
        thr = select_threshold(
            df["actual_label"].astype(int).to_numpy(),
            df["p_tb"].to_numpy(float),
            df["image_id"].tolist(),
            str(datasets[0]),
            str(splits[0]),
            args.method,
            args.min_sensitivity,
            allow_exploratory=args.exploratory,
        )
    except ThresholdLeakageError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    save_threshold(args.out, thr)
    print(f"threshold {thr.value:.6f} ({thr.method}) from {thr.n} {thr.selected_on_split} images -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
