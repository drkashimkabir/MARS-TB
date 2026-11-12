#!/usr/bin/env python
"""Paired DeLong comparison of several models' predictions against one reference, Holm-adjusted.

All prediction files must score the SAME images (matched on image_id). Images repeated per patient
are not independent; pass --note-clustering to record that caveat in the output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mars_tb.evaluation.statistical_tests import delong_test, holm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path, nargs="+")
    ap.add_argument("--names", required=True, nargs="+", help="one name per predictions file")
    ap.add_argument("--reference", required=True, help="name of the reference model")
    ap.add_argument("--out", required=True, type=Path, help="CSV output")
    ap.add_argument("--note-clustering", action="store_true")
    args = ap.parse_args(argv)
    if len(args.predictions) != len(args.names) or args.reference not in args.names:
        print("error: need one name per file and --reference must be one of them", file=sys.stderr)
        return 2
    tables = {
        n: pd.read_csv(p, dtype={"image_id": str}).set_index("image_id")
        for n, p in zip(args.names, args.predictions, strict=True)
    }
    ref = tables[args.reference]
    rows = []
    for name, t in tables.items():
        if name == args.reference:
            continue
        common = ref.index.intersection(t.index)
        if len(common) != len(ref) or len(common) != len(t):
            print(
                f"error: {name} and {args.reference} do not cover identical images "
                f"({len(common)} common of {len(ref)}/{len(t)})",
                file=sys.stderr,
            )
            return 2
        y = ref.loc[common, "actual_label"].astype(int).to_numpy()
        r = delong_test(y, ref.loc[common, "p_tb"].to_numpy(float), t.loc[common, "p_tb"].to_numpy(float))
        rows.append(
            {
                "reference": args.reference,
                "comparator": name,
                "n": len(common),
                "auc_reference": r["auc1"],
                "auc_comparator": r["auc2"],
                "delta_auc": r["diff"],
                "ci_lo": r["lo"],
                "ci_hi": r["hi"],
                "p_raw": r["p"],
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].tolist())
    out["note"] = "images treated as independent" if args.note_clustering else ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
