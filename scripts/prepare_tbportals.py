#!/usr/bin/env python
"""Build a TB Portals evaluation manifest from a user-supplied, access-controlled table.

TB Portals data require a NIAID data-use agreement, so this script never downloads anything. It
REFUSES to produce a manifest unless every TB-negative image carries an explicit provenance string
(``--negative-source-column``): the manuscript could not establish where its 953 negatives came
from, and a cohort with unknown negatives must not be reported as an independently verified
TB-versus-non-TB evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mars_tb.data.metadata import validate_manifest


def build_manifest(
    table: pd.DataFrame, image_col: str, patient_col: str, label_col: str, negative_source_col: str, positive_value: str
) -> pd.DataFrame:
    for col in (image_col, patient_col, label_col, negative_source_col):
        if col not in table.columns:
            raise ValueError(f"column {col!r} not found in the input table")
    is_pos = table[label_col].astype(str) == positive_value
    neg_src = table.loc[~is_pos, negative_source_col]
    if neg_src.isna().any() or (neg_src.astype(str).str.strip() == "").any():
        n = int(neg_src.isna().sum() + (neg_src.astype(str).str.strip() == "").sum())
        raise ValueError(
            f"{n} TB-negative image(s) have no provenance in {negative_source_col!r}; refusing to build "
            "a cohort whose negatives are of unknown origin"
        )
    out = pd.DataFrame(
        {
            "dataset": "tb_portals",
            "split": "external",
            "image_id": table[image_col].astype(str),
            "path": table[image_col].astype(str),
            "patient_id": table[patient_col].astype(str),
            "label_binary": is_pos.astype(int),
            "label_provenance": [
                ("portal_case_label" if p else "negative_source:" + str(s))
                for p, s in zip(is_pos, table[negative_source_col], strict=True)
            ],
            "source": "NIAID TB Portals",
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--image-column", default="image_path")
    ap.add_argument("--patient-column", default="patient_id")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--positive-value", default="TB")
    ap.add_argument("--negative-source-column", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        df = build_manifest(
            pd.read_csv(args.table, dtype=str),
            args.image_column,
            args.patient_column,
            args.label_column,
            args.negative_source_column,
            args.positive_value,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    problems = validate_manifest(df, args.images_root)
    if problems:
        print("manifest invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    c = df["label_binary"].value_counts().to_dict()
    print(f"wrote {args.out}: {len(df)} images (negative={c.get(0, 0)}, positive={c.get(1, 0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
