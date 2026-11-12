#!/usr/bin/env python
"""Build a PadChest-TB *proposed reconstruction* manifest from PadChest's own metadata CSV.

The manuscript's PadChest-TB selection rules (metadata fields, extraction rules, annotation-source
filters) are documented as incomplete, so the 3,000-image cohort cannot be reproduced from the
paper. This script applies an explicit, adjustable rule (positive = report labels contain a
tuberculosis token; negative = the only label is 'normal') and records it in the manifest. The
result is NOT the manuscript's cohort and negatives are radiologically normal, not
microbiologically TB-free.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

from mars_tb.data.metadata import validate_manifest


def _labels(cell: object) -> list[str]:
    if isinstance(cell, list):
        return [str(x).strip().lower() for x in cell]
    try:
        value = ast.literal_eval(str(cell))
        return [str(x).strip().lower() for x in value] if isinstance(value, list | tuple) else [str(value).lower()]
    except (ValueError, SyntaxError):
        return [t.strip().lower() for t in str(cell).split(",") if t.strip()]


def build_manifest(meta: pd.DataFrame, positive: list[str], negative: list[str]) -> pd.DataFrame:
    need = {"ImageID", "PatientID", "Labels"}
    if not need <= set(meta.columns):
        raise ValueError(f"PadChest metadata must contain columns {sorted(need)}")
    rows = []
    for rec in meta.itertuples(index=False):
        labels = [x for x in _labels(rec.Labels) if x and x != "nan"]
        if any(tok in lab for lab in labels for tok in positive):
            y, prov = 1, "report_label:" + "|".join(labels)
        elif labels and all(lab in negative for lab in labels):
            y, prov = 0, "report_label:" + "|".join(labels)
        else:
            continue
        rows.append(
            {
                "dataset": "padchest_tb",
                "split": "external",
                "image_id": str(rec.ImageID),
                "path": str(rec.ImageID),
                "patient_id": str(rec.PatientID),
                "label_binary": y,
                "label_provenance": prov,
                "source": "PadChest",
                "site": "HUSJ Alicante",
                "cohort_definition": "proposed_reconstruction_not_the_manuscript_cohort",
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-csv", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--positive-tokens", default="tuberculosis")
    ap.add_argument("--negative-labels", default="normal")
    ap.add_argument("--max-negatives", type=int, default=None, help="random subsample (seeded) of negatives")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    df = build_manifest(
        pd.read_csv(args.metadata_csv, dtype=str),
        args.positive_tokens.lower().split(","),
        args.negative_labels.lower().split(","),
    )
    if args.max_negatives is not None:
        neg = df[df["label_binary"] == 0].sample(
            n=min(args.max_negatives, int((df["label_binary"] == 0).sum())), random_state=args.seed
        )
        df = pd.concat([df[df["label_binary"] == 1], neg]).sort_values("image_id")
    problems = validate_manifest(df, args.images_root)
    if problems:
        print("manifest invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    c = df["label_binary"].value_counts().to_dict()
    print(f"wrote {args.out}: {len(df)} images (negative={c.get(0, 0)}, positive={c.get(1, 0)})")
    print(
        "WARNING: proposed reconstruction; manuscript cohort = 3,000 (891 positive / 2,109 negative), rules undocumented."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
