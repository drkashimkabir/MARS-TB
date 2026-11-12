#!/usr/bin/env python
"""Build a TBX11K manifest from a local copy of the dataset.

Labels come from filename prefixes (h / s / tb). A TB-prefixed file is labelled TB-positive but its
sub-category (active / latent / both / uncertain) is NOT guessed: supply ``--category-csv``
(columns image_id, diagnostic_category) derived from the dataset's own annotation files to refine
it and to exclude the "uncertain" images. No patient identifiers, pattern or severity labels are
created because TBX11K does not document them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mars_tb.data.label_mapping import category_from_filename, to_binary
from mars_tb.data.metadata import validate_manifest

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def build_manifest(root: Path, category_csv: Path | None = None) -> pd.DataFrame:
    cats: dict[str, str] = {}
    if category_csv is not None:
        table = pd.read_csv(category_csv, dtype=str)
        if not {"image_id", "diagnostic_category"} <= set(table.columns):
            raise ValueError("category CSV needs columns image_id, diagnostic_category")
        cats = dict(zip(table["image_id"], table["diagnostic_category"], strict=True))
    rows, skipped = [], []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        image_id = path.stem
        try:
            category = cats.get(image_id) or category_from_filename(path.name)
        except ValueError:
            skipped.append(path.name)
            continue
        label = to_binary(category)
        if label is None:  # uncertain TB: excluded from the binary cohort
            continue
        rows.append(
            {
                "dataset": "tbx11k",
                "image_id": image_id,
                "path": path.relative_to(root).as_posix(),
                "label_binary": label,
                "diagnostic_category": category,
                "label_provenance": "dataset_annotation_csv" if image_id in cats else "filename_prefix",
            }
        )
    if skipped:
        print(
            f"warning: {len(skipped)} file(s) without an h/s/tb prefix were skipped, e.g. {skipped[:3]}",
            file=sys.stderr,
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="dataset root containing the images")
    ap.add_argument("--category-csv", type=Path, help="optional image_id,diagnostic_category table")
    ap.add_argument("--out", required=True, type=Path, help="output manifest CSV")
    args = ap.parse_args(argv)
    df = build_manifest(args.root, args.category_csv)
    if df.empty:
        print("error: no TBX11K-style images found", file=sys.stderr)
        return 1
    problems = validate_manifest(df, args.root)
    if problems:
        print("manifest invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    c = df["label_binary"].value_counts().to_dict()
    print(f"wrote {args.out}: {len(df)} images (negative={c.get(0, 0)}, positive={c.get(1, 0)})")
    print("NOTE: manuscript reports 11,190 eligible (1,190 positive / 10,000 negative); compare, do not assume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
