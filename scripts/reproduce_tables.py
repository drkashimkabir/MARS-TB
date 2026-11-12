#!/usr/bin/env python
"""Build results tables from prediction files. Refuses to run without them.

Usage: reproduce_tables.py --predictions NAME=path.csv [NAME=other_seed.csv ...] --out-dir results/tables

Entries sharing a NAME are treated as independent training seeds and aggregated (mean, SD, t-CI over
seeds); a single entry gets DeLong / Clopper-Pearson intervals. With --reference NAME, paired DeLong
tests (Holm-adjusted) against that model's single prediction file are added. Every number comes from
the supplied predictions; nothing is read from the manuscript.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from mars_tb.evaluation.metrics import binary_report
from mars_tb.evaluation.statistical_tests import delong_auc_ci, delong_test, holm, seed_summary


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def parse_entries(items: list[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=path, got {item!r}")
        name, path = item.split("=", 1)
        if not Path(path).is_file():
            raise FileNotFoundError(f"prediction file not found: {path}")
        groups[name].append(Path(path))
    return dict(groups)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str}).dropna(subset=["actual_label"])
    if df.empty:
        raise ValueError(f"{path} has no labelled rows")
    return df


def build_main_table(groups: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    for name, paths in groups.items():
        reps = []
        for p in paths:
            df = _read(p)
            y, s = df["actual_label"].astype(int).to_numpy(), df["p_tb"].to_numpy(float)
            reps.append((binary_report(y, s, float(df["threshold"].iloc[0])), delong_auc_ci(y, s)))
        if len(reps) == 1:
            r, d = reps[0]
            rows.append(
                {
                    "model": name,
                    "runs": 1,
                    "n": r["n"],
                    "auc": r["roc_auc"],
                    "auc_lo": d["lo"],
                    "auc_hi": d["hi"],
                    "sensitivity": r["sensitivity"],
                    "sens_lo": r["sensitivity_ci"][0],
                    "sens_hi": r["sensitivity_ci"][1],
                    "specificity": r["specificity"],
                    "spec_lo": r["specificity_ci"][0],
                    "spec_hi": r["specificity_ci"][1],
                    "ppv": r["ppv"],
                    "npv": r["npv"],
                    "mcc": r["mcc"],
                    "interval_type": "DeLong / Clopper-Pearson (image-level)",
                }
            )
        else:
            agg = seed_summary([r["roc_auc"] for r, _ in reps])
            rows.append(
                {
                    "model": name,
                    "runs": len(reps),
                    "n": reps[0][0]["n"],
                    "auc": agg["mean"],
                    "auc_sd": agg["sd"],
                    "auc_lo": agg["lo"],
                    "auc_hi": agg["hi"],
                    "sensitivity": sum(r["sensitivity"] for r, _ in reps) / len(reps),
                    "specificity": sum(r["specificity"] for r, _ in reps) / len(reps),
                    "mcc": sum(r["mcc"] for r, _ in reps) / len(reps),
                    "per_seed_auc": ";".join(f"{r['roc_auc']:.4f}" for r, _ in reps),
                    "interval_type": "t-interval over seeds (conditional on a fixed test set)",
                }
            )
    return pd.DataFrame(rows)


def build_paired_table(groups: dict[str, list[Path]], reference: str) -> pd.DataFrame:
    if reference not in groups or len(groups[reference]) != 1:
        raise ValueError("--reference must name exactly one prediction file")
    ref = _read(groups[reference][0]).set_index("image_id")
    rows = []
    for name, paths in groups.items():
        if name == reference or len(paths) != 1:
            continue
        cmp_ = _read(paths[0]).set_index("image_id")
        if set(cmp_.index) != set(ref.index):
            raise ValueError(f"{name} and {reference} score different images; paired tests need identical cases")
        idx = ref.index
        r = delong_test(
            ref.loc[idx, "actual_label"].astype(int).to_numpy(),
            ref.loc[idx, "p_tb"].to_numpy(float),
            cmp_.loc[idx, "p_tb"].to_numpy(float),
        )
        rows.append(
            {
                "reference": reference,
                "comparator": name,
                "delta_auc": r["diff"],
                "ci_lo": r["lo"],
                "ci_hi": r["hi"],
                "p_raw": r["p"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_raw"].tolist())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    args = ap.parse_args(argv)
    try:
        groups = parse_entries(args.predictions)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_tbl = build_main_table(groups)
    main_tbl.to_csv(args.out_dir / "performance.csv", index=False, lineterminator="\n")
    (args.out_dir / "performance.md").write_text(to_markdown(main_tbl.round(4)), encoding="utf-8")
    print(main_tbl.round(4).to_string(index=False))
    if args.reference:
        paired = build_paired_table(groups, args.reference)
        paired.to_csv(args.out_dir / "paired_auc.csv", index=False, lineterminator="\n")
        print(paired.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
