"""Per-subgroup performance with a minimum-size guard (tiny subgroups give meaningless AUCs)."""

from __future__ import annotations

import pandas as pd

from mars_tb.evaluation.metrics import binary_report

MIN_PER_CLASS = 10


def subgroup_metrics(preds: pd.DataFrame, by: str, min_per_class: int = MIN_PER_CLASS) -> pd.DataFrame:
    """One row per value of column ``by``. Subgroups with too few cases of a class are reported
    with ``reportable=False`` and no metrics, rather than being silently dropped."""
    if by not in preds.columns:
        raise ValueError(f"column {by!r} not in predictions; subgroup analysis needs real metadata")
    rows = []
    for value, g in preds.dropna(subset=["actual_label"]).groupby(by, dropna=False):
        y = g["actual_label"].astype(int).to_numpy()
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        row = {by: value, "n": len(g), "n_pos": n_pos, "n_neg": n_neg, "reportable": min(n_pos, n_neg) >= min_per_class}
        if row["reportable"]:
            rep = binary_report(y, g["p_tb"].to_numpy(), float(g["threshold"].iloc[0]))
            row.update({k: rep[k] for k in ("roc_auc", "sensitivity", "specificity", "ppv", "npv")})
        rows.append(row)
    return pd.DataFrame(rows)
