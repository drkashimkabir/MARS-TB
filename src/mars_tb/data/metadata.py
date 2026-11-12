"""Schema-backed validation of manifests and prediction exports."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
import pandas as pd

from mars_tb.utils.io import UnsafePathError, safe_join

PREDICTION_COLUMNS = (
    "dataset",
    "split",
    "image_id",
    "patient_id",
    "model",
    "seed",
    "actual_label",
    "p_tb",
    "predicted_label",
    "threshold",
    "uncertainty",
    "alpha_negative",
    "alpha_positive",
    "checkpoint_hash",
)


def load_schema(name: str) -> dict[str, Any]:
    text = resources.files("mars_tb.resources").joinpath(f"{name}.schema.json").read_text("utf-8")
    return json.loads(text)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def validate_rows(df: pd.DataFrame, schema_name: str, max_errors: int = 20) -> list[str]:
    """Return a list of human-readable schema violations (empty list = valid)."""
    validator = jsonschema.Draft202012Validator(load_schema(schema_name))
    errors: list[str] = []
    for i, rec in enumerate(_records(df)):
        for err in validator.iter_errors(rec):
            loc = ".".join(str(p) for p in err.absolute_path) or "<row>"
            errors.append(f"row {i} [{rec.get('image_id')}] {loc}: {err.message}")
            if len(errors) >= max_errors:
                return errors
    return errors


def validate_manifest(df: pd.DataFrame, root: str | Path | None = None) -> list[str]:
    """Schema + structural checks. If ``root`` is given every path must resolve inside it."""
    problems = validate_rows(df, "manifest")
    if "image_id" in df and df["image_id"].duplicated().any():
        dups = df.loc[df["image_id"].duplicated(), "image_id"].head(5).tolist()
        problems.append(f"duplicate image_id values, e.g. {dups}")
    if root is not None:
        for rel in df.get("path", []):
            try:
                safe_join(root, rel)
            except UnsafePathError as exc:
                problems.append(str(exc))
                break
    return problems


def validate_predictions(df: pd.DataFrame) -> list[str]:
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing required columns: {missing}"]
    problems = validate_rows(df, "predictions")
    ok = df["alpha_negative"].notna() & df["alpha_positive"].notna()
    if ok.any():
        s = df.loc[ok, "alpha_negative"] + df.loc[ok, "alpha_positive"]
        p = df.loc[ok, "alpha_positive"] / s
        if ((p - df.loc[ok, "p_tb"]).abs() > 1e-4).any():
            problems.append("p_tb is not alpha_positive / (alpha_negative + alpha_positive)")
        if ((2.0 / s - df.loc[ok, "uncertainty"]).abs() > 1e-4).any():
            problems.append("uncertainty is not 2 / (alpha_negative + alpha_positive)")
    pred = (df["p_tb"] >= df["threshold"]).astype(int)
    if (pred != df["predicted_label"]).any():
        problems.append("predicted_label is inconsistent with p_tb >= threshold")
    return problems
