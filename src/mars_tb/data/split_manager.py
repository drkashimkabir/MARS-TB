"""Reproducible, verifiable train/val/test split generation.

A split produced here is ALWAYS labelled ``newly_generated``. It is never the manuscript's
historical partition: those files have not been located (docs/known_discrepancies.md, F/G).
``historical_original`` can only be assigned by :func:`load_historical_split`, which demands the
authors' actual file plus a written attestation of where it came from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mars_tb.data.label_mapping import to_binary
from mars_tb.data.metadata import load_schema
from mars_tb.utils.io import read_json, sha256_file, write_json_atomic

SPLITS = ("train", "val", "test")
GENERATOR = "mars_tb.data.split_manager/0.1"


def canonical_manifest_sha256(df: pd.DataFrame) -> str:
    """Order-independent checksum of (image_id, split, label_binary)."""
    cols = df[["image_id", "split", "label_binary"]].astype({"image_id": str})
    body = cols.sort_values("image_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _allocate(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0
    return n_train, n_val, n_test


def generate_split(
    manifest: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    strata_column: str | None = "diagnostic_category",
    group_column: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``manifest`` with a ``split`` column.

    Without ``group_column``: per-stratum random allocation (input order does not matter).
    With ``group_column`` (e.g. patient_id): whole groups are assigned to one split, each group
    placed in the split with the largest remaining deficit; a group's stratum is that of its first
    image by sorted image_id. Rows whose binary label is missing must be removed beforehand.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    for col in ("image_id", "label_binary"):
        if col not in manifest.columns:
            raise ValueError(f"manifest lacks required column {col!r}")
    if manifest["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    if manifest["label_binary"].isna().any():
        raise ValueError("label_binary contains missing values; exclude those rows explicitly")
    df = manifest.sort_values("image_id").reset_index(drop=True).copy()
    strata = df[strata_column].fillna("NA").astype(str) if strata_column else df["label_binary"].astype(str)
    rng = np.random.default_rng(seed)
    assigned = np.empty(len(df), dtype=object)

    if group_column is None:
        for _, idx in df.groupby(strata, sort=True).groups.items():
            idx = np.array(sorted(idx))
            rng.shuffle(idx)
            a, b, _c = _allocate(len(idx), ratios)
            assigned[idx[:a]] = "train"
            assigned[idx[a : a + b]] = "val"
            assigned[idx[a + b :]] = "test"
    else:
        if group_column not in df.columns or df[group_column].isna().any():
            raise ValueError(f"group column {group_column!r} must exist and have no missing values")
        first = df.groupby(group_column, sort=True).head(1)
        group_ids = first[group_column].to_numpy()
        group_stratum = strata.loc[first.index].to_numpy()
        members_of = df.groupby(group_column).indices
        sizes = np.array([len(members_of[g]) for g in group_ids])
        for stratum in sorted(set(group_stratum)):
            members = np.where(group_stratum == stratum)[0]
            members = members[rng.permutation(len(members))]
            total = int(sizes[members].sum())
            target = np.array(_allocate(total, ratios), dtype=float)
            got = np.zeros(3)
            for m in members:
                k = int(np.argmax(target - got))
                got[k] += sizes[m]
                assigned[members_of[group_ids[m]]] = SPLITS[k]
    df["split"] = assigned
    return df


def split_metadata(
    df: pd.DataFrame,
    name: str,
    seed: int | None,
    ratios: tuple[float, float, float],
    strata_column: str | None,
    group_column: str | None,
    provenance: str = "newly_generated",
    attestation: str | None = None,
) -> dict[str, Any]:
    counts = {
        s: {
            "n": int((df["split"] == s).sum()),
            "positive": int(((df["split"] == s) & (df["label_binary"] == 1)).sum()),
        }
        for s in SPLITS
    }
    return {
        "name": name,
        "provenance": provenance,
        "attestation": attestation,
        "seed": seed,
        "ratios": list(ratios),
        "strata_column": strata_column,
        "group_column": group_column,
        "counts": counts,
        "manifest_sha256": canonical_manifest_sha256(df),
        "generator": GENERATOR,
    }


def save_split(df: pd.DataFrame, meta: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{meta['name']}.csv"
    meta_path = out / f"{meta['name']}.meta.json"
    df.to_csv(csv_path, index=False, lineterminator="\n")
    write_json_atomic(meta_path, meta)
    return csv_path, meta_path


def load_historical_split(csv_path: str | Path, attestation: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the authors' original split file. Requires a non-empty written attestation."""
    if not attestation or len(attestation.strip()) < 20:
        raise ValueError(
            "A historical split needs a written attestation (who supplied the file, when, and "
            "how it was verified) of at least 20 characters."
        )
    df = pd.read_csv(csv_path, dtype={"image_id": str})
    for col in ("image_id", "split", "label_binary"):
        if col not in df.columns:
            raise ValueError(f"historical split file lacks column {col!r}")
    meta = split_metadata(
        df,
        Path(csv_path).stem,
        None,
        (0.0, 0.0, 0.0),
        None,
        None,
        provenance="historical_original",
        attestation=attestation.strip(),
    )
    meta["source_file_sha256"] = sha256_file(csv_path)
    return df, meta


@dataclass
class VerificationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_split(
    df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
    root: str | Path | None = None,
    check_hashes: bool = False,
    prevalence_tolerance: float = 0.01,
) -> VerificationReport:
    """Structural verification of a split table (and optionally its files and sidecar)."""
    rep = VerificationReport()
    for col in ("image_id", "split", "label_binary"):
        if col not in df.columns:
            rep.errors.append(f"missing column {col!r}")
    if rep.errors:
        return rep
    if df["image_id"].duplicated().any():
        rep.errors.append(f"{int(df['image_id'].duplicated().sum())} duplicated image_id rows")
    bad = sorted(set(df["split"].dropna()) - set(SPLITS))
    if bad or df["split"].isna().any():
        rep.errors.append(f"invalid or missing split values: {bad}")
    if df["label_binary"].isna().any() or not set(df["label_binary"].dropna()).issubset({0, 1}):
        rep.errors.append("label_binary must be 0/1 with no missing values")
    if "diagnostic_category" in df.columns:
        for cat, lab in df[["diagnostic_category", "label_binary"]].drop_duplicates().itertuples(index=False):
            if pd.isna(cat):
                continue
            try:
                expected = to_binary(str(cat))
            except ValueError as exc:
                rep.errors.append(str(exc))
                continue
            if expected is None:
                rep.errors.append(f"category {cat!r} is excluded from the binary cohort but is present")
            elif expected != lab:
                rep.errors.append(f"category {cat!r} carries label {lab}, mapping says {expected}")
    if "patient_id" in df.columns and df["patient_id"].notna().any():
        per = df.dropna(subset=["patient_id"]).groupby("patient_id")["split"].nunique()
        n_leaky = int((per > 1).sum())
        rep.summary["patients_spanning_splits"] = n_leaky
        if n_leaky:
            rep.errors.append(f"{n_leaky} patient_id value(s) appear in more than one split")
    else:
        rep.warnings.append("no patient_id: patient-level disjointness CANNOT be verified")
    counts = df.groupby("split")["label_binary"].agg(["size", "sum"])
    rep.summary["counts"] = {s: {"n": int(r["size"]), "positive": int(r["sum"])} for s, r in counts.iterrows()}
    prev = (counts["sum"] / counts["size"]).to_dict()
    if prev and max(prev.values()) - min(prev.values()) > prevalence_tolerance:
        rep.warnings.append(f"prevalence differs across splits by > {prevalence_tolerance}: {prev}")
    if root is not None and "path" in df.columns:
        from mars_tb.utils.io import safe_join

        for row in df.itertuples():
            p = safe_join(root, row.path)
            if not p.is_file():
                rep.errors.append(f"missing file: {row.path}")
                break
            if check_hashes and isinstance(getattr(row, "sha256", None), str) and row.sha256:
                if sha256_file(p) != row.sha256:
                    rep.errors.append(f"sha256 mismatch: {row.path}")
                    break
    if meta is not None:
        for key in load_schema("split")["required"]:
            if key not in meta:
                rep.errors.append(f"sidecar missing key {key!r}")
        if "manifest_sha256" in meta and meta["manifest_sha256"] != canonical_manifest_sha256(df):
            rep.errors.append("manifest_sha256 in sidecar does not match the split table")
        if meta.get("provenance") == "historical_original" and not meta.get("attestation"):
            rep.errors.append("historical_original split without attestation")
    return rep


def read_split(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype={"image_id": str})
    meta_path = csv_path.with_suffix(".meta.json")
    return df, (read_json(meta_path) if meta_path.exists() else None)
