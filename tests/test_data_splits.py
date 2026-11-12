import numpy as np
import pandas as pd
import pytest

from mars_tb.data.split_manager import (
    canonical_manifest_sha256,
    generate_split,
    load_historical_split,
    read_split,
    save_split,
    split_metadata,
    verify_split,
)


def tbx_like_manifest() -> pd.DataFrame:
    """Rows with the category counts the manuscript reports for TBX11K (11,190 eligible)."""
    counts = {"healthy": 5000, "sick_non_tb": 5000, "active_tb": 924, "latent_tb": 212, "active_and_latent_tb": 54}
    rows = []
    for cat, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "image_id": f"{cat}_{i:05d}",
                    "diagnostic_category": cat,
                    "label_binary": int(cat in {"active_tb", "latent_tb", "active_and_latent_tb"}),
                    "path": f"{cat}_{i:05d}.png",
                }
            )
    return pd.DataFrame(rows)


def test_label_stratified_split_reproduces_manuscript_table1_counts_exactly():
    df = generate_split(tbx_like_manifest(), seed=0, strata_column=None)
    c = df.groupby("split")["label_binary"].agg(["size", "sum"])
    assert (c.loc["train", "size"], c.loc["train", "sum"]) == (6714, 714)
    assert (c.loc["val", "size"], c.loc["val", "sum"]) == (2238, 238)
    assert (c.loc["test", "size"], c.loc["test", "sum"]) == (2238, 238)


def test_category_stratified_split_cannot_hit_table1_exactly():
    """Per-category rounding gives 6713/2238/2239 -> Table 1 counts do not pin down the strata."""
    df = generate_split(tbx_like_manifest(), seed=0, strata_column="diagnostic_category")
    sizes = df["split"].value_counts().to_dict()
    assert sizes == {"train": 6713, "val": 2238, "test": 2239}


def test_same_seed_same_split_and_input_order_irrelevant():
    m = tbx_like_manifest().head(500)
    a = generate_split(m, seed=3, strata_column=None)
    b = generate_split(m.sample(frac=1, random_state=9), seed=3, strata_column=None)
    assert a.sort_values("image_id")["split"].tolist() == b.sort_values("image_id")["split"].tolist()
    c = generate_split(m, seed=4, strata_column=None)
    assert a["split"].tolist() != c["split"].tolist()


def test_grouped_split_never_separates_a_patient():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(60):
        label = int(p % 5 == 0)
        for k in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "image_id": f"p{p}_{k}",
                    "patient_id": f"P{p}",
                    "label_binary": label,
                    "diagnostic_category": "active_tb" if label else "healthy",
                }
            )
    df = generate_split(pd.DataFrame(rows), seed=1, group_column="patient_id")
    assert df.groupby("patient_id")["split"].nunique().max() == 1
    assert set(df["split"]) == {"train", "val", "test"}


def test_generated_split_is_labelled_newly_generated():
    df = generate_split(tbx_like_manifest().head(300), seed=1, strata_column=None)
    meta = split_metadata(df, "unit", 1, (0.6, 0.2, 0.2), None, None)
    assert meta["provenance"] == "newly_generated" and meta["attestation"] is None


def test_historical_split_requires_attestation(tmp_path):
    df = generate_split(tbx_like_manifest().head(100), seed=1, strata_column=None)
    path = tmp_path / "orig.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="attestation"):
        load_historical_split(path, "")
    loaded, meta = load_historical_split(path, "Supplied by authors on 2026-11-01, sha256 checked.")
    assert meta["provenance"] == "historical_original" and len(loaded) == 100
    assert "source_file_sha256" in meta


def test_save_read_roundtrip_and_checksum(tmp_path):
    df = generate_split(tbx_like_manifest().head(200), seed=1, strata_column=None)
    meta = split_metadata(df, "s", 1, (0.6, 0.2, 0.2), None, None)
    csv, _ = save_split(df, meta, tmp_path)
    back, meta2 = read_split(csv)
    assert verify_split(back, meta2).ok
    assert canonical_manifest_sha256(back) == meta["manifest_sha256"]


def test_verify_detects_planted_problems():
    df = generate_split(tbx_like_manifest().head(300), seed=1, strata_column=None)
    meta = split_metadata(df, "s", 1, (0.6, 0.2, 0.2), None, None)
    dup = pd.concat([df, df.head(1)])
    assert not verify_split(dup).ok
    bad_split = df.copy()
    bad_split.loc[0, "split"] = "holdout"
    assert not verify_split(bad_split).ok
    mislabel = df.copy()
    mislabel.loc[mislabel["diagnostic_category"] == "healthy", "label_binary"] = 1
    assert any("mapping says" in e for e in verify_split(mislabel).errors)
    excluded = df.copy()
    excluded.loc[0, "diagnostic_category"] = "uncertain_tb"
    assert any("excluded" in e for e in verify_split(excluded).errors)
    tampered = df.copy()
    tampered.loc[0, "split"] = "test" if tampered.loc[0, "split"] != "test" else "train"
    assert any("manifest_sha256" in e for e in verify_split(tampered, meta).errors)


def test_verify_flags_patient_leakage_and_warns_when_patients_unknown():
    df = generate_split(tbx_like_manifest().head(100), seed=1, strata_column=None)
    assert any("patient-level" in w for w in verify_split(df).warnings)
    df["patient_id"] = "same-patient"
    rep = verify_split(df)
    assert not rep.ok and rep.summary["patients_spanning_splits"] == 1


def test_verify_checks_files_and_hashes(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    df = pd.DataFrame(
        {"image_id": ["a"], "split": ["train"], "label_binary": [0], "path": ["a.png"], "sha256": ["0" * 64]}
    )
    assert any("sha256 mismatch" in e for e in verify_split(df, root=tmp_path, check_hashes=True).errors)
    df["path"] = "missing.png"
    assert any("missing file" in e for e in verify_split(df, root=tmp_path).errors)


def test_generate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        generate_split(tbx_like_manifest().head(10), ratios=(0.5, 0.2, 0.2))
    dup = pd.concat([tbx_like_manifest().head(5)] * 2)
    with pytest.raises(ValueError):
        generate_split(dup)
    nolabel = tbx_like_manifest().head(5)
    nolabel.loc[0, "label_binary"] = np.nan
    with pytest.raises(ValueError):
        generate_split(nolabel)
