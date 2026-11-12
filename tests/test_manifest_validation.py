from pathlib import Path

import pandas as pd
import pytest

from mars_tb.data.metadata import validate_manifest, validate_predictions
from mars_tb.utils.io import UnsafePathError, safe_join, sha256_file, write_json_atomic


def _good():
    return pd.DataFrame(
        [
            {
                "dataset": "tbx11k",
                "image_id": "h0001",
                "path": "imgs/h0001.png",
                "label_binary": 0,
                "label_provenance": "filename_prefix",
                "split": "train",
            }
        ]
    )


def test_valid_manifest_passes():
    assert validate_manifest(_good()) == []


@pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.png", "C:\\win\\x.png", "a/../../b.png"])
def test_unsafe_paths_rejected_by_schema(bad):
    df = _good()
    df["path"] = bad
    assert validate_manifest(df)


def test_missing_required_and_bad_enum_values_rejected():
    df = _good().drop(columns=["label_provenance"])
    assert any("label_provenance" in p for p in validate_manifest(df))
    df = _good()
    df["label_binary"] = 2
    assert validate_manifest(df)
    df = _good()
    df["dataset"] = "mystery"
    assert validate_manifest(df)


def test_duplicate_ids_rejected():
    df = pd.concat([_good(), _good()])
    assert any("duplicate" in p for p in validate_manifest(df))


def test_safe_join_confines_to_root(tmp_path):
    assert safe_join(tmp_path, "a/b.png") == (tmp_path / "a" / "b.png").resolve()
    for bad in ["../x", "a/../../x", "/abs/x"]:
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, bad)
    (tmp_path / "real").mkdir()
    # a symlink-free escape via normalisation is still blocked
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path / "real", "../../x")


def _pred(**kw):
    row = {
        "dataset": "d",
        "split": "test",
        "image_id": "i1",
        "patient_id": None,
        "model": "m",
        "seed": 1,
        "actual_label": 1,
        "p_tb": 0.75,
        "predicted_label": 1,
        "threshold": 0.5,
        "uncertainty": 2 / 6,
        "alpha_negative": 1.5,
        "alpha_positive": 4.5,
        "checkpoint_hash": "a" * 64,
    }
    row.update(kw)
    return pd.DataFrame([row])


def test_valid_prediction_row_passes():
    assert validate_predictions(_pred()) == []


def test_prediction_consistency_checks():
    assert any("p_tb is not" in p for p in validate_predictions(_pred(p_tb=0.6, predicted_label=1)))
    assert any("uncertainty is not" in p for p in validate_predictions(_pred(uncertainty=0.9)))
    assert any("predicted_label" in p for p in validate_predictions(_pred(predicted_label=0)))
    assert any("missing required" in p for p in validate_predictions(_pred().drop(columns=["p_tb"])))
    assert validate_predictions(_pred(checkpoint_hash="short"))


def test_softmax_model_rows_may_leave_alpha_empty():
    df = _pred(uncertainty=None, alpha_negative=None, alpha_positive=None)
    assert validate_predictions(df) == []


def test_atomic_json_and_hash(tmp_path):
    p = tmp_path / "x.json"
    write_json_atomic(p, {"b": 1, "a": 2})
    assert Path(p).read_text().startswith('{\n  "a": 2')
    assert len(sha256_file(p)) == 64 and not list(tmp_path.glob("*.tmp"))


def test_schema_copies_in_data_dir_match_package_resources():
    root = Path(__file__).resolve().parents[1]
    for name in ("manifest", "predictions", "split"):
        pkg = (root / "src" / "mars_tb" / "resources" / f"{name}.schema.json").read_bytes()
        pub = root / "data" / "schemas" / f"{name}.schema.json"
        assert pub.is_file() and pub.read_bytes() == pkg, f"{pub} differs from the packaged schema"


def test_shipped_templates_validate_against_their_schemas():
    root = Path(__file__).resolve().parents[1] / "data" / "templates"
    manifest = pd.read_csv(root / "dataset_manifest.example.csv", dtype=str).replace({"": None})
    manifest["label_binary"] = manifest["label_binary"].astype(int)
    manifest = manifest.drop(columns=["split"])
    assert validate_manifest(manifest) == []
    preds = pd.read_csv(root / "prediction_export.example.csv", dtype={"checkpoint_hash": str})
    assert validate_predictions(preds) == []
