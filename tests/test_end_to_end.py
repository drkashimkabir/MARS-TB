"""End-to-end smoke on SYNTHETIC data: manifest -> split -> train -> checkpoint -> predictions.

This demonstrates the plumbing works. It is not an experiment and its numbers mean nothing
clinically.
"""

import dataclasses

import pandas as pd
import pytest

from mars_tb.config import TrainConfig
from mars_tb.data.metadata import validate_manifest, validate_predictions
from mars_tb.data.split_manager import generate_split, save_split, split_metadata, verify_split
from mars_tb.evaluation.evaluator import metrics_from_predictions
from mars_tb.pipeline import predict_from_checkpoint, train_from_split
from mars_tb.testing.synthetic import make_synthetic_dataset
from mars_tb.training.checkpointing import ChecksumError, load_checkpoint
from tests.conftest import tiny_config


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    manifest = make_synthetic_dataset(root, n_per_class=14, size=288, seed=0)
    assert validate_manifest(manifest, root) == []
    split = generate_split(manifest, seed=0, strata_column=None, group_column="patient_id")
    assert verify_split(split).ok
    cfg = TrainConfig(
        epochs=4,
        warmup_epochs=1,
        batch_size_per_gpu=4,
        grad_accum_steps=1,
        early_stopping_patience=10,
        amp=False,
        num_workers=0,
        lr_lora=1e-3,
        lr_new=1e-3,
    )
    out = root / "run"
    summary = train_from_split(split, root, tiny_config(), cfg, out, device="cpu")
    return root, split, out, summary


def test_training_produces_a_verified_checkpoint_selected_on_validation(trained):
    _, _, out, summary = trained
    assert (out / "best.ckpt").is_file() and (out / "best.ckpt.sha256").is_file()
    assert summary["best_epoch"] >= 0 and len(summary["checkpoint_sha256"]) == 64
    blob = load_checkpoint(out / "best.ckpt")
    assert blob["threshold"] is not None and 0.0 <= blob["threshold"] <= 1.0
    assert blob["val_metrics"]["auc"] == pytest.approx(summary["best_val_auc"])
    log = pd.read_json(out / "train_log.jsonl", lines=True)
    assert {"epoch", "val_auc", "train_loss_total"} <= set(log.columns)
    assert summary["effective_batch_size"] == 4 and summary["world_size"] == 1


def test_trainer_never_loads_test_rows(trained, monkeypatch):
    root, split, out, _ = trained
    # poison the test images: a trainer that touched them would crash on decode
    test_rows = split[split["split"] == "test"]
    assert len(test_rows) > 0
    poisoned = split.copy()
    poisoned.loc[poisoned["split"] == "test", "path"] = "images/does_not_exist.png"
    cfg = TrainConfig(epochs=2, warmup_epochs=1, batch_size_per_gpu=4, grad_accum_steps=1, amp=False, num_workers=0)
    try:
        train_from_split(poisoned, root, tiny_config(), cfg, out.parent / "run2", device="cpu")
    except ValueError as exc:  # manifest validation of train/val rows only
        pytest.fail(f"train/val should not depend on test rows: {exc}")


def test_prediction_export_matches_schema_and_is_deterministic(trained):
    root, split, out, _ = trained
    test = split[split["split"] == "test"]
    a = predict_from_checkpoint(out / "best.ckpt", test, root, split="test", dataset="synthetic_fixture", batch_size=4)
    b = predict_from_checkpoint(out / "best.ckpt", test, root, split="test", dataset="synthetic_fixture", batch_size=2)
    assert validate_predictions(a) == []
    assert len(a) == len(test) and a["checkpoint_hash"].nunique() == 1
    assert (a["p_tb"] - b["p_tb"]).abs().max() < 1e-5  # batch-size independent (eval mode)
    rep = metrics_from_predictions(a)
    assert rep["n"] == len(test) and rep["threshold"] == a["threshold"].iloc[0]


def test_threshold_comes_from_the_checkpoint_not_from_test_labels(trained):
    root, split, out, _ = trained
    blob = load_checkpoint(out / "best.ckpt")
    test = split[split["split"] == "test"]
    df = predict_from_checkpoint(out / "best.ckpt", test, root, split="test", dataset="synthetic_fixture")
    assert df["threshold"].nunique() == 1 and df["threshold"].iloc[0] == pytest.approx(blob["threshold"])


def test_tampered_checkpoint_is_refused(trained, tmp_path):
    root, split, out, _ = trained
    bad = tmp_path / "bad.ckpt"
    bad.write_bytes((out / "best.ckpt").read_bytes() + b"\0")
    (tmp_path / "bad.ckpt.sha256").write_text((out / "best.ckpt.sha256").read_text().replace("best.ckpt", "bad.ckpt"))
    with pytest.raises(ChecksumError):
        load_checkpoint(bad)
    (tmp_path / "bad.ckpt.sha256").unlink()
    with pytest.raises(ChecksumError, match="missing checksum"):
        load_checkpoint(bad)


def test_split_sidecar_roundtrip_for_this_run(trained, tmp_path):
    _, split, _, _ = trained
    meta = split_metadata(split, "synthetic", 0, (0.6, 0.2, 0.2), None, "patient_id")
    save_split(split, meta, tmp_path)
    assert dataclasses.is_dataclass(TrainConfig) and verify_split(split, meta).ok
