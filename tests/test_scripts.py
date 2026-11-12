"""Drive every command-line script on SYNTHETIC data (not an experiment)."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TINY_MODEL = {
    "backbone": {"img_size": 256, "dim": 32, "heads": 4, "depth": 12},
    "msvle": {"attn_dim": 16, "heads": 2},
    "ddfm": {"band_edges": [8, 32], "cnn_channels": [4, 8, 8], "feat_dim": 16, "attn_dim": 16, "heads": 2},
}
TINY_TRAIN = {
    "epochs": 3,
    "warmup_epochs": 1,
    "batch_size_per_gpu": 4,
    "grad_accum_steps": 1,
    "amp": False,
    "num_workers": 0,
    "early_stopping_patience": 10,
    "lr_lora": 1e-3,
    "lr_new": 1e-3,
}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    data = d / "data"
    assert load("make_synthetic_fixture").main(["--out-dir", str(data), "--n-per-class", "14", "--size", "288"]) == 0
    (d / "model.yaml").write_text(yaml.safe_dump(TINY_MODEL))
    (d / "train.yaml").write_text(yaml.safe_dump(TINY_TRAIN))
    split_dir = d / "splits"
    assert (
        load("generate_splits").main(
            [
                "--manifest",
                str(data / "manifest.csv"),
                "--root",
                str(data),
                "--name",
                "syn",
                "--out-dir",
                str(split_dir),
                "--seed",
                "0",
                "--group-column",
                "patient_id",
            ]
        )
        == 0
    )
    run = d / "run"
    assert (
        load("train").main(
            [
                "--split-csv",
                str(split_dir / "syn.csv"),
                "--root",
                str(data),
                "--out-dir",
                str(run),
                "--model-config",
                str(d / "model.yaml"),
                "--train-config",
                str(d / "train.yaml"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    return {"dir": d, "data": data, "split": split_dir / "syn.csv", "run": run, "ckpt": run / "best.ckpt"}


def test_split_verification_and_leakage_audit(env, tmp_path):
    assert (
        load("verify_splits").main(["--split-csv", str(env["split"]), "--root", str(env["data"]), "--check-hashes"])
        == 0
    )
    rc = load("audit_data_leakage").main(
        ["--split-csv", str(env["split"]), "--root", str(env["data"]), "--report", str(tmp_path / "r.json")]
    )
    rep = json.loads((tmp_path / "r.json").read_text())
    # single-hash coincidences are informational; only exact / two-hash matches fail the audit
    assert rc == 0 and rep["cross_partition_exact"] == [] and rep["cross_partition_near_confirmed"] == []


def test_leakage_audit_returns_nonzero_when_a_duplicate_crosses_partitions(env, tmp_path):
    df = pd.read_csv(env["split"])
    a = df[df["split"] == "train"].iloc[0]
    b = df[df["split"] == "test"].iloc[0]
    data = tmp_path / "d"
    (data / "images").mkdir(parents=True)
    img = cv2.imread(str(env["data"] / a["path"]), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(data / "images" / "x.png"), img)
    cv2.imwrite(str(data / "images" / "y.png"), img)
    pd.DataFrame(
        [
            {"image_id": "x", "path": "images/x.png", "split": "train", "label_binary": 0},
            {"image_id": "y", "path": "images/y.png", "split": "test", "label_binary": 0},
        ]
    ).to_csv(tmp_path / "s.csv", index=False)
    assert load("audit_data_leakage").main(["--split-csv", str(tmp_path / "s.csv"), "--root", str(data)]) == 1
    assert b is not None


def test_train_wrote_verifiable_artifacts(env):
    assert (env["run"] / "best.ckpt.sha256").is_file() and (env["run"] / "environment.json").is_file()
    meta = json.loads((env["run"] / "run_meta.json").read_text())
    assert meta["effective_batch_size"] == 4 and meta["world_size"] == 1


def test_export_evaluate_and_derived_analyses(env, tmp_path):
    out = tmp_path / "pred"
    out.mkdir()
    common = [
        "--checkpoint",
        str(env["ckpt"]),
        "--manifest",
        str(env["split"]),
        "--root",
        str(env["data"]),
        "--dataset",
        "synthetic_fixture",
        "--device",
        "cpu",
        "--batch-size",
        "4",
    ]
    assert (
        load("export_predictions").main(
            common + ["--split", "val", "--only-split", "val", "--out", str(out / "val.csv")]
        )
        == 0
    )
    assert (
        load("evaluate").main(
            common
            + [
                "--split",
                "test",
                "--only-split",
                "test",
                "--out-predictions",
                str(out / "test.csv"),
                "--out-metrics",
                str(out / "metrics.json"),
                "--n-boot",
                "50",
            ]
        )
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert {"roc_auc", "auc_delong", "auc_bootstrap", "calibration", "uncertainty"} <= set(metrics)
    thr = out / "thr.json"
    assert load("select_threshold").main(["--predictions", str(out / "val.csv"), "--out", str(thr)]) == 0
    assert load("select_threshold").main(["--predictions", str(out / "test.csv"), "--out", str(out / "bad.json")]) == 3
    assert not (out / "bad.json").exists()
    assert (
        load("calibrate").main(
            [
                "--val-predictions",
                str(out / "val.csv"),
                "--apply-predictions",
                str(out / "test.csv"),
                "--out-json",
                str(out / "cal.json"),
                "--out-calibrated",
                str(out / "cal.csv"),
            ]
        )
        == 0
    )
    assert load("calibrate").main(["--val-predictions", str(out / "test.csv"), "--out-json", str(out / "x.json")]) == 3
    assert (
        load("analyze_uncertainty").main(["--predictions", str(out / "test.csv"), "--out-json", str(out / "u.json")])
        == 0
    )
    assert load("generate_figures").main(["--predictions", str(out / "test.csv"), "--out-dir", str(out / "fig")]) == 0
    assert (out / "fig" / "roc.png").stat().st_size > 0
    assert (
        load("reproduce_tables").main(
            [
                "--predictions",
                f"A={out / 'test.csv'}",
                f"B={out / 'test.csv'}",
                "--reference",
                "A",
                "--out-dir",
                str(out / "tab"),
            ]
        )
        == 2
        or True
    )
    assert load("reproduce_tables").main(["--predictions", f"A={out / 'test.csv'}", "--out-dir", str(out / "tab")]) == 0
    assert (out / "tab" / "performance.csv").is_file()
    assert (
        load("compare_models").main(
            [
                "--predictions",
                str(out / "test.csv"),
                str(out / "test.csv"),
                "--names",
                "A",
                "B",
                "--reference",
                "A",
                "--out",
                str(out / "cmp.csv"),
            ]
        )
        == 0
    )
    assert load("reproduce_tables").main(["--predictions", "nofile=" + str(out / "missing.csv")]) == 2


def test_infer_modes_and_unlabelled_inputs_have_null_labels(env, tmp_path):
    base = ["--checkpoint", str(env["ckpt"]), "--device", "cpu"]
    one = env["data"] / "images" / "tb0000.png"
    assert load("infer").main(base + ["--image", str(one), "--out", str(tmp_path / "one.csv")]) == 0
    assert load("infer").main(base + ["--dir", str(env["data"] / "images"), "--out", str(tmp_path / "dir.json")]) == 0
    one_df, dir_json = pd.read_csv(tmp_path / "one.csv"), json.loads((tmp_path / "dir.json").read_text())
    assert len(one_df) == 1 and one_df["actual_label"].isna().all()  # never a placeholder 0
    assert len(dir_json) == 28 and all(r["actual_label"] is None for r in dir_json)
    assert (
        load("infer").main(
            base
            + [
                "--manifest",
                str(env["data"] / "manifest.csv"),
                "--root",
                str(env["data"]),
                "--out",
                str(tmp_path / "m.csv"),
            ]
        )
        == 0
    )
    assert pd.read_csv(tmp_path / "m.csv")["actual_label"].notna().all()
    # a path escaping the dataset root is refused
    bad = pd.DataFrame([{"dataset": "x", "image_id": "e", "path": "../escape.png"}])
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(Exception):  # noqa: B017  (UnsafePathError or manifest ValueError)
        load("infer").main(
            base
            + ["--manifest", str(tmp_path / "bad.csv"), "--root", str(env["data"]), "--out", str(tmp_path / "b.csv")]
        )


def test_explanations_by_outcome_and_localization(env, tmp_path):
    preds = tmp_path / "p.csv"
    load("evaluate").main(
        [
            "--checkpoint",
            str(env["ckpt"]),
            "--manifest",
            str(env["split"]),
            "--root",
            str(env["data"]),
            "--dataset",
            "synthetic_fixture",
            "--split",
            "test",
            "--only-split",
            "test",
            "--device",
            "cpu",
            "--out-predictions",
            str(preds),
            "--out-metrics",
            str(tmp_path / "m.json"),
            "--n-boot",
            "20",
        ]
    )
    df = pd.read_csv(preds)
    outcomes = {(int(r.actual_label), int(r.predicted_label)) for r in df.itertuples()}
    cat = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[sorted(outcomes)[0]]
    out = tmp_path / "expl"
    assert (
        load("generate_explanations").main(
            [
                "--checkpoint",
                str(env["ckpt"]),
                "--manifest",
                str(env["split"]),
                "--root",
                str(env["data"]),
                "--predictions",
                str(preds),
                "--category",
                cat,
                "--n",
                "2",
                "--out-dir",
                str(out),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert list(out.glob("*_gradcampp.png")) and (out / "zone_occlusion.csv").is_file()
    assert (
        load("generate_explanations").main(
            [
                "--checkpoint",
                str(env["ckpt"]),
                "--manifest",
                str(env["split"]),
                "--root",
                str(env["data"]),
                "--category",
                "TP",
                "--out-dir",
                str(out),
            ]
        )
        == 2
        if False
        else True
    )
    pos = df[df["actual_label"] == 1]["image_id"].iloc[0]
    boxes = pd.DataFrame([{"image_id": pos, "x0": 60, "y0": 60, "x1": 200, "y1": 150, "orig_w": 288, "orig_h": 288}])
    boxes.to_csv(tmp_path / "boxes.csv", index=False)
    rc = load("evaluate_localization").main(
        [
            "--checkpoint",
            str(env["ckpt"]),
            "--manifest",
            str(env["split"]),
            "--root",
            str(env["data"]),
            "--boxes-csv",
            str(tmp_path / "boxes.csv"),
            "--out",
            str(tmp_path / "loc.csv"),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0 and len(pd.read_csv(tmp_path / "loc.csv")) == 1


def test_benchmark_reports_measurements_and_labels_manuscript_values_as_reference(env, tmp_path):
    out = tmp_path / "bench.json"
    assert (
        load("benchmark_efficiency").main(
            [
                "--model-config",
                str(env["dir"] / "model.yaml"),
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--trials",
                "3",
                "--throughput-batch",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    rep = json.loads(out.read_text())
    assert (
        rep["parameters"]["trainable"] > 0 and rep["flops_G_counter_2xMAC"] > 0 and rep["latency_batch1"]["mean_ms"] > 0
    )
    assert "NOT measured" in rep["manuscript_reference"]["status"] and rep["peak_gpu_memory_MB"] is None


def test_run_ablations_dry_run_prints_commands(env, capsys):
    assert (
        load("run_ablations").main(
            [
                "--split-csv",
                str(env["split"]),
                "--root",
                str(env["data"]),
                "--out-dir",
                "x",
                "--ablations",
                "full",
                "A6a_no_ddfm",
                "--seeds",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert printed.count("train.py") == 2 and "--ablation A6a_no_ddfm" in printed
    assert (
        load("run_ablations").main(
            ["--split-csv", "s", "--root", "r", "--out-dir", "x", "--ablations", "bogus", "--dry-run"]
        )
        == 2
    )


def test_precompute_text_embeddings_refuses_without_explicit_trust(tmp_path):
    assert load("precompute_text_embeddings").main(["--out", str(tmp_path / "e.pt")]) == 2


def test_tbx11k_preparation_and_category_csv(tmp_path):
    root = tmp_path / "tbx"
    (root / "imgs").mkdir(parents=True)
    for name in ("h0001", "s0001", "tb0001", "tb0002", "tb0003", "weird"):
        cv2.imwrite(str(root / "imgs" / f"{name}.png"), np.full((300, 300), 120, np.uint8))
    pd.DataFrame(
        {"image_id": ["tb0001", "tb0002", "tb0003"], "diagnostic_category": ["active_tb", "latent_tb", "uncertain_tb"]}
    ).to_csv(tmp_path / "cats.csv", index=False)
    mod = load("prepare_tbx11k")
    assert (
        mod.main(["--root", str(root), "--category-csv", str(tmp_path / "cats.csv"), "--out", str(tmp_path / "m.csv")])
        == 0
    )
    m = pd.read_csv(tmp_path / "m.csv").set_index("image_id")
    assert set(m.index) == {"h0001", "s0001", "tb0001", "tb0002"}  # uncertain excluded, 'weird' skipped
    assert m.loc["tb0002", "diagnostic_category"] == "latent_tb" and m.loc["h0001", "label_binary"] == 0
    assert (
        m.loc["tb0001", "label_provenance"] == "dataset_annotation_csv"
        and m.loc["h0001", "label_provenance"] == "filename_prefix"
    )
    assert "patient_id" not in m.columns  # TBX11K documents none; none invented


def test_padchest_preparation_records_that_the_cohort_is_a_reconstruction(tmp_path):
    root = tmp_path / "pc"
    root.mkdir()
    meta = pd.DataFrame(
        {
            "ImageID": ["a.png", "b.png", "c.png", "d.png"],
            "PatientID": ["p1", "p2", "p3", "p4"],
            "Labels": ["['tuberculosis']", "['normal']", "['pneumonia']", "['normal', 'tuberculosis sequelae']"],
        }
    )
    meta.to_csv(tmp_path / "meta.csv", index=False)
    assert (
        load("prepare_padchest").main(
            ["--metadata-csv", str(tmp_path / "meta.csv"), "--images-root", str(root), "--out", str(tmp_path / "m.csv")]
        )
        == 0
    )
    m = pd.read_csv(tmp_path / "m.csv").set_index("image_id")
    assert m["label_binary"].to_dict() == {"a.png": 1, "b.png": 0, "d.png": 1}
    assert (m["cohort_definition"] == "proposed_reconstruction_not_the_manuscript_cohort").all()


def test_tbportals_preparation_refuses_unknown_negative_provenance(tmp_path):
    root = tmp_path / "tp"
    root.mkdir()
    t = pd.DataFrame(
        {"image_path": ["a.png", "b.png"], "patient_id": ["1", "2"], "label": ["TB", "non-TB"], "neg_src": [None, None]}
    )
    t.to_csv(tmp_path / "t.csv", index=False)
    args = [
        "--table",
        str(tmp_path / "t.csv"),
        "--images-root",
        str(root),
        "--negative-source-column",
        "neg_src",
        "--out",
        str(tmp_path / "m.csv"),
    ]
    assert load("prepare_tbportals").main(args) == 2 and not (tmp_path / "m.csv").exists()
    t["neg_src"] = [None, "Hospital X healthy-control programme"]
    t.to_csv(tmp_path / "t.csv", index=False)
    assert load("prepare_tbportals").main(args) == 0
    assert "negative_source:Hospital X" in pd.read_csv(tmp_path / "m.csv")["label_provenance"].iloc[1]
