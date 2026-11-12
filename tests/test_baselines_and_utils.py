import json

import numpy as np
import pandas as pd
import pytest
import torch

from mars_tb.config import ModelConfig
from mars_tb.evaluation.subgroup_analysis import subgroup_metrics
from mars_tb.models.baselines import BASELINES, UNAVAILABLE, build_baseline
from mars_tb.models.lora_adapter import LoRALinear, lora_parameters, merge_lora
from mars_tb.training.checkpointing import restore_model, save_checkpoint
from mars_tb.training.reproducibility import environment_record, library_versions
from mars_tb.utils.io import read_json, sha256_bytes, write_json_atomic
from mars_tb.utils.seed import seed_everything, worker_init_fn


@pytest.mark.parametrize("name", ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "Swin-B"])
def test_convolutional_and_swin_baselines_output_tb_task_only(name):
    m = build_baseline(name).eval()
    out = m(torch.randn(2, 3, 96, 96))
    assert out["tb"]["logits"].shape == (2, 2) and out["tb"]["alpha"] is None and out["tb"]["uncertainty"] is None
    assert torch.allclose(out["tb"]["prob"].sum(-1), torch.ones(2), atol=1e-6)
    assert "pattern" not in out and set(m.parameter_groups()) == {"lora", "backbone", "new"}


def test_vit_baseline_accepts_a_smaller_image_size():
    m = build_baseline("ViT-B/16", image_size=64).eval()
    assert m(torch.randn(1, 3, 64, 64))["tb"]["p_tb"].shape == (1,)
    with pytest.raises(ValueError, match="pretrained"):
        build_baseline("ViT-B/16", pretrained=True)


def test_unavailable_baselines_are_refused_not_approximated():
    for name in UNAVAILABLE:
        with pytest.raises(NotImplementedError, match=name):
            build_baseline(name)
    with pytest.raises(KeyError):
        build_baseline("VGG-16")
    assert set(BASELINES) == {"ResNet-50", "DenseNet-121", "EfficientNet-B4", "ViT-B/16", "Swin-B"}


def test_baseline_checkpoint_roundtrip_rebuilds_the_named_baseline(tmp_path):
    m = build_baseline("ResNet-50").eval()
    save_checkpoint(
        tmp_path / "b.ckpt",
        m,
        ModelConfig(),
        epoch=0,
        seed=1,
        val_metrics={"auc": 0.5},
        threshold=0.5,
        preprocess={"size": 64},
        label_mapping={},
        extra={"baseline": "ResNet-50"},
    )
    restored, blob, _ = restore_model(tmp_path / "b.ckpt")
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(restored.eval()(x)["tb"]["logits"], m(x)["tb"]["logits"], atol=1e-6)
    assert blob["extra"]["baseline"] == "ResNet-50"


def test_subgroup_metrics_guard_small_groups():
    rng = np.random.default_rng(0)
    rows = []
    for site, n in (("A", 80), ("B", 6)):
        y = rng.integers(0, 2, n)
        for yi in y:
            rows.append(
                {
                    "site": site,
                    "actual_label": int(yi),
                    "p_tb": float(np.clip(0.3 + 0.4 * yi + rng.normal(0, 0.1), 0, 1)),
                    "threshold": 0.5,
                }
            )
    out = subgroup_metrics(pd.DataFrame(rows), "site").set_index("site")
    assert bool(out.loc["A", "reportable"]) and not bool(out.loc["B", "reportable"])
    assert pd.isna(out.loc["B", "roc_auc"])  # no metrics are produced for a non-reportable subgroup
    with pytest.raises(ValueError, match="metadata"):
        subgroup_metrics(pd.DataFrame(rows), "device")


def test_lora_helpers_and_merge_count():
    from mars_tb.models.mars_tb import MARSTB
    from tests.conftest import tiny_config

    model = MARSTB(tiny_config())
    assert len(lora_parameters(model)) == 48 and merge_lora(model) == 24
    assert all(m.merged for m in model.modules() if isinstance(m, LoRALinear))


def test_io_helpers(tmp_path):
    write_json_atomic(tmp_path / "x" / "a.json", {"k": [1, 2]})
    assert read_json(tmp_path / "x" / "a.json") == {"k": [1, 2]}
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    class Unserialisable:
        def __str__(self) -> str:
            raise RuntimeError("cannot serialise")

    with pytest.raises(RuntimeError):
        write_json_atomic(tmp_path / "bad.json", {"k": Unserialisable()})
    assert not (tmp_path / "bad.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_seed_and_worker_init_and_environment_record():
    seed_everything(3, deterministic=False)
    a = np.random.rand()
    seed_everything(3, deterministic=False)
    assert a == np.random.rand()
    worker_init_fn(0)
    env = environment_record()
    assert {"python", "torch_cuda_available", "libraries", "git_commit"} <= set(env)
    assert json.dumps(env, default=str) and "torch" in library_versions()
