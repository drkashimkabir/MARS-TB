from pathlib import Path

import pytest

from mars_tb.config import (
    ModelConfig,
    deep_merge,
    from_dict,
    load_model_config,
    load_yaml,
    validate_model_config,
)
from mars_tb.models.ablations import ABLATIONS, get_ablation
from mars_tb.models.mars_tb import MARSTB
from tests.conftest import tiny_config

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        from_dict(ModelConfig, {"backbonee": {}})
    with pytest.raises(ValueError):
        from_dict(ModelConfig, {"lora": {"rnak": 4}})


def test_nested_and_tuple_conversion():
    cfg = from_dict(ModelConfig, {"backbone": {"layers": [2, 6, 10], "depth": 12}, "ddfm": {"band_edges": [8, 32]}})
    assert cfg.backbone.layers == (2, 6, 10) and cfg.ddfm.band_edges == (8, 32)


def test_deep_merge_overrides_leaf_only():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}


@pytest.mark.parametrize(
    "bad",
    [
        {"backbone": {"img_size": 500}},
        {"backbone": {"layers": [1, 2]}},
        {"backbone": {"layers": [4, 8, 13]}},
        {"backbone": {"dim": 770}},
        {"ddfm": {"geometry": "hex"}},
        {"ddfm": {"band_edges": [64, 16]}},
        {"head": {"type": "svm"}},
        {"text": {"source": "file"}},
        {"backbone": {"freeze_backbone": False}},
    ],
)
def test_invalid_model_configs_fail_validation(bad):
    with pytest.raises(ValueError):
        validate_model_config(from_dict(ModelConfig, bad))


def test_all_shipped_model_configs_load_and_validate():
    for path in sorted((ROOT / "configs" / "model").glob("*.yaml")):
        data = load_yaml(path)
        if "ablations" in data or path.name == "ablations.yaml":
            continue
        cfg = load_model_config(path)
        assert cfg.lora.rank == 16 and cfg.lora.alpha == 32 and cfg.lora.scale == 2.0 or not cfg.lora.enabled


def test_default_config_is_the_manuscript_architecture():
    cfg = load_model_config(ROOT / "configs" / "model" / "mars_tb.yaml")
    b = cfg.backbone
    assert (b.img_size, b.patch_size, b.dim, b.depth, b.heads, b.layers) == (512, 16, 768, 12, 12, (4, 8, 12))
    assert (cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.targets) == (16, 32, 0.1, ("q", "v"))
    assert cfg.ddfm.band_edges == (16, 64) and cfg.agrn.heads == 8 and cfg.agrn.hidden == 512


def test_ablation_registry_builds_models():
    base = tiny_config()
    from dataclasses import asdict

    for name in ("A6a_no_ddfm", "A6b_no_agrn", "A6c_no_prm", "A6d_ce_head", "A6e_full_finetune"):
        cfg = from_dict(ModelConfig, deep_merge(asdict(base), dict(get_ablation(name))))
        MARSTB(cfg)


def test_undefined_ablations_are_refused_not_improvised():
    for name in ("A1_resnet50_baseline", "A2_biovil_t_pretraining_configuration"):
        with pytest.raises(KeyError, match="not defined"):
            get_ablation(name)
    with pytest.raises(KeyError):
        get_ablation("nonsense")
    assert "full" in ABLATIONS
