"""Shared fixtures. All data here is SYNTHETIC and is never a clinical experiment."""

from __future__ import annotations

import pytest
import torch

from mars_tb.config import BackboneConfig, DDFMConfig, ModelConfig, MSVLEConfig


def tiny_config(**overrides) -> ModelConfig:
    """Small model: 256 px input (16x16 token grid), width 32, depth 12 (layers 4/8/12)."""
    base = dict(
        backbone=BackboneConfig(img_size=256, dim=32, heads=4, depth=12),
        msvle=MSVLEConfig(attn_dim=16, heads=2),
        ddfm=DDFMConfig(band_edges=(8, 32), cnn_channels=(4, 8, 8), feat_dim=16, attn_dim=16, heads=2),
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture()
def tiny_cfg() -> ModelConfig:
    return tiny_config()


@pytest.fixture()
def tiny_batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 256, 256, generator=g)
