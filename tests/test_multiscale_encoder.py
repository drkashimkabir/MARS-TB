import pytest
import torch

from mars_tb.config import MSVLEConfig, TextConfig
from mars_tb.models.multi_scale_encoder import (
    CrossAttention,
    MultiScaleEncoder,
    PromptBank,
    load_prompts,
    pool_tokens,
)


def test_token_counts_match_manuscript_at_32x32_grid():
    z = torch.randn(2, 1024, 8)
    assert pool_tokens(z, 32, 1).shape == (2, 1024, 8)
    assert pool_tokens(z, 32, 2).shape == (2, 256, 8)  # 16x16
    assert pool_tokens(z, 32, 4).shape == (2, 64, 8)  # 8x8


def test_pooling_is_a_spatial_mean():
    z = torch.arange(16, dtype=torch.float32).view(1, 16, 1)  # 4x4 grid of 0..15
    pooled = pool_tokens(z, 4, 2)
    assert pooled.flatten().tolist() == [2.5, 4.5, 10.5, 12.5]


def test_pool_rejects_nonsquare_token_count():
    with pytest.raises(ValueError):
        pool_tokens(torch.zeros(1, 15, 2), 4, 2)


def test_cross_attention_param_count_matches_supplementary_table():
    # 4 * 768 * 128 = 393,216 per text-guidance block (Supplementary Table 1)
    assert sum(p.numel() for p in CrossAttention(768, 128, 8).parameters()) == 393_216


def test_manuscript_prompts_are_loaded():
    prompts = load_prompts()
    assert [len(v) for v in prompts.values()] == [4, 3, 3]
    assert "miliary nodule" in prompts["scale1_fine"]


def test_encoder_output_shape_and_residual_when_attention_is_zero():
    enc = MultiScaleEncoder(16, 8, MSVLEConfig(attn_dim=8, heads=2), TextConfig())
    for g in enc.guidance:
        torch.nn.init.zeros_(g.o.weight)  # attention contributes nothing -> pure residual path
    z = [torch.randn(2, 64, 16) for _ in range(3)]
    out = enc(*z)
    assert out["z_ms"].shape == (2, 16)
    expected_vecs = torch.stack([z[0].mean(1), pool_tokens(z[1], 8, 2).mean(1), pool_tokens(z[2], 8, 4).mean(1)], dim=1)
    assert torch.allclose(out["scale_vectors"], expected_vecs, atol=1e-6)


def test_fuse_has_no_bias_like_supplementary_table():
    enc = MultiScaleEncoder(768, 32, MSVLEConfig(), TextConfig())
    assert enc.fuse.in_features == 2304 and enc.fuse.bias is None
    assert sum(p.numel() for p in enc.fuse.parameters()) + sum(p.numel() for p in enc.norm.parameters()) == 1_771_008


def test_file_text_embeddings_are_frozen_and_validated(tmp_path):
    path = tmp_path / "emb.pt"
    torch.save(
        {"scale1_fine": torch.randn(4, 16), "scale2_lobe": torch.randn(3, 16), "scale3_whole_lung": torch.randn(3, 16)},
        path,
    )
    bank = PromptBank(16, TextConfig(source="file", path=str(path)))
    assert len(list(bank.parameters())) == 0  # frozen: buffers only
    assert bank.get(0, 5).shape == (5, 4, 16)
    torch.save(
        {"scale1_fine": torch.randn(2, 16), "scale2_lobe": torch.randn(3, 16), "scale3_whole_lung": torch.randn(3, 16)},
        path,
    )
    with pytest.raises(ValueError):
        PromptBank(16, TextConfig(source="file", path=str(path)))


def test_learned_prompts_are_trainable():
    bank = PromptBank(16, TextConfig())
    assert len(list(bank.parameters())) == 3
