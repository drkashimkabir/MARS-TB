import pytest
import torch

from mars_tb.config import DDFMConfig
from mars_tb.models.frequency_fusion import DualDomainFusion, band_masks, split_bands


@pytest.mark.parametrize("geometry", ["radial", "square"])
def test_bands_partition_the_spectrum_exactly(geometry):
    m = band_masks(128, 128, (8, 32), geometry)
    assert m.shape == (3, 128, 128)
    assert torch.equal(m.sum(0), torch.ones(128, 128))  # every bin in exactly one band
    assert m[0, 64, 64] == 1  # DC (centre bin) is in the low band


def test_manuscript_band_edges_at_512():
    m = band_masks(512, 512, (16, 64), "radial")
    assert all(m[i].sum() > 0 for i in range(3))
    assert torch.equal(m.sum(0), torch.ones(512, 512))


def test_geometries_differ():
    assert not torch.equal(band_masks(128, 128, (8, 32), "radial"), band_masks(128, 128, (8, 32), "square"))


def test_empty_band_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        band_masks(32, 32, (4, 64), "radial")  # max radial distance < 64


def test_band_images_sum_back_to_input():
    x = torch.randn(2, 1, 64, 64)
    bands = split_bands(x, band_masks(64, 64, (4, 16)))
    assert bands.shape == (2, 3, 64, 64)
    assert torch.allclose(bands.sum(1, keepdim=True), x, atol=1e-5)


def test_low_band_is_smoother_than_high_band():
    x = torch.randn(1, 1, 64, 64)
    bands = split_bands(x, band_masks(64, 64, (4, 16)))
    dx = lambda t: (t[..., 1:] - t[..., :-1]).abs().mean()  # noqa: E731
    assert dx(bands[:, 0]) < dx(bands[:, 2])


def _ddfm(**kw):
    return DualDomainFusion(
        16, 64, DDFMConfig(band_edges=(4, 16), cnn_channels=(4, 8, 8), feat_dim=8, attn_dim=8, heads=2, **kw)
    )


def test_output_shapes_and_gradients():
    m = _ddfm().train()
    img, z = torch.randn(4, 3, 64, 64), torch.randn(4, 16, requires_grad=True)
    out = m(img, z)
    assert out["z_ddfm"].shape == (4, 16) and out["band_features"].shape == (4, 3, 8)
    out["z_ddfm"].sum().backward()
    assert torch.isfinite(z.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_log_magnitude_mode_runs():
    m = _ddfm(band_input="log_magnitude").eval()
    assert m(torch.randn(2, 3, 64, 64), torch.randn(2, 16))["z_ddfm"].shape == (2, 16)


def test_literal_single_key_attention_is_degenerate_but_band_tokens_is_not():
    img, z = torch.randn(2, 3, 64, 64), torch.randn(2, 16)
    literal = _ddfm(kv_mode="concat_single").eval()
    literal.attn.store_attention = True
    literal(img, z)
    assert torch.allclose(literal.attn.last_attention, torch.ones_like(literal.attn.last_attention))

    tokens = _ddfm(kv_mode="band_tokens").eval()
    tokens.attn.store_attention = True
    tokens(img, z)
    att = tokens.attn.last_attention  # (B, heads, 1, 3 bands)
    assert att.shape[-1] == 3 and torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
    assert (att < 0.9999).all()


def test_three_branch_encoders_are_independent():
    m = _ddfm()
    ids = {id(next(e.parameters())) for e in m.encoders}
    assert len(ids) == 3
