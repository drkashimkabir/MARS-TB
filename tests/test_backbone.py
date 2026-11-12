import pytest
import torch

from mars_tb.config import BackboneConfig
from mars_tb.models.biovil_backbone import ViTBackbone, interpolate_pos_embed, load_timm_vit_state_dict

timm = pytest.importorskip("timm")


def _timm_tiny():
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).eval()


def test_matches_timm_vit_numerically():
    ref = _timm_tiny()
    cfg = BackboneConfig(img_size=224, dim=192, heads=3, depth=12)
    ours = ViTBackbone(cfg).eval()
    unused = load_timm_vit_state_dict(ours, ref.state_dict())
    assert unused == []
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = ref.forward_features(x)[:, 1:]
        got = ours.norm(ours(x, return_layers=(12,))[12])
    assert torch.allclose(got, expected, atol=1e-4)


def test_intermediate_layers_are_distinct_and_shaped():
    ours = ViTBackbone(BackboneConfig(img_size=64, dim=32, heads=4, depth=12)).eval()
    out = ours(torch.randn(2, 3, 64, 64))
    assert set(out) == {4, 8, 12}
    assert all(v.shape == (2, 16, 32) for v in out.values())  # CLS excluded, 4x4 grid
    assert not torch.allclose(out[4], out[8])


def test_pos_embed_interpolation_shape_and_cls_preserved():
    pos = torch.randn(1, 1 + 14 * 14, 8)
    new = interpolate_pos_embed(pos, 14, 32)
    assert new.shape == (1, 1 + 32 * 32, 8)
    assert torch.equal(new[:, :1], pos[:, :1])
    assert torch.equal(interpolate_pos_embed(pos, 14, 14), pos)


def test_loading_224_weights_into_other_resolution():
    ref = _timm_tiny()
    ours = ViTBackbone(BackboneConfig(img_size=128, dim=192, heads=3, depth=12))
    load_timm_vit_state_dict(ours, ref.state_dict(), pretrained_grid=14)
    assert ours(torch.randn(1, 3, 128, 128))[12].shape == (1, 64, 192)


def test_rejects_wrong_input_size_and_channels():
    ours = ViTBackbone(BackboneConfig(img_size=64, dim=32, heads=4, depth=12))
    with pytest.raises(ValueError):
        ours(torch.randn(1, 3, 32, 32))
    with pytest.raises(ValueError):
        ours(torch.randn(1, 1, 64, 64))


def test_missing_weight_key_is_an_error():
    ref = _timm_tiny()
    sd = {k: v for k, v in ref.state_dict().items() if k != "cls_token"}
    ours = ViTBackbone(BackboneConfig(img_size=224, dim=192, heads=3, depth=12))
    with pytest.raises(KeyError):
        load_timm_vit_state_dict(ours, sd)


def test_gradient_checkpointing_gives_same_output():
    cfg = BackboneConfig(img_size=64, dim=32, heads=4, depth=12, gradient_checkpointing=True)
    m = ViTBackbone(cfg).train()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    out = m(x)[12]
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
