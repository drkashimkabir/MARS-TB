import pytest
import torch

from mars_tb.data.augmentation import EvalTransform, TrainTransform, build_transform, rotate_bicubic
from mars_tb.data.preprocessing import normalize


def _img():
    g = torch.Generator().manual_seed(0)
    return torch.rand(3, 64, 64, generator=g)


def test_eval_transform_is_exactly_normalisation_and_deterministic():
    x = _img()
    t = EvalTransform()
    assert torch.equal(t(x), normalize(x))
    assert torch.equal(t(x), t(x))


@pytest.mark.parametrize("split", ["val", "test", "external"])
def test_no_augmentation_outside_training(split):
    t = build_transform(split)
    assert isinstance(t, EvalTransform)
    x = _img()
    torch.manual_seed(0)
    a = t(x)
    torch.manual_seed(1)
    assert torch.equal(a, t(x))


def test_train_split_gets_augmentation():
    assert isinstance(build_transform("train"), TrainTransform)


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        build_transform("holdout")


def test_train_transform_preserves_shape_and_varies():
    t, x = TrainTransform(), _img()
    outs = []
    for seed in range(20):
        torch.manual_seed(seed)
        outs.append(t(x))
    assert all(o.shape == x.shape for o in outs)
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def test_train_transform_is_reproducible_under_a_seed():
    t, x = TrainTransform(), _img()
    torch.manual_seed(7)
    a = t(x)
    torch.manual_seed(7)
    assert torch.equal(a, t(x))


def test_flip_happens_about_half_the_time_and_no_vertical_flip():
    t = TrainTransform()
    # asymmetric image: bright left column
    x = torch.zeros(3, 32, 32)
    x[:, :, 0] = 1.0
    flipped_h = flipped_v = 0
    n = 200
    for seed in range(n):
        torch.manual_seed(seed)
        y = t.geometric.transforms[0](x)
        flipped_h += int(y[0, 0, -1] == 1.0)
        flipped_v += int(torch.equal(y, x.flip(-2)) and not torch.equal(y, x))
    assert 0.35 < flipped_h / n < 0.65
    assert flipped_v == 0


def test_erasing_uses_the_channel_mean_after_normalisation():
    t = TrainTransform()
    assert list(t.erase.value) == [0.0]  # 0 in normalised space == ImageNet channel mean


def test_documented_augmentation_parameters():
    t = TrainTransform()
    flip, rot, jitter = t.geometric.transforms
    assert flip.p == 0.5
    assert tuple(rot.degrees) == (-15.0, 15.0)
    assert tuple(jitter.brightness) == pytest.approx((0.8, 1.2)) and tuple(jitter.contrast) == pytest.approx((0.8, 1.2))
    assert jitter.saturation is None and jitter.hue is None
    assert t.erase.p == 0.1 and tuple(t.erase.scale) == (0.02, 0.20) and tuple(t.erase.ratio) == (0.3, 3.3)


def test_rotation_by_zero_is_identity_and_90_degrees_matches_rot90():
    x = torch.rand(3, 32, 32)
    assert torch.allclose(rotate_bicubic(x, 0.0), x, atol=1e-5)
    r = rotate_bicubic(x, 90.0)
    assert torch.allclose(r, torch.rot90(x, 1, (1, 2)), atol=1e-4) or torch.allclose(
        r, torch.rot90(x, -1, (1, 2)), atol=1e-4
    )


def test_rotation_fills_exposed_corners_with_zero():
    x = torch.ones(3, 64, 64)
    r = rotate_bicubic(x, 15.0)
    assert (r[:, 0, 0].abs() < 1e-6).all()
    assert r[:, 32, 32].allclose(torch.ones(3), atol=1e-4)
