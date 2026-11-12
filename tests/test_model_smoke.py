import torch

from mars_tb.models.mars_tb import MARSTB, count_parameters
from tests.conftest import tiny_config


def test_forward_shapes_and_evidential_identities(tiny_cfg, tiny_batch):
    model = MARSTB(tiny_cfg).eval()
    out = model(tiny_batch)
    tb = out["tb"]
    assert tb["alpha"].shape == (2, 2)
    s = tb["alpha"].sum(-1)
    assert torch.allclose(tb["p_tb"], tb["alpha"][:, 1] / s, atol=1e-6)
    assert torch.allclose(tb["uncertainty"], 2.0 / s, atol=1e-6)
    assert (tb["alpha"] >= 1.0).all()
    assert out["pattern"]["alpha"].shape == (2, 4) and out["severity"]["alpha"].shape == (2, 4)
    assert out["aux"]["node_prob"].shape == (2, 12)
    assert out["features"].shape == (2, 32)


def test_backward_gives_finite_grads_to_trainable_only(tiny_cfg, tiny_batch):
    model = MARSTB(tiny_cfg).train()
    out = model(tiny_batch)
    loss = out["tb"]["alpha"].sum() + out["pattern"]["alpha"].sum() + out["severity"]["alpha"].sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name


def test_ablations_run():
    x = torch.randn(2, 3, 256, 256)
    for kw in ({"use_ddfm": False}, {"use_agrn": False}, {"use_prm": False}):
        out = MARSTB(tiny_config(**kw)).eval()(x)
        assert out["tb"]["p_tb"].shape == (2,)


def test_parameter_counts_consistent(tiny_cfg):
    c = count_parameters(MARSTB(tiny_cfg))
    assert c["total"] == c["trainable"] + c["frozen"] and c["trainable"] > 0
