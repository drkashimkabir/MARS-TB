import pytest
import torch

from mars_tb.config import PRMConfig
from mars_tb.models.progressive_reasoning import ProgressiveReasoning


def test_four_stages_and_dimensions():
    prm = ProgressiveReasoning(768, 12, PRMConfig())
    assert prm.localize.linear.in_features == 780 and prm.localize.linear.out_features == 768
    for stage in (prm.screen, prm.characterize, prm.diagnose):
        assert (stage.linear.in_features, stage.linear.out_features) == (768, 768)
    out = prm(torch.randn(3, 768), torch.rand(3, 12))
    assert out["z_d"].shape == (3, 768) and out["stages"].shape == (3, 4, 768)


def test_stage2_actually_uses_node_activations():
    prm = ProgressiveReasoning(16, 12, PRMConfig()).eval()
    c = torch.randn(2, 16)
    a = prm(c, torch.zeros(2, 12))["z_d"]
    b = prm(c, torch.ones(2, 12))["z_d"]
    assert not torch.allclose(a, b)


def test_stage_outputs_are_layer_normalised():
    prm = ProgressiveReasoning(32, 12, PRMConfig()).eval()
    stages = prm(torch.randn(4, 32) * 5, torch.rand(4, 12))["stages"]
    assert torch.allclose(stages.mean(-1), torch.zeros(4, 4), atol=1e-4)


def test_missing_or_wrong_node_input_is_an_error():
    prm = ProgressiveReasoning(16, 12, PRMConfig())
    with pytest.raises(ValueError):
        prm(torch.randn(2, 16), None)
    with pytest.raises(ValueError):
        prm(torch.randn(2, 16), torch.rand(2, 5))


def test_no_node_variant_for_agrn_ablation():
    prm = ProgressiveReasoning(16, 0, PRMConfig())
    assert prm.localize.linear.in_features == 16
    assert prm(torch.randn(2, 16), None)["z_d"].shape == (2, 16)


def test_optional_gelu_and_residual_variants_run():
    prm = ProgressiveReasoning(16, 12, PRMConfig(use_gelu=True, residual=True))
    out = prm(torch.randn(2, 16), torch.rand(2, 12))["z_d"]
    out.sum().backward()
    assert torch.isfinite(out).all()
