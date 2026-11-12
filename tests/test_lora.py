import pytest
import torch
from torch import nn

from mars_tb.config import BackboneConfig, LoRAConfig
from mars_tb.models.biovil_backbone import ViTBackbone, build_backbone
from mars_tb.models.lora_adapter import LoRALinear, inject_lora, lora_parameters, merge_lora


def test_lora_scale_is_alpha_over_rank():
    assert LoRAConfig().scale == 2.0


def test_lora_starts_as_identity_update():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8, dropout=0.0, init_std=0.01)
    x = torch.randn(3, 8)
    assert torch.equal(lora(x), base(x))  # B = 0 => output equals the frozen layer


def test_lora_merge_matches_unmerged_output():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8, dropout=0.0, init_std=0.5)
    nn.init.normal_(lora.lora_B, std=0.5)
    x = torch.randn(5, 8)
    before = lora(x)
    lora.merge()
    assert torch.allclose(lora(x), before, atol=1e-5)
    lora.unmerge()
    assert torch.allclose(lora(x), before, atol=1e-5)


def test_inject_requires_matches():
    with pytest.raises(ValueError):
        inject_lora(nn.Sequential(nn.Linear(2, 2)), LoRAConfig())


def test_frozen_params_get_no_grad_and_lora_gets_grad():
    cfg = BackboneConfig(img_size=64, dim=32, heads=4, depth=12)
    model = build_backbone(cfg, LoRAConfig())
    # make B non-zero so A also receives a gradient
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.1)
    out = model(torch.randn(2, 3, 64, 64))
    sum(v.sum() for v in out.values()).backward()
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert not p.requires_grad and p.grad is None, name


def test_only_q_and_v_are_adapted_in_all_layers():
    cfg = BackboneConfig(img_size=64, dim=32, heads=4, depth=12)
    model = build_backbone(cfg, LoRAConfig())
    wrapped = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == 24
    assert all(n.endswith((".attn.q", ".attn.v")) for n in wrapped)
    assert not any(n.endswith((".attn.k", ".attn.proj")) for n in wrapped)


def test_full_finetune_config_trains_everything():
    cfg = BackboneConfig(img_size=64, dim=32, heads=4, depth=12, freeze_backbone=False)
    model = build_backbone(cfg, LoRAConfig(enabled=False))
    assert all(p.requires_grad for p in model.parameters())
    assert not lora_parameters(model)


def test_lora_with_unfrozen_backbone_is_rejected():
    from mars_tb.config import ModelConfig, validate_model_config

    with pytest.raises(ValueError):
        validate_model_config(ModelConfig(backbone=BackboneConfig(freeze_backbone=False)))


@pytest.mark.slow
def test_manuscript_lora_parameter_count_at_vit_b16():
    """Manuscript Sec. 2.4.1 / Supp. Table 1: 589,824 trainable LoRA params (~0.69 % of 86 M)."""
    model = build_backbone(BackboneConfig(), LoRAConfig())
    n_lora = sum(p.numel() for p in lora_parameters(model))
    assert n_lora == 589_824
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 589_824
    assert merge_lora(model) == 24
    assert isinstance(model, ViTBackbone)
