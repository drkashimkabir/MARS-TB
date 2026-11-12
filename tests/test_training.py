import numpy as np
import pytest
import torch

from mars_tb.config import TrainConfig, effective_batch_size, load_train_config, validate_train_config
from mars_tb.data.datasets import DistributedWeightedSampler, class_sampling_weights, make_weighted_sampler
from mars_tb.models.mars_tb import MARSTB
from mars_tb.training.optimizer import build_optimizer, lr_factor_schedule, set_lr
from mars_tb.training.trainer import EarlyStopper
from mars_tb.utils.seed import seed_everything
from tests.conftest import tiny_config


def test_lr_schedule_warmup_then_cosine_to_min():
    base, mn = 1e-4, 1e-7
    f = lambda e: lr_factor_schedule(e, 5, 100, base, mn)  # noqa: E731
    assert f(0) == 0.0
    assert f(2.5) == pytest.approx(base / 2)
    assert f(5) == pytest.approx(base)
    assert f(100) == pytest.approx(mn)
    assert f(52.5) == pytest.approx(mn + (base - mn) / 2, rel=1e-6)  # cosine midpoint
    assert all(f(e) >= f(e + 1) for e in range(5, 99))  # monotone decay
    assert lr_factor_schedule(3, 5, 100, 0.0, mn) == 0.0  # frozen group stays at 0


def test_optimizer_groups_use_manuscript_learning_rates():
    model = MARSTB(tiny_config())
    cfg = TrainConfig()
    opt = build_optimizer(model, cfg)
    lrs = {g["name"]: g["base_lr"] for g in opt.param_groups}
    assert lrs == {"lora": 1e-4, "new": 1e-5}  # backbone is frozen: no group at all
    assert opt.defaults["weight_decay"] == 0.01 and opt.defaults["betas"] == (0.9, 0.999)
    assert isinstance(opt, torch.optim.AdamW)
    assert set(set_lr(opt, cfg, 0.0).values()) == {0.0}
    assert set_lr(opt, cfg, 5.0) == {"lora": pytest.approx(1e-4), "new": pytest.approx(1e-5)}
    # every trainable parameter is in exactly one group, nothing frozen leaked in
    in_groups = {id(p) for g in opt.param_groups for p in g["params"]}
    assert in_groups == {id(p) for p in model.parameters() if p.requires_grad}


def test_unfrozen_backbone_with_zero_lr_is_rejected():
    cfg = tiny_config(lora={"enabled": False} if False else tiny_config().lora)
    from dataclasses import replace

    base = tiny_config()
    full = replace(base, lora=replace(base.lora, enabled=False), backbone=replace(base.backbone, freeze_backbone=False))
    with pytest.raises(ValueError, match="never move"):
        build_optimizer(MARSTB(full), TrainConfig())
    assert cfg is not None


def test_effective_batch_size_resolves_manuscript_discrepancy_h():
    cfg = TrainConfig()
    assert effective_batch_size(cfg, world_size=1) == 32  # the manuscript's number = 1 process
    assert effective_batch_size(cfg, world_size=2) == 64  # true global size on 2 GPUs


def test_default_train_config_matches_manuscript_table3():
    c = TrainConfig()
    assert (c.lr_lora, c.lr_new, c.lr_backbone, c.weight_decay, c.betas) == (1e-4, 1e-5, 0.0, 0.01, (0.9, 0.999))
    assert (c.warmup_epochs, c.epochs, c.min_lr, c.grad_clip) == (5, 100, 1e-7, 1.0)
    assert (c.batch_size_per_gpu, c.grad_accum_steps, c.early_stopping_patience) == (16, 2, 15)
    assert c.loss.weights == (1.0, 0.3, 0.3) and c.loss.kl_anneal_epochs == 10


def test_train_config_validation():
    with pytest.raises(ValueError):
        validate_train_config(TrainConfig(grad_accum_steps=0))
    with pytest.raises(ValueError):
        validate_train_config(TrainConfig(warmup_epochs=100))
    with pytest.raises(ValueError):
        validate_train_config(TrainConfig(sampler_weights="inverse"))


def test_sampler_weights_are_derived_from_training_labels():
    revised = np.array([0] * 6000 + [1] * 714)
    w = class_sampling_weights(revised)
    assert w[0] == 1.0 and w[1] == pytest.approx(6000 / 714) and w[1] == pytest.approx(8.403, abs=1e-3)
    stale = np.array([0] * 9085 + [1] * 2019)
    assert class_sampling_weights(stale)[1] == pytest.approx(4.4997, abs=1e-3)  # manuscript's "4.50"
    assert class_sampling_weights(revised, "fixed:4.5")[1] == 4.5
    with pytest.raises(ValueError):
        class_sampling_weights(np.zeros(5))
    with pytest.raises(ValueError):
        class_sampling_weights(revised, "bogus")


def test_weighted_sampler_balances_classes_in_expectation():
    labels = np.array([0] * 900 + [1] * 100)
    idx = list(make_weighted_sampler(labels, seed=0))
    frac_pos = labels[idx].mean()
    assert 0.40 < frac_pos < 0.60  # ~50 % positives after weighting n_neg/n_pos


def test_distributed_sampler_ranks_get_disjoint_equal_slices_of_one_draw():
    labels = np.array([0] * 80 + [1] * 20)
    a = DistributedWeightedSampler(labels, "neg_over_pos", 0, 2, seed=1)
    b = DistributedWeightedSampler(labels, "neg_over_pos", 1, 2, seed=1)
    assert len(a) == len(b) == 50 and len(list(a)) == 50
    assert list(a) != list(b)
    a.set_epoch(1)
    assert list(a) != list(DistributedWeightedSampler(labels, "neg_over_pos", 0, 2, seed=1))


def test_early_stopping_patience_and_best_tracking():
    s = EarlyStopper(patience=3)
    assert [s.step(v) for v in (0.5, 0.6, 0.6, 0.55, 0.59)] == [True, True, False, False, False]
    assert s.should_stop and s.best == 0.6
    s2 = EarlyStopper(patience=2)
    s2.step(0.5)
    s2.step(0.4)
    assert not s2.should_stop
    s2.step(0.7)  # improvement resets the counter
    assert s2.bad_epochs == 0


def test_seed_everything_makes_runs_reproducible():
    seed_everything(11)
    a = torch.randn(3)
    seed_everything(11)
    assert torch.equal(a, torch.randn(3))
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_gradient_accumulation_matches_one_big_batch():
    """Two accumulated micro-batches of 2 give the same gradients as one batch of 4 (no BN/dropout)."""
    from torch import nn

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 1))
    x, y = torch.randn(4, 4), torch.randn(4, 1)
    loss = lambda xb, yb: ((net(xb) - yb) ** 2).mean()  # noqa: E731
    net.zero_grad()
    loss(x, y).backward()
    big = [p.grad.clone() for p in net.parameters()]
    net.zero_grad()
    for xb, yb in ((x[:2], y[:2]), (x[2:], y[2:])):
        (loss(xb, yb) / 2).backward()
    for g, p in zip(big, net.parameters(), strict=True):
        assert torch.allclose(g, p.grad, atol=1e-6)


def test_load_train_config_from_repo_yaml():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_train_config(root / "configs" / "training" / "default.yaml")
    assert cfg.lr_lora == 1e-4 and cfg.epochs == 100
