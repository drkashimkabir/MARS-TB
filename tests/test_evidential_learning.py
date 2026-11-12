import math

import pytest
import torch
from scipy import stats

from mars_tb.config import HeadConfig, LossConfig
from mars_tb.models.evidential_head import MultiTaskHead, evidential_outputs
from mars_tb.models.losses import (
    IGNORE,
    MultiTaskLoss,
    dirichlet_kl_to_uniform,
    evidential_loss,
    evidential_nll,
)


def test_evidential_identities():
    logits = torch.randn(5, 2)
    o = evidential_outputs(logits)
    alpha = torch.nn.functional.softplus(logits) + 1
    assert torch.allclose(o["alpha"], alpha)
    assert torch.allclose(o["prob"].sum(-1), torch.ones(5))
    assert torch.allclose(o["uncertainty"], 2.0 / alpha.sum(-1))


def test_binary_formulas_from_the_task_specification():
    h = MultiTaskHead(8, HeadConfig())
    out = h(torch.randn(4, 8))["tb"]
    a_neg, a_pos = out["alpha"][:, 0], out["alpha"][:, 1]
    assert torch.allclose(out["p_tb"], a_pos / (a_neg + a_pos))
    assert torch.allclose(out["uncertainty"], 2 / (a_neg + a_pos))


def test_supplementary_table_s6_rows_obey_the_same_identities():
    # (alpha_neg, alpha_pos, p_tb, U) as printed in Supplementary Table S6
    rows = [(1.900, 23.100, 0.924, 0.080), (3.225, 9.275, 0.742, 0.160), (15.217, 1.450, 0.087, 0.120)]
    for an, ap, p, u in rows:
        assert ap / (an + ap) == pytest.approx(p, abs=1e-3)
        assert 2 / (an + ap) == pytest.approx(u, abs=1e-3)


def test_head_class_counts():
    h = MultiTaskHead(8, HeadConfig())
    out = h(torch.randn(2, 8))
    assert out["tb"]["alpha"].shape[-1] == 2
    assert out["pattern"]["alpha"].shape[-1] == 4 and out["severity"]["alpha"].shape[-1] == 4


def test_kl_is_zero_for_the_uniform_dirichlet_and_positive_otherwise():
    assert torch.allclose(dirichlet_kl_to_uniform(torch.ones(3, 4)), torch.zeros(3), atol=1e-6)
    assert (dirichlet_kl_to_uniform(torch.tensor([[5.0, 1.0], [1.0, 9.0]])) > 0).all()


def test_kl_matches_beta_entropy_closed_form():
    # KL(Beta(a,b) || Uniform(0,1)) = -differential entropy of Beta(a,b)
    for a, b in [(2.0, 3.0), (5.0, 1.5), (0.7, 4.0), (10.0, 10.0)]:
        got = float(dirichlet_kl_to_uniform(torch.tensor([[a, b]], dtype=torch.float64)))
        assert got == pytest.approx(-float(stats.beta(a, b).entropy()), abs=1e-9)


def test_kl_term_penalises_only_misleading_evidence():
    alpha = torch.tensor([[1.0, 30.0]])  # all evidence on the TRUE class (index 1)
    right = evidential_loss(alpha, torch.tensor([1]), kl_weight=1.0)
    only_nll = evidential_loss(alpha, torch.tensor([1]), kl_weight=0.0)
    assert float(right) == pytest.approx(float(only_nll), abs=1e-6)  # alpha~ = 1 => KL = 0
    wrong = evidential_loss(alpha, torch.tensor([0]), kl_weight=1.0)
    assert float(wrong) > float(right)


def test_nll_variants_prefer_confident_correct_predictions():
    good, bad = torch.tensor([[1.0, 20.0]]), torch.tensor([[20.0, 1.0]])
    onehot = torch.tensor([[0.0, 1.0]])
    for kind in ("log", "digamma", "mse"):
        assert evidential_nll(good, onehot, kind) < evidential_nll(bad, onehot, kind)
    with pytest.raises(ValueError):
        evidential_nll(good, onehot, "nope")


def test_log_nll_equals_negative_log_expected_probability():
    alpha = torch.tensor([[2.0, 6.0]])
    got = evidential_nll(alpha, torch.tensor([[0.0, 1.0]]), "log")
    assert float(got) == pytest.approx(-math.log(6.0 / 8.0), abs=1e-6)


def test_missing_auxiliary_labels_contribute_nothing():
    alpha = torch.rand(4, 4) * 5 + 1
    target = torch.tensor([IGNORE, 2, IGNORE, 1])
    full = evidential_loss(alpha, target, 0.5)
    kept = evidential_loss(alpha[[1, 3]], target[[1, 3]], 0.5)
    assert torch.allclose(full, kept)


def test_all_ignored_gives_zero_loss_with_a_gradient_path():
    alpha = (torch.rand(3, 4) + 1).requires_grad_()
    loss = evidential_loss(alpha, torch.full((3,), IGNORE), 1.0)
    assert float(loss) == 0.0
    loss.backward()
    assert torch.equal(alpha.grad, torch.zeros_like(alpha))


def test_kl_anneal_schedule_min_one_epoch_over_ten():
    crit = MultiTaskLoss(LossConfig())
    assert [crit.kl_weight(e) for e in (0, 5, 10, 50)] == [0.0, 0.5, 1.0, 1.0]


def test_multitask_weights_and_masking():
    head = MultiTaskHead(8, HeadConfig())
    out = head(torch.randn(6, 8))
    targets = {
        "tb": torch.randint(0, 2, (6,)),
        "pattern": torch.full((6,), IGNORE),
        "severity": torch.full((6,), IGNORE),
    }
    total, parts = MultiTaskLoss(LossConfig())(out, targets, epoch=3)
    assert parts["pattern"] == 0.0 and parts["severity"] == 0.0
    assert parts["total"] == pytest.approx(1.0 * parts["tb"])
    targets["pattern"] = torch.randint(0, 4, (6,))
    _, parts2 = MultiTaskLoss(LossConfig())(out, targets, epoch=3)
    assert parts2["total"] == pytest.approx(parts2["tb"] + 0.3 * parts2["pattern"], rel=1e-5)
    total.backward()


def test_cross_entropy_head_has_no_alpha_or_uncertainty():
    head = MultiTaskHead(8, HeadConfig(type="softmax_ce"))
    out = head(torch.randn(3, 8))
    assert out["tb"]["alpha"] is None and out["tb"]["uncertainty"] is None
    assert torch.allclose(out["tb"]["prob"].sum(-1), torch.ones(3))
    targets = {
        "tb": torch.tensor([0, 1, 1]),
        "pattern": torch.tensor([IGNORE, 0, 1]),
        "severity": torch.tensor([IGNORE] * 3),
    }
    total, _ = MultiTaskLoss(LossConfig(), "softmax_ce")(out, targets, 0)
    assert torch.isfinite(total)
