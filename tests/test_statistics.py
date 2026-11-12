import numpy as np
import pytest

from mars_tb.evaluation.metrics import roc_auc
from mars_tb.evaluation.statistical_tests import (
    bootstrap_ci,
    delong_auc_ci,
    delong_test,
    holm,
    paired_bootstrap_diff,
    seed_summary,
)
from mars_tb.evaluation.threshold_selection import (
    ThresholdLeakageError,
    load_threshold,
    save_threshold,
    select_threshold,
    sensitivity_constrained_threshold,
    youden_threshold,
)


def _pair(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    common = rng.normal(0, 1, n)
    s1 = common + 1.4 * y + rng.normal(0, 0.5, n)
    s2 = common + 0.9 * y + rng.normal(0, 0.5, n)
    return y, s1, s2


def _structural_components_bruteforce(y, s):
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(), psi.mean(1), psi.mean(0)  # AUC, V10 per positive, V01 per negative


def test_delong_auc_and_variance_match_the_o_n2_definition():
    y, s1, _ = _pair(150)
    auc, v10, v01 = _structural_components_bruteforce(y, s1)
    var = v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01)
    got = delong_auc_ci(y, s1)
    assert got["auc"] == pytest.approx(auc, abs=1e-12) == pytest.approx(roc_auc(y, s1), abs=1e-12)
    assert got["se"] == pytest.approx(np.sqrt(var), rel=1e-9)


def test_delong_identical_scores_give_zero_difference_and_p_one():
    y, s1, _ = _pair()
    r = delong_test(y, s1, s1)
    assert r["diff"] == 0.0 and (np.isnan(r["p"]) or r["p"] == pytest.approx(1.0))


def test_delong_detects_a_real_difference_and_is_antisymmetric():
    y, s1, s2 = _pair()
    a, b = delong_test(y, s1, s2), delong_test(y, s2, s1)
    assert a["diff"] > 0 and a["p"] < 0.05
    assert a["diff"] == pytest.approx(-b["diff"]) and a["p"] == pytest.approx(b["p"])
    assert a["lo"] < a["diff"] < a["hi"]


def test_delong_se_agrees_with_bootstrap_se():
    y, s1, _ = _pair(300)
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(600):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc(y[idx], s1[idx]))
    assert delong_auc_ci(y, s1)["se"] == pytest.approx(np.std(boots, ddof=1), rel=0.2)


def test_holm_known_values_and_monotonicity():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    adj = holm([0.001, 0.002, 0.9, 0.0004])
    assert all(0 <= a <= 1 for a in adj)
    assert holm([0.4, 0.4, 0.4]) == pytest.approx([1.0, 1.0, 1.0])


def test_bootstrap_ci_is_seeded_and_brackets_the_point_estimate():
    y, s1, _ = _pair()
    a = bootstrap_ci(y, s1, n_boot=300, seed=5)
    b = bootstrap_ci(y, s1, n_boot=300, seed=5)
    assert a == b and a["lo"] <= a["point"] <= a["hi"] and a["clustered"] is False


def test_patient_clustered_bootstrap_is_wider_when_images_repeat_per_patient():
    rng = np.random.default_rng(0)
    n_pat, reps = 80, 6
    y_p = (rng.random(n_pat) < 0.4).astype(int)
    s_p = rng.normal(y_p * 1.0, 1.0)
    y, s = np.repeat(y_p, reps), np.repeat(s_p, reps) + rng.normal(0, 0.01, n_pat * reps)
    groups = np.repeat(np.arange(n_pat), reps)
    naive = bootstrap_ci(y, s, n_boot=400, seed=1)
    clustered = bootstrap_ci(y, s, n_boot=400, seed=1, groups=groups)
    assert (clustered["hi"] - clustered["lo"]) > 1.3 * (naive["hi"] - naive["lo"])
    assert clustered["clustered"] is True


def test_paired_bootstrap_difference_agrees_in_sign_with_delong():
    y, s1, s2 = _pair()
    r = paired_bootstrap_diff(y, s1, s2, n_boot=300, seed=2)
    assert r["diff"] > 0 and r["lo"] > 0 and r["p_bootstrap"] < 0.05


def test_seed_summary_t_interval():
    r = seed_summary([0.97, 0.96, 0.98, 0.972, 0.968])
    assert r["n"] == 5 and r["mean"] == pytest.approx(0.97)
    assert r["lo"] < r["mean"] < r["hi"]
    with pytest.raises(ValueError):
        seed_summary([0.9])


def test_youden_threshold_on_separable_and_noisy_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    assert youden_threshold(y, p) == 0.6
    t = sensitivity_constrained_threshold(y, p, 1.0)
    assert t == 0.6
    with pytest.raises(ValueError):
        sensitivity_constrained_threshold(y, p, 1.01)


@pytest.mark.parametrize("split", ["test", "external", "train"])
def test_threshold_selection_refuses_non_validation_splits(split):
    y, p = np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ThresholdLeakageError):
        select_threshold(y, p, ["a", "b", "c", "d"], "d", split)
    ok = select_threshold(y, p, ["a", "b", "c", "d"], "d", split, allow_exploratory=True)
    assert ok.selected_on_split == split


def test_frozen_threshold_roundtrip_and_digest_is_order_independent(tmp_path):
    y, p = np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    a = select_threshold(y, p, ["a", "b", "c", "d"], "tbx11k", "val")
    b = select_threshold(y, p, ["d", "c", "b", "a"], "tbx11k", "val")
    assert a.image_id_digest == b.image_id_digest
    save_threshold(tmp_path / "t.json", a)
    assert load_threshold(tmp_path / "t.json") == a
