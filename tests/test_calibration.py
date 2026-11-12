import numpy as np
import pytest

from mars_tb.evaluation.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    log_loss,
    reliability_curve,
)


def test_brier_and_log_loss_known_values():
    y, p = np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 0.5, 0.5])
    assert brier_score(y, p) == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)
    assert log_loss(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert np.isfinite(log_loss(np.array([1, 0]), np.array([0.0, 1.0])))  # clipped, not inf


def test_ece_is_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.2, 0.8], 100)
    y = np.concatenate([np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]])
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_known_overconfident_value():
    p = np.full(100, 0.9)
    y = np.r_[np.ones(60), np.zeros(40)]
    assert expected_calibration_error(y, p) == pytest.approx(0.3)


def test_reliability_bins_partition_all_samples_including_p_equal_one():
    p = np.array([0.0, 0.5, 1.0, 0.99])
    r = reliability_curve(np.array([0, 1, 1, 1]), p, n_bins=10)
    assert r["count"].sum() == 4 and r["count"][-1] == 2


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    logit = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-logit))).astype(int)  # truth: calibrated logits
    overconfident = 1 / (1 + np.exp(-3.0 * logit))
    t = fit_temperature(y, overconfident)
    assert t == pytest.approx(3.0, rel=0.15)
    assert expected_calibration_error(y, apply_temperature(overconfident, t)) < expected_calibration_error(
        y, overconfident
    )


def test_temperature_one_is_identity_and_preserves_ranking():
    p = np.array([0.1, 0.4, 0.9])
    assert apply_temperature(p, 1.0) == pytest.approx(p)
    assert np.all(np.diff(apply_temperature(p, 2.5)) > 0)
