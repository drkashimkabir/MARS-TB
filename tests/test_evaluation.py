import math

import numpy as np
import pytest
from sklearn import metrics as skm

from mars_tb.evaluation.metrics import (
    average_precision,
    binary_report,
    clopper_pearson,
    confusion,
    mcc,
    mcc_from_rates,
    roc_auc,
    roc_curve,
)


def _data(n=300, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    s = rng.normal(y * 1.2, 1.0)
    return y, (np.round(s, 1) if ties else s)


@pytest.mark.parametrize("ties", [False, True])
def test_auc_and_ap_match_scikit_learn(ties):
    y, s = _data(ties=ties)
    assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)
    assert average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)


def test_roc_curve_matches_scikit_learn_area():
    y, s = _data()
    fpr, tpr, _ = roc_curve(y, s)
    assert np.trapz(tpr, fpr) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)


def test_auc_edge_cases():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert math.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        roc_auc([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1, float("nan")])


def test_confusion_and_rates_on_a_hand_computed_example():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.2, 0.3, 0.4])
    r = binary_report(y, p, 0.5)
    assert confusion(y, (p >= 0.5).astype(int)) == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}
    assert (r["sensitivity"], r["specificity"]) == (0.75, 5 / 6)
    assert r["ppv"] == 0.75 and r["npv"] == 5 / 6 and r["accuracy"] == 0.8
    assert r["f1"] == pytest.approx(0.75) and r["precision"] == r["ppv"] and r["recall"] == r["sensitivity"]
    expected_mcc = (3 * 5 - 1 * 1) / math.sqrt(4 * 4 * 6 * 6)
    assert r["mcc"] == pytest.approx(expected_mcc)
    assert r["prevalence"] == 0.4


def test_threshold_is_inclusive():
    r = binary_report(np.array([1, 0]), np.array([0.5, 0.5]), 0.5)
    assert r["tp"] == 1 and r["fp"] == 1


def test_ppv_npv_undefined_without_predictions():
    r = binary_report(np.array([1, 0, 1, 0]), np.array([0.1, 0.1, 0.2, 0.3]), 0.9)
    assert math.isnan(r["ppv"]) and r["npv"] == 0.5


def test_mcc_helpers():
    assert mcc(10, 10, 0, 0) == 1.0 and mcc(0, 0, 10, 10) == -1.0
    assert math.isnan(mcc(0, 0, 0, 0))
    assert mcc_from_rates(1.0, 1.0, 5, 5) == 1.0


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-3)
    from scipy import stats

    for k, n in [(5, 10), (1, 20), (87, 100), (238, 238)]:
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        assert clopper_pearson(k, n) == pytest.approx((ref.low, ref.high), abs=1e-9)
    assert clopper_pearson(10, 10)[1] == 1.0
    assert all(math.isnan(v) for v in clopper_pearson(0, 0))


def test_report_confidence_intervals_bracket_the_rates():
    y, s = _data(500)
    r = binary_report(y, s, 0.5)
    assert r["sensitivity_ci"][0] <= r["sensitivity"] <= r["sensitivity_ci"][1]
    assert r["specificity_ci"][0] <= r["specificity"] <= r["specificity_ci"][1]
