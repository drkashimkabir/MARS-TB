import math

import numpy as np
import pytest

from mars_tb.evaluation.uncertainty import error_detection, risk_coverage, selective_accuracy, uncertainty_summary


def _setup():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    yhat = np.array([1, 0, 0, 1, 1, 0, 1, 0])  # errors at indices 1 and 3
    informative = np.array([0.1, 0.9, 0.1, 0.8, 0.1, 0.2, 0.15, 0.1])
    return y, yhat, informative


def test_informative_uncertainty_detects_errors_perfectly():
    y, yhat, u = _setup()
    r = error_detection(y, yhat, u)
    assert r["n_errors"] == 2 and r["auroc"] == 1.0


def test_uninformative_uncertainty_is_chance_and_no_errors_is_nan():
    y, yhat, _ = _setup()
    assert error_detection(y, yhat, np.full(8, 0.5))["auroc"] == 0.5
    assert math.isnan(error_detection(y, y, np.zeros(8))["auroc"])


def test_risk_coverage_is_zero_risk_until_the_first_error_enters():
    y, yhat, u = _setup()
    rc = risk_coverage(y, yhat, u)
    assert rc["coverage"][-1] == 1.0 and rc["risk"][-1] == pytest.approx(2 / 8)
    assert (rc["risk"][:6] == 0).all()  # six most-certain predictions are all correct
    random_aurc = risk_coverage(y, yhat, np.random.default_rng(0).random(8))["aurc"]
    assert rc["aurc"] < random_aurc


def test_selective_accuracy_improves_when_rejecting_uncertain_cases():
    y, yhat, u = _setup()
    assert selective_accuracy(y, yhat, u, 1.0) == pytest.approx(0.75)
    assert selective_accuracy(y, yhat, u, 0.75) == 1.0
    with pytest.raises(ValueError):
        selective_accuracy(y, yhat, u, 0.0)


def test_summary_separates_correct_and_incorrect():
    y, yhat, u = _setup()
    s = uncertainty_summary(y, yhat, u)
    assert s["mean_u_incorrect"] == pytest.approx(0.85) and s["mean_u_correct"] < s["mean_u_incorrect"]
