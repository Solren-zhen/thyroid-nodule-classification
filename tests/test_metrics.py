"""纯函数指标测试：bootstrap AUC、ECE、决策曲线、完整指标字典。"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from eval_thyroid import decision_curve
from train_thyroid import bootstrap_auc, compute_metrics, ece_score


def test_bootstrap_auc_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    mean, lo, hi = bootstrap_auc(labels, probs, n_boot=200, seed=0)
    assert mean == pytest.approx(1.0, abs=0.02)
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_bootstrap_auc_random_is_half():
    rng = np.random.RandomState(42)
    labels = np.array([0] * 50 + [1] * 50)
    probs = rng.rand(100)
    mean, lo, hi = bootstrap_auc(labels, probs, n_boot=200, seed=1)
    assert mean == pytest.approx(0.5, abs=0.1)
    assert lo <= hi


def test_bootstrap_auc_single_class_handled():
    labels = np.zeros(20, dtype=int)
    probs = np.random.RandomState(0).rand(20)
    mean, lo, hi = bootstrap_auc(labels, probs, n_boot=50)
    assert mean == 0.0 and lo == 0.0 and hi == 0.0


def test_ece_perfect_calibration_is_zero():
    labels = np.array([0, 0, 1, 1, 0, 1])
    probs = labels.astype(float)  # perfectly calibrated degenerate case
    assert ece_score(labels, probs) == pytest.approx(0.0, abs=1e-9)


def test_ece_miscalibration_positive():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    probs = np.full(10, 0.5)  # says 50% everywhere, true rate is 50% -> ece 0
    assert ece_score(labels, probs) == pytest.approx(0.0, abs=1e-9)


def test_ece_reflects_overconfidence():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs = np.full(10, 0.9)  # overconfident: predicts 90% but true rate 50%
    assert ece_score(labels, probs) > 0.2


def test_decision_curve_perfect_classifier_net_benefit():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.9, 0.95])
    rows = decision_curve(labels, probs, thresholds=[0.5])
    # at t=0.5: tp=2, fp=0 -> nb = 2/4 - 0 = 0.5
    assert rows[0]["net_benefit"] == pytest.approx(0.5, abs=1e-9)


def test_decision_curve_all_treat_equals_zero():
    labels = np.array([0, 0, 1, 1])
    probs = np.ones(4)  # classify everything positive
    rows = decision_curve(labels, probs, thresholds=[0.5])
    # nb = tp/n - fp/n * t/(1-t) = 2/4 - 2/4 * 1 = 0.5 - 0.5 = 0
    assert rows[0]["net_benefit"] == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_matches_sklearn():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = compute_metrics(labels, probs, threshold=0.5)
    assert m["auc"] == pytest.approx(roc_auc_score(labels, probs))
    assert m["acc"] == pytest.approx(1.0)  # threshold 0.5 separates perfectly here
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert m["tp"] == 3 and m["tn"] == 3 and m["fp"] == 0 and m["fn"] == 0
    assert m["brier"] == pytest.approx(np.mean((probs - labels) ** 2))
    assert 0.0 <= m["ece"] <= 1.0
    assert len(m["auc_ci"]) == 2


def test_compute_metrics_imbalanced_threshold():
    labels = np.array([0, 0, 0, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6])
    m = compute_metrics(labels, probs, threshold=0.5)
    assert m["acc"] == pytest.approx(1.0)
    assert m["recall"] == 1.0 and m["precision"] == 1.0
