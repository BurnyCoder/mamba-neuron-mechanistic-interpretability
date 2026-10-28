"""Tests for the statistics helpers and the auto-interp text/score helpers (no LLM or GPU needed).

Global context: H1/H2 verdicts depend on these numbers being right.
"""

import numpy as np
import pytest

from mamba_neurons import metrics as M
from mamba_neurons.autointerp import audit_units, balanced_accuracy, detect, highlight, unit_key


def test_wilson_and_fisher():
    lo, hi = M.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.19, abs=0.01)
    assert M.fisher_greater(30, 40, 10, 40) < 0.001 and M.fisher_greater(10, 40, 30, 40) > 0.99


def test_bootstraps_cover_truth():
    rng = np.random.default_rng(1)
    d, lo, hi = M.bootstrap_diff_of_props(rng.random(400) < 0.6, rng.random(400) < 0.4)
    assert lo < 0.2 < hi and d == pytest.approx(0.2, abs=0.08)
    clusters = np.repeat(np.arange(50), 2)
    a = rng.normal(1.0, 1, 100)
    m, lo, hi = M.cluster_bootstrap_diff(a, a - 0.5, clusters)
    assert m == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_empirical_p_holm_power_kappa():
    assert M.empirical_p(10, list(range(20))) == pytest.approx(11 / 21)
    assert M.empirical_p(100, list(range(20))) == pytest.approx(1 / 21)
    assert M.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert 35 < M.power_n_per_arm(0.1, 0.4, alpha=0.05 / 3) < 45  # pre-registration: ~40 per arm
    assert M.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_harmonic_mean_and_distinct():
    assert M.harmonic_mean(2, 2) == 2 and M.harmonic_mean(0, 2) == 0 and M.harmonic_mean(1, 2) == pytest.approx(4 / 3)
    assert M.distinct_n([1, 2, 1, 2, 1, 2], 2) == pytest.approx(2 / 5)


def test_audit_units_and_keys():
    units = audit_units(n_layers=3, n_neurons=10, n_dirs=2, per_layer=2, seed=0)
    assert len(units) == 3 * (2 + 2) * 2  # (neurons + directions) x 2 poles per layer
    assert unit_key(units[0]).startswith("neuron:L0:N") and len({unit_key(u) for u in units}) == len(units)


def test_highlight_marks_peak_run():
    ex = {"tokens": ["a", " b", " c", " d"], "acts": [0.0, 5.0, 4.0, 0.1], "peak": 1}
    assert highlight(ex, 1) == "a<< b c>> d"
    assert highlight({**ex, "acts": [0, -5, 0, 0]}, -1) == "a<< b>> c d"


class _FakeLLM:
    def __init__(self, answers):
        self.answers = answers

    def json(self, kind, system, user, schema):
        return {"answers": self.answers}


def test_detect_scores_and_handles_malformed_reply():
    pole = {"detect_top": [{"tokens": ["x"]}] * 2, "detect_random": [{"tokens": ["y"]}] * 2}
    perfect = detect(_FakeLLM(None), "e", pole, seed=0)  # placeholder to obtain the shuffled truth order
    truth = perfect["truth"]
    assert detect(_FakeLLM(truth), "e", pole, seed=0)["balanced_accuracy"] == 1.0
    assert detect(_FakeLLM([1 - t for t in truth]), "e", pole, seed=0)["balanced_accuracy"] == 0.0
    bad = detect(_FakeLLM([1]), "e", pole, seed=0)
    assert bad["valid"] is False and bad["balanced_accuracy"] == 0.5
    assert balanced_accuracy([1, 1, 0, 0], [1, 1, 1, 1]) == 0.5


def test_cluster_bootstrap_of_rates_matches_plain_rates():
    d, lo, hi = M.bootstrap_diff_of_props([1, 1, 0, 0], [0, 0, 0, 1], clusters1=[0, 0, 1, 1], clusters2=[5, 5, 6, 6])
    assert d == pytest.approx(0.25) and lo <= d <= hi
