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


def test_bootstrap_p_matches_the_cluster_ci():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(40), 2)
    b = rng.normal(0, 1, 80)
    assert M.bootstrap_p(b + 1.0, b + rng.normal(0, 0.1, 80), clusters) < 0.001  # clear shift
    assert M.bootstrap_p(b, b, clusters) == 1.0  # every paired difference is 0
    a = b + rng.normal(0, 1, 80)
    _, lo, hi = M.cluster_bootstrap_diff(a, b, clusters)
    assert (M.bootstrap_p(a, b, clusters) < 0.05) == (lo > 0 or hi < 0)  # same resamples as the CI


def test_kappa_ci_brackets_kappa():
    assert M.kappa_ci([0, 1] * 20, [0, 1] * 20) == pytest.approx((1.0, 1.0))
    rng = np.random.default_rng(3)
    r1 = rng.integers(0, 3, 60)
    r2 = np.where(rng.random(60) < 0.7, r1, rng.integers(0, 3, 60))
    lo, hi = M.kappa_ci(r1, r2)
    assert lo < M.cohen_kappa(list(r1), list(r2)) < hi and hi - lo > 0.1


def test_count_word_and_mannwhitney():
    assert M.count_word("As far as I know, he has it as", " as") == 3  # whole words, case-insensitive
    assert M.count_word("see [1] and [a]", "[") == 2  # a non-word token is counted as a substring
    assert M.mannwhitney_greater([3, 4, 5, 6], [0, 1, 2, 3]) < 0.05 < M.mannwhitney_greater([0, 1, 2, 3], [3, 4, 5, 6])


def test_h1_posthoc_mean_ba_and_thresholds(settings, tmp_path):
    from mamba_neurons.jsonio import load_json, save_json
    from mamba_neurons.phases.interp import phase_h1

    s = settings.with_overrides(results_dir=tmp_path)
    ba = {"neuron": [0.9, 0.7, 0.6, 0.6, 0.85, 0.65, 0.7, 0.5], "direction": [0.5, 0.5, 0.6, 0.4, 0.85, 0.5, 0.55, 0.5]}
    res = {f"{k}:L{j}:+{p}": {"unit": {"kind": k, "layer": j, "index": 0, "sign": 1 - 2 * p, "group": "audit"},
                               "balanced_accuracy": b, "valid": True}
           for k, bs in ba.items() for n, b in enumerate(bs) for j, p in [divmod(n, 2)]}
    save_json(res, s.results_path() / "autointerp.json")
    phase_h1(s)
    post = load_json(s.results_path() / "h1.json")["posthoc"]
    assert post["mean_ba_diff"]["diff"] == pytest.approx(np.mean(ba["neuron"]) - np.mean(ba["direction"]))
    assert post["mean_ba_diff"]["mannwhitney_one_sided_p"] < 0.2
    assert post["clean_rate_by_threshold"]["0.6"] == pytest.approx([7 / 8, 2 / 8])
