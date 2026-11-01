"""Confirmation statistics on synthetic judged rows: no model or LLM needed.

Global context: `_confirm_stats` produces results/<model>/h2.json and h2b.json; the post-hoc clamp-vs-baseline
differences and the prompt count it stores back the baseline comparisons in the reports and the viewer, and the
`posthoc_audit` block backs the 2026-11-01 independent audit (pre-registration amendment 9).
"""

import json

import numpy as np
import pytest

from mamba_neurons.metrics import cluster_bootstrap_diff
from mamba_neurons.phases.steering import _confirm_stats

KEY = "neuron:L1:N2:+"
HM = {"unsteered": [0.0, 0.0, 0.67], "clamp": [1.0, 1.33, 2.0], "add": [1.0, 1.0, 2.0],
      "diffmean": [0.0, 1.33, 0.67], "prompting": [0.67, 0.0, 1.0]}  # per prompt, identical for both seeds
TEXT = {"unsteered": "no", "clamp": "as it was, as", "add": "as", "diffmean": "as as as", "prompting": "as"}
PLAN = {KEY: {"label": "l", "concept": "c", "best_level": 0.5}}


def _rows(clamp_fluency: float = 1.0):
    rows = [{"key": KEY, "condition": c, "seed": s, "prompt_idx": i, "hm": h, "concept": h, "nll": 1.0, "distinct2": 0.9,
             "fluency": clamp_fluency if c == "clamp" else 1.0, "text": TEXT[c]}
            for c, hs in HM.items() for s in (0, 1) for i, h in enumerate(hs)]
    rows += [{"key": KEY, "condition": f"random:{j}", "seed": 0, "prompt_idx": i, "hm": 0.0, "concept": 0.0, "fluency": 1.0,
              "nll": 1.0, "distinct2": 0.9, "text": "x"} for j in (5, 6) for i in range(2)]  # null: prompts 0-1, seed 0
    return rows


def _stats(settings, tmp_path, **kw):
    s = settings.with_overrides(results_dir=tmp_path)
    _confirm_stats(s, _rows(**kw), PLAN, "h2", noninferiority=False, targets={KEY: " as"})
    return s, json.loads((s.results_path() / "h2.json").read_text(encoding="utf-8"))["results"][KEY]


def test_confirm_stats_stores_prompt_count_and_paired_baseline_differences(settings, tmp_path):
    s, r = _stats(settings, tmp_path)
    assert r["n_prompts"] == 3
    clusters = [i for _ in (0, 1) for i in range(3)]
    for base in ("prompting", "diffmean"):
        want = cluster_bootstrap_diff(HM["clamp"] * 2, HM[base] * 2, clusters, seed=s.seed)
        got = r[f"clamp_minus_{base}"]
        assert np.isclose(got["diff"], want[0]) and np.allclose(got["ci95"], want[1:])


def test_posthoc_audit_word_counts_like_for_like_and_noninferiority(settings, tmp_path):
    _, r = _stats(settings, tmp_path)
    a = r["posthoc_audit"]
    assert a["target_word"] == "as"
    assert a["word_count"]["clamp"] == 2 and a["word_count"]["unsteered"] == 0 and a["word_count"]["diffmean"] == 3
    assert a["word_count_null"] == [0.0, 0.0] and a["p_word_count_vs_null"] == pytest.approx(1 / 3)
    assert a["word_count_clamp_minus_unsteered"]["diff"] == pytest.approx(2.0)
    lfl = a["like_for_like"]  # the target scored only on the null's prompts (0, 1) and seed (0)
    assert lfl["n_generations"] == 2 and lfl["clamp_hm"] == pytest.approx(np.mean(HM["clamp"][:2]))
    assert lfl["p_hm_vs_null"] == pytest.approx(1 / 3)
    assert a["clamp_minus_unsteered"]["fluency"]["ci95"] == pytest.approx([0.0, 0.0])
    assert a["distinct2_clamp_minus_0.9_unsteered"]["ci95"] == pytest.approx([0.09, 0.09])  # 0.9 - 0.9 * 0.9
    assert a["noninferior_fluency_ci"] is True
    for base in ("prompting", "diffmean"):
        assert 0 <= a[f"p_clamp_vs_{base}"] <= a[f"p_clamp_vs_{base}_holm"] <= 1
    assert a["p_word_count_vs_null_holm"] == pytest.approx(1 / 3)  # one pole: Holm leaves p unchanged


def test_ci_based_noninferiority_fails_when_fluency_drops_beyond_the_margin(settings, tmp_path):
    _, r = _stats(settings, tmp_path, clamp_fluency=0.7)  # clamp - unsteered fluency = -0.3 < -0.2
    assert r["posthoc_audit"]["noninferior_fluency_ci"] is False
