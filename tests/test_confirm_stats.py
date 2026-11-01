"""Confirmation statistics on synthetic judged rows: no model or LLM needed.

Global context: `_confirm_stats` produces results/<model>/h2.json and h2b.json; the post-hoc clamp-vs-baseline
differences and the prompt count it stores back the baseline comparisons in the reports and the viewer.
"""

import json

import numpy as np

from mamba_neurons.metrics import cluster_bootstrap_diff
from mamba_neurons.phases.steering import _confirm_stats

KEY = "neuron:L1:N2:+"
HM = {"unsteered": [0.0, 0.0, 0.67], "clamp": [1.0, 1.33, 2.0], "add": [1.0, 1.0, 2.0],
      "diffmean": [0.0, 1.33, 0.67], "prompting": [0.67, 0.0, 1.0]}  # per prompt, identical for both seeds


def _rows():
    rows = [{"key": KEY, "condition": c, "seed": s, "prompt_idx": i, "hm": h, "concept": h, "fluency": 1.0, "nll": 1.0,
             "distinct2": 0.9} for c, hs in HM.items() for s in (0, 1) for i, h in enumerate(hs)]
    rows += [{"key": KEY, "condition": f"random:{j}", "seed": 0, "prompt_idx": i, "hm": 0.0, "concept": 0.0, "fluency": 1.0,
              "nll": 1.0, "distinct2": 0.9} for j in (5, 6) for i in range(3)]
    return rows


def test_confirm_stats_stores_prompt_count_and_paired_baseline_differences(settings, tmp_path):
    s = settings.with_overrides(results_dir=tmp_path)
    plan = {KEY: {"label": "l", "concept": "c", "best_level": 0.5}}
    _confirm_stats(s, _rows(), plan, "h2", noninferiority=False)
    r = json.loads((s.results_path() / "h2.json").read_text(encoding="utf-8"))["results"][KEY]
    assert r["n_prompts"] == 3
    clusters = [i for _ in (0, 1) for i in range(3)]
    for base in ("prompting", "diffmean"):
        want = cluster_bootstrap_diff(HM["clamp"] * 2, HM[base] * 2, clusters, seed=s.seed)
        got = r[f"clamp_minus_{base}"]
        assert np.isclose(got["diff"], want[0]) and np.allclose(got["ci95"], want[1:])
