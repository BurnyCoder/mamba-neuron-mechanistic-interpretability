"""Sanity tests of the mechanism tools on mamba-130m (consistency identities, not results).

Global context: H3 and the exploratory mechanism results rely on these functions.
"""

import numpy as np

from mamba_neurons.data import Windows
from mamba_neurons.mechanism import context_truncation, factor_split, next_token_logp, patching_effect, top_positions, zero_ablation

LAYER, NEURON = 20, 7


def _windows(sample_ids):
    toks = np.concatenate([np.zeros((2, 1), np.int64), sample_ids.cpu().numpy()], axis=1)
    return Windows(toks.astype(np.int32), np.array([0, 1]), np.array([0, 0]), ["a", "b"])


def test_top_positions_sorted(small_model, sample_ids):
    model, _ = small_model
    rows, pos, vals = top_positions(model, _windows(sample_ids), LAYER, NEURON, 1, n=5)
    assert len(rows) == 5 and (np.diff(vals) <= 0).all() and (pos >= 1).all()


def test_full_context_truncation_ratio_is_one(small_model, sample_ids):
    model, _ = small_model
    w = _windows(sample_ids)
    out = context_truncation(model, w, [0, 1], [10, 9], LAYER, NEURON, ks=(1, None))
    assert abs(out["all"] - 1.0) < 1e-4 and "1" in out


def test_factor_split_reconstructs_activation(small_model, sample_ids):
    model, _ = small_model
    fs = factor_split(model, _windows(sample_ids), [0, 1], [5, 6], LAYER, NEURON)
    np.testing.assert_allclose(np.array(fs["y"]) * np.array(fs["silu_z"]), fs["a"], rtol=1e-4, atol=1e-6)
    assert 0 <= fs["frac_gate_negative"] <= 1


def test_ablation_and_patching_run(small_model, sample_ids):
    model, _ = small_model
    w = _windows(sample_ids)
    clean = next_token_logp(model, w, [0, 1], [3, 4])
    ablated = next_token_logp(model, w, [0, 1], [3, 4], zero_ablation(LAYER, NEURON))
    assert clean.shape == (2,) and (clean <= 0).all() and not np.allclose(clean, ablated)
    assert abs(patching_effect(model, w, 500.0, [0], [3], LAYER, NEURON, [11, 12])) > 0  # a large patch moves the logits
