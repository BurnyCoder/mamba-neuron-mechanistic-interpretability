"""Steering helpers on mamba-130m: conditions change generations, NLL and DiffMean have the right shapes.

Global context: these are the building blocks of the confirmatory H2 experiment.
"""

import numpy as np
import torch

from mamba_neurons.collect import UnitStats
from mamba_neurons.data import Windows
from mamba_neurons.model import record
from mamba_neurons.steer import clamp_setup, continuation_nll, diffmean_vector, generate, neutral_prompts, residual_budget

LAYER, NEURON = 20, 7


def _windows(sample_ids):
    toks = np.concatenate([np.zeros((2, 1), np.int64), sample_ids.cpu().numpy()], axis=1)
    return Windows(toks.astype(np.int32), np.array([0, 1]), np.array([0, 0]), ["a", "b"])


def test_neutral_prompts_disjoint_sets(sample_ids):
    w = _windows(sample_ids)
    a = neutral_prompts(w, 1, 4, seed=0)
    b = neutral_prompts(w, 1, 4, seed=0, skip=1)
    assert a.shape == (1, 5) and not np.array_equal(a, b)


def test_generate_is_seeded_and_clamp_changes_text(small_model, sample_ids):
    model, _ = small_model
    prompts = sample_ids.cpu().numpy()
    g1 = generate(model, prompts, None, seed=3, max_new_tokens=8)
    g2 = generate(model, prompts, None, seed=3, max_new_tokens=8)
    assert g1.shape == (2, 8) and np.array_equal(g1, g2)
    steered = generate(model, prompts, clamp_setup(LAYER, NEURON, 500.0), seed=3, max_new_tokens=8)
    assert not np.array_equal(g1, steered)


def test_nll_and_budget(small_model, sample_ids):
    model, _ = small_model
    prompts = sample_ids.cpu().numpy()
    nll = continuation_nll(model, prompts[:, :8], prompts[:, 8:])
    assert nll.shape == (2,) and (nll > 0).all()
    assert residual_budget(model, LAYER, NEURON, -2.0) == residual_budget(model, LAYER, NEURON, 2.0) > 0


def test_diffmean_vector_shape(small_model, sample_ids):
    model, _ = small_model
    w = _windows(sample_ids)
    acc = UnitStats(n_layers=24, n_units=1536, top_k=3, device="cpu")
    acts = record(model, torch.from_numpy(w.tokens).long().to(model.device), [LAYER])[LAYER][:, 1:].reshape(-1, 1536).cpu()
    acc.update(LAYER, acts, torch.arange(acts.shape[0]))
    v = diffmean_vector(model, w, acc.state(), LAYER, NEURON, 1, seed=0, n_random=4)
    assert v.shape == (model.config.hidden_size,) and torch.isfinite(v).all()
