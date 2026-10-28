"""Tests that the neuron hook site is exactly the input of `out_proj` and that interventions do what they claim.

Global context: every steering / ablation / patching result in the paper relies on these equivalences.
"""

import torch

from mamba_neurons.model import Hooks, add_edit, blocks, clamp_edit, neuron_site, record, write_vectors

LAYER, NEURON, DELTA = 5, 123, 3.0  # arbitrary but fixed test coordinates


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def test_record_shape_and_out_proj_reconstructs_mixer_output(small_model, sample_ids):
    model, _ = small_model
    mixer_out = {}
    handle = blocks(model)[LAYER].mixer.register_forward_hook(lambda m, i, o: mixer_out.__setitem__("o", o.detach()))
    acts = record(model, sample_ids, [LAYER])[LAYER]
    handle.remove()
    assert acts.shape == (*sample_ids.shape, neuron_site(model, LAYER).in_features)
    torch.testing.assert_close(neuron_site(model, LAYER)(acts), mixer_out["o"])  # out_proj(a) == mixer output


def test_add_edit_equals_adding_write_vector_to_mixer_output(small_model, sample_ids):
    model, _ = small_model
    with Hooks(model).edit_neurons(LAYER, add_edit(NEURON, DELTA)):
        steered = _logits(model, sample_ids)
    w = write_vectors(model, LAYER)[:, NEURON].detach()
    handle = blocks(model)[LAYER].mixer.register_forward_hook(lambda m, i, o: o + DELTA * w)
    shifted = _logits(model, sample_ids)
    handle.remove()
    torch.testing.assert_close(steered, shifted, rtol=1e-3, atol=2e-3)  # fp32 rounding differs between the two orderings
    assert not torch.allclose(steered, _logits(model, sample_ids))  # and the edit actually changed something


def test_clamp_to_own_value_is_noop(small_model, sample_ids):
    model, _ = small_model
    own = record(model, sample_ids, [LAYER])[LAYER][..., NEURON]  # (batch, seq) recorded values
    with Hooks(model).edit_neurons(LAYER, clamp_edit(NEURON, own)):
        clamped = _logits(model, sample_ids)
    torch.testing.assert_close(clamped, _logits(model, sample_ids))


def test_hooks_fire_in_cached_decoding_and_identity_edit_preserves_generation(small_model, sample_ids):
    model, tok = small_model
    calls = []
    kwargs = dict(max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
    with torch.no_grad():
        plain = model.generate(sample_ids, **kwargs)
        with Hooks(model).edit_neurons(LAYER, lambda a: calls.append(a.shape) or a):
            hooked = model.generate(sample_ids, **kwargs)
    assert torch.equal(plain, hooked)
    assert calls[0][1] == sample_ids.shape[1]  # prefill sees the whole prompt
    assert all(shape[1] == 1 for shape in calls[1:]) and len(calls) >= 5  # then one call per decoded token


def test_hooks_are_removed_on_exit(small_model):
    model, _ = small_model
    site = neuron_site(model, LAYER)
    with Hooks(model).edit_neurons(LAYER, lambda a: a):
        assert len(site._forward_pre_hooks) == 1
    assert len(site._forward_pre_hooks) == 0
