"""Steering generations: single-neuron clamp / add, and the baselines it is compared against.

Local explanation: every condition is a function `setup(hooks)` that registers edits on a `Hooks` context
(model.py); `generate` samples continuations of the same neutral prompts under that context. Conditions:
  * clamp  - neuron (l, i) fixed to s * c * |pole extreme| at every position (Templeton et al. 2024,
             https://transformer-circuits.pub/2024/scaling-monosemanticity/; Goodfire "turn a neuron up").
  * add    - delta added to the neuron = delta * w_i added to the residual (ActAdd, https://arxiv.org/abs/2308.10248).
  * diffmean - residual vector mean(h | firing) - mean(h | random) at block l, scaled to the same residual norm
             (DiffMean, the strongest simple baseline in AxBench, https://arxiv.org/abs/2501.17148).
  * random neurons - same relative clamp on other neurons of the same layer (null distribution).
  * prompting - unsteered model with a concept prefix.
Global context: outputs (token ids + text) are judged later by `judge.py` in a separate (Ollama) phase so the GPU
never holds Mamba and the judge at once.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .collect import decode_index
from .model import Hooks, add_edit, add_vector_edit, clamp_edit, record_layer, write_vectors

Setup = Callable[[Hooks], None]  # registers the condition's edits on an open Hooks context


def neutral_prompts(windows, n: int, prompt_len: int, seed: int, skip: int = 0) -> np.ndarray:
    """(n, 1 + prompt_len) ids: separator + first `prompt_len` tokens of held-out documents (document openings only).

    Documents are shuffled with `seed`; `skip` lets prompt set B start after prompt set A (disjoint sets).
    """
    openings = np.flatnonzero(windows.start == 0)  # rows that begin a document
    order = np.random.default_rng(seed).permutation(openings)
    rows = order[skip:skip + n]
    return windows.tokens[rows, : prompt_len + 1]


@torch.no_grad()
def pole_values(model, ids: np.ndarray, layer: int, neuron: int, sign: int, batch: int = 32) -> np.ndarray:
    """Signed activation s * a_i on every token of `ids` (used to check prompts are neutral for the neuron)."""
    out = []
    for b in range(0, len(ids), batch):
        a = record_layer(model, torch.from_numpy(ids[b:b + batch]).long().to(model.device), layer)
        out.append((sign * a[..., neuron].float()).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def generate(model, prompt_ids: np.ndarray, setup: Setup | None, seed: int, max_new_tokens: int = 40,
             temperature: float = 0.7, top_p: float = 0.95) -> np.ndarray:
    """(n, max_new_tokens) sampled continuation ids under the condition `setup` (None = unsteered)."""
    ids = torch.from_numpy(prompt_ids).long().to(model.device)
    torch.manual_seed(seed)  # sampling RNG (https://pytorch.org/docs/stable/notes/randomness.html)
    with Hooks(model) as hooks:
        if setup is not None:
            setup(hooks)
        out = model.generate(ids, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                             min_new_tokens=max_new_tokens, pad_token_id=0)
    return out[:, ids.shape[1]:].cpu().numpy()


@torch.no_grad()
def continuation_nll(model, prompt_ids: np.ndarray, cont_ids: np.ndarray) -> np.ndarray:
    """Mean per-token negative log-likelihood of each continuation under the *unsteered* model (fluency proxy)."""
    full = torch.from_numpy(np.concatenate([prompt_ids, cont_ids], axis=1)).long().to(model.device)
    logp = torch.log_softmax(model(input_ids=full, use_cache=False).logits.float(), dim=-1)
    tgt = full[:, prompt_ids.shape[1]:]
    lp = logp[:, prompt_ids.shape[1] - 1:-1].gather(-1, tgt[..., None])[..., 0]  # log p(token_t | tokens_<t)
    return (-lp.mean(1)).cpu().numpy()


@torch.no_grad()
def diffmean_vector(model, windows, stats_group: dict, layer: int, neuron: int, sign: int, seed: int, n_random: int = 200) -> torch.Tensor:
    """mean residual after block `layer` at the pole's held-out top firings minus mean at random held-out positions."""
    ctx = windows.tokens.shape[1] - 1
    idx = (stats_group["top_idx"] if sign > 0 else stats_group["bot_idx"])[layer, :, neuron].numpy()
    pos_rows, pos_pos = decode_index(idx[idx >= 0], ctx)
    rng = np.random.default_rng(seed)
    rnd_rows, rnd_pos = rng.integers(0, len(windows.tokens), n_random), rng.integers(1, ctx + 1, n_random)

    def mean_resid(rows, positions):
        acc = []
        for b in range(0, len(rows), 32):
            ids = torch.from_numpy(windows.tokens[rows[b:b + 32]]).long().to(model.device)
            store = {}

            def grab(h):  # identity edit that keeps a reference to the residual stream
                store["h"] = h
                return h

            with Hooks(model).edit_residual(layer, grab):
                model(input_ids=ids, use_cache=False)
            acc.append(store["h"][torch.arange(len(ids)), torch.as_tensor(positions[b:b + 32], device=model.device)].float())
        return torch.cat(acc).mean(0)

    return mean_resid(pos_rows, pos_pos) - mean_resid(rnd_rows, rnd_pos)


def clamp_setup(layer: int, neuron: int, value: float) -> Setup:
    """Clamp one neuron to `value` at all positions."""
    return lambda hooks: hooks.edit_neurons(layer, clamp_edit(neuron, value))


def add_setup(layer: int, neuron: int, delta: float) -> Setup:
    """Add `delta` to one neuron at all positions."""
    return lambda hooks: hooks.edit_neurons(layer, add_edit(neuron, delta))


def vector_setup(layer: int, vector: torch.Tensor) -> Setup:
    """Add a residual-stream vector after block `layer` at all positions."""
    return lambda hooks: hooks.edit_residual(layer, add_vector_edit(vector))


def residual_budget(model, layer: int, neuron: int, delta: float) -> float:
    """Residual-stream norm of adding `delta` to neuron i: |delta| * ||w_i|| (used to match DiffMean's scale)."""
    return abs(delta) * float(write_vectors(model, layer)[:, neuron].detach().float().norm())
