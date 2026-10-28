"""Causal and mechanistic tests of a single neuron.

Local explanation:
  * `top_positions` - exhaustive scan of a split for a pole's strongest firings (more than the stored top-k).
  * `next_token_logp` + `zero_ablation` - H3: zero-ablate a neuron at all positions and measure the drop in log p(actual
    next token) at given positions (causal necessity; zero-ablation as in Gurnee et al. 2024 https://arxiv.org/abs/2401.12181).
  * `factor_split` - decompose a = y * SiLU(z) at firings: is the pole driven by the gate z or by the SSM output y?
    (the "double-negative regime" is y<0 and z<0, where SiLU(z)<0 makes a>0; cf. Goodfire's SwiGLU note).
  * `context_truncation` - re-run each firing with only the last k tokens of context: k<=4 sits inside Mamba's
    causal conv window (kernel 4), larger k needs the recurrent SSM state, so the curve tells local vs long-range.
  * `patching_effect` - counterfactual activation patching: copy the neuron's firing value into non-firing contexts
    (last position only) and measure the change in log-prob of its top promoted tokens (Meng et al. 2022,
    https://arxiv.org/abs/2202.05262; Heimersheim & Nanda 2024 https://arxiv.org/abs/2404.15255).
Global context: H3 (pre-registered) + exploratory mechanism results reported in reports/ and the viewer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import Hooks, blocks, clamp_edit, record_layer


@torch.no_grad()
def top_positions(model, windows, layer: int, neuron: int, sign: int, n: int, max_windows: int | None = None,
                  batch: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, positions, values) of the `n` largest signed activations s*a_i over the first `max_windows` windows."""
    vals = []
    for b in range(0, len(windows.tokens) if max_windows is None else min(max_windows, len(windows.tokens)), batch):
        a = record_layer(model, torch.from_numpy(windows.tokens[b:b + batch]).long().to(model.device), layer)
        vals.append((sign * a[:, 1:, neuron].float()).cpu().numpy())
    vals = np.concatenate(vals)  # (n_windows, ctx)
    flat = np.argsort(-vals, axis=None)[:n]
    rows, pos = np.unravel_index(flat, vals.shape)
    return rows, pos + 1, vals[rows, pos]


@torch.no_grad()
def _next_token_logp(model, ids: torch.Tensor, positions: torch.Tensor, setup=None) -> torch.Tensor:
    """log p(ids[pos+1] | ids[:pos+1]) for each row, optionally under an intervention `setup(hooks)`."""
    with Hooks(model) as hooks:
        if setup is not None:
            setup(hooks)
        logits = model(input_ids=ids, use_cache=False).logits.float()
    rows = torch.arange(len(ids), device=ids.device)
    logp = F.log_softmax(logits[rows, positions], dim=-1)
    return logp[rows, ids[rows, positions + 1]]


@torch.no_grad()
def next_token_logp(model, windows, rows, positions, setup=None, batch: int = 16) -> np.ndarray:
    """log p(actual next token) at each (row, position), optionally under an intervention `setup(hooks)`.

    Positions on the last window token (no next token inside the window) must be filtered out by the caller.
    """
    rows, positions = np.asarray(rows), np.asarray(positions)
    out = []
    for b in range(0, len(rows), batch):
        ids = torch.from_numpy(windows.tokens[rows[b:b + batch]]).long().to(model.device)
        out.append(_next_token_logp(model, ids, torch.as_tensor(positions[b:b + batch], device=model.device), setup).cpu().numpy())
    return np.concatenate(out)


def zero_ablation(layer: int, neuron: int):
    """Setup that zero-ablates one neuron at every position (H3 intervention)."""
    return lambda hooks: hooks.edit_neurons(layer, clamp_edit(neuron, 0.0))


@torch.no_grad()
def factor_split(model, windows, rows, positions, layer: int, neuron: int, batch: int = 16) -> dict:
    """Gate z_i and SSM factor y_i = a_i / SiLU(z_i) at the given firings (z from the second half of in_proj)."""
    n_inner = blocks(model)[layer].mixer.intermediate_size
    rows, positions = np.asarray(rows), np.asarray(positions)
    a_all, z_all = [], []
    for b in range(0, len(rows), batch):
        store = {}
        handle = blocks(model)[layer].mixer.in_proj.register_forward_hook(lambda m, i, o: store.__setitem__("p", o))
        ids = torch.from_numpy(windows.tokens[rows[b:b + batch]]).long().to(model.device)
        try:
            a = record_layer(model, ids, layer)[..., neuron].float()
        finally:
            handle.remove()
        idx = torch.arange(len(ids), device=ids.device)
        pos = torch.as_tensor(positions[b:b + batch], device=ids.device)
        z_all.append(store["p"][idx, pos, n_inner + neuron].float())  # in_proj output = [x | z] along the last dim
        a_all.append(a[idx, pos])
    a_at, z = torch.cat(a_all), torch.cat(z_all)
    gate = F.silu(z)
    y = a_at / gate.where(gate.abs() > 1e-6, torch.full_like(gate, 1e-6))
    return {"a": a_at.cpu().tolist(), "z": z.cpu().tolist(), "silu_z": gate.cpu().tolist(), "y": y.cpu().tolist(),
            "frac_gate_negative": float((z < 0).float().mean()), "frac_double_negative": float(((z < 0) & (y < 0)).float().mean())}


@torch.no_grad()
def context_truncation(model, windows, rows, positions, layer: int, neuron: int, ks=(1, 2, 4, 8, 16, 32, None)) -> dict:
    """Mean activation ratio a_i(last k tokens) / a_i(full window) at each firing, for each context length k."""
    full = []
    out = {}
    for r, p in zip(rows, positions):
        full.append(float(record_layer(model, torch.from_numpy(windows.tokens[r:r + 1, : p + 1]).long().to(model.device), layer)[0, -1, neuron]))
    full = np.asarray(full)
    for k in ks:
        vals = []
        for r, p in zip(rows, positions):
            start = 1 if k is None else max(1, p - k + 1)
            ids = np.concatenate([[0], windows.tokens[r, start:p + 1]])  # separator + last k tokens
            vals.append(float(record_layer(model, torch.from_numpy(ids[None]).long().to(model.device), layer)[0, -1, neuron]))
        out["all" if k is None else str(k)] = float(np.mean(np.asarray(vals) / full))
    return out


@torch.no_grad()
def patching_effect(model, windows, donor_value: float, rows, positions, layer: int, neuron: int, token_ids, batch: int = 16) -> float:
    """Mean change of mean log p(token_ids) at (rows, positions) when the neuron is set to `donor_value` at that position only."""
    tok = torch.as_tensor(token_ids, device=model.device)
    deltas = []
    for b in range(0, len(rows), batch):
        r, p = np.asarray(rows[b:b + batch]), np.asarray(positions[b:b + batch])
        ids = torch.from_numpy(windows.tokens[r]).long().to(model.device)
        idx = torch.arange(len(ids), device=ids.device)
        pos = torch.as_tensor(p, device=ids.device)

        def patch(a):  # set the neuron only at each row's own position (counterfactual single-site patch)
            a = a.clone()
            if a.shape[1] > 1:  # full-sequence forward (not a cached decode step)
                a[idx, pos, neuron] = donor_value
            return a

        clean = F.log_softmax(model(input_ids=ids, use_cache=False).logits.float()[idx, pos], -1)[:, tok].mean(-1)
        with Hooks(model).edit_neurons(layer, patch):
            patched = F.log_softmax(model(input_ids=ids, use_cache=False).logits.float()[idx, pos], -1)[:, tok].mean(-1)
        deltas.append((patched - clean).cpu().numpy())
    return float(np.concatenate(deltas).mean())
