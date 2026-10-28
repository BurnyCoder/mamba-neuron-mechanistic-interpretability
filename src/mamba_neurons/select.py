"""Pre-registered candidate ranking of neuron poles (see reports/preregistration.md, "Candidate selection").

Local explanation: every pole (layer, neuron, sign) gets three scores:
  1. activation kurtosis of the neuron (detector-like sparsity),
  2. vocab kurtosis of u_i = W_U(g ⊙ w_i) (writes a coherent token set, Gurnee et al. 2024 https://arxiv.org/abs/2401.12181),
  3. input-output coherence: mean over the pole's top-k firing positions of the vocab-z-scored s·u_i[next token],
     i.e. does the neuron push up the token that actually follows where it fires.
The candidate score is the mean of the three rank-percentiles; the top `n` poles are kept with at most `per_layer`
per layer.
Global context: the shortlist is what gets labelled, inspected and screened for steering, so the formula is fixed
in advance to avoid choosing neurons by their steering outcome.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata  # average ranks for ties (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)

from .collect import decode_index
from .vocab import unembed_directions
from .model import write_vectors


@torch.no_grad()
def coherence(model, stats: dict, vstats: dict, windows, sign: int, chunk: int = 512) -> torch.Tensor:
    """(L, N) mean z-scored s·u_i[next token] over the pole's top-k firing positions (NaN-free: last position skipped)."""
    pole_idx = stats["top_idx"] if sign > 0 else stats["bot_idx"]  # (L, k, N) global token indices
    ctx = windows.tokens.shape[1] - 1
    rows, pos = decode_index(pole_idx.numpy(), ctx)
    nxt_pos = np.minimum(pos + 1, ctx)  # token after the firing position (clipped at the window end)
    nxt = torch.from_numpy(windows.tokens[rows, nxt_pos].astype(np.int64))  # (L, k, N) next-token ids
    valid = torch.from_numpy(pos < ctx)  # firing on the last position has no next token inside the window
    w_u, g = unembed_directions(model)
    L, k, N = nxt.shape
    out = torch.zeros(L, N)
    for layer in range(L):
        w = write_vectors(model, layer).float() * g[:, None]  # (d, N) gain-scaled write vectors
        for s in range(0, N, chunk):
            ids = nxt[layer, :, s:s + chunk].to(model.device)  # (k, c)
            u_next = torch.einsum("kcd,dc->kc", w_u[ids], w[:, s:s + chunk])  # u_i[next] for each firing
            z = sign * (u_next.cpu() - vstats["mean"][layer, s:s + chunk]) / vstats["std"][layer, s:s + chunk]
            v = valid[layer, :, s:s + chunk].float()
            out[layer, s:s + chunk] = (z * v).sum(0) / v.sum(0).clamp_min(1)
    return out


def rank_candidates(stats: dict, vstats: dict, coh: dict[int, torch.Tensor], n: int = 60, per_layer: int = 3) -> list[dict]:
    """Return the top-`n` poles as dicts {layer, neuron, sign, score, act_kurtosis, vocab_kurtosis, coherence}."""
    act_k = stats["summary"]["kurtosis"].numpy()
    voc_k = vstats["kurtosis"].numpy()
    L, N = act_k.shape
    feats = []
    for sign in (1, -1):  # both poles share the neuron-level kurtosis terms, coherence is pole-specific
        feats.append(np.stack([act_k, voc_k, coh[sign].numpy()], axis=-1).reshape(-1, 3))
    feats = np.concatenate(feats)  # (2*L*N, 3): positive poles first, then negative poles
    feats = np.nan_to_num(feats, nan=-np.inf)  # dead neurons (std 0) rank last
    pct = np.stack([rankdata(feats[:, j]) / len(feats) for j in range(3)], axis=-1)
    score = pct.mean(-1)
    picked, per = [], {}
    for flat in np.argsort(-score):  # best first
        sign = 1 if flat < L * N else -1
        layer, neuron = divmod(int(flat % (L * N)), N)
        if per.get(layer, 0) >= per_layer:
            continue
        per[layer] = per.get(layer, 0) + 1
        f = feats[flat]
        picked.append({"layer": layer, "neuron": neuron, "sign": sign, "score": float(score[flat]),
                       "act_kurtosis": float(f[0]), "vocab_kurtosis": float(f[1]), "coherence": float(f[2])})
        if len(picked) == n:
            break
    return picked
