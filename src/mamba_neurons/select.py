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


DOMAINS = ("Github", "PubMed Abstracts", "Wikipedia (en)", "USPTO Backgrounds", "FreeLaw", "DM Mathematics", "ArXiv",
           "StackExchange")  # distinctive Pile subsets present in pile-10k (amendment 2 of the pre-registration)


@torch.no_grad()
def domain_selectivity(model, windows, domains=DOMAINS, per_domain: int = 250, n_generic: int = 750, seed: int = 0,
                       batch: int = 12) -> dict[str, torch.Tensor]:
    """Per domain d: d'[l, i] = (mean_d - mean_rest) / sqrt((var_d + var_rest) / 2) of window-mean activations.

    Windows are sampled per Pile subset (`per_domain` each, plus `n_generic` from all other subsets) so every domain
    is compared against the rest of the sample. Returns {domain: (L, N) tensor of d'}.
    """
    from .model import Hooks, n_layers, n_neurons

    rng = np.random.default_rng(seed)
    src = np.asarray([windows.sources[d] for d in windows.doc])  # Pile subset of every window
    picked, labels = [], []
    for k, d in enumerate(domains):
        rows = np.flatnonzero(src == d)
        rows = rng.choice(rows, min(per_domain, len(rows)), replace=False)
        picked += rows.tolist()
        labels += [k] * len(rows)
    other = np.flatnonzero(~np.isin(src, domains))
    picked += rng.choice(other, min(n_generic, len(other)), replace=False).tolist()
    labels += [len(domains)] * min(n_generic, len(other))
    labels_t = torch.tensor(labels)
    L, N, G = n_layers(model), n_neurons(model), len(domains) + 1
    s1 = torch.zeros(G, L, N, dtype=torch.float64, device=model.device)  # per group: sum and sum of squares of window means
    s2 = torch.zeros_like(s1)
    cur = {}

    def sink(layer, a):
        m = a[:, 1:, :].double().mean(1)  # (B, N) window-mean activation (separator excluded)
        s1[:, layer].index_add_(0, cur["g"], m)
        s2[:, layer].index_add_(0, cur["g"], m * m)

    with Hooks(model).record_neurons(range(L), sink):
        for b in range(0, len(picked), batch):
            cur["g"] = labels_t[b:b + batch].to(model.device)
            model(input_ids=torch.from_numpy(windows.tokens[picked[b:b + batch]]).long().to(model.device), use_cache=False)
    counts = torch.bincount(labels_t, minlength=G).double().to(model.device)[:, None, None]
    out = {}
    for k, d in enumerate(domains):
        n_d, n_r = counts[k], counts.sum(0) - counts[k]
        m_d, m_r = s1[k] / n_d, (s1.sum(0) - s1[k]) / n_r
        v_d, v_r = s2[k] / n_d - m_d**2, (s2.sum(0) - s2[k]) / n_r - m_r**2
        out[d] = ((m_d - m_r) / ((v_d + v_r) / 2).clamp_min(1e-12).sqrt()).float().cpu()
    return out


def domain_candidates(dprime: dict[str, torch.Tensor], per_domain: int = 3) -> list[dict]:
    """Top-`per_domain` neurons by |d'| for each domain; the pole sign is the sign of d' (higher vs lower in-domain)."""
    picked = []
    for d, t in dprime.items():
        flat = t.abs().flatten().argsort(descending=True)[:per_domain]
        for f in flat.tolist():
            layer, neuron = divmod(f, t.shape[1])
            picked.append({"layer": layer, "neuron": neuron, "sign": 1 if t[layer, neuron] > 0 else -1,
                           "domain": d, "dprime": float(t[layer, neuron])})
    return picked
