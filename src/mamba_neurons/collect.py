"""Streaming per-neuron activation statistics and top-k examples for both poles.

Local explanation: one pass over a split's windows with record-hooks on every layer. For each unit we keep
float64 raw-moment sums (-> mean, std, skew, kurtosis), the fraction of positive activations, and the top-k
largest (positive pole) and top-k most negative (negative pole) activations with their token positions,
merged batch by batch with `torch.topk` (https://pytorch.org/docs/stable/generated/torch.topk.html).
The same accumulator also tracks *random unit directions* in each layer's neuron space ("virtual neurons"),
the control for the privileged-basis hypothesis H1 (Elhage et al. 2022, https://transformer-circuits.pub/2022/toy_model/index.html#motivation-privileged).
Global context: these statistics feed candidate selection (`select.py`), auto-interpretation (`autointerp.py`),
the viewer export and the split-stability check. Poles follow Goodfire's "positive / negative pole" framing
(https://already-clean.goodfire.com/).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .logging_utils import get_logger
from .model import Hooks, n_layers, n_neurons

log = get_logger(__name__)


class UnitStats:
    """Accumulator for `n_layers` x `n_units` scalar units over a stream of tokens."""

    def __init__(self, n_layers: int, n_units: int, top_k: int, device):
        shape = (n_layers, n_units)
        self.count = 0  # tokens seen per unit (identical for every unit)
        self.sums = torch.zeros((4, *shape), dtype=torch.float64, device=device)  # sum of a, a^2, a^3, a^4
        self.n_pos = torch.zeros(shape, dtype=torch.int64, device=device)  # number of a > 0
        self.top_val = torch.full((n_layers, top_k, n_units), -torch.inf, device=device)  # positive pole values
        self.top_idx = torch.full((n_layers, top_k, n_units), -1, dtype=torch.int64, device=device)
        self.bot_val = torch.full((n_layers, top_k, n_units), -torch.inf, device=device)  # stores -a (negative pole)
        self.bot_idx = torch.full((n_layers, top_k, n_units), -1, dtype=torch.int64, device=device)
        self._layers_seen = 0  # layers updated in the current batch (to count tokens once per batch)

    @staticmethod
    def _merge(vals, idxs, new_vals, new_idx):
        """Top-k of the union of the running buffer (k, U) and a new chunk (T, U) along dim 0."""
        cat_v = torch.cat([vals, new_vals], dim=0)
        cat_i = torch.cat([idxs, new_idx[:, None].expand(-1, new_vals.shape[1])], dim=0)
        top_v, pos = torch.topk(cat_v, k=vals.shape[0], dim=0)  # sorted descending
        return top_v, torch.gather(cat_i, 0, pos)  # carry the token indices along with the values

    def update(self, layer: int, acts: torch.Tensor, token_idx: torch.Tensor) -> None:
        """Add a chunk of activations `acts` (T, U) at global token indices `token_idx` (T,) for `layer`."""
        a64 = acts.double()
        a2 = a64 * a64
        self.sums[0, layer] += a64.sum(0)
        self.sums[1, layer] += a2.sum(0)
        self.sums[2, layer] += (a2 * a64).sum(0)
        self.sums[3, layer] += (a2 * a2).sum(0)
        self.n_pos[layer] += (acts > 0).sum(0)
        a32 = acts.float()
        self.top_val[layer], self.top_idx[layer] = self._merge(self.top_val[layer], self.top_idx[layer], a32, token_idx)
        self.bot_val[layer], self.bot_idx[layer] = self._merge(self.bot_val[layer], self.bot_idx[layer], -a32, token_idx)

    def summary(self) -> dict[str, torch.Tensor]:
        """Per-unit mean, std, skew, (Pearson) kurtosis, max, min, frac_pos from the raw moments (all on CPU)."""
        n = float(self.count)
        m1, m2, m3, m4 = (s / n for s in self.sums)  # raw moments E[a^k]
        var = (m2 - m1**2).clamp_min(1e-30)
        mu3 = m3 - 3 * m1 * m2 + 2 * m1**3  # central moments from raw moments (standard identities)
        mu4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
        return {
            "mean": m1.float().cpu(), "std": var.sqrt().float().cpu(),
            "skew": (mu3 / var**1.5).float().cpu(), "kurtosis": (mu4 / var**2).float().cpu(),
            "max": self.top_val[:, 0].cpu(), "min": (-self.bot_val[:, 0]).cpu(),
            "frac_pos": (self.n_pos.double() / n).float().cpu(),
        }

    def state(self) -> dict:
        """Everything needed to save / reload the accumulator (CPU tensors)."""
        return {"count": self.count, "summary": self.summary(), "top_val": self.top_val.cpu(), "top_idx": self.top_idx.cpu(),
                "bot_val": (-self.bot_val).cpu(), "bot_idx": self.bot_idx.cpu()}  # bot_val stored back in activation units


def random_directions(n_layers: int, n_units: int, per_layer: int, seed: int, device) -> torch.Tensor:
    """(n_layers, n_units, per_layer) isotropic random unit vectors (Gaussian then normalised), seeded."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(n_layers, n_units, per_layer, generator=gen)
    return (q / q.norm(dim=1, keepdim=True)).to(device)


@torch.inference_mode()
def collect_stats(model, windows, settings) -> dict:
    """Run the model over `windows`, returning {"neurons": state, "random_dirs": state, "directions": Q, ...}."""
    device = model.device
    L, N, k = n_layers(model), n_neurons(model), settings.top_k
    ctx = windows.tokens.shape[1] - 1  # analysed positions per window (position 0 = separator is excluded)
    neurons = UnitStats(L, N, k, device)
    q = random_directions(L, N, settings.n_random_directions_per_layer, settings.seed, device)
    rand = UnitStats(L, q.shape[-1], k, device)
    state = {"token_idx": None}

    def sink(layer: int, a: torch.Tensor) -> None:
        flat = a[:, 1:, :].reshape(-1, N).float()  # drop the separator position, flatten (batch*ctx, N)
        neurons.update(layer, flat, state["token_idx"])
        rand.update(layer, flat @ q[layer], state["token_idx"])  # projections onto the random directions

    t0, bs = time.time(), settings.batch_size
    with Hooks(model).record_neurons(range(L), sink):
        for b in range(0, len(windows.tokens), bs):
            ids = torch.from_numpy(windows.tokens[b:b + bs]).long().to(device)
            # global token index = window * ctx + (position - 1); decoded later by `decode_index`
            state["token_idx"] = (torch.arange(b, b + len(ids), device=device)[:, None] * ctx
                                  + torch.arange(ctx, device=device)[None]).reshape(-1)
            model(input_ids=ids, use_cache=False)
            neurons.count += ids.shape[0] * ctx
            rand.count += ids.shape[0] * ctx
            if (b // bs) % 50 == 0:
                rate = neurons.count / (time.time() - t0)
                log.info("collect: %d/%d windows, %.0f tok/s", b + len(ids), len(windows.tokens), rate)
    log.info("collect done: %d tokens in %.1f s", neurons.count, time.time() - t0)
    return {"neurons": neurons.state(), "random_dirs": rand.state(), "directions": q.cpu(), "ctx_len": ctx}


def decode_index(token_idx, ctx_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Global token index -> (window row, position inside `windows.tokens` row, 1-based because of the separator)."""
    token_idx = np.asarray(token_idx)
    return token_idx // ctx_len, token_idx % ctx_len + 1
