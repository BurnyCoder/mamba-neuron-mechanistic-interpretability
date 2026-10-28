"""What each neuron *writes*: projection of its write vector onto the vocabulary.

Local explanation: neuron i of layer l adds a_i * w_i to the residual stream (w_i = column i of out_proj.weight).
Ignoring later layers and the positive RMS scale of the final norm, its direct effect on the logits is proportional to
u_i = W_U (g ⊙ w_i), where g is the gain of `backbone.norm_f` and W_U the (tied) unembedding. This is the
"logit lens on weights" of Geva et al. 2022 (https://arxiv.org/abs/2203.14680) and the W_U·W_out analysis of
Gurnee et al. 2024 (https://arxiv.org/abs/2401.12181), whose high-kurtosis "prediction / suppression neurons"
promote or suppress a coherent token set. The negative pole writes -u_i.
Global context: vocab kurtosis and input-output coherence feed candidate selection; top tokens feed the viewer.
"""

from __future__ import annotations

import torch

from .logging_utils import get_logger
from .model import n_layers, write_vectors

log = get_logger(__name__)


def unembed_directions(model) -> tuple[torch.Tensor, torch.Tensor]:
    """(W_U, g) in float32: unembedding (V, d_model) and final RMSNorm gain (d_model,)."""
    return model.lm_head.weight.float(), model.backbone.norm_f.weight.float()


@torch.no_grad()
def logit_effects(model, layer: int, neurons: torch.Tensor) -> torch.Tensor:
    """(V, len(neurons)) matrix u = W_U (g ⊙ w) for the given neuron indices of `layer`."""
    w_u, g = unembed_directions(model)
    w = write_vectors(model, layer)[:, neurons].float()  # (d_model, n)
    return w_u @ (g[:, None] * w)


@torch.no_grad()
def vocab_stats(model, top_n: int = 20, chunk: int = 512) -> dict[str, torch.Tensor]:
    """Per-neuron vocab mean/std/skew/kurtosis plus top promoted / suppressed token ids and values, all layers."""
    out = {k: [] for k in ("mean", "std", "skew", "kurtosis", "top_ids", "top_vals", "bot_ids", "bot_vals")}
    n_per_layer = write_vectors(model, 0).shape[1]
    for layer in range(n_layers(model)):
        rows = {k: [] for k in out}
        for start in range(0, n_per_layer, chunk):
            u = logit_effects(model, layer, torch.arange(start, min(start + chunk, n_per_layer), device=model.device))
            mu, sd = u.mean(0), u.std(0)
            z = (u - mu) / sd
            rows["mean"].append(mu)
            rows["std"].append(sd)
            rows["skew"].append((z**3).mean(0))  # standardized 3rd moment over the vocabulary
            rows["kurtosis"].append((z**4).mean(0))  # Pearson kurtosis (Gaussian = 3), as in Gurnee et al. 2024
            tv, ti = u.topk(top_n, dim=0)
            bv, bi = (-u).topk(top_n, dim=0)
            rows["top_vals"].append(tv), rows["top_ids"].append(ti), rows["bot_vals"].append(-bv), rows["bot_ids"].append(bi)
        for k in out:
            out[k].append(torch.cat(rows[k], dim=-1).cpu())
        if layer % 8 == 0:
            log.info("vocab stats: layer %d done", layer)
    return {k: torch.stack(v) for k, v in out.items()}  # scalars: (L, N); top lists: (L, top_n, N)
