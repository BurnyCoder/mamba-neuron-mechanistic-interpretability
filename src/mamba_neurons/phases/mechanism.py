"""Mechanism phase: pre-registered H3 (necessity by zero-ablation vs 20 random same-layer neurons) and the exploratory
factor split, context-truncation and activation-patching analyses, for every pole in the confirm plan.

Local explanation: positions come from a scan of the first `SCAN_WINDOWS` held-out windows (the stored top-40 is too
few for 200 positions). The clean log-probs are computed once and reused for all 21 ablations.
Global context: results/<model>/h3_mechanism.json, reported in reports/ and the viewer.
"""

from __future__ import annotations

import numpy as np
import torch

from ..logging_utils import get_logger
from ..mechanism import context_truncation, factor_split, logp_at, next_token_logp, patching_effect, top_positions, zero_ablation
from ..metrics import empirical_p
from ..model import load_model, n_neurons
from .common import free_gpu, get_windows, load_json, load_stats, save_json

log = get_logger(__name__)
N_POS, N_RANDOM, SCAN_WINDOWS, N_TRUNC = 200, 20, 2000, 50


def phase_mechanism(settings) -> None:
    """H3 + exploratory mechanism results for the confirm-plan poles."""
    model, tok = load_model(settings.model_name, settings.dtype, settings.device)
    held = get_windows(settings, tok)["heldout"]
    stats = load_stats(settings)
    vstats = torch.load(settings.run_dir() / "vocab.pt")
    ctx = held.tokens.shape[1] - 1
    rng = np.random.default_rng(settings.seed)
    out = {}
    for p in load_json(settings.results_path() / "confirm_plan.json"):
        L, i, s = p["layer"], p["neuron"], p["sign"]
        rows, pos, vals = top_positions(model, held, L, i, s, N_POS, max_windows=SCAN_WINDOWS)
        keep = pos < ctx  # need a next token inside the window
        r_k, p_k = rows[keep], pos[keep]
        clean = next_token_logp(model, held, r_k, p_k)
        drop = float((clean - next_token_logp(model, held, r_k, p_k, zero_ablation(L, i))).mean())
        others = rng.choice([j for j in range(n_neurons(model)) if j != i], N_RANDOM, replace=False)
        null = [float((clean - next_token_logp(model, held, r_k, p_k, zero_ablation(L, int(j)))).mean()) for j in others]
        p_h3 = empirical_p(drop, null)
        log.info("H3 %s: drop %.4f nats vs random max %.4f (p=%.3f)", p["key"], drop, max(null), p_h3)

        fs = factor_split(model, held, rows, pos, L, i)
        trunc = context_truncation(model, held, rows[:N_TRUNC], pos[:N_TRUNC], L, i)

        # Counterfactual patching into random held-out positions: neuron set to its median firing value at one position.
        nf_rows, nf_pos = rng.integers(0, len(held.tokens), 200), rng.integers(1, ctx + 1, 200)
        promoted = (vstats["top_ids"] if s > 0 else vstats["bot_ids"])[L, :10, i].tolist()
        clean_lp = logp_at(model, held, nf_rows, nf_pos)  # shared by the target and all null patches
        patch = patching_effect(model, held, s * float(np.median(vals)), nf_rows, nf_pos, L, i, promoted, clean_lp)
        patch_null = []
        for j in others:
            group = stats["discovery"]["neurons"]
            donor = s * float(np.median(np.abs((group["top_val"] if s > 0 else group["bot_val"])[L, :, int(j)].numpy())))
            prom_j = (vstats["top_ids"] if s > 0 else vstats["bot_ids"])[L, :10, int(j)].tolist()
            patch_null.append(patching_effect(model, held, donor, nf_rows, nf_pos, L, int(j), prom_j, clean_lp))
        out[p["key"]] = {
            "label": p["label"], "n_positions": int(keep.sum()), "top_values_median": float(np.median(vals)),
            "h3": {"ablation_drop_nats": drop, "random_null": null, "p_empirical": p_h3, "h3_supported": bool(p_h3 <= 1 / 21)},
            "factor_split": {k: fs[k] for k in ("frac_gate_negative", "frac_double_negative")} |
                            {"median_silu_z": float(np.median(fs["silu_z"])), "median_y": float(np.median(fs["y"])), "median_z": float(np.median(fs["z"]))},
            "context_truncation_ratio": trunc,
            "patching": {"delta_logp_promoted": patch, "random_null": patch_null, "p_empirical": empirical_p(patch, patch_null)},
            "promoted_tokens": [tok.decode([t]) for t in promoted],
        }
        log.info("mechanism %s: factor %s | truncation %s | patch %.3f (null max %.3f)", p["key"], out[p["key"]]["factor_split"],
                 trunc, patch, max(patch_null))
    save_json(out, settings.results_path() / "h3_mechanism.json")
    del model  # drop the last reference first, otherwise empty_cache cannot release the weights
    free_gpu()
