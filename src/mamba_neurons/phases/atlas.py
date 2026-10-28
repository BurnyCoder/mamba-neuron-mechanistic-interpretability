"""Atlas phases: scan benchmark, activation collection, vocab projection, candidate selection, example building.

Local explanation: these phases need only the Mamba model (GPU) and produce the neuron atlas: per-neuron statistics
on both splits, write-vector vocab projections, the pre-registered candidate shortlist, the split-stability check
and the highlighted examples used for labelling and the viewer.
Global context: everything downstream (labels, steering, mechanism, viewer) reads these outputs.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import spearmanr  # rank correlation for split stability

from ..autointerp import audit_units, build_examples, unit_key
from ..collect import collect_stats
from ..logging_utils import get_logger
from ..model import load_model, n_layers, n_neurons
from ..select import coherence, rank_candidates
from ..vocab import vocab_stats
from .common import SPLITS, free_gpu, get_windows, load_json, load_stats, save_json

log = get_logger(__name__)


def phase_benchmark(settings) -> None:
    """Throughput, peak memory and next-token loss of the HF reference torch scan on one discovery batch.

    The loss is a sanity check that the bf16 model is healthy on in-distribution Pile text.
    """
    model, tok = load_model(settings.model_name, settings.dtype, settings.device)
    ids = torch.from_numpy(get_windows(settings, tok)["discovery"].tokens[: settings.batch_size]).long().to(model.device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(input_ids=ids[:2], use_cache=False)  # warm-up (CUDA context, kernel selection)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            out = model(input_ids=ids, labels=ids, use_cache=False)  # HF shifts labels internally for the LM loss
        torch.cuda.synchronize()
    results = {"tokens_per_s": 3 * ids[:, 1:].numel() / (time.time() - t0), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "next_token_loss_nats": float(out.loss), "batch_size": settings.batch_size}
    log.info("benchmark: %s", results)
    save_json(results, settings.results_path() / "benchmark.json")
    free_gpu(model)


def phase_collect(settings) -> None:
    """Activation statistics + top-k poles for every neuron and random direction, on both splits."""
    model, tok = load_model(settings.model_name, settings.dtype, settings.device)
    windows = get_windows(settings, tok)
    for split in SPLITS:
        out = settings.run_dir() / f"stats_{split}.pt"
        if out.exists():
            log.info("stats for %s already exist at %s - skipping", split, out)
            continue
        log.info("collecting %s: %d windows, %d tokens", split, len(windows[split].tokens), windows[split].n_tokens)
        torch.save(collect_stats(model, windows[split], settings), out)
    free_gpu(model)


def phase_vocab(settings) -> None:
    """W_U(g ⊙ w_i) statistics for every neuron; per-layer median vocab kurtosis as a small summary."""
    model, _ = load_model(settings.model_name, settings.dtype, settings.device)
    v = vocab_stats(model)
    torch.save(v, settings.run_dir() / "vocab.pt")
    save_json({"median_vocab_kurtosis_per_layer": v["kurtosis"].median(1).values.tolist()}, settings.results_path() / "vocab_summary.json")
    free_gpu(model)


def phase_select(settings) -> None:
    """Pre-registered candidate ranking + split-stability check."""
    model, tok = load_model(settings.model_name, settings.dtype, settings.device)
    windows = get_windows(settings, tok)
    stats = load_stats(settings)
    vstats = torch.load(settings.run_dir() / "vocab.pt")
    disc = stats["discovery"]["neurons"]
    coh = {s: coherence(model, disc, vstats, windows["discovery"], s) for s in (1, -1)}
    torch.save(coh, settings.run_dir() / "coherence.pt")
    cands = rank_candidates(disc, vstats, coh, n=60, per_layer=3)
    for c in cands:  # attach the pole's promoted tokens for readability (negative pole promotes -u)
        ids = (vstats["top_ids"] if c["sign"] > 0 else vstats["bot_ids"])[c["layer"], :10, c["neuron"]].tolist()
        c["promoted_tokens"] = [tok.decode([t]) for t in ids]
    save_json(cands, settings.results_path() / "candidates.json")
    log.info("top candidates: %s", [(c["layer"], c["neuron"], c["sign"], c["promoted_tokens"][:5]) for c in cands[:10]])
    stability = {}
    for key in ("mean", "std", "max", "min", "kurtosis", "frac_pos"):
        a = stats["discovery"]["neurons"]["summary"][key].flatten().numpy()
        b = stats["heldout"]["neurons"]["summary"][key].flatten().numpy()
        stability[key] = float(spearmanr(a, b, nan_policy="omit").statistic)
    save_json(stability, settings.results_path() / "split_stability.json")
    log.info("split stability (Spearman discovery vs held-out): %s", stability)
    free_gpu(model)


def phase_examples(settings) -> None:
    """Highlighted examples for the H1 audit sample (random neurons + random directions) and the candidates."""
    model, tok = load_model(settings.model_name, settings.dtype, settings.device)
    windows = get_windows(settings, tok)
    stats = load_stats(settings)
    units = audit_units(n_layers(model), n_neurons(model), settings.n_random_directions_per_layer, per_layer=2, seed=settings.seed)
    for u in units:
        u["group"] = "audit"
    keys = {unit_key(u) for u in units}
    for c in load_json(settings.results_path() / "candidates.json"):
        u = {"kind": "neuron", "layer": c["layer"], "index": c["neuron"], "sign": c["sign"], "group": "candidate"}
        if unit_key(u) not in keys:
            units.append(u)
    examples = build_examples(model, tok, windows, stats, stats["discovery"]["directions"], units, seed=settings.seed)
    save_json(examples, settings.run_dir() / "examples.json")
    log.info("built examples for %d poles", len(examples))
    free_gpu(model)
