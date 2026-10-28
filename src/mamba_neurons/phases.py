"""Pipeline phases: thin functions that wire the library modules to on-disk inputs/outputs.

Local explanation: each `phase_*(settings)` loads what it needs from `settings.run_dir()` (large artefacts) or
`settings.results_path()` (small committed results), calls the modular functions, and saves its outputs.
Global context: `pipeline.py` runs them by name, in the order of `PHASES`.
"""

from __future__ import annotations

import gc  # explicit garbage collection between GPU-heavy phases
import json
import time

import torch

from .collect import collect_stats
from .data import build_splits, load_windows, save_windows
from .logging_utils import get_logger
from .model import load_model

log = get_logger(__name__)
SPLITS = ("discovery", "heldout")


def _save_json(obj, path) -> None:
    """Pretty JSON writer used for every small result file."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _free(*objs) -> None:
    """Drop references and release cached GPU memory before the next GPU user (e.g. Ollama)."""
    del objs
    gc.collect()
    torch.cuda.empty_cache()  # https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html


def get_windows(settings, tokenizer) -> dict:
    """Load cached windows for both splits, building and saving them on first use."""
    paths = {s: settings.run_dir() / f"windows_{s}.npz" for s in SPLITS}
    if not all(p.exists() for p in paths.values()):
        for split, windows in build_splits(tokenizer, settings).items():
            save_windows(windows, paths[split])
    return {s: load_windows(p) for s, p in paths.items()}


def phase_benchmark(settings) -> None:
    """Throughput + numerical agreement of the sequential scan vs the mamba.py parallel scan (use_mambapy)."""
    results = {}
    logits = {}
    for use_mambapy in (False, True):
        model, tok = load_model(settings.model_name, settings.dtype, settings.device, use_mambapy=use_mambapy)
        windows = get_windows(settings, tok)["discovery"]
        ids = torch.from_numpy(windows.tokens[: settings.batch_size]).long().to(model.device)
        with torch.inference_mode():
            model(input_ids=ids[:2], use_cache=False)  # warm-up (CUDA context, kernel selection)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                out = model(input_ids=ids, use_cache=False).logits
            torch.cuda.synchronize()
        rate = 3 * ids[:, 1:].numel() / (time.time() - t0)
        logits[use_mambapy] = out.float().cpu()
        results[f"use_mambapy={use_mambapy}"] = {"tokens_per_s": rate, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        log.info("benchmark use_mambapy=%s: %.0f tok/s, peak %.2f GB", use_mambapy, rate, results[f"use_mambapy={use_mambapy}"]["peak_mem_gb"])
        _free(model)
        torch.cuda.reset_peak_memory_stats()
    diff = (logits[True] - logits[False]).abs()
    results["max_abs_logit_diff"] = diff.max().item()
    results["argmax_agreement"] = (logits[True].argmax(-1) == logits[False].argmax(-1)).float().mean().item()
    log.info("benchmark agreement: max |dlogit| %.4f, argmax agreement %.4f", results["max_abs_logit_diff"], results["argmax_agreement"])
    _save_json(results, settings.results_path() / "benchmark.json")


def phase_collect(settings) -> None:
    """Activation statistics + top-k poles for every neuron and random direction, on both splits."""
    model, tok = load_model(settings.model_name, settings.dtype, settings.device, settings.use_mambapy)
    windows = get_windows(settings, tok)
    for split in SPLITS:
        out = settings.run_dir() / f"stats_{split}.pt"
        if out.exists():
            log.info("stats for %s already exist at %s - skipping", split, out)
            continue
        log.info("collecting %s: %d windows, %d tokens", split, len(windows[split].tokens), windows[split].n_tokens)
        torch.save(collect_stats(model, windows[split], settings), out)
    _free(model)


PHASES = {
    "benchmark": phase_benchmark,
    "collect": phase_collect,
}
