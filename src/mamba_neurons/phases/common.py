"""Shared I/O helpers for the pipeline phases.

Local explanation: JSON read/write, GPU memory release, cached windows and cached statistics loading.
Global context: every phase module (atlas, interp, steering, mechanism, publish) uses these so file names and
formats are defined in exactly one place.
"""

from __future__ import annotations

import gc
import json

import torch

from ..data import build_splits, load_windows, save_windows

SPLITS = ("discovery", "heldout")


def save_json(obj, path) -> None:
    """Pretty UTF-8 JSON writer used for every result file."""
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    """Inverse of `save_json`."""
    return json.loads(path.read_text(encoding="utf-8"))


def free_gpu(*objs) -> None:
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


def load_stats(settings) -> dict:
    """{"discovery": stats, "heldout": stats} as saved by the collect phase."""
    return {s: torch.load(settings.run_dir() / f"stats_{s}.pt") for s in SPLITS}


def pole_extreme(stats_split: dict, layer: int, neuron: int, sign: int) -> float:
    """|max| (positive pole) or |min| (negative pole) of a neuron in a split's summary."""
    summary = stats_split["neurons"]["summary"]
    return abs(float((summary["max"] if sign > 0 else summary["min"])[layer, neuron]))
