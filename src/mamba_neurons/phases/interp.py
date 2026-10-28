"""Auto-interpretation phases: LLM labels + detection scores (resumable), and the pre-registered H1 analysis.

Local explanation: `phase_label` asks the local LLM for an explanation of every pole in examples.json and scores it
by held-out detection; results are appended to results/<model>/autointerp.json every few poles so a long run can
resume. `phase_h1` compares the fraction of "clean" poles (balanced accuracy >= 0.8) between random neurons and random
directions exactly as pre-registered.
Global context: labels are reused as the concept descriptions for steering; H1 tests the privileged-basis claim.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..autointerp import detect, explain
from ..llm import LocalLLM
from ..logging_utils import get_logger
from ..metrics import bootstrap_diff_of_props, cluster_bootstrap_mean, fisher_greater, wilson
from .common import load_json, save_json

log = get_logger(__name__)
CLEAN = 0.8  # pre-registered threshold on detection balanced accuracy


def phase_label(settings) -> None:
    """Explain + detection-score every pole (resumes from a partial results file)."""
    examples = load_json(settings.run_dir() / "examples.json")
    out_path = settings.results_path() / "autointerp.json"
    done = load_json(out_path) if out_path.exists() else {}
    llm = LocalLLM(settings.ollama_host, settings.llm_model)
    todo = [k for k in examples if k not in done]
    log.info("labelling %d poles (%d already done)", len(todo), len(done))
    for n, key in enumerate(todo, 1):
        pole = examples[key]
        explanation = explain(llm, pole)
        det = detect(llm, explanation, pole, seed=zlib.crc32(key.encode()))  # per-pole deterministic shuffle
        done[key] = {"unit": pole["unit"], "explanation": explanation, **det}
        log.info("[%d/%d] %s | BA=%.2f | %s", n, len(todo), key, det["balanced_accuracy"], explanation)
        if n % 10 == 0 or n == len(todo):
            save_json(done, out_path)
    llm.unload()


def phase_h1(settings) -> None:
    """Pre-registered H1: clean-pole rate of random neurons vs random directions (audit sample only)."""
    res = load_json(settings.results_path() / "autointerp.json")
    groups = {"neuron": [], "direction": []}
    for r in res.values():
        if r["unit"].get("group", "audit") == "audit":
            groups[r["unit"]["kind"]].append(r)
    out = {}
    for kind, rows in groups.items():
        ba = np.array([r["balanced_accuracy"] for r in rows])
        clean = (ba >= CLEAN).astype(float)
        clusters = np.array([f"{r['unit']['layer']}:{r['unit']['index']}" for r in rows])  # both poles of a unit
        mean, lo, hi = cluster_bootstrap_mean(ba, clusters)
        out[kind] = {"n_poles": len(rows), "n_clean": int(clean.sum()), "clean_rate": float(clean.mean()),
                     "clean_rate_wilson95": wilson(int(clean.sum()), len(rows)), "mean_ba": mean, "mean_ba_ci95": [lo, hi],
                     "n_invalid_llm_replies": int(sum(not r["valid"] for r in rows))}
    n, d = groups["neuron"], groups["direction"]
    cn, cd = [r["balanced_accuracy"] >= CLEAN for r in n], [r["balanced_accuracy"] >= CLEAN for r in d]
    unit_of = lambda rows: [f"{r['unit']['layer']}:{r['unit']['index']}" for r in rows]  # both poles share a unit
    diff, lo, hi = bootstrap_diff_of_props(cn, cd, unit_of(n), unit_of(d), seed=settings.seed)  # resample units
    p = fisher_greater(sum(cn), len(cn), sum(cd), len(cd))
    out["difference"] = {"clean_rate_diff": diff, "ci95": [lo, hi], "fisher_one_sided_p": p,
                         "h1_supported": bool(diff > 0 and lo > 0 and p < 0.05)}
    save_json(out, settings.results_path() / "h1.json")
    log.info("H1: %s", out)
