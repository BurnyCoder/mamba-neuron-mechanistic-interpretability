"""Export the neuron atlas to small JSON files for the static viewer (site/) and to a table for the HF dataset.

Local explanation: joins examples.json (token-level activations), autointerp.json (labels, detection scores),
candidates.json, vocab.pt (promoted / suppressed tokens), per-neuron statistics and the steering / mechanism results
into one `index.json` (one row per labelled pole) plus one detail file per pole.
Global context: the viewer is static (GitHub Pages), so everything it shows is precomputed here.
"""

from __future__ import annotations

import re

import torch

from .jsonio import load_json, save_json


def safe_name(key: str) -> str:
    """File-system-safe version of a pole key ('neuron:L3:N7:+' -> 'neuron_L3_N7_pos')."""
    return re.sub(r"[^A-Za-z0-9_]", "_", key.replace(":+", ":pos").replace(":-", ":neg"))


def _vocab(vstats, tok, layer: int, neuron: int, sign: int, n: int = 12) -> dict:
    """Promoted / suppressed tokens of a *neuron pole* (the negative pole writes -u, so the lists swap)."""
    up_ids, up_vals = ("top_ids", "top_vals") if sign > 0 else ("bot_ids", "bot_vals")
    dn_ids, dn_vals = ("bot_ids", "bot_vals") if sign > 0 else ("top_ids", "top_vals")
    row = lambda ids, vals: [[tok.decode([int(t)]), round(abs(float(v)), 3)] for t, v in zip(vstats[ids][layer, :n, neuron], vstats[vals][layer, :n, neuron])]
    return {"promoted": row(up_ids, up_vals), "suppressed": row(dn_ids, dn_vals), "vocab_kurtosis": float(vstats["kurtosis"][layer, neuron])}


def export_site(settings, tokenizer, out_dir) -> int:
    """Write site data into `out_dir` (index.json + poles/*.json); returns the number of poles exported."""
    res = settings.results_path()
    run = settings.run_dir()
    examples = load_json(run / "examples.json")
    labels = load_json(res / "autointerp.json")
    cands = {f"neuron:L{c['layer']}:N{c['neuron']}:{'+' if c['sign'] > 0 else '-'}": c for c in load_json(res / "candidates.json")}
    vstats = torch.load(run / "vocab.pt")
    summary = torch.load(run / "stats_discovery.pt")["neurons"]["summary"]
    screen = load_json(res / "screen.json") if (res / "screen.json").exists() else {"generations": [], "summary": []}
    h2 = load_json(res / "h2.json")["results"] if (res / "h2.json").exists() else {}
    h3 = load_json(res / "h3_mechanism.json") if (res / "h3_mechanism.json").exists() else {}
    screen_summary = {s["key"]: s for s in screen["summary"]}
    (out_dir / "poles").mkdir(parents=True, exist_ok=True)
    index = []
    for key, lab in labels.items():
        u = lab["unit"]
        row = {"key": key, "file": safe_name(key), "kind": u["kind"], "layer": u["layer"], "index": u["index"], "sign": u["sign"],
               "group": u.get("group", "audit"), "label": lab["explanation"], "ba": lab["balanced_accuracy"],
               "candidate_score": cands.get(key, {}).get("score"), "steering": key in screen_summary, "confirmed": key in h2}
        detail = {**row, "examples": {k: examples[key][k] for k in ("label", "detect_top", "detect_random")},
                  "detection": {k: lab[k] for k in ("answers", "truth", "valid")}}
        if u["kind"] == "neuron":
            L, i = u["layer"], u["index"]
            detail["stats"] = {k: round(float(v[L, i]), 4) for k, v in summary.items()}
            detail["vocab"] = _vocab(vstats, tokenizer, L, i, u["sign"])
        if key in screen_summary:
            detail["screen"] = {"levels": screen_summary[key]["levels"], "best_level": screen_summary[key]["best_level"],
                                "generations": [g for g in screen["generations"] if g["key"] == key]}
        if key in h2:
            detail["confirm"] = h2[key]
        if key in h3:
            detail["mechanism"] = h3[key]
        save_json(detail, out_dir / "poles" / f"{row['file']}.json")
        index.append(row)
    extras = {name: load_json(res / f"{name}.json") for name in ("h1", "split_stability", "benchmark") if (res / f"{name}.json").exists()}
    save_json({"model": settings.model_name, "poles": index, "h2": h2, **extras}, out_dir / "index.json")
    return len(index)


def export_hf(settings, tokenizer, out_dir) -> list[str]:
    """Write the HF dataset files (parquet via pandas/pyarrow) into `out_dir`; returns the file names.

    neuron_stats.parquet - one row per neuron (196,608): activation statistics on both splits + vocab projection
    poles.parquet        - one row per labelled pole: label, detection score, group, examples (JSON)
    steering.parquet     - every judged generation of the screen and the confirmation
    """
    import json

    import pandas as pd

    res, run = settings.results_path(), settings.run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {s: torch.load(run / f"stats_{s}.pt")["neurons"]["summary"] for s in ("discovery", "heldout")}
    vstats = torch.load(run / "vocab.pt")
    L, N = stats["discovery"]["max"].shape
    cols = {"layer": torch.arange(L).repeat_interleave(N).numpy(), "neuron": torch.arange(N).repeat(L).numpy()}
    for split, s in stats.items():
        cols.update({f"{k}_{split}": v.flatten().numpy() for k, v in s.items()})
    cols["vocab_kurtosis"] = vstats["kurtosis"].flatten().numpy()
    top = vstats["top_ids"][:, :5].permute(0, 2, 1).reshape(L * N, 5).numpy()  # (L*N, 5) token ids
    bot = vstats["bot_ids"][:, :5].permute(0, 2, 1).reshape(L * N, 5).numpy()
    vocab = {i: tokenizer.decode([i]) for i in set(top.flatten().tolist()) | set(bot.flatten().tolist())}
    cols["promoted_top5"] = [[vocab[t] for t in row] for row in top]
    cols["suppressed_top5"] = [[vocab[t] for t in row] for row in bot]
    pd.DataFrame(cols).to_parquet(out_dir / "neuron_stats.parquet")
    labels, examples = load_json(res / "autointerp.json"), load_json(run / "examples.json")
    pd.DataFrame([{"key": k, **{f: v["unit"][f] for f in ("kind", "layer", "index", "sign")}, "group": v["unit"].get("group", "audit"),
                   "label": v["explanation"], "detection_balanced_accuracy": v["balanced_accuracy"],
                   "promoted": examples[k]["promoted"], "examples_json": json.dumps({x: examples[k][x] for x in ("label", "detect_top", "detect_random")}, ensure_ascii=False)}
                  for k, v in labels.items()]).to_parquet(out_dir / "poles.parquet")
    rows = []
    if (res / "screen.json").exists():
        rows += [{"stage": "screen", **{k: g.get(k) for k in ("key", "level", "prompt_idx", "prompt", "text", "concept", "fluency", "hm", "nll", "distinct2")}}
                 for g in load_json(res / "screen.json")["generations"]]
    if (res / "h2.json").exists():
        rows += [{"stage": "confirm", **{k: g.get(k) for k in ("key", "condition", "seed", "prompt_idx", "prompt", "text", "concept", "fluency", "hm", "nll", "distinct2")}}
                 for g in load_json(res / "h2.json")["generations"]]
    if rows:
        pd.DataFrame(rows).to_parquet(out_dir / "steering.parquet")
    return sorted(p.name for p in out_dir.glob("*.parquet"))
