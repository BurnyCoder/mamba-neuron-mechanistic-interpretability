"""Automated interpretability: label each unit pole from top examples, score the label by held-out detection.

Local explanation:
  * A *unit* is a neuron (layer, i) or a random direction q in a layer's neuron space; a *pole* is the unit with a
    sign s (+1: large positive activations, -1: large negative activations), following Goodfire's poles.
  * `build_examples` (GPU, Mamba) recomputes per-token activations for the top-firing windows of each pole: discovery
    windows for labelling, held-out windows + random held-out snippets for detection.
  * `explain` (LLM) asks for a one-phrase explanation from highlighted discovery examples; `detect` (LLM) shows the
    explanation and 20 shuffled *unhighlighted* held-out snippets (10 top, 10 random) and asks which match; the
    score is balanced accuracy. Both prompts are adapted from EleutherAI delphi
    (https://github.com/EleutherAI/delphi/blob/main/delphi/explainers/default/prompts.py and
    .../scorers/classifier/prompts/detection_prompt.py; Paulo et al. 2024, https://arxiv.org/abs/2410.13928).
Global context: balanced accuracy on held-out data is the H1 metric (neurons vs random directions) and also ranks
the steering candidates' label quality.
"""

from __future__ import annotations

import numpy as np
import torch

from .collect import decode_index
from .data import decode_tokens
from .logging_utils import get_logger
from .model import record_layer
from .vocab import direction_logit_effect

log = get_logger(__name__)
SPAN_BEFORE, SPAN_AFTER = 24, 8  # snippet = 24 tokens before the peak, the peak, and 8 after (delphi uses ~32-token contexts)


def audit_units(n_layers: int, n_neurons: int, n_dirs: int, per_layer: int, seed: int) -> list[dict]:
    """Pre-registered H1 sample: `per_layer` random neurons per layer + every random direction (both poles each)."""
    rng = np.random.default_rng(seed)
    units = []
    for layer in range(n_layers):
        for neuron in rng.choice(n_neurons, size=per_layer, replace=False):
            units += [{"kind": "neuron", "layer": layer, "index": int(neuron), "sign": s} for s in (1, -1)]
        for j in range(n_dirs):
            units += [{"kind": "direction", "layer": layer, "index": j, "sign": s} for s in (1, -1)]
    return units


def unit_key(u: dict) -> str:
    """Stable string id of a pole, e.g. 'neuron:L31:N2047:+'."""
    return f"{u['kind']}:L{u['layer']}:N{u['index']}:{'+' if u['sign'] > 0 else '-'}"


def _pole_rows(stats: dict, u: dict, ctx: int, n: int) -> list[tuple[int, int]]:
    """Up to `n` (window row, position) pairs of the pole's strongest firings, one per window (diversity)."""
    group = stats["neurons"] if u["kind"] == "neuron" else stats["random_dirs"]
    idx = (group["top_idx"] if u["sign"] > 0 else group["bot_idx"])[u["layer"], :, u["index"]].numpy()
    rows, pos = decode_index(idx[idx >= 0], ctx)
    seen, out = set(), []
    for r, p in zip(rows.tolist(), pos.tolist()):
        if r not in seen:
            seen.add(r)
            out.append((r, p))
        if len(out) == n:
            break
    return out


def _span(pos: int, ctx: int) -> tuple[int, int]:
    """Inclusive token range of the snippet around `pos` inside a window row (position 0 = separator excluded)."""
    return max(1, pos - SPAN_BEFORE), min(ctx, pos + SPAN_AFTER)


@torch.no_grad()
def build_examples(model, tokenizer, windows: dict, stats: dict, directions: torch.Tensor, units: list[dict],
                   n_label: int = 20, n_detect: int = 10, seed: int = 0, batch: int = 16) -> dict[str, dict]:
    """Per pole: {"label": [...], "detect_top": [...], "detect_random": [...]}, each example = {tokens, acts, peak}."""
    disc, held = windows["discovery"], windows["heldout"]
    ctx = disc.tokens.shape[1] - 1
    rng = np.random.default_rng(seed)
    plan = {}  # pole key -> {set name: [(split, row, pos)]}
    for u in units:
        label = [("discovery", r, p) for r, p in _pole_rows(stats["discovery"], u, ctx, n_label)]
        top = [("heldout", r, p) for r, p in _pole_rows(stats["heldout"], u, ctx, n_detect)]
        taken = {r for _, r, _ in top}
        rand_rows = rng.choice([r for r in range(len(held.tokens)) if r not in taken] if len(taken) else len(held.tokens),
                               size=n_detect, replace=False)
        rand = [("heldout", int(r), int(rng.integers(SPAN_BEFORE + 1, ctx + 1))) for r in rand_rows]
        plan[unit_key(u)] = {"unit": u, "label": label, "detect_top": top, "detect_random": rand}

    # Recompute per-token activations, grouped by (layer, split) so each window is run at most once per layer.
    acts_cache: dict[tuple, np.ndarray] = {}
    by_layer: dict[tuple, set] = {}
    for p in plan.values():
        for name in ("label", "detect_top", "detect_random"):
            for split, r, _ in p[name]:
                by_layer.setdefault((p["unit"]["layer"], split), set()).add(r)
    units_by_layer: dict[int, list[dict]] = {}
    for u in units:
        units_by_layer.setdefault(u["layer"], []).append(u)
    for (layer, split), rows in sorted(by_layer.items()):
        rows = sorted(rows)
        win = disc if split == "discovery" else held
        cols = sorted({u["index"] for u in units_by_layer[layer] if u["kind"] == "neuron"})
        q = directions[layer].to(model.device)
        for b in range(0, len(rows), batch):
            ids = torch.from_numpy(win.tokens[rows[b:b + batch]]).long().to(model.device)
            a = record_layer(model, ids, layer).float()  # (B, ctx+1, N)
            neuron_acts = a[..., cols].cpu().numpy()
            dir_acts = (a @ q).cpu().numpy()  # (B, ctx+1, n_dirs)
            for j, r in enumerate(rows[b:b + batch]):
                for c_i, c in enumerate(cols):
                    acts_cache[(layer, split, r, "neuron", c)] = neuron_acts[j, :, c_i]
                for d in range(dir_acts.shape[-1]):
                    acts_cache[(layer, split, r, "direction", d)] = dir_acts[j, :, d]
        log.info("examples: layer %d %s, %d windows", layer, split, len(rows))

    out = {}
    for key, p in plan.items():
        u = p["unit"]
        vec = directions[u["layer"], :, u["index"]] if u["kind"] == "direction" else torch.nn.functional.one_hot(
            torch.tensor(u["index"]), directions.shape[1]).float()
        top_ids = (u["sign"] * direction_logit_effect(model, u["layer"], vec)).topk(10).indices.tolist()
        out[key] = {"unit": u, "promoted": [tokenizer.decode([t]) for t in top_ids]}  # what this pole pushes the model to say
        for name in ("label", "detect_top", "detect_random"):
            exs = []
            for split, r, pos in p[name]:
                win = disc if split == "discovery" else held
                lo, hi = _span(pos, ctx)
                acts = acts_cache[(u["layer"], split, r, u["kind"], u["index"])]
                exs.append({"split": split, "row": r, "peak": pos - lo,
                            "tokens": decode_tokens(tokenizer, win.tokens[r, lo:hi + 1]),
                            "acts": [round(float(x), 4) for x in acts[lo:hi + 1]]})
            out[key][name] = exs
    return out


def highlight(example: dict, sign: int, frac: float = 0.4) -> str:
    """Snippet text with tokens whose signed activation >= frac * signed peak wrapped in <<...>> (runs merged)."""
    acts = np.asarray(example["acts"]) * sign
    peak = acts[example["peak"]]
    on = acts >= frac * peak if peak > 0 else np.zeros_like(acts, dtype=bool)
    parts, open_ = [], False
    for tok, flag in zip(example["tokens"], on):
        if flag and not open_:
            parts.append("<<")
        if not flag and open_:
            parts.append(">>")
        parts.append(tok)
        open_ = bool(flag)
    if open_:
        parts.append(">>")
    return "".join(parts).replace("\n", "\\n")


def plain(example: dict) -> str:
    """Snippet text without highlights (used for detection, so the scorer cannot see where the unit fires)."""
    return "".join(example["tokens"]).replace("\n", "\\n")


EXPLAIN_SYSTEM = (
    "You are a meticulous AI researcher investigating a single neuron inside a language model. "
    "You will see text examples where the neuron fires strongly; the tokens on which it fires most are marked "
    "between delimiters like <<this>> (consecutive marked tokens are wrapped together <<like this>>). "
    "You also get the tokens the neuron pushes the model to predict next. Many neurons fire on the token *before* "
    "a predictable word, so check whether the words right after the marked tokens match that list. "
    "Find what the marked tokens and their context have in common: a specific token or word, the word that comes "
    "next, a concept or topic, a syntactic role, a language, or a code or formatting pattern. "
    "Write ONE concise explanation of at most 15 words describing when the neuron fires. Do not mention the "
    "delimiters, do not list alternatives, and be as specific as the examples allow. "
    "Reply as JSON: {\"explanation\": \"...\"}."
)
EXPLAIN_SCHEMA = {"type": "object", "properties": {"explanation": {"type": "string"}}, "required": ["explanation"]}

DETECT_SYSTEM = (
    "You are an intelligent and meticulous linguistics researcher. You will be given an explanation of when a "
    "neuron in a language model fires, followed by numbered text examples. For each example in turn, answer 1 if "
    "the explanation applies to some token in the example (the neuron would fire), otherwise 0. Reply as JSON: "
    "{\"answers\": [0 or 1 for every example, in order]}."
)


def explain(llm, pole: dict) -> str:
    """One-phrase explanation of a pole from its highlighted discovery examples."""
    shown = "\n".join(f"Example {i + 1}: {highlight(ex, pole['unit']['sign'])}" for i, ex in enumerate(pole["label"]))
    reply = llm.json("explain", EXPLAIN_SYSTEM, f"Examples:\n{shown}\n\nExplanation as JSON:", EXPLAIN_SCHEMA)
    return str(reply.get("explanation", "")).strip()


def detect(llm, explanation: str, pole: dict, seed: int) -> dict:
    """Balanced accuracy of `explanation` at telling held-out top snippets from random held-out snippets."""
    items = [(ex, 1) for ex in pole["detect_top"]] + [(ex, 0) for ex in pole["detect_random"]]
    order = np.random.default_rng(seed).permutation(len(items))  # shuffle so position carries no signal
    items = [items[i] for i in order]
    shown = "\n".join(f"Example {i + 1}: {plain(ex)}" for i, (ex, _) in enumerate(items))
    schema = {"type": "object", "required": ["answers"],
              "properties": {"answers": {"type": "array", "items": {"type": "integer", "enum": [0, 1]},
                                         "minItems": len(items), "maxItems": len(items)}}}
    reply = llm.json("detect", DETECT_SYSTEM, f"Explanation: {explanation}\n\nText examples:\n{shown}\n\nAnswers as JSON:", schema)
    answers = reply.get("answers") or []  # None / missing -> treated as malformed below
    truth = [t for _, t in items]
    if len(answers) != len(truth):  # malformed reply -> chance-level score, flagged
        return {"balanced_accuracy": 0.5, "valid": False, "answers": answers, "truth": truth}
    return {"balanced_accuracy": balanced_accuracy(truth, answers), "valid": True, "answers": answers, "truth": truth}


def balanced_accuracy(truth, pred) -> float:
    """Mean of true-positive rate and true-negative rate (robust to class imbalance)."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    tpr = (pred[truth == 1] == 1).mean() if (truth == 1).any() else 0.0
    tnr = (pred[truth == 0] == 0).mean() if (truth == 0).any() else 0.0
    return float((tpr + tnr) / 2)
