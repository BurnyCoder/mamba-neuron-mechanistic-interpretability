"""Steering phases: exploratory screen (prompt set A) and pre-registered confirmation (prompt set B), each split into
a generation phase (Mamba on GPU) and a judging phase (Ollama on GPU), never both at once.

Local explanation:
  * screen: clamp each well-labelled candidate pole at c x |pole extreme| for c in LEVELS on 10 prompts; judge concept
    (the pole's output-side steering concept, amendment 4) + fluency; pick <= 3 poles by mean harmonic-mean gain over
    unsteered -> results/confirm_plan.json (the post-hoc low-dose sweep does the same -> confirm_plan_h2b.json).
  * confirm: for each planned pole, 50 neutral prompts (neuron inactive) x 2 seeds under unsteered / clamp / add /
    DiffMean / prompting, plus 80 random same-layer neurons x 10 prompts (amendment 5); judge; H2 statistics.
Global context: H2 in reports/preregistration.md; the confirm plan must be appended there before `confirm` runs.
"""

from __future__ import annotations

import numpy as np

from ..judge import rate_concept, rate_fluency, steering_concept
from ..llm import LocalLLM
from ..logging_utils import get_logger
from ..metrics import (bootstrap_p, cluster_bootstrap_diff, cluster_bootstrap_mean, count_word, distinct_n, empirical_p,
                       harmonic_mean, holm)
from ..model import load_model, n_neurons, record_layer
from ..steer import (add_setup, clamp_setup, continuation_nll, diffmean_vector, generate, neutral_prompts,
                     pole_values, residual_budget, vector_setup)
from .common import free_gpu, get_windows, load_json, load_stats, pole_extreme, save_json

log = get_logger(__name__)
LEVELS = (2, 5, 10, 20)  # clamp multiples of the pole extreme (pre-registered)
LOW_LEVELS = (0.25, 0.5, 1.0)  # post-hoc gentler doses (amendment 6)
PROMPT_LEN, N_SCREEN, N_CONFIRM, POOL = 16, 10, 50, 400
N_RANDOM, N_RANDOM_PROMPTS = 80, 10  # amendment 5: min empirical p = 1/81, so Holm over 3 poles can reach < 0.05
JUDGE_CTX = 2048  # judge prompts are < 1k tokens; a small KV cache keeps the 9B judge entirely on an 8 GB GPU
CLEAN = 0.8


def _record_generations(model, tok, prompts, conts, meta: dict) -> list[dict]:
    """Rows of {meta..., prompt_idx, text, nll, distinct2} for a batch of continuations."""
    nll = continuation_nll(model, prompts, conts)
    return [{**meta, "prompt_idx": i, "prompt": tok.decode(prompts[i, 1:]), "text": tok.decode(conts[i]),
             "nll": float(nll[i]), "distinct2": distinct_n(conts[i].tolist(), 2)} for i in range(len(conts))]


def _screen_poles(settings) -> list[dict]:
    """Candidate poles (both streams) whose label is clean (BA >= 0.8), in stream order (amendment 2 in the prereg)."""
    from .atlas import candidate_poles
    from ..autointerp import unit_key

    labels = load_json(settings.results_path() / "autointerp.json")
    poles = []
    for u in candidate_poles(settings):
        key = unit_key(u)
        if key in labels and labels[key]["balanced_accuracy"] >= CLEAN:
            poles.append({"key": key, "layer": u["layer"], "neuron": u["index"], "sign": u["sign"], "group": u["group"],
                          "label": labels[key]["explanation"], "ba": labels[key]["balanced_accuracy"], **u["meta"]})
    return poles


def _screen_generate(settings, levels, tag: str) -> None:
    """Clamp sweep on prompt set A at `levels` x |pole extreme| for every clean candidate pole."""
    model, tok = load_model(settings.model_name, settings.dtype, settings.device)
    held = get_windows(settings, tok)["heldout"]
    stats = load_stats(settings)
    prompts = neutral_prompts(held, N_SCREEN, PROMPT_LEN, settings.seed)  # prompt set A
    rows = _record_generations(model, tok, prompts, generate(model, prompts, None, settings.seed), {"key": "unsteered", "level": 0})
    poles = _screen_poles(settings)
    log.info("%s: %d clean candidate poles at levels %s", tag, len(poles), levels)
    for p in poles:
        ext = pole_extreme(stats["discovery"], p["layer"], p["neuron"], p["sign"])
        for c in levels:
            setup = clamp_setup(p["layer"], p["neuron"], p["sign"] * c * ext)
            conts = generate(model, prompts, setup, settings.seed)
            rows += _record_generations(model, tok, prompts, conts, {"key": p["key"], "level": c})
        log.info("%s %s (%s): e.g. c=%s -> %r", tag, p["key"], p["label"], levels[-1], rows[-1]["text"][:120])
    save_json({"poles": poles, "generations": rows}, settings.run_dir() / f"{tag}_generations.json")
    del model  # drop the last reference first, otherwise empty_cache cannot release the weights
    free_gpu()


def phase_screen(settings) -> None:
    """Pre-registered exploratory screen: clamp at 2, 5, 10, 20 x max on prompt set A."""
    _screen_generate(settings, LEVELS, "screen")


def phase_screen_low(settings) -> None:
    """POST HOC (amendment 6): the same sweep at gentler doses 0.25, 0.5, 1 x max."""
    _screen_generate(settings, LOW_LEVELS, "screen_low")


def _judge(llm, rows: list[dict], concept_of) -> None:
    """Add concept / fluency / hm to each row in place; fluency is rated once per distinct text."""
    fluency = {}
    for t in sorted({r["text"] for r in rows}):
        fluency[t] = rate_fluency(llm, t)
    log.info("fluency judged for %d distinct texts", len(fluency))
    for n, r in enumerate(rows, 1):
        r["concept"], r["fluency"] = rate_concept(llm, concept_of(r), r["text"]), fluency[r["text"]]
        r["hm"] = harmonic_mean(r["concept"], r["fluency"])
        if n % 200 == 0:
            log.info("concept judged %d/%d", n, len(rows))


def _screen_judge(settings, levels, tag: str, reuse_concepts: bool) -> list[dict]:
    """Judge a sweep; returns the per-pole summary sorted by mean HM gain over unsteered (saved to results)."""
    data = load_json(settings.run_dir() / f"{tag}_generations.json")
    poles = {p["key"]: p for p in data["poles"]}
    llm = LocalLLM(settings.ollama_host, settings.llm_model, num_ctx=JUDGE_CTX)
    if reuse_concepts:  # identical concept text as the pre-registered screen
        known = {x["key"]: x["concept"] for x in load_json(settings.results_path() / "screen.json")["summary"]}
        for key, p in poles.items():
            p["concept"] = known[key]
    else:
        promoted = {k: v["promoted"] for k, v in load_json(settings.run_dir() / "examples.json").items() if k in poles}
        for key, p in poles.items():  # fixed before any steered text is judged
            p["concept"] = steering_concept(llm, p["label"], promoted[key])
            log.info("steering concept %s: %s -> %s", key, p["label"], p["concept"])
    base = [r for r in data["generations"] if r["key"] == "unsteered"]
    rows = [r for r in data["generations"] if r["key"] != "unsteered"]
    rows += [{**r, "key": k, "level": 0} for k in poles for r in base]  # unsteered judged against every pole's concept
    _judge(llm, rows, lambda r: poles[r["key"]]["concept"])
    llm.unload()
    summary = []
    for key, p in poles.items():
        by_level = {c: [r for r in rows if r["key"] == key and r["level"] == c] for c in (0, *levels)}
        stats = {c: {m: float(np.mean([r[m] for r in rs])) for m in ("concept", "fluency", "hm", "nll")} for c, rs in by_level.items()}
        best = max(levels, key=lambda c: stats[c]["hm"])
        summary.append({**p, "levels": stats, "best_level": best, "hm_gain": stats[best]["hm"] - stats[0]["hm"]})
    summary.sort(key=lambda s: -s["hm_gain"])
    save_json({"summary": summary, "generations": rows}, settings.results_path() / f"{tag}.json")
    return summary


def _plan(summary: list[dict]) -> list[dict]:
    """Confirmation plan: the first 3 poles of a sweep summary (already sorted by mean HM gain) at their best level."""
    return [{k: s[k] for k in ("key", "layer", "neuron", "sign", "label", "concept", "best_level", "hm_gain")} for s in summary[:3]]


def phase_screen_judge(settings) -> None:
    """Judge the screen; choose <= 3 poles (and their level) for confirmation by mean HM gain over unsteered."""
    summary = _screen_judge(settings, LEVELS, "screen", reuse_concepts=False)
    plan = _plan(summary)
    save_json(plan, settings.results_path() / "confirm_plan.json")
    log.info("confirm plan: %s", plan)


def phase_screen_low_judge(settings) -> None:
    """POST HOC: judge the low-dose sweep with the screen's steering concepts; the top 3 poles by HM gain, at their
    best dose, become the H2b plan (results/confirm_plan_h2b.json, frozen in amendment 7 before `confirm_h2b`)."""
    summary = _screen_judge(settings, LOW_LEVELS, "screen_low", reuse_concepts=True)
    log.info("low-dose sweep, top 5 by HM gain: %s", [(s["key"], s["concept"], s["best_level"], round(s["hm_gain"], 2)) for s in summary[:5]])
    save_json(_plan(summary), settings.results_path() / "confirm_plan_h2b.json")


def _neutral_prompt_set(model, held, disc, layer, neuron, sign, seed, skip: int) -> np.ndarray:
    """50 openings (from a pool of 400 starting at `skip`) on which s*a_i stays below its 90th discovery percentile."""
    q90 = np.quantile(pole_values(model, disc.tokens[:200], layer, neuron, sign)[:, 1:], 0.9)
    pool = neutral_prompts(held, POOL, PROMPT_LEN, seed, skip=skip)  # B: skip=10 (after set A); C: skip=410 (after B)
    ok = pole_values(model, pool, layer, neuron, sign)[:, 1:].max(1) < q90
    log.info("neutral prompts for L%d N%d: %d/%d eligible (q90=%.3f)", layer, neuron, ok.sum(), len(pool), q90)
    return pool[ok][:N_CONFIRM]


def _confirm_generate(settings, plan_name: str, out_name: str, skip: int) -> None:
    """Generate every confirmation condition for each pole of results/<plan_name>.json on a disjoint prompt set."""
    model, tok = load_model(settings.model_name, settings.dtype, settings.device)
    windows = get_windows(settings, tok)
    stats = load_stats(settings)
    rng = np.random.default_rng(settings.seed)
    rows = []
    for p in load_json(settings.results_path() / f"{plan_name}.json"):
        L, i, s, c = p["layer"], p["neuron"], p["sign"], p["best_level"]
        prompts = _neutral_prompt_set(model, windows["heldout"], windows["discovery"], L, i, s, settings.seed, skip)
        value = s * c * pole_extreme(stats["discovery"], L, i, s)
        budget = residual_budget(model, L, i, value)
        dm = diffmean_vector(model, windows["heldout"], stats["heldout"]["neurons"], L, i, s, settings.seed)
        prefix = np.asarray(tok(f"{p['concept']}:\n").input_ids, dtype=prompts.dtype)  # prompting baseline (amendment 4)
        prompted = np.concatenate([prompts[:, :1], np.tile(prefix, (len(prompts), 1)), prompts[:, 1:]], axis=1)
        conditions = {"unsteered": None, "clamp": clamp_setup(L, i, value), "add": add_setup(L, i, value),
                      "diffmean": vector_setup(L, dm / dm.norm() * budget)}
        for seed in (settings.seed, settings.seed + 1):
            for name, setup in conditions.items():
                rows += _record_generations(model, tok, prompts, generate(model, prompts, setup, seed),
                                            {"key": p["key"], "condition": name, "seed": seed})
            conts = generate(model, prompted, None, seed)
            rows += _record_generations(model, tok, prompts, conts, {"key": p["key"], "condition": "prompting", "seed": seed})
        others = rng.choice([j for j in range(n_neurons(model)) if j != i], N_RANDOM, replace=False)
        for j in others:
            vj = s * c * pole_extreme(stats["discovery"], L, int(j), s)
            sub = prompts[:N_RANDOM_PROMPTS]  # first 10 of the 50 confirmation prompts
            rows += _record_generations(model, tok, sub, generate(model, sub, clamp_setup(L, int(j), vj), settings.seed),
                                        {"key": p["key"], "condition": f"random:{int(j)}", "seed": settings.seed})
        log.info("confirm %s: %d generations so far; clamp e.g. %r", p["key"], len(rows),
                 next(r["text"] for r in rows if r["key"] == p["key"] and r["condition"] == "clamp")[:150])
    save_json(rows, settings.run_dir() / f"{out_name}.json")
    del model  # drop the last reference first, otherwise empty_cache cannot release the weights
    free_gpu()


def phase_confirm(settings) -> None:
    """Pre-registered H2 confirmation on prompt set B."""
    _confirm_generate(settings, "confirm_plan", "confirm_generations", skip=N_SCREEN)


def phase_confirm_h2b(settings) -> None:
    """H2b (pre-registered after the post-hoc low-dose sweep) on prompt set C, disjoint from A and B."""
    _confirm_generate(settings, "confirm_plan_h2b", "confirm_h2b_generations", skip=N_SCREEN + POOL)


def _confirm_judge(settings, plan_name: str, gen_name: str, out_name: str, noninferiority: bool) -> None:
    """Judge confirmation generations and run the pre-registered tests.

    Fluency criterion: H2 (absolute) requires judged fluency >= 1.0; H2b (amendment 7, non-inferiority) requires
    distinct-2 >= 0.9 x unsteered and judged fluency >= unsteered - 0.2. Both verdicts are always stored.
    """
    rows = load_json(settings.run_dir() / f"{gen_name}.json")
    plan = {p["key"]: p for p in load_json(settings.results_path() / f"{plan_name}.json")}
    llm = LocalLLM(settings.ollama_host, settings.llm_model, num_ctx=JUDGE_CTX)
    _judge(llm, rows, lambda r: plan[r["key"]]["concept"])
    llm.unload()
    _confirm_stats(settings, rows, plan, out_name, noninferiority, _target_words(settings, plan))


def _target_words(settings, plan: dict) -> dict[str, str]:
    """Each planned pole's top promoted token (from the screen summary): the word counted by the judge-free check."""
    summary = {s["key"]: s for s in load_json(settings.results_path() / "screen.json")["summary"]}
    return {key: summary[key]["promoted_tokens"][0] for key in plan}  # formula candidates store their promoted tokens


def _posthoc_audit(by: dict, word: str, seed: int) -> dict:
    """POST HOC (2026-11-01 independent audit, pre-registration amendment 9): robustness statistics for one pole.

    * clamp - unsteered paired CIs of fluency, distinct-2 and NLL, and a CI-based non-inferiority verdict: lower CI
      bounds of (fluency_clamp - fluency_unsteered) >= -0.2 and of (distinct2_clamp - 0.9 * distinct2_unsteered) >= 0,
      the pre-registered margins tested on CIs instead of point estimates (Piaggio et al. 2012,
      https://doi.org/10.1001/jama.2012.87802);
    * judge-free concept score: occurrences of the pole's top promoted word per generation, against the null;
    * like-for-like null: the target scored only on the null's own prompts and seed (the null has 10 generations per
      neuron, the target 100);
    * bootstrap p-values of the clamp-vs-baseline HM differences (Holm across poles is applied by the caller).
    """
    order = lambda rs: sorted(rs, key=lambda r: (r["seed"], r["prompt_idx"]))  # pair identical prompt + seed
    clamp, unst = order(by["clamp"]), order(by["unsteered"])
    clusters = [r["prompt_idx"] for r in clamp]
    count = lambda rs: [count_word(r["text"], word) for r in rs]

    def diff(a, b) -> dict:
        """Paired clamp - unsteered difference with its prompt-cluster bootstrap CI."""
        d = cluster_bootstrap_diff(a, b, clusters, seed=seed)
        return {"diff": d[0], "ci95": [d[1], d[2]]}

    paired ={m: diff([r[m] for r in clamp], [r[m] for r in unst]) for m in ("fluency", "distinct2", "nll")}
    d2_margin = diff([r["distinct2"] for r in clamp], [0.9 * r["distinct2"] for r in unst])
    nulls = [rs for name, rs in by.items() if name.startswith("random:")]
    null_cells = {(r["seed"], r["prompt_idx"]) for rs in nulls for r in rs}
    sub = [r for r in clamp if (r["seed"], r["prompt_idx"]) in null_cells]  # target on the null's prompts and seed
    null_hm, null_count = [float(np.mean([r["hm"] for r in rs])) for rs in nulls], [float(np.mean(count(rs))) for rs in nulls]
    return {
        "target_word": word.strip(),
        "clamp_minus_unsteered": paired,
        "distinct2_clamp_minus_0.9_unsteered": d2_margin,
        "noninferior_fluency_ci": bool(paired["fluency"]["ci95"][0] >= -0.2 and d2_margin["ci95"][0] >= 0),
        "word_count": {name: float(np.mean(count(rs))) for name, rs in by.items() if not name.startswith("random:")},
        "word_count_null": null_count,
        "word_count_clamp_minus_unsteered": diff(count(clamp), count(unst)),
        "p_word_count_vs_null": empirical_p(float(np.mean(count(clamp))), null_count),
        "like_for_like": {"n_generations": len(sub), "clamp_hm": float(np.mean([r["hm"] for r in sub])),
                          "p_hm_vs_null": empirical_p(float(np.mean([r["hm"] for r in sub])), null_hm),
                          "clamp_word_count": float(np.mean(count(sub))),
                          "p_word_count_vs_null": empirical_p(float(np.mean(count(sub))), null_count)},
        **{f"p_clamp_vs_{base}": bootstrap_p([r["hm"] for r in clamp], [r["hm"] for r in order(by[base])], clusters, seed=seed)
           for base in ("prompting", "diffmean")},
    }


def _holm_into(blocks: list[dict], field: str, suffix: str = "_holm") -> None:
    """Holm-adjust `field` across `blocks` (one per pole) and store it next to it as `field + suffix`."""
    for b, p in zip(blocks, holm([b[field] for b in blocks])):
        b[field + suffix] = p


def _confirm_stats(settings, rows: list[dict], plan: dict, out_name: str, noninferiority: bool,
                   targets: dict[str, str]) -> None:
    """Pre-registered statistics on judged confirmation rows (no LLM calls), saved to results/<out_name>.json.

    `targets` maps each pole to the word counted by the post-hoc judge-free check (`_target_words`).
    """
    results, pvals = {}, []
    for key in plan:
        mine = [r for r in rows if r["key"] == key]
        by = {}
        for r in mine:
            by.setdefault(r["condition"], []).append(r)
        cond = {}
        for name, rs in by.items():
            if name.startswith("random:"):
                continue
            hm = np.array([r["hm"] for r in rs])
            clusters = np.array([r["prompt_idx"] for r in rs])
            cond[name] = {m: float(np.mean([r[m] for r in rs])) for m in ("concept", "fluency", "hm", "nll", "distinct2")}
            cond[name]["hm_ci95"] = list(cluster_bootstrap_mean(hm, clusters, seed=settings.seed)[1:])
        nulls = [rs for name, rs in by.items() if name.startswith("random:")]
        null = [float(np.mean([r["hm"] for r in rs])) for rs in nulls]
        order = lambda rs: sorted(rs, key=lambda r: (r["seed"], r["prompt_idx"]))  # pair identical prompt + seed
        clamp = order(by["clamp"])

        def clamp_minus(other: str) -> dict:
            """Paired clamp - `other` HM difference with a prompt-cluster bootstrap CI."""
            d = cluster_bootstrap_diff([r["hm"] for r in clamp], [r["hm"] for r in order(by[other])],
                                       [r["prompt_idx"] for r in clamp], seed=settings.seed)
            return {"diff": d[0], "ci95": [d[1], d[2]]}

        p_null = empirical_p(cond["clamp"]["hm"], null)
        pvals.append(p_null)
        results[key] = {"label": plan[key]["label"], "concept": plan[key]["concept"], "level": plan[key]["best_level"],
                        "n_prompts": len({r["prompt_idx"] for r in clamp}), "conditions": cond, "random_null_hm": null,
                        "random_null_concept": [float(np.mean([r["concept"] for r in rs])) for rs in nulls],
                        "random_null_fluency": [float(np.mean([r["fluency"] for r in rs])) for rs in nulls],
                        "p_concept_vs_random_null": empirical_p(cond["clamp"]["concept"], [float(np.mean([r["concept"] for r in rs])) for rs in nulls]),
                        "clamp_minus_unsteered": clamp_minus("unsteered"), "p_vs_random_null": p_null,
                        # POST HOC (added in the 2026-11-01 audit): neuron clamp vs the two baselines, same pairing
                        "clamp_minus_prompting": clamp_minus("prompting"), "clamp_minus_diffmean": clamp_minus("diffmean"),
                        "posthoc_audit": _posthoc_audit(by, targets[key], settings.seed)}
    audits = [r["posthoc_audit"] for r in results.values()]  # Holm across the poles of this confirmation
    _holm_into(audits, "p_word_count_vs_null")
    for field in ("p_hm_vs_null", "p_word_count_vs_null"):
        _holm_into([a["like_for_like"] for a in audits], field)
    cells = [(a, f"p_clamp_vs_{b}") for a in audits for b in ("prompting", "diffmean")]
    for (a, field), p in zip(cells, holm([a[f] for a, f in cells])):  # one family: every clamp-vs-baseline test
        a[field + "_holm"] = p
    for key, p_adj in zip(results, holm(pvals)):
        r = results[key]
        r["p_holm"] = p_adj
        c, u = r["conditions"]["clamp"], r["conditions"]["unsteered"]
        effect = bool(r["clamp_minus_unsteered"]["ci95"][0] > 0 and p_adj < 0.05)
        r["verdict_absolute_fluency"] = bool(effect and c["fluency"] >= 1.0)
        r["verdict_noninferior_fluency"] = bool(effect and c["distinct2"] >= 0.9 * u["distinct2"] and c["fluency"] >= u["fluency"] - 0.2)
        r["h2_supported"] = r["verdict_noninferior_fluency"] if noninferiority else r["verdict_absolute_fluency"]
    save_json({"results": results, "generations": rows}, settings.results_path() / f"{out_name}.json")
    log.info("%s: %s", out_name, {k: (v["clamp_minus_unsteered"], v["p_holm"], v["h2_supported"]) for k, v in results.items()})


def phase_confirm_judge(settings) -> None:
    """Judge the H2 confirmation."""
    _confirm_judge(settings, "confirm_plan", "confirm_generations", "h2", noninferiority=False)


def _restats(settings, plan_name: str, out_name: str, noninferiority: bool) -> None:
    """Recompute confirmation statistics from the already-judged rows in results/<out_name>.json (no LLM calls)."""
    plan = {p["key"]: p for p in load_json(settings.results_path() / f"{plan_name}.json")}
    _confirm_stats(settings, load_json(settings.results_path() / f"{out_name}.json")["generations"], plan, out_name,
                   noninferiority, _target_words(settings, plan))


def phase_h2_stats(settings) -> None:
    """Recompute the H2 statistics without re-judging (e.g. after adding verdicts or post-hoc comparisons)."""
    _restats(settings, "confirm_plan", "h2", noninferiority=False)


def phase_h2b_stats(settings) -> None:
    """Recompute the H2b statistics without re-judging."""
    _restats(settings, "confirm_plan_h2b", "h2b", noninferiority=True)


def phase_confirm_h2b_judge(settings) -> None:
    """Judge the H2b confirmation."""
    _confirm_judge(settings, "confirm_plan_h2b", "confirm_h2b_generations", "h2b", noninferiority=True)
