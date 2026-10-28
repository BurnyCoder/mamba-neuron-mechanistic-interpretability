"""Validate the local-LLM detection judge against independent manual ratings (pre-registration, H1 "Judge validity").

Usage:
  uv run python scripts/judge_validation.py sample   # writes results/<model>/judge_validation_items.json (blind: no answers)
  (rate every item 0/1 in results/<model>/judge_validation_ratings.json as {"<id>": 0 or 1})
  uv run python scripts/judge_validation.py score    # Cohen's kappa (manual vs LLM) and both raters' accuracy vs truth
  uv run python scripts/judge_validation.py steer_sample / steer_score   # same for the steering judge (concept, fluency 0-2)
Local explanation: 40 (explanation, snippet) detection decisions are sampled with a fixed seed from autointerp.json;
the rater sees only the explanation and the plain snippet. Ground truth = whether the snippet was a top firing.
Global context: the judge is accepted if kappa >= 0.6 ("substantial", Landis & Koch 1977).
"""

import json
import sys

import numpy as np

from mamba_neurons.autointerp import plain
from mamba_neurons.config import load_settings
from mamba_neurons.jsonio import load_json, save_json
from mamba_neurons.metrics import cohen_kappa

N_ITEMS, SEED = 40, 0


def sample(res, run) -> None:
    auto = load_json(res / "autointerp.json")
    examples = load_json(run / "examples.json")
    rng = np.random.default_rng(SEED)
    keys = rng.choice(sorted(auto), N_ITEMS, replace=False)
    items, hidden = [], {}
    for n, key in enumerate(keys):
        r = auto[key]
        pool = [(ex, 1) for ex in examples[key]["detect_top"]] + [(ex, 0) for ex in examples[key]["detect_random"]]
        order = np.random.default_rng(__import__("zlib").crc32(key.encode())).permutation(len(pool))  # same shuffle as detect()
        j = int(rng.integers(len(pool)))
        ex, truth = pool[order[j]]
        items.append({"id": n, "pole": key, "explanation": r["explanation"], "snippet": plain(ex)})
        assert r["truth"][j] == truth  # same shuffle as detect() -> item j is the j-th scored snippet
        hidden[str(n)] = {"truth": truth, "llm": r["answers"][j] if r["valid"] else None}
    save_json(items, res / "judge_validation_items.json")
    save_json(hidden, res / "judge_validation_hidden.json")


def score(res) -> None:
    hidden = load_json(res / "judge_validation_hidden.json")
    mine = load_json(res / "judge_validation_ratings.json")
    ids = [i for i in hidden if hidden[i]["llm"] is not None]
    manual, llm, truth = ([int(mine[i]) for i in ids], [int(hidden[i]["llm"]) for i in ids], [hidden[i]["truth"] for i in ids])
    out = {"n": len(ids), "kappa_manual_vs_llm": cohen_kappa(manual, llm),
           "accuracy_manual_vs_truth": float(np.mean(np.array(manual) == np.array(truth))),
           "accuracy_llm_vs_truth": float(np.mean(np.array(llm) == np.array(truth))),
           "agreement_manual_llm": float(np.mean(np.array(manual) == np.array(llm)))}
    out["judge_accepted"] = out["kappa_manual_vs_llm"] >= 0.6
    save_json(out, res / "judge_validation.json")
    print(out)


def steer_sample(res) -> None:
    """40 screen generations (seeded, stratified over clamp levels) shown blind: concept description + text only."""
    gens = load_json(res / "screen.json")["generations"]
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(gens), N_ITEMS, replace=False)
    labels = {p["key"]: p["label"] for p in load_json(res / "screen.json")["summary"]}
    items = [{"id": n, "concept": labels[gens[i]["key"]], "text": gens[i]["text"]} for n, i in enumerate(picks)]
    hidden = {str(n): {"concept": gens[i]["concept"], "fluency": gens[i]["fluency"], "level": gens[i]["level"]} for n, i in enumerate(picks)}
    save_json(items, res / "steer_validation_items.json")
    save_json(hidden, res / "steer_validation_hidden.json")


def steer_score(res) -> None:
    """Cohen's kappa between blind manual and LLM 0-2 ratings, per metric, plus exact agreement."""
    hidden = load_json(res / "steer_validation_hidden.json")
    mine = load_json(res / "steer_validation_ratings.json")  # {"<id>": {"concept": 0-2, "fluency": 0-2}}
    out = {}
    for m in ("concept", "fluency"):
        a, b = [int(mine[i][m]) for i in hidden], [int(hidden[i][m]) for i in hidden]
        out[m] = {"kappa": cohen_kappa(a, b), "exact_agreement": float(np.mean(np.array(a) == np.array(b))),
                  "manual_mean": float(np.mean(a)), "llm_mean": float(np.mean(b))}
    save_json(out, res / "steer_validation.json")
    print(out)


if __name__ == "__main__":
    s = load_settings()
    {"sample": lambda: sample(s.results_path(), s.run_dir()), "score": lambda: score(s.results_path()),
     "steer_sample": lambda: steer_sample(s.results_path()), "steer_score": lambda: steer_score(s.results_path())}[sys.argv[1]]()
