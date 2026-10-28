"""Pick the label/judge model on mamba-130m smoke poles only (pre-registration amendment 2).

Usage: OUTPUT_DIR=<smoke dir> uv run python scripts/compare_labelers.py qwen3.5:9b qwen3.5:4b
Reads <OUTPUT_DIR>/mamba-130m-hf/examples.json, labels + detection-scores the first 20 candidate poles with each model,
and writes results/labeler_comparison_smoke.json (mean balanced accuracy, number of clean poles, wall time).
"""
import json, sys, time, zlib
from pathlib import Path
import numpy as np
from mamba_neurons.config import load_settings
from mamba_neurons.llm import LocalLLM
from mamba_neurons.autointerp import explain, detect
from mamba_neurons.logging_utils import setup_logging

s = load_settings()
setup_logging("compare_labelers", Path("logs/smoke"))
ex = json.load(open(s.output_dir / "mamba-130m-hf" / "examples.json", encoding="utf-8"))
keys = [k for k in ex if ex[k]["unit"].get("group") == "candidate"][:20]
out = {}
for model in sys.argv[1:]:
    llm = LocalLLM(s.ollama_host, model)
    bas, t0 = [], time.time()
    for k in keys:
        e = explain(llm, ex[k]); d = detect(llm, e, ex[k], seed=zlib.crc32(k.encode()))
        bas.append(d["balanced_accuracy"]); print(model, k, f"BA={d['balanced_accuracy']:.2f}", e, flush=True)
    out[model] = {"mean_ba": float(np.mean(bas)), "clean": int(np.sum(np.array(bas) >= 0.8)), "n": len(bas), "sec": time.time() - t0}
    print("SUMMARY", model, out[model], flush=True)
    llm.unload()
Path("results/labeler_comparison_smoke.json").write_text(json.dumps(out, indent=1))
