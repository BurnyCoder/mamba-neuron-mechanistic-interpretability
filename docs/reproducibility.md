# Reproducibility

## Environment used for the reported results
| | |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB, compute capability 12.0, driver 591.74 |
| Python | 3.12 (uv-managed, `.python-version`) |
| Key packages | torch 2.14.1+cu130, transformers 5.21.0, datasets, numpy, scipy, statsmodels 0.15.0, ollama (python) 0.6.3 |
| LLM server | Ollama 0.40.12, model `qwen3.5:9b` (labels, detection, judge) |
| Exact versions | `uv.lock` |

## Determinism
- **Data and seeds.** The document split, window sampling, random directions, audit sample, prompt sets and random-neuron nulls are all seeded from `SEED=0`. Generation seeds are `SEED` and `SEED+1`.
- **Activations.** Statistics are deterministic up to GPU floating-point non-associativity (bf16 weights, fp32 accumulation in the scan).
- **LLM calls.** Temperature 0 with a fixed seed per attempt. Llama.cpp results can still differ slightly across hardware and context sizes. For example, the same fluency ratings differed in 1 of 24 cases between `num_ctx` 8192 and 2048. Every prompt and reply is logged in full in `logs/*_llm_calls.jsonl`.

## Re-running
1. Copy `.env.example` to `.env`. Adjust `OUTPUT_DIR` and `HF_HOME` to a drive with about 15 GB free, and `HF_TOKEN_PATH` to your token.
2. Install Ollama and pull the model:
   ```bash
   ollama pull qwen3.5:9b
   ```
3. Run the pipeline in three stages:
   ```bash
   uv run python -m mamba_neurons.pipeline --phases benchmark,collect,vocab,select,domains,examples,label,h1,screen,screen_judge
   ```
   Then append the generated `results/<model>/confirm_plan.json` to `reports/preregistration.md` as an amendment, and run:
   ```bash
   uv run python -m mamba_neurons.pipeline --phases confirm,confirm_judge,mechanism,export,plots
   ```
4. Optionally publish to the Hugging Face Hub (needs `hf auth login`):
   ```bash
   uv run python -m mamba_neurons.pipeline --phases hf
   ```

Approximate wall time on the test machine:

| Phase | Time |
|---|---|
| Collection | 50 + 11 min |
| Domains | 8 min |
| Examples | 45 min |
| Labels | 30 min |
| Screen | 16 min generation + about 50 min judging |
| Confirmation | about 3 h (mostly judging) |

## Smoke test
Use tiny budgets and the 130M model, and keep the outputs separate:
```bash
OUTPUT_DIR=D:/AI/mamba-neurons/smoke RESULTS_DIR=D:/AI/mamba-neurons/smoke_results MAX_DISCOVERY_TOKENS=60000 MAX_HELDOUT_TOKENS=30000 uv run python -m mamba_neurons.pipeline --model state-spaces/mamba-130m-hf --phases collect,vocab,select,domains,examples
```
