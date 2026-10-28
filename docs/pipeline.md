# Pipeline and data flow

`uv run python -m mamba_neurons.pipeline --phases <list>` runs the phases below in order. Each phase reads its inputs from disk, so it can be re-run on its own.

**Where files go:**
- Large artefacts: `OUTPUT_DIR/<model>/` (default `D:/AI/mamba-neurons/outputs/mamba-1.4b-hf/`).
- Small results: `results/<model>/`, committed to git.

**What uses the GPU:** "GPU" means the Mamba model is loaded. "LLM" means Ollama is used. They are never loaded at the same time.

| # | Phase | Uses | Reads | Writes | What happens |
|---|---|---|---|---|---|
| 1 | `benchmark` | GPU | corpus | `results/benchmark.json` | Throughput, peak memory and next-token loss of the torch scan on one batch, as a sanity check. |
| 2 | `collect` | GPU | `NeelNanda/pile-10k` | `windows_*.npz`, `stats_{discovery,heldout}.pt` | Splits documents 70/30 and cuts 128-token windows. One pass per split records all 48 layers, keeping per-neuron moments, top-40 / bottom-40 firings, and the same for 96 random directions. |
| 3 | `vocab` | GPU | weights | `vocab.pt`, `results/vocab_summary.json` | Computes `W_U(g ⊙ w_i)` for all 196,608 neurons: promoted and suppressed tokens, and vocab kurtosis. |
| 4 | `select` | GPU | stats, vocab | `coherence.pt`, `results/candidates.json`, `results/split_stability.json` | Pre-registered ranking that keeps 60 candidate poles (at most 3 per layer). Also the Spearman stability of statistics across splits. |
| 5 | `examples` | GPU | stats, windows | `examples.json` | For every audit and candidate pole, recomputes per-token activations for its top-20 discovery firings, 10 held-out top firings and 10 random held-out snippets, plus its 10 promoted tokens. |
| 6 | `label` | LLM | `examples.json` | `results/autointerp.json`, `logs/*_llm_calls.jsonl` | Explanation and held-out detection scoring per pole. It can resume, and every prompt and output is logged in full. |
| 7 | `h1` | – | autointerp | `results/h1.json` | Pre-registered H1 test: neurons against random directions. |
| 8 | `screen` | GPU | labels, candidates | `screen_generations.json` | Clamp sweep (2, 5, 10, 20 × max) on 10 neutral prompts for well-labelled candidates. |
| 9 | `screen_judge` | LLM | screen generations | `results/screen.json`, `results/confirm_plan.json` | AxBench concept and fluency judging. Picks ≤ 3 poles and their clamp levels. |
| – | *(manual)* | – | confirm plan | `reports/preregistration.md` | Append the plan as an amendment, then commit and push **before** step 10. |
| 10 | `confirm` | GPU | plan | `confirm_generations.json` | 50 neutral prompts × 2 seeds for each condition, plus 20 random-neuron nulls. |
| 11 | `confirm_judge` | LLM | generations | `results/h2.json` | Judging and the pre-registered H2 statistics. |
| 12 | `mechanism` | GPU | plan | `results/h3_mechanism.json` | H3 ablation, factor split, context truncation, patching. |
| 13 | `export` | – | all results | `site/data/` | Viewer data. |
| 14 | `plots` | – | all results | `results/<model>/figures/*.png` | Figures for the README and paper. |

Logs: each invocation writes `logs/<timestamp>_<phases>.log`, with the same lines streamed to the terminal. Every LLM call goes in full to `logs/<timestamp>_<phases>_llm_calls.jsonl`.
