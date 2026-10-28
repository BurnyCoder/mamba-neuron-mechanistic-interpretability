# All experiments: summary, key terms, models

Detailed reports:
- [experiment_01_atlas.md](experiment_01_atlas.md): statistics of all neurons.
- [experiment_02_autointerp_h1.md](experiment_02_autointerp_h1.md): labels and the privileged-basis test.
- [experiment_03_steering_h2.md](experiment_03_steering_h2.md): steering screen and confirmation.
- [experiment_04_mechanism_h3.md](experiment_04_mechanism_h3.md): ablation and mechanism.

Pre-registration and all amendments: [preregistration.md](preregistration.md).

## Summary table
| # | Experiment | What was tested | Method | Data | Main result | Verdict |
|---|---|---|---|---|---|---|
| 1 | Atlas | activation shape of all 196,608 neurons vs 96 random directions | streaming moments + top-k poles | 2.32M + 0.65M Pile tokens | median kurtosis 354 vs 6.9; 86% positively skewed; split ρ 0.94–0.999 | descriptive (privileged basis visible in statistics) |
| 2 | H1 audit | clean-pole rate (LLM label, held-out detection BA ≥ 0.8), neurons vs directions | qwen3.5:9b explain + detect | 192 + 192 poles | 15.1% vs 13.5%, p = 0.39; formula candidates 50% clean | **H1 not supported** (judge κ 0.55) |
| 3a | Screen | clamp 2–20× max on 39 clean poles | AxBench judge | prompt set A | concept 0.08 → 0.82, fluency 0.60 → 0.11 | exploratory |
| 3b | H2 | 3 poles at screen doses vs 80-neuron null, DiffMean, prompting | as pre-registered | set B, 100 gens/condition | concept > all 80 random neurons (p = 1/81) for all 3; HM Holm p 0.037/0.074/0.037; fluency < 1.0 | **H2 not supported** (fluency rule) |
| 3c | Low-dose sweep | clamp 0.25–1× max | same judge | set A | 0.25×: concept 0.52, distinct-2 0.83 | post hoc |
| 3d | H2b | 3 low-dose poles, non-inferior fluency | pre-registered after 3c | fresh set C | HM 0.46/0.24/0.35 vs 0.06/0.08/0.00; Holm p 0.037 each; fluency non-inferior | **H2b supported (3/3)** |
| 4a | H3 | zero-ablation drop at top firings vs 20 random neurons | causal ablation | held-out | 0.0018 and 0.012 nats (p = 1/21); "as" −0.003 (p = 0.76) | **2/3 supported** (uncorrected), tiny effects |
| 4b | Mechanism | gate vs SSM factor; context truncation; patching | interventions | held-out | gate always open; k ≤ 4 keeps 0–11% of activation; patching ≈ null | exploratory |
| 4c | Truncation survey | k = 4 ratio across 39 poles by depth | interventions | held-out | layers 0–1: 0.92; ≥ 20: 0.10–0.24 | post hoc |

## Models involved
| Model | Role | Architecture | Size | Training data and objective | Notes |
|---|---|---|---|---|---|
| [state-spaces/mamba-1.4b-hf](https://huggingface.co/state-spaces/mamba-1.4b-hf) | **subject** | Mamba-1: 48 blocks of selective SSM + gated conv, d_model 2048, d_inner 4096, state 16, conv kernel 4, tied embeddings | 1.4B | 300B tokens of the Pile, next-token prediction ([state-spaces/mamba](https://github.com/state-spaces/mamba#pretrained-models)) | bf16 inference, torch scan path; next-token loss 3.05 nats on 8 discovery windows (128-token context) |
| [state-spaces/mamba-130m-hf](https://huggingface.co/state-spaces/mamba-130m-hf) | tests, smoke runs, labeller selection | same family: 24 blocks, d_model 768, d_inner 1536 | 130M | same | used only for tests and model selection, never for reported main results |
| [qwen3.5:9b](https://ollama.com/library/qwen3.5) (Ollama) | labeller, detection scorer, steering judge | decoder-only LLM (Qwen3.5), quantized for llama.cpp | 9B | Qwen pre- and post-training (not by us) | temperature 0, thinking off; chosen over 4B on smoke data |
| [qwen3.5:4b](https://ollama.com/library/qwen3.5) (Ollama) | rejected labeller | same family | 4B | same | mean smoke detection BA 0.61 vs 0.66 for 9B |

No model is trained or fine-tuned in this project. Everything is analysis and inference-time intervention.

## Key terms, in plain language
- **Neuron (ℓ, i).** One of the 4096 numbers that each Mamba block computes right before writing back into the model's running memory (the residual stream). Technically it's coordinate i of `a = SSM output ⊙ SiLU(gate)`, the input of `out_proj`.
- **Pole.** A neuron can be strongly positive or strongly negative, and the two sides may mean different things. Each side is a "pole".
- **Write vector.** The direction a neuron adds to the residual stream (column i of `out_proj`). Projected onto the vocabulary, it gives the tokens the neuron *promotes*.
- **Kurtosis.** How heavy-tailed a distribution is. A Gaussian has 3. A neuron with kurtosis in the hundreds is near zero almost always and occasionally fires very strongly, like a detector.
- **Random direction.** A random combination of all 4096 neurons of a layer. It is the control for "is the neuron basis special?" (the privileged basis).
- **Label / explanation.** The LLM's one-sentence guess of when a pole fires.
- **Detection balanced accuracy (BA).** The LLM sees the label and 20 unhighlighted held-out snippets (10 where the pole fires most, 10 random) and guesses which ones fire. BA is the mean of the hit rate and the correct-rejection rate: 0.5 is chance, 1 is perfect. A pole is **clean** if BA ≥ 0.8.
- **Clamp.** Overwrite a neuron's value with a constant at every position during generation; "turn it up". The dose is given in multiples of the pole's maximum activation on discovery text.
- **Additive steering.** Add a constant to the neuron, which is the same as adding the constant times the write vector to the residual stream.
- **DiffMean.** A steering vector: the mean residual state where the neuron fires minus the mean elsewhere. It is the strongest simple baseline in AxBench.
- **Concept / fluency score.** AxBench judge ratings from 0 to 2: is the concept present in the generated text, and is the text fluent? **HM** is their harmonic mean, which is 0 if either is 0.
- **Steering concept.** The output-side description used for judging, e.g. "mentions of the word 'all'". It is derived from the input-side label before any steered text is seen.
- **Random-neuron null.** The same clamp applied to random neurons of the same layer, to check that the effect is specific to the chosen neuron.
- **Empirical p-value.** (1 + number of null neurons at least as good) / (1 + number of null neurons).
- **Holm correction.** Adjusts p-values when several neurons are tested, so the chance of any false positive stays at 5%.
- **Cluster bootstrap.** Confidence intervals obtained by resampling whole prompts (or units), because generations from the same prompt are correlated.
- **Cohen's κ.** Agreement between two raters beyond chance: 0 is chance, 1 is perfect, and ≥ 0.6 counts as "substantial".
- **Zero-ablation.** Set a neuron to 0 everywhere and measure how much worse the model predicts the next token where the neuron normally fires.
- **Context truncation.** Feed only the last k tokens. Mamba's convolution sees 4 tokens, and anything longer must come through the recurrent SSM state.
