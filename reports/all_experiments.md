# All experiments: summary, key terms, models

Detailed reports:
- [experiment_01_atlas.md](experiment_01_atlas.md): statistics of all neurons.
- [experiment_02_autointerp_h1.md](experiment_02_autointerp_h1.md): labels and the privileged-basis test.
- [experiment_03_steering_h2.md](experiment_03_steering_h2.md): steering screen and confirmation.
- [experiment_04_mechanism_h3.md](experiment_04_mechanism_h3.md): ablation and mechanism.

Pre-registration and all amendments: [preregistration.md](preregistration.md).

## Summary table
EXPERIMENT_TABLE_PLACEHOLDER

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
