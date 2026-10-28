# Methodology

This page holds the technical details. The README has the overview, and `reports/preregistration.md` has the hypotheses, tests and thresholds. All code references are to `src/mamba_neurons/`.

## 1. Model and the neuron site
`state-spaces/mamba-1.4b-hf` is the HF conversion of the original Mamba-1 1.4B checkpoint ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)), trained on 300B Pile tokens ([state-spaces/mamba README](https://github.com/state-spaces/mamba#pretrained-models)). It has 48 blocks, with d_model = 2048, d_inner = 4096, state size N = 16 and conv kernel 4. The GPT-NeoX-20B tokenizer has 50,280 tokens, and the unembedding is tied to the embedding.

The mixer of block ℓ is computed in HF `MambaMixer.forward`, using the torch path because the `mamba_ssm` kernels do not build on Windows:

```
[x ; z]  = W_in · RMSNorm(h)                     x, z ∈ R^4096
x̃        = SiLU(conv1d_k=4(x))                   depthwise causal conv
Δ, B, C  = from x̃ (input-dependent; "selective")
s_t      = exp(Δ_t A) ⊙ s_{t-1} + Δ_t B_t x̃_t    per-channel state s ∈ R^{4096×16}
y_t      = C_t · s_t + D ⊙ x̃_t
a_t      = y_t ⊙ SiLU(z_t)                       ← neurons
h_t     += W_out a_t                             W_out ∈ R^{2048×4096}, no bias
```

**Neuron (ℓ, i) is `a_i`**, the i-th input coordinate of `mixer.out_proj`. It is read and edited through `register_forward_pre_hook` on `out_proj` (`model.py::Hooks`).

**Why this site:**
1. Each `a_i` has its own write vector `w_i = W_out[:, i]`, so it is a unit of computation in the same sense as a transformer MLP neuron.
2. The elementwise gate `SiLU(z_i)` and the per-channel SSM make the coordinate basis *privileged*: the computation is not rotation-invariant there ([Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html)).
3. It matches the SwiGLU neuron `SiLU(W_g x) ⊙ (W_u x)` that Goodfire studies, including two signed factors. Hence two poles and a "double-negative" regime: `y < 0` and `z < 0` give `a > 0` with a small magnitude, because `min SiLU ≈ −0.28`.

Unit tests (`tests/test_model.py`) check four things:
- `out_proj(a)` equals the mixer output.
- Adding δ to `a_i` gives the same logits as adding `δ·w_i` to the mixer output.
- Clamping a neuron to its own value is a no-op.
- Hooks fire in both prefill and cached decoding.

## 2. Data
- **Corpus.** `NeelNanda/pile-10k`: 10,000 Pile documents, in-distribution for the model.
- **Split.** Documents are shuffled with seed 0, and 30% are held out *by document*.
- **Windows.** Each document is tokenized and cut into non-overlapping 128-token windows, at most 4 per document. Shorter tails are dropped. Each window is prefixed with the `<|endoftext|>` separator (id 0), and statistics exclude that position.
- **Size.** The discovery split has 18,111 windows (2.32M tokens) from 6,193 documents. The held-out split is capped at 650k tokens: 5,078 windows from 1,758 documents. The cap is for compute: the reference scan runs at about 960 tok/s on the test GPU.

## 3. Activation statistics (`collect.py`)
- **Pass.** One streaming pass per split records every layer.
- **Per neuron.** Running float64 sums of `a, a², a³, a⁴` give mean, std, skew and Pearson kurtosis. The accumulator also keeps the fraction `a > 0`, the top-40 activations (positive pole) and the bottom-40 (negative pole), each with its token position. These are merged with `torch.topk` per batch, and the merge is tested against brute force.
- **Random-direction controls.** Two isotropic random unit vectors `q` per layer (seed 0), applied as `a·q`, go through the same accumulator.

## 4. What a neuron writes (`vocab.py`)
- **Direct logit effect.** `u_i = W_U (g ⊙ w_i)`, where `g` is the final RMSNorm gain. This ignores later blocks and the positive RMS scale. The negative pole writes `−u_i`.
- **Vocab kurtosis.** The kurtosis of `u_i` over the 50,280 tokens identifies prediction/suppression neurons ([Gurnee et al. 2024](https://arxiv.org/abs/2401.12181); [Geva et al. 2022](https://arxiv.org/abs/2203.14680)).

## 5. Candidate selection (`select.py`, pre-registered)
Each of the 393,216 poles gets three scores:
1. **Activation kurtosis** of the neuron.
2. **Vocab kurtosis** of `u_i`.
3. **Input-output coherence.** The mean over the pole's top-40 firing positions of the vocab-z-scored `s·u_i[next token]`.

The score is the mean of the three rank-percentiles. The top 60 poles go forward, with at most 3 per layer.

## 6. Labels and detection scores (`autointerp.py`, `llm.py`)
- **Local LLM.** `qwen3.5:4b` through Ollama, at temperature 0, with thinking off and JSON-schema structured output. Every prompt and reply is logged in full.
- **Explanation.** Shows the pole's top-20 discovery firings, one per window, as 33-token snippets (24 before the peak, 8 after). Tokens with signed activation ≥ 0.4 × peak are marked `<<like this>>`. The prompt is adapted from [delphi's explainer](https://github.com/EleutherAI/delphi).
- **Detection.** Uses 10 held-out top-firing snippets and 10 random held-out snippets, unhighlighted, shuffled with a per-pole seed. The LLM answers 0/1 for each, and the score is balanced accuracy ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)). A malformed reply scores 0.5 and is counted.
- **H1 sample.** 2 random neurons per layer (96 neurons, 192 poles) against 2 random directions per layer (96 directions, 192 poles).

## 7. Steering (`steer.py`, `judge.py`)
- **Interventions.** These act at every position, prompt included, during sampled generation: 40 new tokens, temperature 0.7, top-p 0.95, seeded.
  - **clamp:** `a_i ← s·c·|pole extreme|`.
  - **add:** `a_i ← a_i + s·c·|extreme|`, which is equivalent to adding `δ·w_i` to the residual.
  - **DiffMean:** mean residual at the pole's held-out top firings minus mean at random positions, after block ℓ, scaled to the add condition's residual norm `|δ|·‖w_i‖` ([AxBench](https://arxiv.org/abs/2501.17148)).
  - **prompting:** a `<label>:\n` prefix.
  - **random neurons:** the same relative clamp on 20 other neurons of the same layer.
- **Prompts.** The first 16 tokens of held-out documents. Set A (10 prompts) is used for the exploratory screen. Set B (50 prompts) is used for the confirmation, keeping only prompts where the pole stays below its 90th discovery percentile.
- **Judge.** The AxBench concept and fluency templates (0–2), answered by the local LLM as JSON. The primary metric is the per-generation harmonic mean.
- **Other metrics.** Continuation NLL under the unsteered model, and distinct-2.

## 8. Mechanism (`mechanism.py`)
- **H3 ablation.** Zero-ablate the neuron at all positions and measure the drop in log p of the actual next token at its 200 strongest held-out firings, against 20 random same-layer neurons at the same positions.
- **Factor split.** At the firings, compare `z_i` (from `in_proj`), `SiLU(z_i)` and `y_i = a_i / SiLU(z_i)`.
- **Context truncation.** Re-run each firing with only the last k ∈ {1, 2, 4, 8, 16, 32} tokens plus the separator, and report the activation as a fraction of its full-context value. k ≤ 4 is the conv receptive field, so anything beyond needs the SSM state.
- **Activation patching.** Set the neuron to its median top-firing value at one random held-out position. Measure the change in mean log p of its top-10 promoted tokens, against 20 random neurons patched with their own typical firing value.

## 9. Statistics (`metrics.py`)
- **Intervals.** Wilson intervals for proportions. Percentile bootstrap (10,000 resamples) for differences. Resampling is by unit for H1, because both poles of a unit form a cluster, and by prompt for H2.
- **Tests.** One-sided Fisher exact test. Empirical p-values `(1 + #null ≥ obs)/(1 + n)` against random-unit nulls. Holm correction across the confirmatory poles.
- **Other.** Power analysis uses Cohen's h (statsmodels). Judge agreement uses Cohen's κ.
