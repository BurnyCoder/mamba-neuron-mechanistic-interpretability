---
title: "Turning up single neurons in Mamba: a pre-registered neuron-level study of a 1.4B state-space language model"
author: "Libor Burian"
date: "October 2026"
---

## Abstract
ABSTRACT_PLACEHOLDER

## 1 Introduction
Goodfire's essay *Neural Networks Are Already Clean* ([Cammarata et al., 2026](https://already-clean.goodfire.com/)) argues that most individual neurons of modern networks have one main job. In a preliminary audit of Qwen3-32B, about 90% of neuron poles had a single interpretable role. Turning a neuron up changes what the model writes. Almost all such evidence comes from transformers, whose MLP neurons are the classic object of study ([Geva et al., 2021](https://arxiv.org/abs/2012.14913); [Geva et al., 2022](https://arxiv.org/abs/2203.14680); [Voita et al., 2023](https://arxiv.org/abs/2309.04827); [Gurnee et al., 2023](https://arxiv.org/abs/2305.01610); [Gurnee et al., 2024](https://arxiv.org/abs/2401.12181); [Bills et al., 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)).

Selective state-space models such as Mamba ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)) replace attention with an input-dependent linear recurrence. Their interpretability has been studied at the level of layers, circuits, hidden attention and subspaces:
- factual recall ([Sharma et al., 2024](https://arxiv.org/abs/2404.03646));
- the IOI circuit ([Ensign & Garriga-Alonso, 2024](https://arxiv.org/abs/2407.14008));
- hidden attention ([Ali et al., 2024](https://arxiv.org/abs/2403.01590));
- token-interaction decomposition ([Pitorro & Treviso, 2025](https://arxiv.org/abs/2502.15612));
- SAE feature universality with transformers ([Wang et al., 2025](https://arxiv.org/abs/2410.06672));
- activation-subspace steering ([Sunku Mohan et al., 2026](https://arxiv.org/abs/2602.22719)).

In our literature search we found no neuron-level atlas of a Mamba language model and no controlled single-neuron steering study.

We ask whether Mamba's natural neuron unit is interpretable and causally steerable. That unit is the gated SSM output feeding the output projection, the direct analogue of a SwiGLU neuron. Our contributions:
1. A tested definition of Mamba neurons and an open atlas of all 196,608 neurons of Mamba-1.4B: activation shape, both poles, top firings, write-side vocabulary projections and split stability.
2. A pre-registered privileged-basis test (H1). It compares local-LLM interpretability of random neurons against random directions in the same space.
3. A pre-registered single-neuron steering protocol (H2) with random-neuron, DiffMean, additive and prompting baselines, cluster bootstrap and Holm correction. It is followed by a post-hoc dose exploration and a second pre-registered confirmation (H2b) on fresh prompts.
4. Causal and mechanistic follow-ups (H3): ablation against random neurons, gate-vs-SSM factor decomposition, context truncation (convolution window vs SSM memory) and activation patching.

All code, data, logs (including every LLM prompt and reply), the pre-registration with its dated amendments, and an interactive viewer are public.

## 2 Background
**Mamba blocks.** A block maps the residual stream `h` through RMSNorm and an input projection to two width-4096 branches `x` and `z`. The `x` branch passes a depthwise causal convolution (kernel 4) and SiLU, and then a selective SSM with input-dependent step size Δ and matrices B, C and a diagonal A. There is one 16-dimensional state per channel. The output `y = SSM(x̃) + D ⊙ x̃` is gated elementwise by `SiLU(z)` and written back through `out_proj` ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752); HF [`modeling_mamba.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba/modeling_mamba.py)).

**Neurons and poles.** In a SwiGLU transformer MLP, the neuron `SiLU(W_g x) ⊙ (W_u x)` multiplies two signed factors before `down_proj` ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)). Both signs can carry meaning, which Goodfire calls a neuron's poles. Mamba's gated SSM output has the same structure. A coordinate basis is *privileged* when the computation is not rotation-invariant there ([Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html)). Here that holds because of the elementwise gate and the per-channel recurrence.

**Explaining and scoring neurons.** Automated pipelines show an LLM the top-activating examples, ask for an explanation, and score it by simulation ([Bills et al., 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)) or by detection and fuzzing ([Paulo et al., 2024](https://arxiv.org/abs/2410.13928); [delphi](https://github.com/EleutherAI/delphi)). Transluce scaled this to every neuron of Llama-3.1-8B ([Choi et al., 2024](https://transluce.org/neuron-descriptions)). The vocabulary projection of a neuron's write vector identifies prediction and suppression neurons ([Geva et al., 2022](https://arxiv.org/abs/2203.14680); [Gurnee et al., 2024](https://arxiv.org/abs/2401.12181)).

**Steering.** Activation addition ([Turner et al., 2023](https://arxiv.org/abs/2308.10248)), contrastive activation addition ([Rimsky et al., 2024](https://arxiv.org/abs/2312.06681)) and SAE-feature clamping ([Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)) change model behaviour by editing activations. AxBench ([Wu et al., 2025](https://arxiv.org/abs/2501.17148)) evaluates steering with an LLM judge for concept relevance and fluency. It finds that simple baselines such as DiffMean and prompting often beat SAE-based steering. Single-neuron steering in language models has been shown for TopK LMs ([Takahashi et al., 2025](https://arxiv.org/abs/2506.21468)) and analysed through "control windows" ([Liu, 2026](https://arxiv.org/abs/2606.19831)).

## 3 Method
**Model and neuron site.** We study `state-spaces/mamba-1.4b-hf`: 48 blocks, d_model = 2048, d_inner = 4096, state size 16, trained on 300B Pile tokens. It runs in bf16 with the reference PyTorch scan of HF `transformers` 5.21 on one 8 GB laptop GPU. Neuron (ℓ, i) is coordinate i of the gated SSM output `a` at the input of `out_proj`. We read and edit it with a forward pre-hook. Unit tests verify three properties:
- `out_proj(a)` reproduces the mixer output.
- Adding δ to `a_i` equals adding `δ·W_out[:, i]` to the residual stream.
- Clamping a neuron to its own value is a no-op, in both prefill and cached decoding.

**Data.** `NeelNanda/pile-10k` documents are split by document (seed 0) into discovery (70%) and held-out (30%). They are cut into non-overlapping 128-token windows, at most 4 per document, each prefixed with `<|endoftext|>` (excluded from statistics). Discovery has 18,111 windows (2.32M tokens) and held-out has 5,078 windows (0.65M tokens). Labels, candidate selection and the steering screen use discovery data only. Detection scoring, prompts, DiffMean and ablations use held-out data only.

**Atlas.** A streaming pass per split records, for every neuron:
- float64 moments (mean, std, skew, Pearson kurtosis);
- the fraction of positive activations;
- the top-40 and bottom-40 activations with positions (the two poles).

The same quantities are recorded for 2 isotropic random unit directions per layer. The write side is `u_i = W_U(g ⊙ w_i)`, with `g` the final RMSNorm gain, together with its vocabulary kurtosis.

**Candidates.** Each of the 393,216 poles gets three rank-percentile scores, which are averaged:
- activation kurtosis;
- vocab kurtosis;
- input-output coherence: the mean vocab-z-scored `s·u_i` of the token that follows the pole's top firings.

The top 60 poles are kept, at most 3 per layer. An exploratory second stream (amendment 2) adds, for each of 8 distinctive Pile subsets, the 3 neurons with the largest |d′| of window-mean activation in-domain vs rest.

**Labels and detection (H1).** A local `qwen3.5:9b` (Ollama, temperature 0) writes a ≤15-word explanation per pole. Its input:
- the top-20 discovery firings as 33-token snippets with `<<highlights>>`;
- the most common peak and next tokens;
- the pole's 10 promoted tokens (computed identically for neurons and random directions).

It then classifies 20 shuffled, unhighlighted held-out snippets: the pole's 10 top firings and 10 random ones. The score is balanced accuracy, and a pole is clean if BA ≥ 0.8. H1 compares the clean rate of 192 random-neuron poles (2 neurons per layer, both poles) with 192 random-direction poles. It uses a one-sided Fisher exact test and a unit-cluster bootstrap CI.

**Steering (H2).** Neutral prompts are the first 16 tokens of held-out documents. Interventions apply at every position during sampled generation (40 tokens, temperature 0.7, top-p 0.95):
- *clamp*: `a_i ← s·c·|pole max|`;
- *add*: `a_i ← a_i + s·c·|pole max|`;
- *DiffMean*: the mean residual at the pole's held-out top firings minus the mean at random positions, after block ℓ, scaled to the add condition's residual norm ([AxBench](https://arxiv.org/abs/2501.17148));
- *prompting*: a `<concept>:` prefix;
- *random-neuron null*: the same relative clamp on random neurons of the same layer.

An LLM judge scores concept relevance and fluency (0–2) with the AxBench templates. The primary metric is their per-generation harmonic mean (HM). The concept is an output-side "steering concept" (e.g. "mentions of the word 'all'"), derived by the LLM from the label and promoted tokens before any steered text is judged (amendment 4).

The protocol runs in three stages:
1. **Screen** (prompt set A, 10 prompts). Every candidate pole with a clean label (39 poles) is clamped at c ∈ {2, 5, 10, 20}.
2. **Confirmation.** The ≤ 3 poles with the largest HM gain over unsteered are confirmed on prompt set B: 50 prompts on which the neuron is inactive, × 2 seeds, against an 80-neuron × 10-prompt random null (amendment 5).
3. **Decision rule.** H2 is supported if the paired clamp − unsteered HM difference has a 95% prompt-cluster-bootstrap CI above 0, the Holm-adjusted empirical p against the null is < 0.05, and the clamped judged fluency is ≥ 1.0.

**Mechanism (H3 and exploratory).** For each confirmed pole we use its 200 strongest firings in 2,000 held-out windows.
- **H3 ablation.** Zero-ablate the neuron at all positions and measure the drop in log p(actual next token), against 20 random same-layer neurons at the same positions.
- **Factor split.** Decompose `a = y·SiLU(z)` at the firings.
- **Context truncation.** Re-run each firing with only the last k ∈ {1, 2, 4, 8, 16, 32} tokens, to separate the 4-token convolution window from SSM memory.
- **Activation patching.** Set the neuron to its median firing value at one random held-out position, and measure the change in log p of its 10 promoted tokens against 20 random neurons.

**Pre-registration.** Hypotheses, metrics, tests and thresholds were committed to the public repository before the main-model analyses ([preregistration.md](https://github.com/BurnyCoder/mamba-neuron-mechanistic-interpretability/blob/main/reports/preregistration.md)). Every later change is a dated amendment listing what data existed at the time (Appendix A).

## 4 Results
RESULTS_PLACEHOLDER

## 5 Discussion
DISCUSSION_PLACEHOLDER

## 6 Limitations
LIMITATIONS_PLACEHOLDER

## 7 Conclusion
CONCLUSION_PLACEHOLDER

## References
REFERENCES_PLACEHOLDER

## Appendix A: pre-registration amendments
APPENDIX_PLACEHOLDER

## AI statement
Claude Code (Anthropic's coding agent, model Claude Opus 5.5) wrote this codebase and this paper: the code, the experiments, the analyses and the text. A human, Libor Burian, provided the exact specification that was executed. Human cleanup and rewriting of the paper are in progress. The "manual" judge-validation ratings reported here were produced by Claude Code, not by a human annotator.
