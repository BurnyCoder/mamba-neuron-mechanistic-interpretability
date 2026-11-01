---
title: "Turning up single neurons in Mamba: a pre-registered neuron-level study of a 1.4B state-space language model"
author: "Libor “Burny” Burian"
date: "October 2026"
---

## Abstract
We study the natural neuron unit of the Mamba-1.4B state-space language model: the 196,608 coordinates of the gated SSM output that feed each block's output projection, the analogue of a SwiGLU MLP neuron. All analyses were pre-registered, with dated amendments.

- **Activation shape.** Almost all Mamba neurons are sparse, heavy-tailed detectors. Their median activation kurtosis is 354 (90% are above 46), against 6.9 for random directions in the same space.

- **Interpretability audit (H1, negative).** A local 9B LLM's labels predict held-out firings no better for random neurons than for random directions: 15.1% vs 13.5% clean poles, p = 0.39. Agreement with blind reference ratings was only moderate (κ = 0.55), so this null result is limited by the instrument.

- **Selected neurons are interpretable.** A pre-registered selection formula finds neurons whose labels transfer to held-out text (50% clean), mostly next-token-prediction neurons.

- **Steering at the pre-registered doses (H2, negative).** Clamping a single neuron at ≥ 2× its maximum injects its concept more strongly than every one of 80 random same-layer neurons (p = 1/81) and more than prompting. It also degrades text, so H2's pre-registered fluency criterion fails.

- **Steering at gentle doses (H2b, positive).** A post-hoc dose sweep revealed a gentle regime (0.25–0.5× max). In a second pre-registered confirmation on fresh prompts (H2b), three neurons, steering toward "from", "as" and "first", raised a concept–fluency harmonic mean from 0.00–0.08 to 0.24–0.46. They beat the random-neuron null (Holm p = 0.037 each) with non-inferior fluency. They also had a higher mean HM than prompting and DiffMean; in a post-hoc paired test, 5 of these 6 differences are significant.

- **Necessity is weak.** Zero-ablation costs at most 0.012 nats where these neurons fire.

- **Mechanism.** For the three H2 neurons, the SiLU gate was open at all 600 examined firings, so the SSM factor decides the pole. In a post-hoc survey of 39 poles, those from layer 20 on keep only 10–24% (median per layer bin) of their activation when they see just the last 4 tokens, against 92% for the 4 poles in layers 0–1. Deep neurons therefore need longer context. This test does not tell whether that context arrives through the SSM state or through convolutions stacked across layers.

Code, data, all LLM prompts and replies, and an interactive viewer are public.

## 1 Introduction
Goodfire's essay *Neural Networks Are Already Clean* ([Cammarata, 2026](https://already-clean.goodfire.com/)) argues that most individual neurons of modern networks have one main job. From preliminary audits of Qwen3-32B, it estimates that roughly 90% of neuron poles have one main job. Turning a neuron up changes what the model writes. Almost all such evidence comes from transformers, whose MLP neurons are the classic object of study ([Geva et al., 2021](https://arxiv.org/abs/2012.14913); [Geva et al., 2022](https://arxiv.org/abs/2203.14680); [Voita et al., 2024](https://arxiv.org/abs/2309.04827); [Gurnee et al., 2023](https://arxiv.org/abs/2305.01610); [Gurnee et al., 2024](https://arxiv.org/abs/2401.12181); [Bills et al., 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)).

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

4. Causal and mechanistic follow-ups (H3): ablation against random neurons, gate-vs-SSM factor decomposition, context truncation (how much context a firing needs) and activation patching.

All code, data, logs (including every LLM prompt and reply), the pre-registration with its dated amendments, and an interactive viewer are public.

## 2 Background
**Mamba blocks.** A block maps the residual stream `h` through RMSNorm and an input projection to two width-4096 branches `x` and `z`. The `x` branch passes a depthwise causal convolution (kernel 4) and SiLU, and then a selective SSM with input-dependent step size Δ and matrices B, C and a diagonal A. There is one 16-dimensional state per channel. The output `y = SSM(x̃) + D ⊙ x̃` is gated elementwise by `SiLU(z)` and written back through `out_proj` ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752); HF [`modeling_mamba.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba/modeling_mamba.py)).

**Neurons and poles.** In a SwiGLU transformer MLP, the neuron `SiLU(W_g x) ⊙ (W_u x)` multiplies two signed factors before `down_proj` ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)). Both signs can carry meaning, which Goodfire calls a neuron's poles. Mamba's gated SSM output has the same structure. A coordinate basis is *privileged* when the computation is not rotation-invariant there ([Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html)). Here that holds because of the elementwise gate and the per-channel recurrence.

**Explaining and scoring neurons.** Automated pipelines show an LLM the top-activating examples, ask for an explanation, and score it by simulation ([Bills et al., 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)) or by detection and fuzzing ([Paulo et al., 2024](https://arxiv.org/abs/2410.13928); [delphi](https://github.com/EleutherAI/delphi)). Transluce scaled this to every MLP neuron of Llama-3.1-8B-Instruct ([Choi et al., 2024](https://transluce.org/neuron-descriptions)). The vocabulary projection of a neuron's write vector identifies prediction and suppression neurons ([Geva et al., 2022](https://arxiv.org/abs/2203.14680); [Gurnee et al., 2024](https://arxiv.org/abs/2401.12181)).

**Steering.** Activation addition ([Turner et al., 2023](https://arxiv.org/abs/2308.10248)), contrastive activation addition ([Panickssery (formerly Rimsky) et al., 2024](https://arxiv.org/abs/2312.06681)) and SAE-feature clamping ([Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)) change model behaviour by editing activations. AxBench ([Wu et al., 2025](https://arxiv.org/abs/2501.17148)) evaluates steering with an LLM judge that rates concept relevance, instruction relevance and fluency, and takes their harmonic mean. Its findings:

- prompting steers best, followed by finetuning;

- representation-based methods such as DiffMean detect concepts best;

- SAEs are not competitive on either task.

Single-neuron steering in language models has been shown for TopK LMs ([Takahashi et al., 2025](https://arxiv.org/abs/2506.21468)) and analysed through "control windows" ([Liu, 2026](https://arxiv.org/abs/2606.19831)).

## 3 Method
**Model and neuron site.** We study `state-spaces/mamba-1.4b-hf`: 48 blocks, d_model = 2048, d_inner = 4096, state size 16, trained on 300B Pile tokens. It runs in bf16 with the reference PyTorch scan of HF `transformers` 5.21 on one 8 GB laptop GPU. Neuron (ℓ, i) is coordinate i of the gated SSM output `a` at the input of `out_proj`. We read and edit it with a forward pre-hook. Unit tests verify three properties:

- `out_proj(a)` reproduces the mixer output.

- Adding δ to `a_i` equals adding `δ·W_out[:, i]` to the residual stream.

- Clamping a neuron to its own value is a no-op, and hooks fire at every step of cached decoding without changing the generated tokens.

**Data.** `NeelNanda/pile-10k` documents are split by document (seed 0) into discovery (70%) and held-out (30%). They are cut into non-overlapping 128-token windows, at most 4 per document, each prefixed with `<|endoftext|>` (excluded from statistics). Discovery has 18,111 windows (2.32M tokens) and held-out has 5,078 windows (0.65M tokens). Labels and candidate selection use discovery data only. Held-out data is used for detection scoring, all steering prompts (disjoint sets A, B and C of held-out document openings), DiffMean and ablations. The pre-registration had listed the steering screen under discovery data, but its prompts were held-out openings, like all neutral prompts (Appendix A, amendment 8).

**Atlas.** A streaming pass per split records, for every neuron:

- float64 moments (mean, std, skew, Pearson kurtosis);

- the fraction of positive activations;

- the top-40 and bottom-40 activations with positions (the two poles).

The same quantities are recorded for 2 isotropic random unit directions per layer. The write side is `u_i = W_U(g ⊙ w_i)`, with `g` the final RMSNorm gain, together with its vocabulary kurtosis.

**Candidates.** Each of the 393,216 poles gets three rank-percentile scores, which are averaged:

- activation kurtosis;

- vocab kurtosis;

- input-output coherence: the mean vocab-z-scored `s·u_i` of the token that follows the pole's 40 stored top firings (the pre-registration text said 20; amendment 8).

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

An LLM judge scores concept relevance and fluency (0–2) with the AxBench templates. The primary metric is their per-generation harmonic mean (HM). AxBench also scores instruction relevance; we drop it because our prompts are text to continue, not instructions. The concept is an output-side "steering concept" (e.g. "mentions of the word 'all'"), derived by the LLM from the label and promoted tokens before any steered text is judged (amendment 4).

The protocol runs in three stages:

1. **Screen** (prompt set A, 10 prompts). Every candidate pole with a clean label (39 poles) is clamped at c ∈ {2, 5, 10, 20}.

2. **Confirmation.** The ≤ 3 poles with the largest HM gain over unsteered are confirmed on prompt set B: 50 prompts on which the neuron stays below its 90th percentile (estimated on the first 200 discovery windows), × 2 seeds, against an 80-neuron × 10-prompt random null (amendment 5).

3. **Decision rule.** H2 is supported if the paired clamp − unsteered HM difference has a 95% prompt-cluster-bootstrap CI above 0, the Holm-adjusted empirical p against the null is < 0.05, and the clamped judged fluency is ≥ 1.0.

**Mechanism (H3 and exploratory).** For each confirmed pole we use its 200 strongest firings in 2,000 held-out windows. The ablation uses the 196–199 of them that have a next token inside the window.

- **H3 ablation.** Zero-ablate the neuron at all positions and measure the drop in log p(actual next token), against 20 random same-layer neurons at the same positions.

- **Factor split.** Decompose `a = y·SiLU(z)` at the firings.

- **Context truncation.** Re-run each firing with only the last k ∈ {1, 2, 4, 8, 16, 32} tokens, to measure how much context it needs. k = 4 is one block's convolution kernel, but convolutions stacked over layers reach back 3ℓ + 4 tokens at layer ℓ. This test therefore does not separate the convolution path from SSM memory.

- **Activation patching.** Set the neuron to its median firing value at one random held-out position, and measure the change in log p of its 10 promoted tokens against 20 random neurons. Each null neuron is set to the median of its 40 strongest discovery activations, a more extreme value than the target's donor, so this comparison is not like-for-like.

**Pre-registration.** Hypotheses, metrics, tests and thresholds were committed to the public repository before the main-model analyses ([preregistration.md](https://github.com/BurnyCoder/mamba-neuron-mechanistic-interpretability/blob/main/reports/preregistration.md)). Every later change is a dated amendment listing what data existed at the time (Appendix A).

## 4 Results
### 4.1 Atlas: Mamba neurons are sparse, heavy-tailed detectors
Over 2.32M discovery tokens, the median neuron has kurtosis **354** (10th–90th percentile 46–3,064) and skew +7.4. 85.6% of neurons are positively skewed. For the 96 random directions the median kurtosis is **6.9** (Gaussian = 3) and the skew −0.03 (Table 1). Statistics replicate on the disjoint held-out documents, with Spearman ρ between splits of 0.96 (max), 0.94 (kurtosis) and 0.998 (std). Activation scale grows with depth: the median per-neuron maximum is 0.5 at layer 0 and plateaus at 24–26 in layers 40–45 (22.1 at layer 47).

By contrast, the write-side vocabulary projections are close to Gaussian. The median vocab kurtosis is 3.23–3.55 per layer, highest in layers 31–40. Prediction-like neurons are therefore a heavy tail, not the norm. The pre-registered candidate formula selected 60 late-layer poles (layers 27–47 apart from three). Almost all are next-token-prediction neurons for one word family, such as `all`, `way`, `nor`, `time`, `from`, `as`, `per` and `respectively`. Examples:

- L37 N1812 fires on the token before "all" ("after all", "at all", "of all") and promotes `all`.

- L38 N918 fires on "make(s)" before "sense".

: **Table 1.** Activation shape of neurons and random directions (discovery split).

| | Neurons (196,608) | Random directions (96) |
|---|---|---|
| Kurtosis, median [p10, p90] | 354 [46, 3,064] | 6.9 [5.1, 19.7] |
| Skew, median [p10, p90] | +7.4 [−1.5, +31.4] | −0.03 [−0.44, +0.45] |
| Positively skewed | 85.6% | 45.8% |

### 4.2 H1: a local-LLM audit does not separate neurons from random directions
192 random-neuron poles and 192 random-direction poles were labelled and scored on held-out text (Figure 1). Clean rates (detection balanced accuracy ≥ 0.8) were **15.1%** (Wilson 95% CI 10.7–20.9%) for neurons and **13.5%** (9.4–19.1%) for directions. The difference is +1.6 points (unit-bootstrap 95% CI −5.7 to +8.3), with one-sided Fisher p = 0.39. **H1 is not supported.**

Mean balanced accuracy was 0.617 (0.597–0.637) for neurons against 0.583 (0.560–0.607) for directions; this comparison is post hoc. By depth, neurons stay flat at 0.63, 0.60 and 0.62 for layers 0–15, 16–31 and 32–47, while directions fall from 0.63 to 0.56. Formula candidates, which are not part of H1, reached a mean BA of 0.768 with 30 of 60 clean.

The detection judge agreed only moderately with blind reference ratings made by Claude Code: κ = 0.55 on 40 decisions, below the pre-registered acceptance bar of 0.6. The audit's measurement floor is therefore high. With a 9B local labeller only about 15% of any units count as clean. That is far from the roughly 90% that Goodfire estimates for Qwen3-32B, from preliminary audits whose procedure the essay does not detail.

![**Figure 1.** H1. Left: clean-pole rates with Wilson 95% CIs. Right: distribution of held-out detection balanced accuracy.](../results/mamba-1.4b-hf/figures/h1_clean_rate.png)

### 4.3 Steering screen: concepts in, fluency out
Clamping each of 39 clean candidate poles at 2× its maximum (10 prompts) raised the mean judged concept score from 0.08 to **0.82**, and the share of generations with the concept present (score ≥ 1) from 6% to 48%. Judged fluency fell from 0.60 to 0.11, and distinct-2 from 0.91 to 0.39. Higher doses only made it worse (Table 2). The typical failure is repetition of the promoted token ("all all all …", "@@@@").

The steering judge was validated on 40 blind items, rated by Claude Code. The concept judge is reliable (κ = 0.79, 87.5% exact agreement). The fluency judge is not (κ = 0.34): it is systematically harsh, with a mean of 0.25 against 0.65 for manual ratings.

: **Table 2.** Screen and post-hoc low-dose sweep, averaged over 39 poles (10 prompts each, prompt set A).

| Dose (× pole max) | Concept | P(concept ≥ 1) | Fluency (judge) | distinct-2 | HM |
|---|---|---|---|---|---|
| unsteered | 0.08 | 0.06 | 0.60 | 0.91 | 0.050 |
| 0.25 (post hoc) | 0.52 | 0.32 | 0.42 | 0.83 | **0.092** |
| 0.5 (post hoc) | 0.75 | 0.45 | 0.31 | 0.68 | 0.080 |
| 1 (post hoc) | 0.80 | 0.48 | 0.19 | 0.51 | 0.054 |
| 2 | 0.82 | 0.48 | 0.11 | 0.39 | 0.022 |
| 5 | 0.67 | 0.40 | 0.05 | 0.27 | 0.008 |
| 10 | 0.76 | 0.45 | 0.02 | 0.21 | 0.000 |
| 20 | 0.70 | 0.41 | 0.01 | 0.17 | 0.000 |

### 4.4 H2: pre-registered confirmation
The three pre-registered poles were each confirmed on 50 fresh neutral prompts (47 for L32 N2669) × 2 seeds, against an 80-neuron random null (Table 4, Figure 2). Clamping raised the HM above unsteered for all three. The paired 95% CIs exclude 0: +0.29 for "as", +0.09 for "[", +0.27 for "set".

Against the random-neuron null:

- **Concept alone.** Every target beat all 80 null neurons on the concept score (p = 1/81).

- **Primary HM metric.** L30 N2121 and L32 N2669 beat the null after Holm correction (p = 0.037); L28 N953 did not (p = 0.074).

- **Fluency.** No pole reached the pre-registered absolute judged fluency of ≥ 1.0. Unsteered text scored only 0.41–0.46 with this judge.

**H2 is therefore not supported as pre-registered.** Under the non-inferiority fluency rule later registered for H2b (distinct-2 ≥ 0.9 × unsteered and judged fluency ≥ unsteered − 0.2), L30 N2121 would pass. That reading is post hoc for H2.

The baselines behaved differently:

- **Prompting** with the concept as a prefix produced lower HM than neuron clamping for all three poles (0.15, 0.02, 0.07). A post-hoc paired bootstrap gives clamp − prompting of +0.23 [0.09, 0.39] for "as" and +0.20 [0.08, 0.33] for "set", but only +0.07 [−0.01, 0.16] for "[".

- **DiffMean** at the same residual-norm budget produced maximal concept but degenerate text (distinct-2 0.03–0.17, HM 0), whereas the neuron's write direction kept distinct-2 at 0.75–0.85.

- **Additive steering** gave the same concept scores as clamping, because these neurons are near zero outside their firings. Its HM was nearly equal for "as" and "[" but lower for "set" (0.18 vs 0.27).

: **Table 4.** H2 confirmation (prompt set B). HM = harmonic mean of the judge's concept and fluency scores (0–2).

| Pole · dose | Unsteered HM | Clamp HM [95% CI] | Prompting HM | DiffMean HM | Null HM mean / max | Holm p | Clamp concept | Clamp distinct-2 |
|---|---|---|---|---|---|---|---|---|
| L30 N2121 "as" · 2× | 0.09 | **0.38** [0.26, 0.52] | 0.15 | 0.00 | 0.05 / 0.27 | **0.037** | 1.68 | 0.85 |
| L28 N953 "[" · 5× | 0.00 | 0.09 [0.03, 0.17] | 0.02 | 0.00 | 0.01 / 0.10 | 0.074 | 0.83 | 0.77 |
| L32 N2669 "set" · 2× | 0.00 | **0.27** [0.17, 0.39] | 0.07 | 0.00 | 0.01 / 0.13 | **0.037** | 1.78 | 0.75 |

![**Figure 2.** H2. Left: HM per condition with 95% prompt-cluster-bootstrap CIs. Right: concept score. Gray dots: 80 random same-layer neurons at the same relative clamp.](../results/mamba-1.4b-hf/figures/h2_conditions.png)

### 4.5 Post-hoc low-dose regime and H2b
The pre-registered doses (≥ 2× the maximum) sit far outside these neurons' natural range. For the three H2 poles, the median of the 200 strongest firings in 256k held-out tokens is 9–39× below the discovery maximum. A post-hoc sweep at 0.25×, 0.5× and 1× found a regime where the concept enters without breaking the text (Table 2). The best poles kept judged fluency within 0.1–0.2 of the unsteered level (0.5–0.8 vs 0.6) and distinct-2 at 0.82–0.93 (unsteered 0.91):

- L30 N815 ("from", 0.5×): HM 0.67, concept 1.6.

- L30 N2121 ("as", 0.5×): HM 0.60, concept 1.6.

- L37 N71 ("first", 0.25×): HM 0.53, concept 1.6.

Example (L30 N2121, 0.5×): "…objects, and more specifically, **as** used in a mobile device." These three poles and doses were then pre-registered as H2b and tested on a fresh prompt set C.

On fresh prompt set C (Table 5, Figure 4), all three low-dose poles raised the HM well above unsteered:

| Pole | Dose | Clamp − unsteered HM [95% CI] |
|---|---|---|
| "from" | 0.5× | +0.40 [0.25, 0.56] |
| "as" | 0.5× | +0.16 [0.07, 0.26] |
| "first" | 0.25× | +0.35 [0.23, 0.49] |

Each beat all 80 random same-layer neurons (Holm p = 0.037) and met the non-inferiority fluency rule. Judged fluency stayed within 0.09 of unsteered, and distinct-2 was ≥ 0.87 against 0.93. **H2b is supported for all three neurons.**

Neuron clamping also had a higher mean HM than both baselines for every pole: prompting (0.10–0.14) and DiffMean at the same residual norm (0.15–0.19). In a post-hoc paired bootstrap, five of the six differences exclude 0; the exception is "as" vs DiffMean, +0.05 [−0.06, 0.17]. DiffMean matched the neuron's concept score only for "from", and then at a clear fluency cost (distinct-2 0.67 vs 0.90). Example (L37 N71, 0.25×): "…have been recognized as the **first** step in understanding the biomechanical behavior of the bone."

: **Table 5.** H2b confirmation (prompt set C).

| Pole · dose | Unsteered HM | Clamp HM [95% CI] | Prompting HM | DiffMean HM | Null HM max | Holm p | Fluency (clamp / unst.) | distinct-2 (clamp / unst.) |
|---|---|---|---|---|---|---|---|---|
| L30 N815 "from" · 0.5× | 0.06 | **0.46** [0.31, 0.62] | 0.14 | 0.16 | 0.13 | 0.037 | 0.45 / 0.48 | 0.90 / 0.93 |
| L30 N2121 "as" · 0.5× | 0.08 | **0.24** [0.14, 0.35] | 0.10 | 0.19 | 0.13 | 0.037 | 0.32 / 0.41 | 0.92 / 0.93 |
| L37 N71 "first" · 0.25× | 0.00 | **0.35** [0.23, 0.49] | 0.11 | 0.15 | 0.00 | 0.037 | 0.42 / 0.42 | 0.87 / 0.93 |

![**Figure 4.** H2b. HM (left, 95% CIs) and concept score (right) per condition; gray dots are 80 random same-layer neurons at the same relative dose.](../results/mamba-1.4b-hf/figures/h2b_conditions.png)

![**Figure 5.** Dose-response of the H2b neurons on prompt set A (10 prompts per point, no CIs). Doses left of the dotted line are the post-hoc sweep, doses right of it the pre-registered screen.](../results/mamba-1.4b-hf/figures/dose_response_h2b.png)

### 4.6 H3 and mechanism: sufficient but barely necessary, and dependent on long context
Results for the three H2 poles (Table 3):

- **Ablation (H3).** Zero-ablation lowered log p(next token) at the strongest held-out firings (196–199 per pole) by 0.0018 (L28 N953) and 0.0123 nats (L32 N2669). Both exceed all 20 random same-layer neurons (empirical p = 1/21, uncorrected). The "as" neuron L30 N2121 showed no effect (−0.003 nats, p = 0.76).

- **Gate.** At all 600 firings the gate was open (z > 0, median SiLU(z) 3.1–4.2), so the SSM factor y decides the pole. The double-negative regime never occurred.

- **Context truncation.** With only the last ≤ 4 tokens (one block's convolution kernel), the activations reached 0–11% of their full-context value. They recovered to 40–84% with 32 tokens. These neurons therefore need dozens of tokens of context. The test does not show whether that context is carried by the recurrent state or by convolutions stacked across layers.

- **Patching.** Writing a typical firing value at a single position changed promoted-token log-probabilities no more than random neurons did (p 0.33–0.86). The null neurons were patched with relatively stronger values (Section 3), so this comparison is not like-for-like.

: **Table 3.** H3 and mechanism for the H2 poles (activation ratio = truncated / full context).

| Pole | Ablation drop (nats) | Null max | p | Ratio, k = 4 | Ratio, k = 32 | Gate open |
|---|---|---|---|---|---|---|
| L30 N2121 ("as") | −0.0030 | 0.0030 | 0.76 | 0.00 | 0.40 | 100% |
| L28 N953 ("[") | 0.0018 | 0.0014 | 0.048 | 0.00 | 0.74 | 100% |
| L32 N2669 ("set") | 0.0123 | 0.0069 | 0.048 | 0.11 | 0.84 | 100% |

**Post-hoc truncation survey (Figure 3).** Each of the 39 screened poles is tested on up to 40 held-out top firings, one per window. The median activation kept with only 4 tokens of context falls with depth:

- **Layers 0–1 (4 poles):** 0.92. Three of the four early detectors are local (0.84–1.10), which shows that short inputs do not suppress activations per se. The fourth, L0 N2107, keeps 0.20.

- **Layers 20–29 (8 poles):** 0.10.

- **Layers 30–39 (19 poles):** 0.18.

- **Layers 40–47 (8 poles):** 0.24.

The layer-27–32 prediction neurons ("as", "from", "set", "per", "[", digits) are among the most long-range: only 50–70% of their activation is recovered with 32 tokens. L22 N1915 (separator lines, 0.32) and L31 N2412 (periods before legal citations, 0.41) recover even less. In Mamba-1.4B, then, mid-to-late neurons depend on dozens of tokens of context. Whether that context flows through the recurrent state or through convolutions stacked across layers (3ℓ + 4 tokens at layer ℓ) is not identified. Resetting the SSM state would test it causally. We did not test whether transformer MLP neurons at comparable depth show a similar context dependence.

![**Figure 3.** Fraction of a neuron's activation retained with only the last k tokens of context (k = 4 is one block's convolution kernel), for 39 clean candidate poles, by layer.](../results/mamba-1.4b-hf/figures/truncation_survey.png)

## 5 Discussion
**Are Mamba neurons "already clean"?** Our evidence is mixed, and it depends on the instrument.

- **Statistics: strongly privileged.** The neuron basis is privileged in its statistics. Neurons are far more heavy-tailed than random directions (median kurtosis about 50× higher), which is the signature of sparse, detector-like units ([Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html)).

- **Audit: no separation.** A local 9B labeller could not separate random neurons from random directions (H1). It also reached only ~15% clean poles overall, and its judgments agreed only moderately with blind reference ratings.

- **Contrast with Goodfire.** Goodfire's roughly 90% estimate for Qwen3-32B comes from preliminary audits with a different procedure that the essay does not detail. Our null result therefore says more about cheap automated auditing than about Mamba. Neurons that the selection formula picks, mostly prediction neurons, were described well (50% clean, mean BA 0.77).

**Steering works, at the right dose.**

- **Specific and controllable.** Single Mamba neurons are causally steerable, and the effect is specific. At both confirmation doses (H2 and H2b), the target concept appeared far more often than under 80 random neurons pushed equally hard. The prompting baseline raised the concept only modestly: 0.11–0.48, against 0.83–1.78 for clamping.

- **Dose matters.** The pre-registered doses, borrowed from SAE-feature clamping at multiples of the maximum, are wrong for neurons whose maxima are 9–39× their typical strong firings. Those doses produce degenerate repetition. A quarter to half of the maximum injects the concept while fluency stays non-inferior to unsteered text (by the pre-registered rule), which H2b confirmed on fresh prompts.

- **Mostly better than DiffMean here.** In this regime the neuron's own write direction outperformed DiffMean at the same norm on our metric for "from" and "first"; for "as" the difference was not significant (post hoc). That differs from AxBench's finding that DiffMean is a strong representation baseline, but our concepts are single function words rather than topics, and only one norm was compared.

**Sufficient but not necessary.** Turning a neuron up reliably changes output, but removing it barely hurts prediction where it fires (≤ 0.012 nats). This asymmetry suggests distributed or redundant computation, or downstream compensation ([McGrath et al., 2023](https://arxiv.org/abs/2307.15771)). It warns against reading steerability as evidence that a neuron is *the* mechanism for its token.

**What is Mamba-specific.** Two observations stand out:

- **Where the pole comes from.** The SiLU gate was always open at the firings we examined, so the pole is decided by the SSM output y. The double-negative regime Goodfire flags for SwiGLU did not arise.

- **Long context.** Mid and late neurons barely respond to the last 4 tokens alone and recover their activation only with tens of tokens of context. Most early-layer detectors are local. Our truncation test cannot say whether the deep neurons get that context from the recurrent state or from convolutions stacked across layers. If it is the state, interpreting them requires tracking what the state carries, for example with the hidden-attention view of [Ali et al. (2024)](https://arxiv.org/abs/2403.01590).

## 6 Limitations

- **Labeller and judge.** All labels, detection decisions and steering ratings come from a local 9B model (qwen3.5:9b).
  - Its detection decisions agree only moderately with blind reference ratings (κ = 0.55).
  - Its fluency ratings are systematically harsh and only fairly reliable (κ = 0.34), although its concept ratings are good (κ = 0.79).
  - The "manual" reference ratings were produced by Claude Code, not by human annotators.
  - AxBench's instruction-relevance score is not used, so our HM is a two-score version of its metric.
  - H1's null result may therefore reflect the instrument more than the network.

- **Scale and data.** One model (Mamba-1.4B), one corpus (2.32M + 0.65M Pile tokens, 128-token windows) and one seed for data and sampling. 128-token windows understate any long-range neuron behaviour beyond 128 tokens.

- **Doses.** The pre-registered clamp doses copied the SAE-clamping convention (multiples of the maximum), which is badly suited to neurons with kurtosis in the hundreds to tens of thousands. The working low-dose regime was found post hoc on the screen prompts and then confirmed on fresh prompts (H2b), not anticipated.

- **Steering metrics.** For token-prediction neurons the steering concept ("the word 'as'") is narrow, and concept presence can be judged high for degenerate repetition. We therefore report distinct-2 and NLL alongside the judge. NLL under the unsteered model is a weak fluency proxy for repetitive text.

- **Baselines.** DiffMean vectors come from held-out documents that may share documents with the held-out prompt pools. If anything, this favours the baseline.

- **Necessity and mechanism.** H3 p-values are uncorrected per pole (two of three reach the minimum 1/21). Context truncation feeds inputs that begin right after a separator, which is itself unusual context. The post-hoc survey across depths is meant to control for this, but its early-layer control has only 4 poles. Truncation shows how much context a neuron needs, not which path (SSM state or stacked convolutions) carries it. The patching null uses stronger donor values than the target.

- **Compute.** Without the `mamba_ssm` CUDA kernels on Windows, the reference torch scan averaged 786 tok/s over the discovery pass (batch 12). This limited corpus size and the number of confirmation conditions.

## 7 Conclusion
On an 8 GB laptop GPU and with only local models, a pre-registered neuron-level study of Mamba-1.4B yielded the following:

- **An open atlas** of every neuron.

- **A falsified hypothesis.** Automated labels from a local 9B LLM did not separate neurons from random directions (H1).

- **A falsified-as-registered steering test.** H2's fluency rule failed at strong doses.

- **A confirmed refinement.** H2b showed that single neurons, turned up gently, steer the model toward their word with non-inferior fluency, beating random neurons. Their mean HM was also higher than prompting and DiffMean, significantly so in 5 of 6 post-hoc comparisons.

The neurons we confirmed, for "from", "as" and "first", are lexical, and their necessity is weak. Natural next steps:

1. A stronger labeller or human audit to retest H1.

2. Topic-level neurons, for which the domain stream found candidates.

3. Dose-calibrated steering in units of a neuron's typical firing rather than its maximum.

4. Direct interventions on the SSM state, such as resetting it, to test whether the long-range context flows through the state or through stacked convolutions.

## References

- Ali, Zimerman & Wolf (2024). The Hidden Attention of Mamba Models. [arXiv:2403.01590](https://arxiv.org/abs/2403.01590)

- Bills et al. (2023). Language models can explain neurons in language models. [OpenAI](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)

- Cammarata (2026). Neural Networks Are Already Clean. [Goodfire](https://already-clean.goodfire.com/)

- Choi et al. (2024). Scaling Automatic Neuron Description. [Transluce](https://transluce.org/neuron-descriptions)

- Elhage et al. (2022). Toy Models of Superposition. [Transformer Circuits](https://transformer-circuits.pub/2022/toy_model/index.html)

- Ensign & Garriga-Alonso (2024). Investigating the Indirect Object Identification circuit in Mamba. [arXiv:2407.14008](https://arxiv.org/abs/2407.14008)

- Gao et al. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)

- Geva et al. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)

- Geva et al. (2022). Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. [arXiv:2203.14680](https://arxiv.org/abs/2203.14680)

- Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

- Gurnee et al. (2023). Finding Neurons in a Haystack: Case Studies with Sparse Probing. [arXiv:2305.01610](https://arxiv.org/abs/2305.01610)

- Gurnee et al. (2024). Universal Neurons in GPT2 Language Models. [arXiv:2401.12181](https://arxiv.org/abs/2401.12181)

- Landis & Koch (1977). The Measurement of Observer Agreement for Categorical Data. [Biometrics](https://doi.org/10.2307/2529310)

- Liu (2026). Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models. [arXiv:2606.19831](https://arxiv.org/abs/2606.19831)

- Lones (2021). How to avoid machine learning pitfalls. [arXiv:2108.02497](https://arxiv.org/abs/2108.02497)

- McGrath et al. (2023). The Hydra Effect: Emergent Self-repair in Language Model Computations. [arXiv:2307.15771](https://arxiv.org/abs/2307.15771)

- Meng et al. (2022). Locating and Editing Factual Associations in GPT. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)

- Panickssery (formerly Rimsky) et al. (2024). Steering Llama 2 via Contrastive Activation Addition. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)

- Paulo et al. (2024). Automatically Interpreting Millions of Features in Large Language Models. [arXiv:2410.13928](https://arxiv.org/abs/2410.13928)

- Phipson & Smyth (2010). Permutation P-values Should Never Be Zero. [SAGMB](https://doi.org/10.2202/1544-6115.1585)

- Pitorro & Treviso (2025). LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models. [arXiv:2502.15612](https://arxiv.org/abs/2502.15612)

- Sharma, Atkinson & Bau (2024). Locating and Editing Factual Associations in Mamba. [arXiv:2404.03646](https://arxiv.org/abs/2404.03646)

- Shazeer (2020). GLU Variants Improve Transformer. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

- Sunku Mohan et al. (2026). Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks. [arXiv:2602.22719](https://arxiv.org/abs/2602.22719)

- Takahashi et al. (2025). TopK Language Models. [arXiv:2506.21468](https://arxiv.org/abs/2506.21468)

- Templeton et al. (2024). Scaling Monosemanticity. [Transformer Circuits](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

- Turner et al. (2023). Steering Language Models With Activation Engineering. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)

- Voita, Ferrando & Nalmpantis (2024). Neurons in Large Language Models: Dead, N-gram, Positional. [arXiv:2309.04827](https://arxiv.org/abs/2309.04827)

- Wang et al. (2025). Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures. [arXiv:2410.06672](https://arxiv.org/abs/2410.06672)

- Wu et al. (2025). AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. [arXiv:2501.17148](https://arxiv.org/abs/2501.17148)

## Appendix A: pre-registration amendments
The full text is in [reports/preregistration.md](https://github.com/BurnyCoder/mamba-neuron-mechanistic-interpretability/blob/main/reports/preregistration.md). Each amendment records what data existed when it was written.

| # | When | Change | Data seen at the time |
|---|---|---|---|
| 1 | before any analysis | power figure corrected from "~45" to "~40 per arm" (n = 100 unchanged) | none |
| 2 | during the activation pass | promoted tokens in the label prompt (symmetric for neurons and directions); byte-correct token decoding; domain-neuron candidate stream; screen restricted to clean labels | smoke runs on mamba-130m only |
| 3 | before any main-model label | labeller/judge `qwen3.5:9b`, chosen over 4B on 20 mamba-130m smoke poles (mean BA 0.66 vs 0.61); token-count hints | smoke data only |
| 4 | screen generated, first judge calls seen | judge steering against an output-side *steering concept*; one-sentence judge explanations; 2k context | 9 discarded judge calls (8 concept, 1 fluency; the amendment text says 7) |
| 5 | before any confirmation data | random null 20 → 80 neurons × 10 prompts, because 1/21 can never survive Holm over 3 poles | none for confirmation |
| 6 | screen judged | confirmation plan frozen (3 poles, doses); post-hoc low-dose sweep announced | screen results (prompt set A) |
| 7 | low-dose sweep judged, before any prompt-set-C data | H2b: 3 low-dose poles on fresh prompt set C; non-inferiority fluency rule (judged fluency κ = 0.34) | screen and low-dose results (set A), H2 not yet judged |
| 8 | 2026-11-01 audit, after all results | erratum only, no analysis changed: coherence used 40 (not 20) top firings; amendment 4 discarded 9 (not 7) judge calls; screen prompts are held-out openings; neutrality percentile from the first 200 discovery windows; Landis & Koch's "substantial" band starts at κ = 0.61 | all results |

## AI statement
Claude Code (Anthropic's coding agent, model Claude Opus 5.5) wrote this codebase and this paper: the code, the experiments, the analyses and the text. A human, Libor Burian, provided the exact specification that was executed. Human cleanup and rewriting of the paper are in progress. The "manual" judge-validation ratings reported here were produced by Claude Code, not by a human annotator.
