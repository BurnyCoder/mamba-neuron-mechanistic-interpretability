# Pre-registration: neurons of Mamba-1.4B

Written **2026-10-28, before any analysis run on the main model** (only unit tests on mamba-130m had been run).
Any change after this point is logged in the *Amendments* section with a date and a reason. Analyses that are
not listed here are reported as **post hoc**.

## Domain and question
- **Model.** `state-spaces/mamba-1.4b-hf`: 48 Mamba-1 blocks, d_model 2048, d_inner 4096, trained on the Pile ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)).
- **Unit of analysis.** Neuron (ℓ, i) is coordinate `a_i` of `a = (SSM(x̃) + D·x̃) ⊙ SiLU(z)`, the input to `mixer.out_proj` in block ℓ. That gives 196,608 neurons. Its write vector is column `i` of `out_proj.weight`.
- **Question.** Do individual Mamba neurons behave as interpretable, causally steerable units, as Goodfire argues for transformer neurons ([Neural Networks Are Already Clean](https://already-clean.goodfire.com/))?

## Data and leakage control
- **Corpus.** `NeelNanda/pile-10k`, split by document with seed 0: 70% *discovery*, 30% *held-out*.
- **Windows.** 128 tokens with the `<|endoftext|>` separator prepended, at most 4 per document. The separator position is excluded from all statistics.
- **What each split is used for.** Labels, candidate selection and the steering screen use discovery data only. Detection scoring, confirmatory steering prompts, DiffMean vectors and ablations use held-out data only.
- **Neutral steering prompts.** These are the first 16 tokens of held-out documents. Each pre-registered neuron also gets a set where it is inactive: its activation stays below its 90th discovery percentile on every prompt token. The screen and the confirmation use disjoint prompt sets (A and B).

## H1: privileged basis (interpretability audit)
- **Sample.** 96 neurons (2 per layer, seeded) and 96 random unit directions in the same `a`-space (2 per layer, seed 0). Both poles of each unit are evaluated, giving 192 neuron poles and 192 direction poles.
- **Label.** The local LLM (`qwen3.5:4b` via Ollama) writes a one-phrase explanation from the top-20 discovery examples of the pole.
- **Score.** Held-out detection balanced accuracy, following [Paulo et al. 2024](https://arxiv.org/abs/2410.13928). The LLM sees the label and 20 shuffled held-out snippets: the pole's 10 top-activating snippets and 10 random snippets. It predicts which snippets match the label.
- **Primary metric.** The fraction of poles that are "clean", meaning balanced accuracy ≥ 0.8.
- **Test.** Neuron fraction minus random-direction fraction, using a one-sided Fisher exact test (α = 0.05), a 95% bootstrap CI on the difference (10,000 resamples of units), and Wilson 95% CIs on each fraction.
- **Falsified if** the difference is ≤ 0, or its CI includes 0.
- **Judge validity.** I rate about 40 label/snippet decisions by hand. The judge is accepted if Cohen's κ ≥ 0.6, which is "substantial" agreement on the [Landis & Koch 1977](https://doi.org/10.2307/2529310) scale.

## Candidate selection (fixed formula, discovery data only)
Each neuron pole gets three scores. The pole sign is s = +1 for the positive pole and −1 for the negative pole.
1. **Activation kurtosis.** Detector-like sparsity.
2. **Vocab kurtosis.** The kurtosis of `u_i = W_U (g ⊙ w_i)` ([Gurnee et al. 2024](https://arxiv.org/abs/2401.12181); `g` is the gain of `norm_f`).
3. **Input-output coherence.** The mean vocab-z-scored `s·u_i[next token]` at the pole's top-20 firing positions. In plain terms: does the neuron push up the tokens that actually come next where it fires?

The candidate score is the mean of the three ranks. The top 60 poles go forward, with at most 3 per layer.

## H2: steerability (main claim)
- **Screen** (exploratory; prompt set A; 10 prompts × 1 seed). Clamp the pole at s·c·|pole extreme| for c ∈ {2, 5, 10, 20} at all positions. Generate 40 new tokens (temperature 0.7, top-p 0.95).
- **Screen scoring.** The judge rates concept (0–2) and fluency (0–2), following [AxBench](https://arxiv.org/abs/2501.17148). It is given the pole's label as the concept. Each generation gets HM = the harmonic mean of the two.
- **Selection.** At most 3 neurons go forward, each with its best c, ranked by mean HM. They are written into the amendment below before the confirmation runs.
- **Confirmation** (prompt set B; 50 prompts × 2 seeds = 100 generations per condition). Conditions:
  1. Unsteered.
  2. Neuron clamp at the chosen c.
  3. Neuron additive steering with the same residual-norm budget.
  4. **Random-neuron null:** 20 random neurons from the same layer, each clamped at the same relative level c × its own pole extreme, with 50 generations each.
  5. **DiffMean:** the residual direction at block ℓ output, from mean(held-out top-firing positions) − mean(random positions), scaled to the same residual-norm budget.
  6. **Prompting:** prefix "`<label>`:\n" on the unsteered model.
- **Primary metric.** Mean per-generation HM(concept, fluency).
- **Tests.** Paired cluster bootstrap over prompts versus unsteered (95% CI). An empirical p-value versus the random-neuron null: p = (1 + #random ≥ observed) / 21. Holm correction across ≤ 3 neurons.
- **Supported if** at least one neuron has a CI that excludes 0 versus unsteered, a Holm-adjusted p < 0.05 versus the null, **and** mean fluency ≥ 1.0.
- **Secondary metrics.** Concept alone; fluency alone; NLL of the continuation under the unsteered model; distinct-2; dose–response over c.
- **Power.** With a binary concept ≥ 1 proxy, p0 = 0.1 versus p1 = 0.4 at α = 0.05/3 and power 0.8 needs about 40 per arm (39.7; statsmodels `NormalIndPower`, Cohen's h). 100 per arm was chosen.

## H3: necessity
- **Ablation.** On held-out windows, zero-ablate the neuron at all positions. Measure the drop in log p(actual next token) at its 200 top-firing positions.
- **Comparison.** The same measurement for 20 random same-layer neurons.
- **Supported if** the neuron's mean drop exceeds every random neuron's drop (empirical p ≤ 1/21).

## Exploratory (reported as such)
- Activation patching from firing into non-firing contexts.
- Direct versus total logit effect.
- Gate (`SiLU(z)`) versus SSM-factor decomposition, and the double-negative regime.
- The context-truncation test (1/4/16/all previous tokens).
- Split stability (Spearman correlation of per-neuron statistics, discovery vs held-out).
- Cross-size universality (mamba-130m/790m).
- Seed and prompt-set sensitivity.

## Amendments
- **2026-10-28 (before any analysis):** the power figure in the first pushed version ("about 45 per arm") was a hand estimate. The tested `metrics.power_n_per_arm` gives 39.7, so the text now says about 40. The chosen n = 100 per arm is unchanged.
- **2026-10-28, amendment 2 (written while the main-model activation pass was running; no label, steering or mechanism result existed).** These changes come from smoke tests on mamba-130m.
  - **Label input.** The explanation prompt now also lists the 10 tokens each pole promotes, `s·W_U(g ⊙ W_out v)`. This is computed identically for neurons (v = one-hot) and random directions (v = q), so the H1 comparison stays symmetric. Token strings are now decoded byte-correctly.
  - **Label model.** The 4B model mislabelled two obvious smoke-test neurons ("fires before *that*", "fires before a closing dash"). The labeller and judge model will be fixed after a comparison with `qwen3.5:9b` on these smoke poles. That comparison uses only mamba-130m data, not the main model.
  - **Second, exploratory candidate stream ("domain neurons").** Window-mean activations come from 250 discovery windows for each of 8 distinctive Pile subsets (Github, PubMed Abstracts, Wikipedia (en), USPTO Backgrounds, FreeLaw, DM Mathematics, ArXiv, StackExchange) plus 750 generic windows. For each subset, the 3 neurons with the largest |d'| (subset vs rest) are added, with pole sign = sign(d'). The goal is to find topic- or domain-level neurons; the fixed formula above mostly finds next-token-prediction neurons.
  - **Screen.** The screen covers candidate poles from both streams whose label has detection balanced accuracy ≥ 0.8. The confirmation protocol (H2) is unchanged.
- **2026-10-28, amendment 3 (before any label of the main model).** Labeller and judge model: **`qwen3.5:9b`**. The explanation prompt also lists the 5 most common peak tokens and next tokens of the shown examples (same for every unit type). On the 20 mamba-130m smoke candidates (`scripts/compare_labelers.py`, `results/labeler_comparison_smoke.json`), 9B reached mean detection BA 0.66 with 5/20 clean, against 0.61 and 2/20 for 4B.
- **2026-10-28, amendment 4 (screen generations existed; no screen judgement had been scored or inspected).** The first 7 judge calls showed that the concept passed to the judge was the *input-side* detection label (e.g. "fires on tokens preceding 'all'"). The judge then rated "all all all …" as 0 because no token *precedes* 'all'. Those 7 calls are discarded.
  - **Steering concept.** Each screened pole now gets an output-side *steering concept*, written once by the same LLM from its label and promoted tokens before any steered text is judged (`judge.steering_concept`, e.g. "mentions of the word 'all'"). This concept is what the AxBench judge scores, and it is also the prefix of the prompting baseline.
  - **Judge settings.** The judge system prompt asks for a one-sentence explanation, and judge calls use a 2,048-token context. Together these cut judge time from about 20 s to about 1.2 s per call. The AxBench rating templates are unchanged.
- **2026-10-28, amendment 5 (before any confirmation data existed; found by reviewing the analysis code, not from data).** With 20 random-neuron nulls the smallest empirical p is 1/21 = 0.048. Holm correction over 2–3 confirmatory poles then gives an adjusted p ≥ 0.095, so H2 could never be supported. The random-neuron null is changed to **80 random same-layer neurons × 10 prompts** (the first 10 of the 50 confirmation prompts, seed 0). The smallest empirical p becomes 1/81 = 0.012, and Holm × 3 = 0.037. Each null neuron's mean is noisier (10 generations) than the target's (100). This widens the null distribution and makes the test conservative. The H3 ablation null stays at 20 neurons, as pre-registered, and its p-values are reported uncorrected per pole.
- **2026-10-28, amendment 6: confirmation plan (written by `screen_judge` and frozen here before `confirm` ran).** The three poles with the largest mean HM gain over unsteered on prompt set A:

  | Pole | Label | Steering concept | Clamp | HM gain |
  |---|---|---|---|---|
  | `neuron:L30:N2121:+` | "fires on tokens immediately preceding the word 'as'" | "the word 'as'" | 2× | 0.40 |
  | `neuron:L28:N953:+` | "fires on tokens immediately preceding an opening square bracket" | "opening square brackets" | 5× | 0.30 |
  | `neuron:L32:N2669:+` | "fires when the word 'set' appears in code or text" | "mentions of the word 'set'" | 2× | 0.27 |

  The screen showed that every pre-registered dose (≥ 2× the pole maximum) drives judged fluency to about 0 for nearly all poles; unsteered text gets 0.6. The H2 fluency criterion (≥ 1.0) is therefore expected to fail. The confirmation is run exactly as registered anyway.
- **Planned post-hoc exploration (labelled as such everywhere).** A low-dose sweep at 0.25×, 0.5× and 1× the pole maximum, on prompt set A with the same steering concepts. If it reveals a fluent regime, a new hypothesis **H2b** will be pre-registered here *before* it is tested on a fresh prompt set C, disjoint from A and B.
- **2026-10-28, amendment 7: H2b, pre-registered after the post-hoc low-dose sweep and before any prompt-set-C data existed.** The sweep on prompt set A showed a regime where concept injection keeps the text fluent: 0.25× gives concept 0.52, distinct-2 0.83 and HM 0.092, against 0.08, 0.91 and 0.050 unsteered. Steering judge validation showed the fluency judge to be unreliable and harsh (κ = 0.34; it scores fluent unsteered text at 0.60).
  - **Poles and doses.** The three poles with the largest low-dose HM gain, at their best dose (`results/mamba-1.4b-hf/confirm_plan_h2b.json`):

    | Pole | Steering concept | Dose |
    |---|---|---|
    | `neuron:L30:N815:+` | "the word 'from'" | 0.5× |
    | `neuron:L30:N2121:+` | "the word 'as'" | 0.5× |
    | `neuron:L37:N71:+` | "mentions of the word 'first'" | 0.25× |

  - **Prompts.** Prompt set C: 50 held-out document openings from a pool of 400 that starts *after* the pool of prompt set B (`skip = 410`). The same neutrality filter applies, with 2 seeds.
  - **Conditions.** Identical to H2: unsteered, clamp, add, DiffMean, prompting, and the 80-neuron × 10-prompt random null at the same relative dose.
  - **H2b is supported for a pole if all of the following hold:**
    1. The paired clamp − unsteered HM difference has a 95% prompt-cluster-bootstrap CI above 0.
    2. The Holm-adjusted (over the 3 poles) empirical p against the random-neuron null is < 0.05.
    3. Fluency is non-inferior: clamp distinct-2 ≥ 0.9 × unsteered distinct-2 **and** clamp judged fluency ≥ unsteered judged fluency − 0.2.
  - **Falsified if** no pole meets all three. Secondary metrics: concept alone against the null, P(concept ≥ 1), and NLL.
  - **H2 itself is unchanged and is judged by its original absolute fluency rule.** The non-inferiority verdict is also stored for H2 and is reported as post hoc.
