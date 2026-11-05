# Experiment 3: turning single neurons up (steering screen, H2, post-hoc dose sweep, H2b)

**Status:** done.
**Code:** `steer.py`, `judge.py`, `phases/steering.py`.
**Results:** `results/mamba-1.4b-hf/{screen,screen_low,confirm_plan,confirm_plan_h2b,h2,h2b,steer_validation}.json`.

## Protocol
- **Neutral prompts.** The first 16 tokens of held-out documents.
- **Interventions.** These act at every position (prompt included) during sampled generation: 40 new tokens, temperature 0.7, top-p 0.95, seed 0.
- **Judge.** The judge (`qwen3.5:9b`) uses the AxBench concept and fluency templates, scoring each from 0 to 2. Each generation gets their harmonic mean (HM). AxBench's third score, instruction relevance, is dropped because the prompts are text to continue.
- **Steering concept.** The concept is an output-side "steering concept" written once per pole from its label and promoted tokens (amendment 4).

## Screen (pre-registered, exploratory): 39 clean candidate poles × {2, 5, 10, 20}× pole max × 10 prompts
Averaged over all 39 poles. The unsteered row repeats the same 10 unsteered texts, judged against each pole's concept.

| Dose | Concept | P(concept ≥ 1) | Fluency (judge) | HM | NLL (unsteered model) | distinct-2 |
|---|---|---|---|---|---|---|
| unsteered | 0.08 | 0.06 | 0.60 | 0.050 | 1.13 | 0.91 |
| 2× | **0.82** | **0.48** | 0.11 | 0.022 | 1.80 | 0.39 |
| 5× | 0.67 | 0.40 | 0.05 | 0.008 | 2.10 | 0.27 |
| 10× | 0.76 | 0.45 | 0.02 | 0.000 | 2.14 | 0.21 |
| 20× | 0.69 | 0.41 | 0.01 | 0.000 | 2.02 | 0.17 |

**Reading.** Clamping a single neuron at ≥ 2× its maximum injects its concept about 8 to 10 times more often than unsteered text (concept 0.08 → 0.82). It also wrecks fluency: the text degenerates into repetition of the promoted token, and distinct-2 falls from 0.91 to 0.39. Examples (prompt set A, 2× unless noted):
- **L37 N1812** ("after all" → `all`): " all all all all …"
- **L36 N109** (email): "@@@@…"; at 5×: " AT AT AT …"
- **L23 N334** (academic citation markers), at 5×, stays fluent: "two-dimensional objects.\n2. Background\nIn [@state]{} [@state]{} the question is, 'Are these questions sufficient…'"

The pre-registered selection rule (largest mean HM gain over unsteered) picked three poles, frozen in amendment 6:

| Pole | Steering concept | Dose | HM gain |
|---|---|---|---|
| L30 N2121 | "the word 'as'" | 2× | 0.40 |
| L28 N953 | "opening square brackets" | 5× | 0.30 |
| L32 N2669 | "mentions of the word 'set'" | 2× | 0.27 |

## Judge validity (40 screen items, rated by Claude Code from an item file without the judge's ratings)
| Judge metric | Cohen's κ [95% CI] | Exact agreement | Mean, LLM vs reference |
|---|---|---|---|
| Concept | **0.79** [0.62, 0.95] | 0.875 | 0.85 vs 0.98 |
| Fluency | **0.34** [0.17, 0.53] | 0.70 | 0.25 vs 0.65 |

The concept judge agrees well with the reference. The fluency judge is harsh: it often gives 0–1 to fluent base-model text, which explains why unsteered text scores only 0.60. Post hoc, per clamp level (`steer_validation.json` → `by_level`):

- **Unsteered items (n = 9).** The reference gives fluency 1.89 and the judge 0.67. The fluency κ comes mostly from agreement on degenerate high-dose text, which both raters score 0, so the judge cannot reliably rank *fluent* texts.
- **Coverage.** The 40 items come from levels 0, 2, 5, 10 and 20× (9, 10, 10, 6 and 5 items). None comes from the 0.25–1× regime where H2b is tested, so the judge is unvalidated there.
- **Reference rater.** The ratings were made by Claude Code, not by hand as the pre-registration said (amendment 9). The pre-registered H2 fluency criterion (≥ 1.0) is therefore close to unreachable even without steering. We report NLL and distinct-2 as objective fluency proxies. NLL under the unsteered model is a weak proxy for degeneration, because repetitive text becomes predictable; distinct-2 is better.

## H2 confirmation (pre-registered)
Prompt set B: 50 held-out openings on which the neuron is inactive (47 for L32 N2669), × 2 seeds = 100 generations per condition. The null is 80 random same-layer neurons × 10 prompts at the same relative clamp. Values are means, with HM 95% prompt-cluster-bootstrap CIs.

| Pole · dose · concept | Condition | HM [95% CI] | Concept | Fluency (judge) | distinct-2 | NLL |
|---|---|---|---|---|---|---|
| **L30 N2121 · 2× · "the word 'as'"** | unsteered | 0.090 [0.033, 0.157] | 0.24 | 0.45 | 0.93 | 1.22 |
| | **clamp** | **0.383 [0.257, 0.517]** | **1.68** | 0.33 | 0.85 | 2.31 |
| | add | 0.383 [0.257, 0.513] | 1.66 | 0.33 | 0.85 | 2.33 |
| | DiffMean | 0.000 | 1.92 | 0.00 | 0.05 | 1.26 |
| | prompting | 0.150 [0.077, 0.223] | 0.48 | 0.52 | 0.91 | 1.40 |
| | 80 random neurons | mean 0.052, max 0.267 | mean 0.19, max 0.60 | | | |
| **L28 N953 · 5× · "opening square brackets"** | unsteered | 0.000 | 0.05 | 0.41 | 0.94 | 1.25 |
| | **clamp** | **0.093 [0.027, 0.173]** | **0.83** | 0.21 | 0.77 | 2.42 |
| | add | 0.093 [0.027, 0.173] | 0.82 | 0.21 | 0.77 | 2.40 |
| | DiffMean | 0.000 | 1.57 | 0.00 | 0.17 | 1.94 |
| | prompting | 0.023 [0.000, 0.060] | 0.11 | 0.46 | 0.92 | 1.44 |
| | 80 random neurons | mean 0.006, max 0.100 | mean 0.03, max 0.40 | | | |
| **L32 N2669 · 2× · "mentions of the word 'set'"** | unsteered | 0.000 | 0.01 | 0.46 | 0.92 | 1.30 |
| | **clamp** | **0.273 [0.167, 0.387]** | **1.78** | 0.22 | 0.75 | 2.57 |
| | add | 0.184 [0.096, 0.284] | 1.78 | 0.17 | 0.76 | 2.57 |
| | DiffMean | 0.000 | 2.00 | 0.00 | 0.03 | 1.35 |
| | prompting | 0.071 [0.014, 0.149] | 0.47 | 0.37 | 0.90 | 1.55 |
| | 80 random neurons | mean 0.008, max 0.133 | mean 0.03, max 0.40 | | | |

| Pole | Clamp − unsteered HM [95% CI] | Empirical p vs null (HM) | Holm p | p vs null (concept only) | Absolute fluency rule (pre-registered) | Non-inferior fluency rule (post hoc for H2) |
|---|---|---|---|---|---|---|
| L30 N2121 | +0.293 [0.157, 0.433] | 1/81 = 0.012 | **0.037** | 0.012 | ✗ (0.33 < 1.0) | ✓ |
| L28 N953 | +0.093 [0.027, 0.173] | 0.074 | 0.074 | 0.012 | ✗ | ✗ |
| L32 N2669 | +0.273 [0.167, 0.387] | 0.012 | **0.037** | 0.012 | ✗ | ✗ (distinct-2 0.75 < 0.83) |

**Verdict: H2 as pre-registered is not supported.** No pole reaches judged fluency ≥ 1.0, and unsteered text does not either (0.41–0.46).

**What the data do show:**
- **Specific concept injection.** For all three neurons the clamped concept score is higher than for every one of the 80 random neurons clamped at the same relative dose (p = 1/81, Holm 0.037). For L30 N2121 and L32 N2669 the primary HM metric also beats the null after Holm correction (p = 0.037). Clamping a random neuron in the same layer does not produce the concept, so the effect is specific to the chosen neuron. A judge-free count of the target token agrees (see "Judge-free robustness check" below): 1.98, 6.91 and 4.79 per 40 tokens, against at most 1.0 for any random neuron.
- **Neuron steering has a higher HM than prompting** for all three poles: 0.38 vs 0.15, 0.09 vs 0.02, 0.27 vs 0.07. Post hoc, the paired clamp − prompting difference is +0.23 [0.09, 0.39] for "as" and +0.20 [0.08, 0.33] for "set", but +0.07 [−0.01, 0.16] for "[" (not significant). The same holds after Holm correction over the 6 clamp-vs-baseline comparisons. The prompting prefix ("the word 'as':") raises the concept only modestly (0.24 → 0.48 for "as"). This baseline is a bare prefix on a base model that is not instruction-tuned, much weaker than AxBench's instruction prompts.
- **DiffMean at the same residual-norm budget is far too strong.** It drives distinct-2 to 0.03–0.17 ("as as as …", "set set set …", "[X['['X…"), so concept is maximal but HM = 0. The neuron's write direction, at the same norm, keeps text far more intact (distinct-2 0.75–0.85). That suggests the neuron direction is much better aligned with "fluent use of the concept" than the DiffMean direction, at least at this budget. This is post hoc: the budget was not tuned for DiffMean.
- **Add ≈ clamp** in concept score, as expected, because these neurons sit at about 0 outside their firings. Their HM differs for "set" (0.18 vs 0.27) through the fluency judge.

Examples (clamp, seed 0, prompt set B):
- **L30 N2121:** "…let your children know you are **as** proud of them **as** they are. In addition, the holiday is also a great time…" (concept 2, fluency 1)
- **L32 N2669:** "**Sets** of five Wistar-plus-Lewis adult mice was **set** up for five successive **sets** of experiments…" (concept 2, fluency 0)
- **L28 N953:** "…last day of the month of […]]]…"

## Post-hoc low-dose sweep (0.25×, 0.5×, 1× pole max; same prompts and concepts)
Averaged over the 39 poles:

| Dose | Concept | Fluency | distinct-2 | HM |
|---|---|---|---|---|
| unsteered | 0.08 | 0.60 | 0.91 | 0.050 |
| 0.25× | 0.52 | 0.42 | 0.83 | 0.092 |
| 0.5× | 0.75 | 0.31 | 0.68 | 0.080 |
| 1× | 0.80 | 0.19 | 0.51 | 0.054 |

The low-dose regime injects most of the concept while keeping most of the judged fluency (10 prompts per point; the fluency judge is unreliable on fluent text). The top three poles by HM gain were frozen as H2b (amendment 7):

| Pole | Concept | Dose | HM | Concept score | Fluency | distinct-2 |
|---|---|---|---|---|---|---|
| L30 N815 | "from" | 0.5× | 0.67 | 1.6 | 0.6 | 0.93 |
| L30 N2121 | "as" | 0.5× | 0.60 | 1.6 | 0.8 | 0.92 |
| L37 N71 | "first" | 0.25× | 0.53 | 1.6 | 0.5 | 0.82 |

## H2b confirmation (pre-registered after the sweep; fresh prompt set C)
| Pole · dose | Condition | HM [95% CI] | Concept | Fluency | distinct-2 | NLL |
|---|---|---|---|---|---|---|
| **L30 N815 · 0.5× · "from"** | unsteered | 0.057 [0.010, 0.117] | 0.16 | 0.48 | 0.93 | 1.07 |
| | **clamp** | **0.460 [0.310, 0.620]** | **1.53** | 0.45 | 0.90 | 1.58 |
| | add | 0.447 [0.303, 0.597] | 1.53 | 0.44 | 0.90 | 1.56 |
| | DiffMean | 0.157 [0.063, 0.273] | 1.48 | 0.17 | 0.67 | 1.62 |
| | prompting | 0.137 [0.057, 0.227] | 0.45 | 0.54 | 0.93 | 1.28 |
| | 80 random neurons | mean 0.009, max 0.133 | mean 0.05, max 0.40 | | | |
| **L30 N2121 · 0.5× · "as"** | unsteered | 0.077 [0.020, 0.150] | 0.19 | 0.41 | 0.93 | 1.13 |
| | **clamp** | **0.240 [0.140, 0.350]** | **1.17** | 0.32 | 0.92 | 1.41 |
| | add | 0.270 [0.160, 0.390] | 1.16 | 0.33 | 0.92 | 1.41 |
| | DiffMean | 0.187 [0.097, 0.290] | 0.94 | 0.34 | 0.85 | 1.50 |
| | prompting | 0.097 [0.037, 0.170] | 0.47 | 0.43 | 0.91 | 1.28 |
| | 80 random neurons | mean 0.013, max 0.133 | mean 0.11, max 0.30 | | | |
| **L37 N71 · 0.25× · "first"** | unsteered | 0.000 | 0.10 | 0.42 | 0.93 | 1.18 |
| | **clamp** | **0.353 [0.227, 0.487]** | **1.62** | 0.42 | 0.87 | 1.64 |
| | add | 0.363 [0.217, 0.527] | 1.65 | 0.43 | 0.88 | 1.64 |
| | DiffMean | 0.150 [0.067, 0.243] | 1.23 | 0.23 | 0.84 | 1.69 |
| | prompting | 0.107 [0.040, 0.187] | 0.44 | 0.46 | 0.93 | 1.39 |
| | 80 random neurons | mean 0.000, max 0.000 | mean 0.00, max 0.20 | | | |

| Pole | Clamp − unsteered HM [95% CI] | p vs null (HM) | Holm p | Judged fluency ≥ unsteered − 0.2? (point estimates) | distinct-2 ≥ 0.9 × unsteered? (point estimates) | **H2b** (pre-registered rule) |
|---|---|---|---|---|---|---|
| L30 N815 | +0.403 [0.253, 0.560] | 1/81 | **0.037** | ✓ (0.45 ≥ 0.28) | ✓ (0.90 ≥ 0.84) | **supported** |
| L30 N2121 | +0.163 [0.070, 0.263] | 1/81 | **0.037** | ✓ (0.32 ≥ 0.21) | ✓ (0.92 ≥ 0.84) | **supported** |
| L37 N71 | +0.353 [0.227, 0.487] | 1/81 | **0.037** | ✓ (0.42 ≥ 0.22) | ✓ (0.87 ≥ 0.84) | **supported** |

**Verdict: H2b is supported for all three neurons by its pre-registered rule.** At a quarter to half of its maximum activation, each single neuron makes the model use its word:
- far more than unsteered text (concept 0.10–0.19 → 1.17–1.62);
- more than prompting with the concept (0.44–0.47);
- more than any of 80 random neurons pushed equally hard.

Even these doses are several times the neuron's typical strong firing, applied at every token. For L30 N2121, 0.5× max is about 19× the median of its 200 strongest held-out firings.

**Fluency: met on point estimates, not established** (post hoc, `h2b.json` → `posthoc_audit`). The pre-registered margins hold for the point estimates: judged fluency is within 0.09 of unsteered, and distinct-2 within 0.07. Tested on confidence intervals, as standard non-inferiority analysis requires ([Piaggio et al. 2012](https://doi.org/10.1001/jama.2012.87802)), the result is weaker:

| Pole | Fluency clamp − unsteered [95% CI] | distinct-2 clamp − 0.9 × unsteered [95% CI] | NLL clamp − unsteered [95% CI] (nats/token) | CI-based non-inferiority |
|---|---|---|---|---|
| L30 N815 "from" | −0.03 [−0.20, 0.13] | +0.065 [0.032, 0.091] | +0.51 [0.37, 0.67] | ✓ (fluency bound exactly at −0.2) |
| L30 N2121 "as" | −0.09 [−0.22, 0.04] | +0.083 [0.061, 0.103] | +0.28 [0.16, 0.42] | ✗ (fluency) |
| L37 N71 "first" | 0.00 [−0.17, 0.17] | +0.029 [−0.004, 0.057] | +0.46 [0.26, 0.68] | ✗ (distinct-2, narrowly) |

Three reasons for caution:

- **Unreliable judge.** The fluency judge is near its floor on natural text and unvalidated in this dose regime (see "Judge validity" above). That floor makes a fixed 0.2-point margin easy to meet.
- **NLL.** NLL under the unsteered model, a pre-registered secondary metric, rises for every pole. Part of the rise is mechanical, because each injected word is itself improbable under the unsteered model. The stored mean NLLs cannot separate that from a real fluency cost.
- **Conclusion.** The concept injection is established; fluency preservation is not.

At the same residual-norm budget, DiffMean has a lower HM for every pole, with much less fluent text for "from" (distinct-2 0.67). By the judge-free count, though, DiffMean writes "from" and "as" *more* often than the neuron (below). Post-hoc paired HM differences, with Holm-adjusted bootstrap p over the 6 comparisons:

| Pole | Clamp − prompting | Clamp − DiffMean |
|---|---|---|
| L30 N815 "from" | +0.32 [0.17, 0.48], Holm p < 0.001 | +0.30 [0.13, 0.48], Holm p = 0.003 |
| L30 N2121 "as" | +0.14 [0.02, 0.27], Holm p = 0.039 | +0.05 [−0.06, 0.17], Holm p = 0.37 (n.s.) |
| L37 N71 "first" | +0.25 [0.11, 0.38], Holm p = 0.002 | +0.20 [0.06, 0.35], Holm p = 0.022 |

The dose was chosen to maximise the neuron's HM on prompt set A. Neither baseline was tuned: DiffMean ran at this single norm, and prompting is a bare prefix. Additive steering and clamping are equivalent within their CIs.

## Judge-free robustness check (post hoc, 2026-11-01 audit)
Every steering concept here is one token, so we also counted the pole's top promoted token in each 40-token continuation: whole words, case-insensitive (`metrics.count_word`, `posthoc_audit.word_count`).

| Pole · dose (stage) | Token | Unsteered | Clamp | Clamp − unsteered [95% CI] | Prompting | DiffMean | Null mean / max |
|---|---|---|---|---|---|---|---|
| L30 N2121 · 2× (H2) | as | 0.07 | 1.98 | +1.91 [1.57, 2.25] | 0.27 | 37.2 | 0.09 / 0.40 |
| L28 N953 · 5× (H2) | [ | 0.01 | 6.91 | +6.90 [4.94, 9.09] | 0.10 | 18.1 | 0.02 / 1.00 |
| L32 N2669 · 2× (H2) | set | 0.00 | 4.79 | +4.79 [4.18, 5.39] | 0.30 | 39.0 | 0.01 / 0.20 |
| L30 N815 · 0.5× (H2b) | from | 0.08 | 2.18 | +2.10 [1.61, 2.72] | 0.27 | 10.3 | 0.03 / 0.20 |
| L30 N2121 · 0.5× (H2b) | as | 0.10 | 1.02 | +0.92 [0.70, 1.15] | 0.37 | 2.96 | 0.10 / 0.30 |
| L37 N71 · 0.25× (H2b) | first | 0.04 | 2.27 | +2.23 [1.74, 2.78] | 0.27 | 1.15 | 0.00 / 0.10 |

- **Count vs the null.** All six targets exceed all 80 null neurons (p = 1/81, Holm 0.037). They also do so like-for-like, when the target is scored only on the null's own 10 prompts and seed (p = 1/81 for all six). Specific injection therefore does not depend on the LLM judge.
- **HM like-for-like.** The pre-registered HM test is weaker in this form for two poles. H2 "[" has HM 0.00 on those 10 generations (p = 1.0), and H2b "as" has 0.10 (p = 0.136, Holm 0.136). The registered design (a 100-generation target mean against 10-generation null means) is conservative in expectation, so the verdicts stand. The H2b "as" HM significance depends on that design; its word-count significance does not.

Examples (clamp, seed 0, prompt set C):
- **N815:** "…have been evaluated **from** the analysis of the stress values and the mechanical properties of the bone."
- **N2121:** "…Redistribution and use in source **as** well **as** in binary form, **as** either…"
- **N71:** "…have been recognized as the **first** step in understanding the biomechanical behavior of the bone. The **first** step…"

**Caveats.**

- **Selection.** Doses and poles were chosen post hoc on prompt set A and then confirmed on disjoint prompts; the confirmation is the evidence, not the sweep.
- **Lexical concepts.** The concepts are single function words, so this is lexical rather than topical steering.
- **Fluency judge.** It is unreliable on fluent text (κ = 0.34 [0.17, 0.53]), which is why distinct-2 is part of the rule; see the fluency section above.
- **What the null shows.** The poles were selected for write vectors with a strong direct effect on one token. Beating random neurons therefore shows that the effect is not a generic perturbation. It does not show that the neuron basis is privileged for steering, compared with other directions that promote the same token.

## Learnings
1. **Judge the right concept.** We first judged steering against the *input-side* detection label. The judge then answered a mechanism question ("is there a token before 'all'?") and gave 0 even to "all all all". Steering must be judged against what the neuron *writes*. We caught this in the first judge calls, 9 according to the archived log (amendment 4 says 7; erratum in amendment 8), and recorded it as amendment 4.
2. **Pre-registered doses were too strong.** Doses were chosen as in SAE-feature clamping (multiples of max), but Mamba neurons are extremely heavy-tailed (median kurtosis 354). For the H2 poles, the median of the 200 strongest held-out firings is 9–39× below the discovery maximum, so 2× max is already far outside the natural range.
3. **The original null could never pass Holm.** A 20-neuron random null caps the empirical p at 1/21, which Holm over 3 poles can never bring below 0.05. We found this by code review before any confirmation data existed and fixed it to an 80-neuron null (amendment 5).
4. **Faster judging.** Asking for one-sentence explanations and using a 2k context cut judge latency from about 20 s to about 1.2 s per call. Parallel Ollama requests did not help on this GPU (1.55 vs 1.35 s/call).
5. **Neutral prompts.** L32 N2669 had only 47 of 400 openings below its 90th-percentile activation (its q90 is −0.015), so its confirmation uses 47 prompts instead of 50.
6. **Test non-inferiority on CIs, and validate the judge where it is used** (2026-11-01 independent audit).
   - The H2b fluency rule compared point estimates with a margin. With 100 generations and a noisy 0–2 judge, the CIs reach past that margin for 2 of 3 poles.
   - The fluency judge had been validated mainly on degenerate high-dose text, where it is trivially right. On fluent text it is not reliable.
   - A judge-free word count was the cheapest way to check the concept claim independently, and it confirmed it.
