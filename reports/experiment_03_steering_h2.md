# Experiment 3: turning single neurons up (steering screen, H2, post-hoc dose sweep, H2b)

**Status:** screen, low-dose sweep and H2 done; H2b in progress.
**Code:** `steer.py`, `judge.py`, `phases/steering.py`.
**Results:** `results/mamba-1.4b-hf/{screen,screen_low,confirm_plan,h2,steer_validation}.json`.

## Protocol
- **Neutral prompts.** The first 16 tokens of held-out documents.
- **Interventions.** These act at every position (prompt included) during sampled generation: 40 new tokens, temperature 0.7, top-p 0.95, seed 0.
- **Judge.** The judge (`qwen3.5:9b`) uses the AxBench concept and fluency templates, scoring each from 0 to 2. Each generation gets their harmonic mean (HM).
- **Steering concept.** The concept is an output-side "steering concept" written once per pole from its label and promoted tokens (amendment 4).

## Screen (pre-registered, exploratory): 39 clean candidate poles × {2, 5, 10, 20}× pole max × 10 prompts
Averaged over all 39 poles. The unsteered row repeats the same 10 unsteered texts, judged against each pole's concept.

| Dose | Concept | P(concept ≥ 1) | Fluency (judge) | HM | NLL (unsteered model) | distinct-2 |
|---|---|---|---|---|---|---|
| unsteered | 0.08 | 0.06 | 0.60 | 0.050 | 1.14 | 0.91 |
| 2× | **0.82** | **0.48** | 0.11 | 0.022 | 1.80 | 0.39 |
| 5× | 0.67 | 0.40 | 0.05 | 0.008 | 2.10 | 0.27 |
| 10× | 0.76 | 0.45 | 0.02 | 0.000 | 2.14 | 0.21 |
| 20× | 0.70 | 0.41 | 0.01 | 0.000 | 2.02 | 0.17 |

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

## Judge validity (40 blind items, rated by Claude Code)
| Judge metric | Cohen's κ | Exact agreement | Mean, LLM vs manual |
|---|---|---|---|
| Concept | **0.79** | 0.875 | 0.85 vs 0.98 |
| Fluency | **0.34** | 0.70 | 0.25 vs 0.65 |

The concept judge is reliable. The fluency judge is harsh: it often gives 0–1 to fluent base-model text, which explains why unsteered text scores only 0.60. The pre-registered H2 fluency criterion (≥ 1.0) is therefore close to unreachable even without steering. We report NLL and distinct-2 as objective fluency proxies. NLL under the unsteered model is a weak proxy for degeneration, because repetitive text becomes predictable; distinct-2 is better.

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
- **Specific concept injection.** For all three neurons the clamped concept score is higher than for every one of the 80 random neurons clamped at the same relative dose (p = 1/81). For L30 N2121 and L32 N2669 the primary HM metric also beats the null after Holm correction (p = 0.037). Clamping a random neuron in the same layer does not produce the concept, so the effect is specific to the chosen neuron.
- **Neuron steering beats prompting** on HM for all three poles: 0.38 vs 0.15, 0.09 vs 0.02, 0.27 vs 0.07. The prompting prefix ("the word 'as':") barely raises the concept at all.
- **DiffMean at the same residual-norm budget is far too strong.** It drives distinct-2 to 0.03–0.17 ("as as as …", "set set set …", "[X['['X…"), so concept is maximal but HM = 0. The neuron's write direction, at the same norm, keeps text far more intact (distinct-2 0.75–0.85). That suggests the neuron direction is much better aligned with "fluent use of the concept" than the DiffMean direction, at least at this budget. This is post hoc: the budget was not tuned for DiffMean.
- **Add ≈ clamp**, as expected, because these neurons sit at about 0 outside their firings.

Examples (clamp, seed 0, prompt set B):
- **L30 N2121:** "…let your children know you are **as** proud of them **as** they are. In addition, the holiday is also a great time…" (concept 2, fluency 1)
- **L32 N2669:** "**Sets** of five Wistar-plus-Lewis adult mice was **set** up for five successive **sets** of experiments…" (concept 2, fluency 0)
- **L28 N953:** "…last day of the month of […]]]…"

## Post-hoc low-dose sweep (0.25×, 0.5×, 1× pole max; same prompts and concepts)
LOW_PLACEHOLDER

## Learnings
1. **Judge the right concept.** We first judged steering against the *input-side* detection label. The judge then answered a mechanism question ("is there a token before 'all'?") and gave 0 even to "all all all". Steering must be judged against what the neuron *writes*. We caught this after 7 judge calls and recorded it as amendment 4.
2. **Pre-registered doses were too strong.** Doses were chosen as in SAE-feature clamping (multiples of max), but Mamba neurons are extremely heavy-tailed (median kurtosis 354). For the H2 poles, the median of the 200 strongest held-out firings is 9–38× below the discovery maximum, so 2× max is already far outside the natural range.
3. **The original null could never pass Holm.** A 20-neuron random null caps the empirical p at 1/21, which Holm over 3 poles can never bring below 0.05. We found this by code review before any confirmation data existed and fixed it to an 80-neuron null (amendment 5).
4. **Faster judging.** Asking for one-sentence explanations and using a 2k context cut judge latency from about 20 s to about 1.2 s per call. Parallel Ollama requests did not help on this GPU (1.55 vs 1.35 s/call).
5. **Neutral prompts.** L32 N2669 had only 47 of 400 openings below its 90th-percentile activation (its q90 is −0.015), so its confirmation uses 47 prompts instead of 50.
