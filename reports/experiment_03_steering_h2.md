# Experiment 3: turning single neurons up (steering screen, H2, post-hoc dose sweep, H2b)

**Status:** screen and low-dose sweep done; H2 confirmation judging in progress.
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
H2_PLACEHOLDER

## Post-hoc low-dose sweep (0.25×, 0.5×, 1× pole max; same prompts and concepts)
LOW_PLACEHOLDER

## Learnings
1. **Judge the right concept.** We first judged steering against the *input-side* detection label. The judge then answered a mechanism question ("is there a token before 'all'?") and gave 0 even to "all all all". Steering must be judged against what the neuron *writes*. We caught this after 7 judge calls and recorded it as amendment 4.
2. **Pre-registered doses were too strong.** Doses were chosen as in SAE-feature clamping (multiples of max), but Mamba neurons are extremely heavy-tailed (median kurtosis 354). For the H2 poles, the median of the 200 strongest held-out firings is 9–38× below the discovery maximum, so 2× max is already far outside the natural range.
3. **The original null could never pass Holm.** A 20-neuron random null caps the empirical p at 1/21, which Holm over 3 poles can never bring below 0.05. We found this by code review before any confirmation data existed and fixed it to an 80-neuron null (amendment 5).
4. **Faster judging.** Asking for one-sentence explanations and using a 2k context cut judge latency from about 20 s to about 1.2 s per call. Parallel Ollama requests did not help on this GPU (1.55 vs 1.35 s/call).
5. **Neutral prompts.** L32 N2669 had only 47 of 400 openings below its 90th-percentile activation (its q90 is −0.015), so its confirmation uses 47 prompts instead of 50.
