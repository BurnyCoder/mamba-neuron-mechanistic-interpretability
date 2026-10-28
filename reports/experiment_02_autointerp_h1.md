# Experiment 2: automated labels and the privileged-basis test (H1)

**Status:** done (2026-10-28). Code: `autointerp.py`, `phases/interp.py`, `scripts/compare_labelers.py`, `scripts/judge_validation.py`.
**Results:** `results/mamba-1.4b-hf/{autointerp,h1,judge_validation}.json`, `results/labeler_comparison_smoke.json`.

## Protocol (pre-registered, see `reports/preregistration.md`)
- **Labeller.** A local `qwen3.5:9b` (Ollama, temperature 0, thinking off, JSON output) labels each pole. Its input:
  - the top-20 discovery firings, one per window, as 33-token snippets with `<<highlights>>`;
  - the 5 most common peak and next tokens;
  - the 10 tokens the pole promotes.
- **Scorer.** The same model classifies 10 held-out top firings and 10 random held-out snippets, unhighlighted and shuffled. The score is balanced accuracy (BA), and "clean" means BA ≥ 0.8.
- **Audit sample.** 2 random neurons and 2 random directions per layer, with both poles: 192 neuron poles and 192 direction poles.

## Choosing the labeller (smoke data only, mamba-130m)
| Labeller | Mean BA (20 poles) | Clean | Time |
|---|---|---|---|
| qwen3.5:4b, plain prompt | 0.586 | 2 | 60 s |
| qwen3.5:9b, plain prompt | 0.594 | 3 | 86 s |
| qwen3.5:4b, + promoted tokens + token counts | 0.611 | 2 | 44 s |
| **qwen3.5:9b, + promoted tokens + token counts** | **0.662** | **5** | 83 s |

The first two rows ran with the promoted-token block accidentally missing from the prompt (a failed code replacement), which made the 4B labeller miss "fires before *that*"-type neurons. With the token-count hints both models improved, and 9B was chosen (amendment 3).

## H1 result: **not supported**
| | Neuron poles | Random-direction poles |
|---|---|---|
| Clean (BA ≥ 0.8) | 29 / 192 = **15.1%** (Wilson 95% CI 10.7–20.9%) | 26 / 192 = **13.5%** (9.4–19.1%) |
| Mean BA (95% CI, unit-cluster bootstrap) | 0.617 (0.597–0.637) | 0.583 (0.560–0.607) |
| Invalid LLM replies | 0 | 0 |

The difference in clean rate is **+1.6 points (95% unit-bootstrap CI −5.7 to +8.3), one-sided Fisher p = 0.39**. The pre-registered criterion fails, so we cannot claim that neurons are more *interpretable by this pipeline* than random directions.

**Post hoc, not pre-registered:**
- **Mean BA.** It is 0.03 higher for neurons, with barely overlapping CIs. Treat this as suggestive at most.
- **By depth.** Neurons stay flat (0.63 / 0.60 / 0.62 mean BA for layers 0–15 / 16–31 / 32–47). Random directions degrade (0.63 / 0.56 / 0.56). The neuron advantage appears only in the middle and late layers.
- **By pole.** Neuron positive poles: 0.625 mean BA, 18 clean. Negative poles: 0.608, 11 clean.

**Judge validity (pre-registered check): fails the bar.**
- **Agreement.** On 40 randomly sampled detection decisions, rated blind by Claude Code (the agent that wrote this project, *not* a human), agreement with the LLM is κ = **0.55**, below the pre-registered 0.6.
- **Accuracy vs ground truth.** The manual ratings got 0.78 and the LLM 0.70.
- **Why the bar fails.** Many labels are vague ("tokens containing the letter 'i'"), and both raters then call almost any snippet a match.

**Interpretation.** The null H1 result coexists with a huge difference in activation *shape* (experiment 1: median kurtosis 354 vs 6.9). The most likely explanation is a measurement floor. A 9B local labeller gives about 15% clean poles for anything, far below the ~90% that Goodfire reports for Qwen3-32B with expert human auditing ([Goodfire](https://already-clean.goodfire.com/)). Its judge only moderately agrees with careful rating. A stronger labeller, such as a frontier model or human audit, is needed before concluding anything about Mamba neuron interpretability. This is a measurement limitation, not evidence against clean neurons.

**Candidate streams** (not part of H1):
- **Formula candidates:** 30 of 60 clean, mean BA **0.768**, much higher than random neurons (0.617).
- **Domain neurons:** 9 of 24 clean, mean BA 0.633.

So the pre-registered selection formula does find neurons whose behaviour a 9B model can describe and predict. Examples:

| Neuron | Label | BA |
|---|---|---|
| L37 N1812 | "after" in "after all" | 0.95 |
| L34 N1725 | "neither … nor" | 0.95 |
| L38 N918 | "make" before "sense" | 0.95 |
| L43 N2475 | before "each other / one another" | 1.0 |
| L34 N269 | before "respectively" | 1.0 |
| L36 N109 | inside email addresses | 0.95 |
| L45 N2113 | the colon in timestamps | 1.0 |
