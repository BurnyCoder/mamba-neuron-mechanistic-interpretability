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

The first two rows ran with the promoted-token block accidentally missing from the prompt (a failed code replacement), which made the 4B labeller miss "fires before *that*"-type neurons. Only the last two rows are saved in `results/labeler_comparison_smoke.json`; the first two come from the development run's console output. With the token-count hints both models improved, and 9B was chosen (amendment 3).

## H1 result: **not supported**
| | Neuron poles | Random-direction poles |
|---|---|---|
| Clean (BA ≥ 0.8) | 29 / 192 = **15.1%** (Wilson 95% CI 10.7–20.9%) | 26 / 192 = **13.5%** (9.4–19.1%) |
| Mean BA (95% CI, unit-cluster bootstrap) | 0.617 (0.597–0.637) | 0.583 (0.560–0.607) |
| Invalid LLM replies | 0 | 0 |

The difference in clean rate is **+1.6 points (95% unit-bootstrap CI −5.7 to +8.3), one-sided Fisher p = 0.39**. The pre-registered criterion fails, so we cannot claim that neurons are more *interpretable by this pipeline* than random directions.

The CI rules out a clean-rate advantage larger than 8.3 points under this instrument. It does not show that there is no difference.

**Post hoc, not pre-registered** (added by the 2026-11-01 independent audit, amendment 9; `h1.json` → `posthoc`):
- **Mean BA.** It is higher for neurons by **+0.034 (unit-cluster bootstrap 95% CI 0.003 to 0.065)**, one-sided Mann-Whitney p = 0.0015 on per-unit means. The 0.8 threshold dichotomises a noisy 20-snippet score and hides this small difference.
- **Other thresholds.** The clean rate (neurons vs directions) is 55.7% vs 40.1% at ≥ 0.6, 29.2% vs 22.4% at ≥ 0.7, 15.1% vs 13.5% at ≥ 0.8, and 4.2% vs 7.3% at ≥ 0.9.
- **By depth.** Neurons stay flat (0.63 / 0.60 / 0.62 mean BA for layers 0–15 / 16–31 / 32–47). Random directions degrade (0.63 / 0.56 / 0.56). The neuron advantage appears only in the middle and late layers.
- **By pole.** Neuron positive poles: 0.625 mean BA, 18 clean. Negative poles: 0.608, 11 clean.

**Judge validity (pre-registered check): fails the bar.**
- **Agreement.** On 40 randomly sampled detection decisions, agreement with the LLM is κ = **0.55** (item-bootstrap 95% CI 0.22–0.82), below the pre-registered 0.6. The reference ratings were made by Claude Code, the agent that wrote this project, *not* a human. It worked from an item file without the answers, but it had file access to them, so blinding rests on file separation only. The pre-registration said these ratings would be made by hand (deviation declared in amendment 9). With n = 40 the CI is so wide that the pass/fail outcome itself is uncertain.
- **Consequence.** By the pre-registration the judge was not accepted, so H1 is best read as inconclusive rather than falsified.
- **Accuracy vs ground truth.** The reference ratings got 0.78 and the LLM 0.70.
- **Why the bar fails.** Many labels are vague ("tokens containing the letter 'i'"), and both raters then call almost any snippet a match.

**Interpretation.** The null H1 result coexists with a huge difference in activation *shape* (experiment 1: median kurtosis 354 vs 6.9). The most likely explanation is a measurement floor. A 9B local labeller gives about 15% clean poles for anything. That is far below the roughly 90% that Goodfire estimates for Qwen3-32B from preliminary audits ([Goodfire](https://already-clean.goodfire.com/)); the essay does not describe that audit's procedure. Its judge only moderately agrees with careful rating. A stronger labeller, such as a frontier model or human audit, is needed before concluding anything about Mamba neuron interpretability. This is a measurement limitation, not evidence against clean neurons.

**Candidate streams** (not part of H1):
- **Formula candidates:** 30 of 60 clean, mean BA **0.768**, much higher than random neurons (0.617).
- **Domain neurons:** 9 of 24 clean, mean BA 0.633.

So the pre-registered selection formula does find neurons whose behaviour a 9B model can describe and predict. Examples:

| Neuron | Label | BA |
|---|---|---|
| L37 N1812 | the token before "all" (after all, at all, of all) | 0.95 |
| L34 N1725 | "neither … nor" | 0.95 |
| L38 N918 | "make" before "sense" | 0.95 |
| L43 N2475 | before "each other / one another" | 1.0 |
| L34 N269 | before "respectively" | 1.0 |
| L36 N109 | inside email addresses | 0.95 |
| L45 N2113 | the colon in timestamps | 1.0 |
