# Independent audit of the methodology and claims (2026-11-01)

**Status:** done. **Pre-registration:** amendment 9. **Code:** `metrics.py` (`bootstrap_p`, `kappa_ci`, `count_word`, `mannwhitney_greater`), `phases/steering.py::_posthoc_audit`, `phases/interp.py::_h1_posthoc`, `scripts/judge_validation.py`. **Results:** the `posthoc_audit` block in `results/mamba-1.4b-hf/h2.json` and `h2b.json`, `posthoc` in `h1.json`, and `kappa_ci95` / `by_level` in `judge_validation.json` and `steer_validation.json`.

## Scope and method
This audit was done independently of the earlier fact-check (amendment 8). It asked whether any methodological error invalidates the findings, and whether the claims match the evidence.

- **Code read.** The pre-registration, paper, README, reports and all analysis code: the hook site, steering conditions, judge, auto-interp, mechanism, data split and statistics.
- **Statistics recomputed.** Every headline statistic was recomputed from the committed results files.
- **Timing checked.** Amendment timing was checked against commit times and the archived run logs.
- **Citations spot-checked:** Goodfire's 90% estimate, Liu 2026, Sunku Mohan et al. 2026, Piaggio et al. 2012 and Efron & Tibshirani 1993.

New analyses use only stored data (judged generations, labels, validation ratings). They are post hoc. To keep the pre-registered values intact, the results files were diffed before and after: every pre-registered field is unchanged, and only new keys were added.

## Bottom line
**No error invalidates a pre-registered verdict.** H1 is not supported, H2 is not supported, H2b is supported by its registered rule, and H3 is as reported per pole. The central positive finding, that single neurons inject their concept specifically, also holds without the LLM judge.

Several *interpretations* overstated the evidence: fluency non-inferiority, H1 as "falsified / no better", the gate analysis, "Mamba-specific", and the truncation and necessity readings. They are corrected in the paper, README and reports.

## What holds
- **Timing.**
  - Amendment 6 (12:30:31) precedes `confirm` (12:32:08).
  - Amendment 7 (14:48:04) precedes every H2 judge call. A `confirm_judge` run that started at 14:46:55 logged no call before it was stopped, and the rerun began at 14:49:07 (`reports/logs/20261028-123208_all.log.gz`, `20261028-144907_confirm_judge.log.gz`).
  - `confirm_h2b` ran at 16:50.
  - The hand-written `confirm_plan_h2b.json` matches the top 3 that `screen_low_judge` logged.
- **Data separation.** Prompt sets A, B and C are disjoint slices of one seeded permutation of held-out document openings (`steer.neutral_prompts`, skip 0 / 10 / 410).
- **Interventions.**
  - The neuron hook is a forward pre-hook on `out_proj`, and `z` is the second half of `in_proj` (HF `modeling_mamba.py`).
  - DiffMean is matched to the residual norm |value|·‖w_i‖.
  - All conditions share sampling seeds, so comparisons are paired.
- **Statistics.** The prompt-cluster bootstrap, the empirical p-values (Phipson & Smyth) and Holm correction are implemented correctly. The table numbers in the paper reproduce from the results files.
- **Baseline comparisons.** They survive multiplicity correction. With Holm over the 6 clamp-vs-baseline comparisons of each confirmation, 10 of 12 remain significant. The exceptions are H2 "[" vs prompting and H2b "as" vs DiffMean, which were already reported as not significant.
- **Concept injection without the judge.** Counting each pole's top promoted token per 40-token continuation (Table below), every target beats all 80 random neurons (p = 1/81, Holm 0.037). This also holds like-for-like on the null's own 10 prompts and seed (p = 1/81 for all six).

| Pole · dose (stage) | Token | Unsteered | Clamp | Clamp − unsteered [95% CI] | Prompting | DiffMean | Null mean / max |
|---|---|---|---|---|---|---|---|
| L30 N2121 · 2× (H2) | as | 0.07 | 1.98 | +1.91 [1.57, 2.25] | 0.27 | 37.2 | 0.09 / 0.40 |
| L28 N953 · 5× (H2) | [ | 0.01 | 6.91 | +6.90 [4.94, 9.09] | 0.10 | 18.1 | 0.02 / 1.00 |
| L32 N2669 · 2× (H2) | set | 0.00 | 4.79 | +4.79 [4.18, 5.39] | 0.30 | 39.0 | 0.01 / 0.20 |
| L30 N815 · 0.5× (H2b) | from | 0.08 | 2.18 | +2.10 [1.61, 2.72] | 0.27 | 10.3 | 0.03 / 0.20 |
| L30 N2121 · 0.5× (H2b) | as | 0.10 | 1.02 | +0.92 [0.70, 1.15] | 0.37 | 2.96 | 0.10 / 0.30 |
| L37 N71 · 0.25× (H2b) | first | 0.04 | 2.27 | +2.23 [1.74, 2.78] | 0.27 | 1.15 | 0.00 / 0.10 |

## Findings and fixes (most severe first)

| # | Claim | Problem and evidence | Fix |
|---|---|---|---|
| M1 | H2b "keeps fluency non-inferior" | **Point-estimate rule.** The registered rule compares point estimates, whereas non-inferiority is normally shown by the CI ([Piaggio et al. 2012](https://doi.org/10.1001/jama.2012.87802)). Judged-fluency differences: −0.03 [−0.20, 0.13] ("from"), −0.09 [−0.22, 0.04] ("as"), 0.00 [−0.17, 0.17] ("first"). The distinct-2 margin (clamp − 0.9 × unsteered) for "first" is +0.029 [−0.004, 0.057]. The CI-based version therefore passes only for "from", exactly at the margin. **Instrument.** The fluency judge rates the 9 unsteered validation items 0.67 where the reference gives 1.89. Its κ (0.34) comes mostly from degenerate text, and no validation item is from the 0.25–1× regime. **NLL.** Under the unsteered model it rises by +0.51 [0.37, 0.67], +0.28 [0.16, 0.42] and +0.46 [0.26, 0.68] nats/token. Partly mechanical, but this pre-registered secondary metric was not reported before. | Verdict kept ("supported by the pre-registered rule"). The claim is now "fluency preservation not established". The CIs, NLL and CI-based verdict are in Table 5 of the paper and in report 3. |
| M2 | H1 "no better", "no separation", "a falsified hypothesis" | **Continuous measure.** Mean detection BA is higher for neurons by +0.034 [0.003, 0.065] (one-sided Mann-Whitney p = 0.0015 on unit means). The 0.8 threshold hides this; at ≥ 0.6 the rates are 55.7% vs 40.1%. **Judge gate.** The detection judge failed its pre-registered validity gate (κ 0.55 [0.22, 0.82]). **Absence of evidence.** A CI that includes 0 does not show no difference. | "Not supported, inconclusive"; the CI bounds the clean-rate advantage at ≤ 8.3 points; the post-hoc mean-BA result is reported. |
| M3 | "Gate open, so the SSM factor decides the pole"; "Mamba-specific" | **Expected result.** SiLU(z) ≥ −0.28, so a negative gate caps \|a\| at 0.28·\|y\|. An open gate at the 200 strongest firings of positive poles is therefore nearly forced, in SwiGLU too. **Selectivity untested.** Only firings were examined, so which factor makes a neuron *selective* was not tested. **No comparison.** Neither observation was compared with a transformer. | Reworded to "the sign of these firings comes from y". The section is renamed "Mamba observations (not compared with transformers)". |
| m4 | Judge validity | **Wide CIs:** detection κ 0.55 [0.22, 0.82], concept 0.79 [0.62, 0.95], fluency 0.34 [0.17, 0.53]. **Rater.** The reference rater is Claude Code, the agent that built the pipeline, not a human as pre-registered. Blinding rests on file separation only. | CIs reported; the deviation is declared in amendment 9; "blind manual ratings" is replaced by "reference ratings made by Claude Code". |
| m5 | "Deep neurons need longer context" | **Depth vs type.** No screened pole lies in layers 2–21, and the early control is 4 domain-stream poles, so depth is confounded with neuron type. **Short input.** Truncation removes context *and* shortens the input, and a short-input effect at deep layers is not ruled out. **Selection.** Firings were chosen as maxima under the full context. | Hedged to "need longer inputs; not separated from a short-input effect or neuron type". |
| m6 | "Gentle doses"; "sufficient but not necessary" | **Dose.** 0.25–0.5× max is still several times the typical strong firing: for L30 N2121, 0.5× max ≈ 19× the median of its 200 strongest held-out firings, at every token. **Other explanations.** The steering/ablation asymmetry may also come from the dose mismatch or from log p saturation at confident positions. **H3.** Two of three pass at uncorrected p = 1/21 and none after Holm (0.14). The L28 N953 effect (0.0018 nats) is smaller than the spread of random-neuron ablation effects (−0.0055 to +0.0069). | "Lower doses"; the alternative explanations and the Holm values are now stated. |
| m7 | "Higher HM than prompting and DiffMean" | **Prompting.** The baseline is a bare `<concept>:` prefix on a base model, far weaker than AxBench's instruction prompts. **DiffMean.** It was compared only at the neuron-tuned norm. By word count it injects "from" and "as" more often than the neuron, with more repetition. | Caveats added wherever the comparison is claimed; the Holm-corrected p-values are reported. |
| m8 | Specificity as evidence for a privileged basis | The poles were selected for write vectors with a strong direct effect on one token. Beating random neurons therefore shows the effect is not a generic perturbation; it does not show the neuron basis is privileged for steering. | One sentence added to the Discussion and to report 3. |
| n9 | Robustness of the null comparison | The registered null compares a 100-generation target mean with 10-generation null means. Like-for-like on the null's prompts, HM gives p = 0.136 for H2b "as" and p = 1.0 for H2 "[". The registered design is conservative in expectation, so the verdicts stand. The word-count version is p = 1/81 for all six. | Reported as a sensitivity analysis. |
| n10 | Minor | Limitations said "one seed for data and sampling", but the confirmations used two sampling seeds. The concept-only p vs the null is now also given Holm-adjusted (0.037). | Corrected. |
| n11 | Process | The numbers in `paper.md` and `README.md` are typed by hand. Every number added by this audit comes from the results JSON written by code, but the documents are not generated from it. | Open follow-up: template the tables from results. |

## Not tested (alternative explanations that remain open)
The user chose to correct claims and add post-hoc statistics from existing data, without new model runs. These controls would test the open explanations:

1. **Gate selectivity.** The distribution of SiLU(z) and y at random non-firing positions, against firings (M3).
2. **Same-length prefix.** Keep the last k tokens and restore the original input length with an unrelated held-out prefix. This separates missing context from a short-input effect (m5).
3. **H3 sensitivity.** The change in the target-token logit and the KL instead of log p, plus repeated clean runs to measure the bf16 noise floor (m6).
4. **Fluency without the injected word.** H2b continuation NLL that excludes the injected-token positions, or human fluency ratings of the H2b generations (M1).

## Learnings
- **Test the claim, not only the rule.** A pre-registered rule can be met while the claim it was meant to support, here fluency preservation, stays unsupported because the instrument is not valid in the regime where it is used. Validation items should come from the regime of the claim.
- **Use a judge-free check when one exists.** For single-word concepts a word count is cheaper and more objective than an LLM judge, and here it confirmed the main result.
- **Do not throw away a continuous score.** Thresholding a noisy 20-item score at 0.8 hid a small but consistent difference. Report the continuous comparison alongside the thresholded one.
- **Constraint or finding?** Before calling a result a property of Mamba, check whether it follows from the functional form (SiLU's bounded negative range) or from how the sample was chosen.
