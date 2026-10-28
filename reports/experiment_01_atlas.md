# Experiment 1: activation atlas of all 196,608 Mamba-1.4B neurons

**Status:** done (2026-10-28). Code: `phases/atlas.py` (`collect`, `vocab`, `select`, `domains`).
**Results:** `results/mamba-1.4b-hf/{benchmark,split_stability,activation_shape,candidates,domain_candidates,vocab_summary}.json`.

## What was run
- **Model.** `state-spaces/mamba-1.4b-hf` in bf16 on an RTX 5070 Laptop GPU (8 GB). The HF reference torch scan was used, because the `mamba_ssm` kernels are unavailable on Windows.
- **Data.** `NeelNanda/pile-10k`, split by document (seed 0) into 70% discovery and 30% held-out. Windows are 128 tokens plus the separator, at most 4 per document.
  - Discovery: 18,111 windows, 2,318,208 tokens, 6,193 documents.
  - Held-out: 5,078 windows, 649,984 tokens, 1,758 documents (capped for compute).
- **Pass.** One streaming pass per split over all 48 layers. Each neuron and each of 96 random directions gets float64 moments, fraction positive, and top/bottom-40 firings.
- **Time.** Discovery took 2,950 s (786 tok/s on average) and held-out 651 s, at batch size 12.

## Results
| Quantity | Neurons (n = 196,608) | Random directions (n = 96) |
|---|---|---|
| Kurtosis, median [p10, p90] | **354** [46, 3,064] | 6.9 [5.1, 19.7] |
| Skew, median [p10, p90] | +7.4 [−1.5, +31.4] | −0.03 [−0.44, +0.45] |
| Fraction with positive skew | 85.6% | 45.8% |
| Fraction of tokens with a > 0, median | 0.52 | 0.49 |

**Reading.** A Gaussian has kurtosis 3. A typical Mamba neuron sits near zero and occasionally fires very strongly, and 86% of neurons fire on their *positive* pole. Random directions in the same 4096-d space are much closer to Gaussian. This is direct statistical evidence for a privileged neuron basis, which is what you'd expect from the elementwise `SiLU(z)` gate and the per-channel SSM. It is correlational and exploratory; H1 tests the interpretability side (experiment 2).

**Reliability.** The statistics are stable across disjoint document sets. Spearman ρ between the discovery and held-out splits, over all neurons:

| Statistic | ρ |
|---|---|
| mean | 0.985 |
| std | 0.998 |
| max | 0.960 |
| min | 0.973 |
| kurtosis | 0.942 |
| frac_pos | 0.999 |

**Scale grows with depth.** The per-layer median of each neuron's maximum activation rises from 0.5 (layer 0) to 24.4 (layer 42). The median std rises from 0.02 to 0.28. In the vocab projection, the per-layer median kurtosis of `W_U(g ⊙ w_i)` stays near Gaussian at every layer (3.23–3.55). It is highest in layers 31–40 (peak 3.55 at layer 35), the same layers where the prediction-neuron candidates sit. Prediction-like neurons are a heavy tail, not the norm.

**Candidates (pre-registered formula).** All 60 are late-layer (27–47, plus one at layer 9 and two at 22) and positive-pole (59 of 60). Nearly all are next-token-prediction neurons, which promote one word family: `all`, `way`, `self`, `nor`, `time`, `per`, `from`, `basis`, `with`, `others`, and numbers. This follows from the formula: high vocab kurtosis plus input-output coherence selects exactly Gurnee-style prediction neurons ([Gurnee et al. 2024](https://arxiv.org/abs/2401.12181)).

**Domain stream (amendment 2).** For each domain, the 3 largest |d′| of window-mean activation, in-domain vs rest:

| Domain | Neurons (d′) |
|---|---|
| Github | L0 N2107 (+2.17), L32 N3719 (−2.16), L0 N1037 (+2.13) |
| PubMed Abstracts | L47 N194 (−3.40), L35 N3257 (+3.20), L1 N3532 (+3.04) |
| Wikipedia | L30 N36 (+3.00), L47 N846 (+2.17), L36 N732 (+1.96) |
| USPTO Backgrounds | L0 N84 (+3.67), L37 N3610 (−3.58), L23 N397 (−3.54) |
| FreeLaw | L1 N1296 (−3.23), L45 N2751 (+2.90), L0 N2634 (+2.67) |
| DM Mathematics | L45 N3322 (−9.48), L24 N4057 (−8.85), L26 N1780 (−6.59) |
| ArXiv | L31 N2093 (+3.65), L23 N334 (+3.15), L47 N444 (+3.09) |
| StackExchange | L34 N3185 (−2.68), L38 N4078 (+2.28), L33 N196 (−2.10) |

The DM Mathematics signal is by far the strongest (|d′| up to 9.5): those synthetic arithmetic documents are stylistically very distinct.

## Learnings and issues
1. **Fast kernels.** `mamba_ssm` kernels are not available on Windows, so the HF reference sequential scan runs at about 1,300 tok/s at batch 16.
2. **Parallel scan rejected.** `use_mambapy=True` (the mamba.py Blelloch scan) was *slower* (2.2k vs 3.6k tok/s on mamba-130m in fp32). Its logits also differed by up to 153, plausibly because pscan expects power-of-two lengths and ours is 129. In bf16 it crashed with a dtype mismatch inside transformers. Kept the reference path.
3. **Memory spill.** At batch 16 the first run spilled into Windows shared GPU memory (7.7 / 8.1 GB used, 185 tok/s). Batch 12 (about 5 GB) fixed it. The held-out budget was cut from 1.3M to 650k tokens to keep the collection under an hour.
4. **Stale caches.** Cached windows are now keyed by a settings fingerprint (`windows_tag`), so a budget change can never silently reuse old windows.
