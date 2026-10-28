"""Statistics and text metrics used by the pre-registered analyses.

Local explanation: thin, tested wrappers around SciPy / statsmodels for confidence intervals (Wilson, cluster
bootstrap), tests (one-sided Fisher exact, empirical permutation-style p-values, Holm correction), power analysis,
inter-rater agreement (Cohen's kappa) and generation metrics (AxBench harmonic mean, distinct-n).
Global context: H1 uses `wilson`, `fisher_greater`, `bootstrap_diff_of_props`; H2 uses `cluster_bootstrap_diff`,
`empirical_p`, `holm`, `harmonic_mean`; judge validation uses `cohen_kappa`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import fisher_exact  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
from statsmodels.stats.inter_rater import cohens_kappa  # https://www.statsmodels.org/stable/generated/statsmodels.stats.inter_rater.cohens_kappa.html
from statsmodels.stats.multitest import multipletests  # https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
from statsmodels.stats.power import NormalIndPower  # https://www.statsmodels.org/stable/generated/statsmodels.stats.power.NormalIndPower.html
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def fisher_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact p-value for H: rate1 > rate2."""
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="greater").pvalue)


def bootstrap_diff_of_props(x1, x2, clusters1=None, clusters2=None, n_boot: int = 10_000, seed: int = 0,
                            alpha: float = 0.05) -> tuple[float, float, float]:
    """(rate1 - rate2, CI low, CI high) of two independent 0/1 samples, percentile bootstrap.

    With `clusters*` (e.g. the unit id shared by a neuron's two poles) whole clusters are resampled and the rate is
    sum(successes) / sum(trials) over the resampled clusters.
    """
    rng = np.random.default_rng(seed)

    def per_cluster(x, c):
        x = np.asarray(x, float)
        c = np.arange(len(x)) if c is None else np.asarray(c)
        ids = np.unique(c)
        return np.array([x[c == i].sum() for i in ids]), np.array([(c == i).sum() for i in ids])

    (s1, n1), (s2, n2) = per_cluster(x1, clusters1), per_cluster(x2, clusters2)
    i1, i2 = rng.integers(0, len(s1), (n_boot, len(s1))), rng.integers(0, len(s2), (n_boot, len(s2)))
    boots = s1[i1].sum(1) / n1[i1].sum(1) - s2[i2].sum(1) / n2[i2].sum(1)
    return float(s1.sum() / n1.sum() - s2.sum() / n2.sum()), *map(float, np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def cluster_bootstrap_diff(a, b, clusters, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired difference mean(a - b) with a CI that resamples whole clusters (prompts), as generations share prompts."""
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    ids = np.unique(clusters)
    per = np.array([(a[clusters == c] - b[clusters == c]).mean() for c in ids])  # one paired mean per cluster
    rng = np.random.default_rng(seed)
    boots = rng.choice(per, (n_boot, len(per))).mean(1)
    return float(per.mean()), *map(float, np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def cluster_bootstrap_mean(x, clusters, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of x with a cluster-bootstrap CI (clusters = prompts)."""
    return cluster_bootstrap_diff(x, np.zeros(len(x)), clusters, n_boot, seed, alpha)


def empirical_p(observed: float, null) -> float:
    """(1 + #null >= observed) / (1 + n): exact-size p-value against a sample from the null (Phipson & Smyth 2010)."""
    null = np.asarray(null, float)
    return float((1 + (null >= observed).sum()) / (1 + len(null)))


def holm(pvals) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    return [float(p) for p in multipletests(pvals, method="holm")[1]]


def power_n_per_arm(p0: float, p1: float, alpha: float, power: float = 0.8) -> float:
    """Samples per arm for a two-sided two-proportion z-test (Cohen's h effect size)."""
    return float(NormalIndPower().solve_power(effect_size=proportion_effectsize(p1, p0), alpha=alpha, power=power, ratio=1.0))


def cohen_kappa(r1, r2) -> float:
    """Cohen's kappa between two raters' categorical labels."""
    cats = sorted(set(r1) | set(r2))
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(r1, r2):
        table[cats.index(x), cats.index(y)] += 1
    return float(cohens_kappa(table).kappa)


def harmonic_mean(concept: float, fluency: float) -> float:
    """AxBench aggregate of two 0-2 judge scores; 0 if either is 0 (https://arxiv.org/abs/2501.17148)."""
    return 0.0 if concept + fluency == 0 else 2 * concept * fluency / (concept + fluency)


def distinct_n(tokens, n: int = 2) -> float:
    """Fraction of distinct n-grams in a token sequence (low = repetitive degeneration; Li et al. 2016)."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0
