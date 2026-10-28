"""Static figures for README and paper (matplotlib, Agg backend).

Local explanation: one function per figure, each reading the small result JSONs and writing a PNG. Colours follow the
validated reference palette of the dataviz guidelines: slot 1 blue (#2a78d6) for the method / neurons, slot 2 orange
(#eb6834) for a second measure, neutral gray for controls and null distributions; one y-axis per panel, error bars
are 95% intervals as stated in each caption.
Global context: called by the `plots` phase; figures land in results/<model>/figures and are embedded in README.md
and reports/paper.md.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only, no display (https://matplotlib.org/stable/users/explain/figure/backends.html)
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e4e2dd", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_h1(h1: dict, autointerp: dict, path) -> None:
    """Left: clean-pole rate (BA >= 0.8) for random neurons vs random directions, Wilson 95% CI. Right: BA distributions."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    kinds, colors = ["neuron", "direction"], [BLUE, GRAY]
    rates = [h1[k]["clean_rate"] for k in kinds]
    lo = [r - h1[k]["clean_rate_wilson95"][0] for r, k in zip(rates, kinds)]
    hi = [h1[k]["clean_rate_wilson95"][1] - r for r, k in zip(rates, kinds)]
    a.bar([0, 1], rates, width=0.55, color=colors, yerr=[lo, hi], capsize=4, ecolor=INK)
    a.set_xticks([0, 1], [f"random neurons\n(n={h1['neuron']['n_poles']} poles)", f"random directions\n(n={h1['direction']['n_poles']} poles)"])
    a.set_ylabel("fraction of clean poles")
    a.set_ylim(0, 1)
    for x, r in enumerate(rates):
        a.text(x, r + hi[x] + 0.03, f"{100 * r:.0f}%", ha="center", color=INK)
    bins = np.linspace(0.3, 1.0, 15)
    for k, c in zip(kinds, colors):
        ba = [r["balanced_accuracy"] for r in autointerp.values() if r["unit"]["kind"] == k and r["unit"].get("group", "audit") == "audit"]
        b.hist(ba, bins=bins, histtype="step", linewidth=2, color=c, label=f"random {k}s")
    b.axvline(0.8, color=INK, linestyle="--", linewidth=1)
    b.text(0.805, b.get_ylim()[1] * 0.92, "clean ≥ 0.8", color=INK, fontsize=9)
    b.set_xlabel("held-out detection balanced accuracy of the LLM label")
    b.set_ylabel("poles")
    b.legend(frameon=False)
    _save(fig, path)


def fig_dose_response(screen: dict, keys: list[str], path) -> None:
    """Concept and fluency judge scores (0-2) vs clamp level for the confirmed poles (screen data, 10 prompts)."""
    poles = [s for s in screen["summary"] if s["key"] in keys]
    fig, axes = plt.subplots(1, len(poles), figsize=(3.6 * len(poles), 3.2), squeeze=False)
    for ax, s in zip(axes[0], poles):
        levels = sorted(int(c) for c in s["levels"])
        x = np.arange(len(levels))
        for metric, color in (("concept", BLUE), ("fluency", ORANGE)):
            ys = [s["levels"][str(c)][metric] for c in levels]
            ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=5, label=metric)
            ax.text(x[-1] + 0.1, ys[-1], metric, color=color, va="center", fontsize=9)
        ax.set_xticks(x, ["off" if c == 0 else f"{c}×" for c in levels])
        ax.set_ylim(-0.05, 2.05)
        ax.set_xlabel("clamp (× max activation)")
        ax.set_title(f"L{s['layer']} N{s['neuron']} ({'+' if s['sign'] > 0 else '−'})\n{s['label'][:48]}", fontsize=9)
    axes[0][0].set_ylabel("judge score (0–2)")
    axes[0][0].legend(frameon=False, loc="upper left")
    _save(fig, path)


def fig_h2(h2: dict, path) -> None:
    """Per confirmed pole: harmonic mean (concept, fluency) per condition with 95% cluster-bootstrap CI + random-neuron null."""
    order = ["unsteered", "prompting", "diffmean", "add", "clamp"]
    fig, axes = plt.subplots(1, len(h2), figsize=(4.2 * len(h2), 3.2), squeeze=False, sharey=True)
    for ax, (key, r) in zip(axes[0], h2.items()):
        conds = [c for c in order if c in r["conditions"]]
        for y, c in enumerate(conds):
            m, (lo, hi) = r["conditions"][c]["hm"], r["conditions"][c]["hm_ci95"]
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=BLUE if c == "clamp" else INK, markersize=6, capsize=3)
        y_null = len(conds)
        ax.scatter(r["random_null_hm"], np.full(len(r["random_null_hm"]), y_null) + np.random.default_rng(0).uniform(-0.15, 0.15, len(r["random_null_hm"])),
                   color=GRAY, s=14)
        ax.set_yticks(range(len(conds) + 1), conds + ["20 random neurons\n(same clamp)"])
        ax.set_xlim(-0.05, 2.05)
        ax.set_xlabel("harmonic mean of concept & fluency (0–2)")
        ax.set_title(f"{key}\n{r['label'][:50]}", fontsize=9)
    _save(fig, path)


def fig_h3(mech: dict, path) -> None:
    """Zero-ablation drop in log p(next token) at the pole's top firings: neuron vs 20 random same-layer neurons."""
    fig, ax = plt.subplots(figsize=(6, 0.9 + 0.7 * len(mech)))
    for y, (key, m) in enumerate(mech.items()):
        null = m["h3"]["random_null"]
        ax.scatter(null, np.full(len(null), y) + np.random.default_rng(y).uniform(-0.12, 0.12, len(null)), color=GRAY, s=14,
                   label="random neurons" if y == 0 else None)
        ax.scatter([m["h3"]["ablation_drop_nats"]], [y], color=BLUE, s=50, zorder=3, label="target neuron" if y == 0 else None)
    ax.set_yticks(range(len(mech)), list(mech))
    ax.set_xlabel("drop in log p(next token) when zero-ablated (nats)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def fig_truncation(mech: dict, path) -> None:
    """Activation kept (fraction of full-context value) when only the last k tokens are visible."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for n, (key, m) in enumerate(mech.items()):
        items = [(int(k), v) for k, v in m["context_truncation_ratio"].items() if k != "all"]
        ks, vs = zip(*sorted(items))
        ax.plot(ks, vs, marker="o", markersize=5, linewidth=2, color=[BLUE, ORANGE, "#1baf7a"][n % 3], label=key)
    ax.axvline(4, color=GRAY, linestyle=":", linewidth=1)
    ax.text(4.2, 0.05, "conv kernel = 4", color=MUTED, fontsize=8)
    ax.axhline(1, color=GRAY, linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context tokens kept (k, log scale)")
    ax.set_ylabel("activation / full-context activation")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_layers(autointerp: dict, n_layers: int, path) -> None:
    """Mean held-out detection accuracy per layer block (6 layers each) for random neurons vs random directions."""
    fig, ax = plt.subplots(figsize=(6, 3))
    block = 6
    for kind, color in (("neuron", BLUE), ("direction", GRAY)):
        xs, ys = [], []
        for start in range(0, n_layers, block):
            ba = [r["balanced_accuracy"] for r in autointerp.values() if r["unit"]["kind"] == kind and r["unit"].get("group", "audit") == "audit"
                  and start <= r["unit"]["layer"] < start + block]
            if ba:
                xs.append(start + block / 2 - 0.5)
                ys.append(np.mean(ba))
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=color, label=f"random {kind}s")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean detection accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False)
    _save(fig, path)
