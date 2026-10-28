# Neuron viewer

A static single-page app in `site/` (vanilla JS, no build step). It's deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main` that touches `site/`.

## Run locally
```bash
uv run python -m mamba_neurons.pipeline --phases export
```
```bash
uv run python -m http.server 8000 --directory site
```
Then open http://localhost:8000.

## What it shows
- **Summary cards.**
  - The pre-registered H1 result: clean-pole rate for random neurons vs random directions.
  - The H2 result for each confirmed neuron: harmonic mean of concept and fluency when clamped vs unsteered, the Holm p-value against the random-neuron null, and the verdict.
  - Split stability.
- **Pole list.**
  - Filters: steering-screened, formula candidates, domain neurons, the random audit sample, or everything.
  - A minimum detection-accuracy slider.
  - Text search, or `L37 N1812` to jump to a neuron.
- **Pole page**, deep-linkable with `?key=neuron:L37:N1812:+` or `?layer=37&neuron=1812&sign=1`:
  - **Label and held-out detection accuracy.** From the local LLM.
  - **Activation statistics.** Discovery split: mean, std, skew, kurtosis, max, min, fraction positive.
  - **Writes up / writes down.** Top promoted and suppressed tokens of `W_U(g ⊙ w_i)`, sign-flipped for negative poles.
  - **Turn it up.** The clamp-sweep table (concept, fluency, harmonic mean, NLL) with buttons that show the 10 generations at each level.
  - **Confirmation and Mechanism tables.** Only for the confirmed poles.
  - **Strongest firings.** Token heatmaps; colour intensity is activation divided by the snippet's peak, and the peak token is outlined. Orange is the positive pole, blue the negative.
  - **Held-out detection test.** The 10 top firings (★) and 10 random snippets (·) the scorer saw. Random snippets are coloured relative to the pole's strongest discovery firing.

## Data format
- `site/data/index.json` holds one row per labelled pole, plus the H1 / H2 / stability / benchmark summaries.
- `site/data/poles/<key>.json` holds the full detail of each pole (`export.py::export_site`).
