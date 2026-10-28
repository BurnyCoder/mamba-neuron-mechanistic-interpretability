"""Publishing phases: viewer data export (site/data) and static figures (results/<model>/figures).

Local explanation: both only read result files (tokenizer only, no GPU), so they can be re-run any time.
Global context: site/ is deployed to GitHub Pages by .github/workflows/pages.yml; figures are embedded in README.md and
reports/paper.md.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from .. import plots
from ..export import export_site
from ..logging_utils import get_logger
from .common import load_json

log = get_logger(__name__)
SITE_DATA = Path("site/data")


def phase_export(settings) -> None:
    """Write site/data/index.json and site/data/poles/*.json."""
    tok = AutoTokenizer.from_pretrained(settings.model_name)
    n = export_site(settings, tok, SITE_DATA)
    log.info("exported %d poles to %s", n, SITE_DATA)


def phase_plots(settings) -> None:
    """Render every figure whose inputs exist."""
    res = settings.results_path()
    out = res / "figures"
    out.mkdir(exist_ok=True)
    have = lambda name: (res / f"{name}.json").exists()
    if have("h1") and have("autointerp"):
        auto = load_json(res / "autointerp.json")
        plots.fig_h1(load_json(res / "h1.json"), auto, out / "h1_clean_rate.png")
        plots.fig_layers(auto, 1 + max(r["unit"]["layer"] for r in auto.values()), out / "layers_detection.png")
    if have("screen") and have("confirm_plan"):
        screens = [load_json(res / f"{n}.json") for n in ("screen", "screen_low") if have(n)]
        plots.fig_dose_response(screens, [p["key"] for p in load_json(res / "confirm_plan.json")], out / "dose_response.png")
        if have("confirm_plan_h2b"):
            plots.fig_dose_response(screens, [p["key"] for p in load_json(res / "confirm_plan_h2b.json")], out / "dose_response_h2b.png")
    for name in ("h2", "h2b"):
        if have(name):
            plots.fig_h2(load_json(res / f"{name}.json")["results"], out / f"{name}_conditions.png")
    if have("h3_mechanism"):
        mech = load_json(res / "h3_mechanism.json")
        plots.fig_h3(mech, out / "h3_ablation.png")
        plots.fig_truncation(mech, out / "context_truncation.png")
    if have("truncation_survey"):
        plots.fig_truncation_survey(load_json(res / "truncation_survey.json"), out / "truncation_survey.png")
    log.info("figures in %s: %s", out, sorted(p.name for p in out.glob("*.png")))


HF_CARD = """---
license: other
license_name: mixed-pile-snippets
task_categories: [text-generation]
tags: [mechanistic-interpretability, mamba, state-space-model, neurons, steering]
pretty_name: Mamba-1.4B neuron atlas
---
# Mamba-1.4B neuron atlas

Neuron-level statistics, automated labels and single-neuron steering generations for
[state-spaces/mamba-1.4b-hf](https://huggingface.co/state-spaces/mamba-1.4b-hf). Produced by
https://github.com/BurnyCoder/mamba-neuron-mechanistic-interpretability (methods, pre-registration and paper there).

A *neuron* (layer l, index i) is coordinate i of the gated SSM output `a = y * SiLU(z)`, i.e. the input of
`mixer.out_proj` (48 layers x 4096 = 196,608 neurons).

| file | rows | content |
|---|---|---|
| `neuron_stats.parquet` | 196,608 | mean / std / skew / kurtosis / max / min / frac_pos on the discovery (2.32M tokens) and held-out (0.65M tokens) splits of [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k); vocab kurtosis and top-5 promoted / suppressed tokens of `W_U(g * w_i)` |
| `poles.parquet` | labelled poles | local-LLM (qwen3.5:9b) label, held-out detection balanced accuracy, promoted tokens, highlighted top-firing snippets (JSON) |
| `steering.parquet` | generations | clamp-sweep screen and pre-registered confirmation generations with AxBench-style judge scores |

Text snippets are short excerpts of the Pile (Gao et al. 2020) and keep the licences of their original sources;
labels and judge scores are produced by a local LLM and are noisy (judge-vs-manual Cohen's kappa 0.55 on detection).
"""


def phase_hf(settings) -> None:
    """Export parquet files + dataset card and upload them to the HF dataset repo, then add it to a collection."""
    from huggingface_hub import HfApi  # https://huggingface.co/docs/huggingface_hub/guides/upload

    from ..export import export_hf

    tok = AutoTokenizer.from_pretrained(settings.model_name)
    out = settings.output_dir / "hf_dataset"
    files = export_hf(settings, tok, out)
    (out / "README.md").write_text(HF_CARD, encoding="utf-8")
    api = HfApi()
    api.create_repo(settings.hf_dataset_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=settings.hf_dataset_repo, repo_type="dataset",
                      commit_message="Upload Mamba-1.4B neuron atlas")
    owner = settings.hf_dataset_repo.split("/")[0]
    collection = api.create_collection(title="Mamba neuron interpretability", namespace=owner, exists_ok=True,
                                       description="Neuron-level mechanistic interpretability of Mamba state-space LMs")
    api.add_collection_item(collection.slug, settings.hf_dataset_repo, item_type="dataset", exists_ok=True)
    log.info("uploaded %s to https://huggingface.co/datasets/%s (collection %s)", files, settings.hf_dataset_repo, collection.slug)
