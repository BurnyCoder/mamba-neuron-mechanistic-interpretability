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
        plots.fig_dose_response(load_json(res / "screen.json"), [p["key"] for p in load_json(res / "confirm_plan.json")], out / "dose_response.png")
    if have("h2"):
        plots.fig_h2(load_json(res / "h2.json")["results"], out / "h2_conditions.png")
    if have("h3_mechanism"):
        mech = load_json(res / "h3_mechanism.json")
        plots.fig_h3(mech, out / "h3_ablation.png")
        plots.fig_truncation(mech, out / "context_truncation.png")
    log.info("figures in %s: %s", out, sorted(p.name for p in out.glob("*.png")))
