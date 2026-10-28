"""Single entry point: runs the named pipeline phases in order.

Local explanation: `python -m mamba_neurons.pipeline --phases collect,vocab` loads `.env` settings, sets up
timestamped logging and calls each phase function from `phases.py`.
Global context: this is the "one abstracted wrapper" of the project; each phase reads its inputs from and writes its
outputs to `settings.run_dir()` / `settings.results_path()`, so phases can be re-run independently.
"""

from __future__ import annotations

import argparse  # stdlib CLI parsing (https://docs.python.org/3/library/argparse.html)

from . import phases
from .config import load_settings
from .logging_utils import get_logger, setup_logging

log = get_logger(__name__)


def main(argv=None) -> None:
    """Parse `--phases` (comma list or "all") and `--model` override, then run the phases in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default="all", help=f"comma-separated subset of: {','.join(phases.PHASES)} (or 'all')")
    parser.add_argument("--model", default=None, help="override MODEL_NAME from .env (e.g. state-spaces/mamba-130m-hf)")
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.model:
        settings = settings.with_overrides(model_name=args.model)
    names = list(phases.PHASES) if args.phases == "all" else [p.strip() for p in args.phases.split(",")]
    setup_logging("-".join(names) if len(names) < 4 else "all", settings.log_dir)
    log.info("Settings: %s", settings.to_dict())
    for name in names:
        log.info("=== phase %s ===", name)
        phases.PHASES[name](settings)  # KeyError on a typo is intentional: fail loudly


if __name__ == "__main__":
    main()
