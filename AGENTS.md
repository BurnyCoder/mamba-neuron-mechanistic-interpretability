# AGENTS.md: notes for coding agents working on this repo

- **Entry point:** `uv run python -m mamba_neurons.pipeline --phases <comma list|all>`. Phases are registered in `src/mamba_neurons/phases.py::PHASES` and run in that order.
- **Config:** everything comes from `.env`, documented in `.env.example`. Add new knobs to `config.py::Settings` and to `.env.example` together.
- **Neuron site:** the only place that touches model internals is `model.py` (`neuron_site`, `Hooks`). Interventions are `EditFn`s (`clamp_edit`, `add_edit`, `add_vector_edit`). Never hook anything else directly.
- **Outputs:** big artefacts go to `OUTPUT_DIR/<model>` (gitignored, default `D:/AI/mamba-neurons/outputs`). Small results go to `results/<model>` (committed).
- **Logging:** use `get_logger(__name__)`. Every LLM call goes through `llm.py`, which logs the full prompt and output via `log_llm_call`.
- **Tests:** `uv run pytest` (needs about 1 GB to download mamba-130m on first run). Write the test first for any new intervention or statistic.
- **Science rules:** confirmatory analyses must match `reports/preregistration.md`. Anything else is labelled post hoc, and amendments are dated in that file.
- **GPU:** 8 GB laptop GPU. Mamba phases and Ollama phases must not overlap, so call `phases._free` before LLM phases.
