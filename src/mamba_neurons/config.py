"""Run configuration loaded from `.env`.

Local explanation: reads every pipeline knob from environment variables (populated from `.env` by
python-dotenv, https://github.com/theskumar/python-dotenv#getting-started) into one frozen dataclass.
Global context: every phase in `pipeline.py` receives the same `Settings` object, so a run is fully
described by its `.env` file (saved next to the outputs for reproducibility).
"""

from __future__ import annotations  # postpone annotation evaluation (PEP 563) so `Path | None` works everywhere

import os  # os.environ holds the values loaded from .env
from dataclasses import asdict, dataclass, replace  # frozen config record + helpers to dump/override it
from pathlib import Path  # OS-independent paths (Windows backslashes vs POSIX slashes)

from dotenv import load_dotenv  # parses KEY=VALUE lines from .env into os.environ

# Load .env once at import time. `override=False` keeps variables already set in the shell, so a user can
# override any key ad hoc: `MODEL_NAME=... uv run ...`. HF_HOME must be set *before* transformers/datasets
# are imported, which is why `mamba_neurons/__init__.py` imports this module first.
load_dotenv(override=False)


def _get(name: str, default: str) -> str:
    """Return env var `name` or `default` (string form; callers cast)."""
    return os.environ.get(name, default)  # dict lookup with fallback



@dataclass(frozen=True)
class Settings:
    """All tunable parameters of the pipeline (see `.env.example` for meaning and defaults)."""

    model_name: str  # HF id of the Mamba model under study
    test_model_name: str  # small model used by tests / smoke runs
    dtype: str  # torch dtype name for weights ("bfloat16", "float32")
    device: str  # "cuda" or "cpu"
    dataset_name: str  # HF dataset id of the text corpus
    heldout_fraction: float  # fraction of *documents* reserved for held-out evaluation
    ctx_len: int  # tokens per analysis window (excluding the separator token)
    max_windows_per_doc: int  # cap so long documents do not dominate the statistics
    max_discovery_tokens: int  # token budget for the discovery split
    max_heldout_tokens: int  # token budget for the held-out split
    batch_size: int  # windows per forward pass
    top_k: int  # top activations stored per neuron pole
    n_random_directions_per_layer: int  # random-direction control units per layer (H1 control)
    seed: int  # global RNG seed
    ollama_host: str  # URL of the local Ollama server
    llm_model: str  # Ollama model tag used for labels / judging
    output_dir: Path  # large artefacts (activation stats, generations)
    results_dir: Path  # small committed results (JSON, figures)
    log_dir: Path  # timestamped run logs
    hf_dataset_repo: str  # HF dataset repo for publishing the neuron atlas

    @property
    def model_slug(self) -> str:
        """Filesystem-safe model name, e.g. 'mamba-1.4b-hf' (used to namespace outputs per model)."""
        return self.model_name.split("/")[-1]  # drop the org prefix

    def run_dir(self) -> Path:
        """Directory holding this model's large artefacts; created on demand."""
        path = self.output_dir / self.model_slug  # one sub-folder per model so runs never collide
        path.mkdir(parents=True, exist_ok=True)  # idempotent creation
        return path

    def results_path(self) -> Path:
        """Directory holding this model's small committed results; created on demand."""
        path = self.results_dir / self.model_slug  # mirror of run_dir inside the repo
        path.mkdir(parents=True, exist_ok=True)
        return path

    def with_overrides(self, **kwargs) -> "Settings":
        """Return a copy with some fields replaced (used by tests and smoke runs)."""
        return replace(self, **kwargs)  # dataclasses.replace builds a new frozen instance

    def to_dict(self) -> dict:
        """JSON-serialisable view (Paths -> str) for saving next to outputs."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_settings() -> Settings:
    """Build `Settings` from the current environment (after `.env` was loaded)."""
    return Settings(
        model_name=_get("MODEL_NAME", "state-spaces/mamba-1.4b-hf"),
        test_model_name=_get("TEST_MODEL_NAME", "state-spaces/mamba-130m-hf"),
        dtype=_get("DTYPE", "bfloat16"),
        device=_get("DEVICE", "cuda"),
        dataset_name=_get("DATASET_NAME", "NeelNanda/pile-10k"),
        heldout_fraction=float(_get("HELDOUT_FRACTION", "0.3")),
        ctx_len=int(_get("CTX_LEN", "128")),
        max_windows_per_doc=int(_get("MAX_WINDOWS_PER_DOC", "4")),
        max_discovery_tokens=int(_get("MAX_DISCOVERY_TOKENS", "3000000")),
        max_heldout_tokens=int(_get("MAX_HELDOUT_TOKENS", "650000")),
        batch_size=int(_get("BATCH_SIZE", "12")),
        top_k=int(_get("TOP_K", "40")),
        n_random_directions_per_layer=int(_get("N_RANDOM_DIRECTIONS_PER_LAYER", "2")),
        seed=int(_get("SEED", "0")),
        ollama_host=_get("OLLAMA_HOST", "http://localhost:11434"),
        llm_model=_get("LLM_MODEL", "qwen3.5:9b"),
        output_dir=Path(_get("OUTPUT_DIR", "outputs")),
        results_dir=Path(_get("RESULTS_DIR", "results")),
        log_dir=Path(_get("LOG_DIR", "logs")),
        hf_dataset_repo=_get("HF_DATASET_REPO", "BurnyCoder/mamba-1.4b-neuron-atlas"),
    )
