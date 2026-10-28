"""Shared pytest fixtures.

Local explanation: loads the small `TEST_MODEL_NAME` (mamba-130m) once per test session in float32, so equivalence
tests can use tight tolerances (https://docs.pytest.org/en/stable/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session).
Global context: the same hook code then runs unchanged on mamba-1.4b in bfloat16 in the real pipeline.
"""

import pytest
import torch

from mamba_neurons.config import load_settings
from mamba_neurons.model import load_model


@pytest.fixture(scope="session")
def settings():
    """Settings from .env (tests only read them)."""
    return load_settings()


@pytest.fixture(scope="session")
def small_model(settings):
    """(model, tokenizer) for mamba-130m in fp32 on the configured device."""
    device = settings.device if torch.cuda.is_available() else "cpu"  # fall back to CPU on GPU-less CI
    return load_model(settings.test_model_name, dtype="float32", device=device)


@pytest.fixture(scope="session")
def sample_ids(small_model):
    """A fixed 2x12 batch of token ids built from real text."""
    model, tok = small_model
    texts = ["The capital of France is Paris, and the capital of Germany is", "def add(a, b):\n    return a + b\n\nprint(add(2,"]
    ids = [tok(t).input_ids[:12] for t in texts]  # truncate both to 12 tokens -> equal length, no padding needed
    return torch.tensor(ids, device=model.device)
