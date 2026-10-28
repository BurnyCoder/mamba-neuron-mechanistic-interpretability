"""mamba_neurons: neuron-level interpretability and single-neuron steering for Mamba language models.

Local explanation: importing the package first imports `config`, which loads `.env` (so HF_HOME and
friends are set before transformers/datasets are imported anywhere else).
Global context: the phases live in sibling modules and are orchestrated by `pipeline.py`.
"""

from . import config as _config  # noqa: F401  (side effect: load .env before any HF import)
