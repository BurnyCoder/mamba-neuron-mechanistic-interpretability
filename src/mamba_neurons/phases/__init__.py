"""Phase registry: name -> function(settings), in pipeline order.

Global context: `pipeline.py --phases all` runs them top to bottom; GPU (Mamba) and Ollama phases alternate so
only one model occupies the 8 GB GPU at a time.
"""

from .atlas import phase_benchmark, phase_collect, phase_domains, phase_examples, phase_select, phase_vocab
from .interp import phase_h1, phase_label
from .mechanism import phase_mechanism
from .publish import phase_export, phase_hf, phase_plots
from .steering import (phase_confirm, phase_confirm_judge, phase_screen, phase_screen_judge, phase_screen_low,
                       phase_screen_low_judge)

PHASES = {
    "benchmark": phase_benchmark,
    "collect": phase_collect,
    "vocab": phase_vocab,
    "select": phase_select,
    "domains": phase_domains,
    "examples": phase_examples,
    "label": phase_label,  # Ollama
    "h1": phase_h1,
    "screen": phase_screen,
    "screen_judge": phase_screen_judge,  # Ollama
    "screen_low": phase_screen_low,  # post hoc
    "screen_low_judge": phase_screen_low_judge,  # Ollama, post hoc
    "confirm": phase_confirm,
    "confirm_judge": phase_confirm_judge,  # Ollama
    "mechanism": phase_mechanism,
    "export": phase_export,
    "plots": phase_plots,
    "hf": phase_hf,  # uploads to the Hugging Face Hub
}
