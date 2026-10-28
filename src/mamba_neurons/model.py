"""Model loading and the neuron hook site for Mamba.

Local explanation: a Mamba block computes (HF `modeling_mamba.py`, Gu & Dao 2023 https://arxiv.org/abs/2312.00752 Fig. 3)
    x, z = in_proj(RMSNorm(h)).chunk(2)
    a    = (SSM(SiLU(conv1d(x))) + D * SiLU(conv1d(x))) * SiLU(z)      # gated SSM output, width d_inner
    h'   = h + out_proj(a)
We call coordinate a_i of layer l "neuron (l, i)": the input to `mixer.out_proj`, exactly like a SwiGLU MLP neuron
SiLU(W_g x) * (W_u x) is the input to `down_proj` (Shazeer 2020, https://arxiv.org/abs/2002.05202). Its *write
vector* is column i of `out_proj.weight`. We read/edit it with a PyTorch forward-pre-hook on `out_proj`
(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook), which
fires both for full-sequence forwards and for cached one-token decoding steps.
Global context: every phase (collection, steering, ablation, patching) goes through `Hooks` so all interventions
act on the same, unit-tested site.
"""

from __future__ import annotations

from typing import Callable  # type of edit functions

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer  # HF auto classes resolve MambaForCausalLM

from .logging_utils import get_logger

log = get_logger(__name__)

EditFn = Callable[[torch.Tensor], torch.Tensor]  # maps a (batch, seq, width) activation to its edited version


def load_model(name: str, dtype: str = "bfloat16", device: str = "cuda", use_mambapy: bool = False):
    """Load a HF Mamba causal LM + tokenizer in eval mode on `device`.

    `dtype=` is the transformers-5 spelling of the former `torch_dtype=` argument
    (https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained);
    `use_mambapy` overrides the matching `MambaConfig` field (parallel scan from https://github.com/alxndrTL/mamba.py).
    """
    tokenizer = AutoTokenizer.from_pretrained(name)  # GPT-NeoX-20B BPE tokenizer for state-spaces/* models
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), use_mambapy=use_mambapy)
    model.to(device).eval()  # eval(): no dropout and, importantly, no fused training-only path in MambaMixer
    log.info("Loaded %s (%s, %s): %d layers x %d neurons", name, dtype, device, n_layers(model), n_neurons(model))
    return model, tokenizer


def blocks(model) -> nn.ModuleList:
    """The residual blocks `backbone.layers` of a HF Mamba model."""
    return model.backbone.layers


def neuron_site(model, layer: int) -> nn.Linear:
    """The `out_proj` Linear whose *input* is the neuron vector `a` of `layer`."""
    return blocks(model)[layer].mixer.out_proj


def n_layers(model) -> int:
    """Number of Mamba blocks (48 for mamba-1.4b)."""
    return len(blocks(model))


def n_neurons(model) -> int:
    """Neurons per layer = d_inner = expand * d_model (4096 for mamba-1.4b)."""
    return neuron_site(model, 0).in_features


def write_vectors(model, layer: int) -> torch.Tensor:
    """(d_model, d_inner) matrix whose column i is neuron i's write direction into the residual stream."""
    return neuron_site(model, layer).weight  # nn.Linear stores weight as (out_features, in_features)


class Hooks:
    """Context manager that owns hook handles and removes them on exit (no hook ever leaks between runs)."""

    def __init__(self, model):
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []  # returned by register_*_hook

    def __enter__(self) -> "Hooks":
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:  # RemovableHandle.remove() detaches the hook
            handle.remove()
        self.handles.clear()

    def record_neurons(self, layers, sink: Callable[[int, torch.Tensor], None]) -> "Hooks":
        """Call `sink(layer, a)` with the (batch, seq, d_inner) neuron activations of each layer in `layers`."""
        for layer in layers:
            def pre_hook(_module, inputs, layer=layer):  # default arg binds the loop variable (late-binding fix)
                sink(layer, inputs[0].detach())  # inputs is the positional-args tuple of out_proj.forward
            self.handles.append(neuron_site(self.model, layer).register_forward_pre_hook(pre_hook))
        return self

    def edit_neurons(self, layer: int, fn: EditFn) -> "Hooks":
        """Replace layer's neuron activations `a` by `fn(a)` before `out_proj` (returning a tuple replaces the args)."""
        def pre_hook(_module, inputs):
            return (fn(inputs[0]),) + tuple(inputs[1:])
        self.handles.append(neuron_site(self.model, layer).register_forward_pre_hook(pre_hook))
        return self

    def edit_residual(self, layer: int, fn: EditFn) -> "Hooks":
        """Replace the residual stream *after* block `layer` by `fn(h)` (forward hook: a returned value replaces output)."""
        def hook(_module, _inputs, output):
            return fn(output)
        self.handles.append(blocks(self.model)[layer].register_forward_hook(hook))
        return self


def clamp_edit(index, value) -> EditFn:
    """Set neuron(s) `index` to `value` at every position (Templeton et al. 2024 style clamping)."""
    def fn(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()  # never mutate the tensor the model may still reference
        a[..., index] = torch.as_tensor(value, dtype=a.dtype, device=a.device)
        return a
    return fn


def add_edit(index, delta) -> EditFn:
    """Add `delta` to neuron(s) `index` at every position (ActAdd-style additive steering on one coordinate)."""
    def fn(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        a[..., index] = a[..., index] + torch.as_tensor(delta, dtype=a.dtype, device=a.device)
        return a
    return fn


def add_vector_edit(vector: torch.Tensor) -> EditFn:
    """Add a fixed residual-stream vector at every position (DiffMean / ActAdd baselines, AxBench https://arxiv.org/abs/2501.17148)."""
    def fn(h: torch.Tensor) -> torch.Tensor:
        return h + vector.to(dtype=h.dtype, device=h.device)
    return fn


@torch.no_grad()
def record(model, input_ids: torch.Tensor, layers) -> dict[int, torch.Tensor]:
    """One forward pass; returns {layer: (batch, seq, d_inner) activations} for `layers`."""
    store: dict[int, torch.Tensor] = {}
    with Hooks(model).record_neurons(layers, lambda layer, a: store.__setitem__(layer, a)):
        model(input_ids=input_ids, use_cache=False)  # use_cache=False: plain full-sequence forward
    return store
