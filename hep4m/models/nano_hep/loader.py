"""Rebuild a nanoHEP model and vocabulary from a checkpoint written by NanoHepLightning.

The checkpoint holds ``state_dict`` (GPT weights under the ``model.`` prefix),
``gpt_config`` (GPTConfig keyword arguments), ``vocab_args`` (Vocab.build keyword
arguments), ``global_step`` and ``config`` (optimiser settings, kept for reference).
Other keys are ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from .model import GPT, GPTConfig
from .vocab import Vocab


def load_model_from_ckpt(ckpt_path: str | Path, device: str = "cuda") -> Tuple:
    """Load a nanoHEP checkpoint.

    Returns ``(model, vocab, cfg, step)``: the GPT in eval mode on ``device``, the
    Vocab, the ``config`` dict of the checkpoint (may be empty) and the global step.
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    vocab = Vocab.build(**state["vocab_args"])
    model = GPT(GPTConfig(**state["gpt_config"]))
    model_sd = {k[len("model."):]: v for k, v in state["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(model_sd)
    model.to(device).eval()
    return model, vocab, state.get("config", {}), int(state.get("global_step", 0))
