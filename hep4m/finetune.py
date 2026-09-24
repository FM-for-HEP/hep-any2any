"""Fine-tuning a HEP4M model from a trained checkpoint (weights only).

``python -m hep4m.train_hep4m`` calls ``apply_init_weights`` right after building the
model when ``config_t`` has ``init_weights_path`` (or ``--init_weights_path`` is given)
and the run is not resuming. It loads the model weights of the checkpoint (strict). The
optimiser state, the learning-rate schedule and the epoch/step counters are not loaded,
so the run starts at epoch 0 with a fresh optimiser and its own warm-up.

A resumed run (``last.ckpt`` in the run directory) is restored by Lightning as usual and
ignores ``init_weights_path``.
"""
from __future__ import annotations

import logging
import pickle
from typing import Callable, Optional

import torch


def load_state_dict_file(path: str) -> dict:
    """The ``state_dict`` of a Lightning checkpoint (or a bare state dict).

    Tries ``torch.load(weights_only=True)`` first (the released checkpoints load this
    way); full training checkpoints that need unpickling fall back to
    ``weights_only=False``, so only use checkpoints you trust.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def apply_init_weights(lightning_model, config_t: dict, path: Optional[str] = None,
                       log: Optional[Callable[[str], None]] = None) -> bool:
    """Load weights from ``path`` (default ``config_t['init_weights_path']``) into a
    ``HEP4MLightning``. Returns False when there is nothing to load."""
    log = log or logging.getLogger(__name__).info
    path = path or (config_t or {}).get("init_weights_path")
    if not path:
        return False
    res = lightning_model.load_state_dict(load_state_dict_file(path), strict=True)
    log(f"loaded model weights from {path} "
        f"(missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}); "
        f"fresh optimiser, epoch 0")
    return True
