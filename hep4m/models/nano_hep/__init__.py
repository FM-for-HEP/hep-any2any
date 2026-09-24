"""nanoHEP: decoder-only autoregressive transformer over tokenised event modalities.

``model.py``   GPT model (adapted from nanoGPT)
``vocab.py``   joint vocabulary: per-modality token ranges, ``<MOD_IN:m>`` / ``<MOD_OUT:m>``
               role tags, ``<TASK_SEP>``, ``<EOS>``, ``<PAD>``
``decode.py``  batched autoregressive decoding
``loader.py``  rebuilds model and vocabulary from a training checkpoint
"""

from .model import GPT, GPTConfig
from .vocab import Vocab
from .loader import load_model_from_ckpt

__all__ = [
    "GPT",
    "GPTConfig",
    "Vocab",
    "load_model_from_ckpt",
]
