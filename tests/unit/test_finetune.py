"""Weights-only initialisation for fine-tuning (hep4m.finetune), on a small stand-in
module with the same attribute layout as HEP4MLightning (no data needed)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from hep4m.finetune import apply_init_weights, load_state_dict_file


class _Embedder(nn.Module):
    def __init__(self, n_q=3, vocab=16, dim=8, pos=True):
        super().__init__()
        self.token_emb = nn.ModuleList([nn.Embedding(vocab, dim) for _ in range(n_q)])
        if pos:
            self.pos_token_emb = nn.ModuleList([nn.Embedding(vocab, dim) for _ in range(2)])


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedders = nn.ModuleDict({"topo": _Embedder(), "truthpart": _Embedder()})
        self.head = nn.Linear(8, 4)


class _Lit(nn.Module):
    """Same layout as HEP4MLightning: the network is ``.model``."""

    def __init__(self):
        super().__init__()
        self.model = _Net()


def _save(tmp_path, lit, full=True):
    p = tmp_path / "src.ckpt"
    torch.save({"state_dict": lit.state_dict(), "epoch": 33, "global_step": 46138} if full
               else lit.state_dict(), p)
    return p


@pytest.mark.parametrize("full", [True, False], ids=["lightning-ckpt", "bare-state-dict"])
def test_load_weights(tmp_path, full):
    torch.manual_seed(1)
    src = _Lit()
    path = _save(tmp_path, src, full)

    torch.manual_seed(2)
    dst = _Lit()
    logs = []
    assert apply_init_weights(dst, {"init_weights_path": str(path)}, log=logs.append)
    sd_src, sd_dst = src.state_dict(), dst.state_dict()
    for k in sd_dst:
        assert torch.equal(sd_dst[k], sd_src[k]), k
    assert any("loaded model weights" in m for m in logs)


def test_no_path_is_a_no_op():
    lit = _Lit()
    before = {k: v.clone() for k, v in lit.state_dict().items()}
    assert not apply_init_weights(lit, {})
    assert all(torch.equal(before[k], v) for k, v in lit.state_dict().items())


def test_cli_path_overrides_config(tmp_path):
    src = _Lit()
    path = _save(tmp_path, src)
    dst = _Lit()
    cfg = {"init_weights_path": str(tmp_path / "missing.ckpt")}
    assert apply_init_weights(dst, cfg, path=str(path), log=lambda *_: None)
    assert torch.equal(dst.model.head.weight, src.model.head.weight)


def test_architecture_mismatch_fails(tmp_path):
    src = _Lit()
    src.model.head = nn.Linear(8, 5)
    path = _save(tmp_path, src)
    with pytest.raises(RuntimeError):
        apply_init_weights(_Lit(), {"init_weights_path": str(path)}, log=lambda *_: None)


def test_load_state_dict_file_bare(tmp_path):
    lit = _Lit()
    path = _save(tmp_path, lit, full=False)
    assert set(load_state_dict_file(str(path))) == set(lit.state_dict())
