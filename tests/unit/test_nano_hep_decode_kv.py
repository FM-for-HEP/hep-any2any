"""Regression tests for the KV-cache path in ``batched_ar_decode``.

The cache must be *mathematically identical* to the full-recompute path. The
scripted-stub model in ``test_nano_hep_decode.py`` can't catch a cache bug —
its output ignores the input sequence. These tests use a REAL (small, random)
``GPT`` whose logits depend on every prior token, so a wrong cache (bad
positions, dropped keys, off-by-one) produces different logits and fails.

Two levels:
  1. forward-equivalence: incremental KV forward reproduces the non-cached
     last-position logits at EVERY sequence length (the property decode relies on).
  2. decode-equivalence: ``batched_ar_decode(use_kv=True)`` produces byte-identical
     token streams to ``use_kv=False`` for both layouts (argmax, structural decode).
"""
import pytest
import torch

from hep4m.models.nano_hep import Vocab
from hep4m.models.nano_hep.model import GPT, GPTConfig
from hep4m.models.nano_hep.decode import batched_ar_decode


def _small_gpt(vocab_total, block_size=256, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(block_size=block_size, vocab_size=vocab_total,
                    n_layer=3, n_head=4, n_embd=32, dropout=0.0, bias=True)
    model = GPT(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def vocab():
    return Vocab.build(modalities=["topo", "track", "truthjet", "truthpart"])


# ---------------------------------------------------------------------------
# 1. forward-equivalence: cached single-step logits == recompute at every length
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kv_forward_matches_recompute_every_length(vocab, seed):
    torch.manual_seed(100 + seed)
    model = _small_gpt(vocab.total, block_size=128, seed=seed)
    B, L = 4, 40
    seq = torch.randint(0, vocab.total, (B, L), dtype=torch.long)

    # incremental cached pass: prime on token 0, then feed one token at a time.
    with torch.no_grad():
        logit, _, kv = model(seq[:, :1], return_kv_list=True)
        inc = [logit]
        for k in range(2, L + 1):
            logit, _, kv = model(seq[:, k - 1:k], past_kv_list=kv, return_kv_list=True)
            inc.append(logit)

        # reference: full recompute over each prefix length (last-pos logits).
        for k in range(1, L + 1):
            ref = model(seq[:, :k])[0]          # (B, 1, V)
            assert ref.shape == inc[k - 1].shape
            assert torch.allclose(ref, inc[k - 1], atol=1e-5, rtol=1e-4), (
                f"KV logits diverge at length {k} (seed {seed}): "
                f"max|d|={(ref - inc[k - 1]).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# 2. decode-equivalence: cached vs non-cached produce identical token streams
# ---------------------------------------------------------------------------
def _random_prefix(vocab, m_first, B, T_prefix, seed):
    torch.manual_seed(7000 + seed)
    body = torch.randint(0, vocab.total, (B, T_prefix - 1), dtype=torch.long)
    cue = torch.full((B, 1), vocab.mod_out[m_first], dtype=torch.long)
    return torch.cat([body, cue], dim=1)


@pytest.mark.parametrize("layout", ["grouped", "interleaved"])
def test_decode_kv_equals_recompute(vocab, layout):
    v = vocab
    out_mods = ["topo", "track"]
    m_first = sorted(out_mods)[0]
    total_emitted = 0
    for seed in range(6):
        model = _small_gpt(v.total, block_size=256, seed=seed)
        prefix = _random_prefix(v, m_first, B=3, T_prefix=8, seed=seed)

        kw = dict(output_layout=layout, argmax=True, structural_decode=True,
                  max_new_tokens=80)
        out_recompute = batched_ar_decode(model, v, prefix.clone(), out_mods, use_kv=False, **kw)
        out_cached = batched_ar_decode(model, v, prefix.clone(), out_mods, use_kv=True, **kw)

        for m in out_mods:
            assert len(out_recompute[m]) == len(out_cached[m])
            for b in range(len(out_recompute[m])):
                a = out_recompute[m][b].tolist()
                c = out_cached[m][b].tolist()
                total_emitted += len(a)
                assert a == c, (
                    f"[{layout} seed={seed} mod={m} row={b}] cached != recompute\n"
                    f"  recompute={a}\n  cached   ={c}"
                )
    # guard against a vacuous pass (all rows EOS-ing immediately): the cache must
    # have been exercised over many steps across the seeds.
    assert total_emitted > 30, f"decode produced too few tokens ({total_emitted}) to be meaningful"


def test_use_kv_default_is_on(vocab):
    """The production path must default to the fast cache without any flag."""
    import inspect
    sig = inspect.signature(batched_ar_decode)
    assert sig.parameters["use_kv"].default is True
