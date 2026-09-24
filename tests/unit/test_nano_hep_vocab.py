"""Tests for the nanoHEP vocabulary (``hep4m.models.nano_hep.vocab``).

The vocabulary has per-modality content + pos token blocks, ``<MOD_IN:m>`` /
``<MOD_OUT:m>`` role tags and a ``<TASK_SEP>`` separator. The key property exploited downstream is that content
tokens self-identify their modality via vocab range
(``offsets[m] + q*codebook_size``), so the interleaved-layout decoder can
partition emissions without inline modality boundaries.

These tests are pure-python (no filesystem, no GPU) and fast.
"""
import numpy as np
import pytest


MODALITIES = ["topo", "track", "truthjet", "truthpart"]


@pytest.fixture(scope="module")
def vocab():
    from hep4m.models.nano_hep import Vocab
    return Vocab.build(modalities=MODALITIES)


class TestVocabConstruction:

    def test_total_size(self, vocab):
        """total = sum(content + pos blocks) + 2*N specials + 3 (task_sep, eos, pad)."""
        expected = 0
        for m in MODALITIES:
            expected += vocab.num_quantizers[m] * vocab.codebook_sizes[m]
            expected += vocab.num_q_pos[m] * vocab.pos_codebook_sizes[m]
        expected += 2 * len(MODALITIES) + 3
        assert vocab.total == expected

    def test_specials_distinct(self, vocab):
        ids = set()
        for m in MODALITIES:
            ids.add(vocab.mod_in[m])
            ids.add(vocab.mod_out[m])
        ids.update({vocab.task_sep, vocab.eos, vocab.pad})
        # 8 mod-tags + 3 = 11 distinct
        assert len(ids) == 2 * len(MODALITIES) + 3

    def test_mod_in_out_disjoint(self, vocab):
        in_ids = set(vocab.mod_in.values())
        out_ids = set(vocab.mod_out.values())
        assert in_ids.isdisjoint(out_ids)

    def test_specials_outside_content_range(self, vocab):
        """Specials must not collide with any modality's content/pos range."""
        for m in MODALITIES:
            c_lo = vocab.offsets[m]
            c_hi = c_lo + vocab.num_quantizers[m] * vocab.codebook_sizes[m]
            p_lo = vocab.pos_offsets[m]
            p_hi = p_lo + vocab.num_q_pos[m] * vocab.pos_codebook_sizes[m]
            for tid in (
                *vocab.mod_in.values(), *vocab.mod_out.values(),
                vocab.task_sep, vocab.eos, vocab.pad,
            ):
                assert not (c_lo <= tid < c_hi)
                assert not (p_lo <= tid < p_hi)


class TestVocabRangeModalityLookup:
    """The interleaved-layout decoder depends on this lookup to partition
    emitted content tokens into per-modality buckets without needing inline
    boundary tokens."""

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_content_token_self_identifies(self, vocab, modality):
        c_lo = vocab.offsets[modality]
        c_hi = c_lo + vocab.num_quantizers[modality] * vocab.codebook_sizes[modality]
        for tid in [c_lo, c_lo + 1, c_hi - 1]:
            assert vocab.modality_of(tid) == modality

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_pos_token_self_identifies(self, vocab, modality):
        p_lo = vocab.pos_offsets[modality]
        p_hi = p_lo + vocab.num_q_pos[modality] * vocab.pos_codebook_sizes[modality]
        for tid in [p_lo, p_lo + 1, p_hi - 1]:
            assert vocab.modality_of(tid) == modality

    def test_specials_return_empty_for_modality_of(self, vocab):
        """modality_of(tok) is best-effort -- specials return ''."""
        for tid in (
            vocab.mod_in["topo"], vocab.mod_out["topo"],
            vocab.task_sep, vocab.eos, vocab.pad,
        ):
            # Specials may either return '' or fail; we accept either as
            # "not a content/pos token" -- the partition logic checks for empty.
            mod = vocab.modality_of(tid)
            assert mod == "" or mod not in MODALITIES

    def test_content_modality_disjoint(self, vocab):
        """No two modalities' content blocks overlap."""
        ranges = {}
        for m in MODALITIES:
            lo = vocab.offsets[m]
            hi = lo + vocab.num_quantizers[m] * vocab.codebook_sizes[m]
            ranges[m] = (lo, hi)
        for m1 in MODALITIES:
            for m2 in MODALITIES:
                if m1 == m2:
                    continue
                lo1, hi1 = ranges[m1]
                lo2, hi2 = ranges[m2]
                assert hi1 <= lo2 or hi2 <= lo1


class TestEncodeDecodeRoundtrip:

    def test_roundtrip_simple(self, vocab):
        content = np.array([[5, 6, 7], [10, 11, 12], [20, 21, 22]], dtype=np.int64)
        pos = np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]], dtype=np.int64)
        flat = vocab.encode_element_with_pos("truthpart", content, pos)
        c_back, p_back = vocab.decode_element_with_pos("truthpart", flat)
        assert np.array_equal(c_back, content)
        assert np.array_equal(p_back, pos)

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_roundtrip_all_modalities(self, vocab, modality):
        nq = vocab.num_quantizers[modality]
        cb = vocab.codebook_sizes[modality]
        nqp = vocab.num_q_pos[modality]
        cbp = vocab.pos_codebook_sizes[modality]
        # 4 elements, each drawn from valid local ranges.
        content = np.random.RandomState(0).randint(0, cb, size=(4, nq), dtype=np.int64)
        pos = np.random.RandomState(1).randint(0, cbp, size=(4, nqp), dtype=np.int64)
        flat = vocab.encode_element_with_pos(modality, content, pos)
        c_back, p_back = vocab.decode_element_with_pos(modality, flat)
        assert np.array_equal(c_back, content)
        assert np.array_equal(p_back, pos)

    def test_decode_rejects_oob_content(self, vocab):
        # Fabricate a flat sequence with an out-of-range content slot.
        flat = vocab.encode_element_with_pos(
            "truthpart",
            np.array([[0, 0, 0]], dtype=np.int64),
            np.array([[0, 0, 0]], dtype=np.int64),
        )
        flat[0] = vocab.eos  # corrupt first content slot
        with pytest.raises(ValueError):
            vocab.decode_element_with_pos("truthpart", flat)
