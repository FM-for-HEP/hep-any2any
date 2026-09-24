"""Tests for ``batched_ar_decode``.

Uses a deterministic stub model that emits a canned token script, so the
tests verify the *partitioning logic* (grouped: split at inline boundary
tokens; interleaved: split by vocab range) without depending on a real
trained checkpoint.
"""
import pytest
import torch
import torch.nn as nn


class _ScriptedModel(nn.Module):
    """Stub model: at each forward step, return logits that put +inf at
    the next token in a pre-baked script. One register parameter so
    ``next(model.parameters())`` works for the decode helper's device
    inference."""

    def __init__(self, vocab_total: int, script: list):
        super().__init__()
        self.vocab_total = vocab_total
        self.script = list(script)
        self.step = 0
        self._dummy = nn.Linear(1, 1)

    def forward(self, idx, targets=None, **kwargs):
        B = idx.shape[0]
        logits = torch.full((B, 1, self.vocab_total), -1e9, device=idx.device)
        if self.step < len(self.script):
            logits[:, 0, int(self.script[self.step])] = 1e9
        self.step += 1
        return logits, None


@pytest.fixture(scope="module")
def vocab():
    from hep4m.models.nano_hep import Vocab
    return Vocab.build(modalities=["topo", "track", "truthjet", "truthpart"])


class TestBatchedArDecodeGrouped:
    """Grouped layout: model emits modality content tokens; switches modality
    by emitting an internal ``<MOD_OUT:m>`` boundary; partitioner uses
    those inline boundaries."""

    def test_two_output_modalities_partition_correctly(self, vocab):
        from hep4m.models.nano_hep.decode import batched_ar_decode

        v = vocab
        # 6 topo content tokens (3 content quantizers + 3 pos), then boundary,
        # then 6 track content tokens, then EOS.
        # All content tokens chosen in [0, 7680) which covers topo+track ranges.
        # We don't care about exact validity here; we just verify partition.
        script = [
            # topo element 0
            100, 200, 300, 900, 1500, 2500,
            v.mod_out["track"],  # boundary
            # track element 0 (in track content range [3840, 4608))
            3850, 3900, 3950, 4700, 5700, 6700,
            v.eos,
        ]
        model = _ScriptedModel(v.total, script)
        # Prefix ends with <MOD_OUT:topo> as the begin-gen cue.
        prefix = torch.tensor([[v.mod_out["topo"]], [v.mod_out["topo"]]], dtype=torch.long)
        out = batched_ar_decode(
            model, v, prefix, ["topo", "track"],
            output_layout="grouped", argmax=True, structural_decode=False,
            max_new_tokens=40,
        )
        # topo gets the first 6 tokens; track gets the next 6.
        assert out["topo"][0].tolist() == [100, 200, 300, 900, 1500, 2500]
        assert out["track"][0].tolist() == [3850, 3900, 3950, 4700, 5700, 6700]

    def test_eos_terminates_decode_for_batched_rows(self, vocab):
        from hep4m.models.nano_hep.decode import batched_ar_decode
        v = vocab
        # Single OUT modality, 6 content tokens, then EOS.
        script = [100, 200, 300, 900, 1500, 2500, v.eos]
        model = _ScriptedModel(v.total, script)
        prefix = torch.tensor([[v.mod_out["topo"]]], dtype=torch.long)
        out = batched_ar_decode(
            model, v, prefix, ["topo"],
            output_layout="grouped", argmax=True, structural_decode=False,
            max_new_tokens=20,
        )
        assert out["topo"][0].tolist() == [100, 200, 300, 900, 1500, 2500]


class TestBatchedArDecodeInterleaved:
    """Interleaved layout: no inline boundary tokens; modality of each
    emitted content token is read from its vocab range via
    ``vocab.modality_of(t)``."""

    def test_round_robin_partition_by_vocab_range(self, vocab):
        from hep4m.models.nano_hep.decode import batched_ar_decode

        v = vocab
        # Round-robin: topo element, track element, topo, track, EOS.
        # All content tokens picked to fall in topo / track content ranges.
        script = [
            # topo elem 0  (topo content range [0, 768))
            100, 200, 300, 900, 1500, 2500,
            # track elem 0  (track content range [3840, 4608))
            3850, 3900, 3950, 4700, 5700, 6700,
            # topo elem 1
            150, 250, 350, 950, 1550, 2550,
            # track elem 1
            3855, 3905, 3955, 4705, 5705, 6705,
            v.eos,
        ]
        model = _ScriptedModel(v.total, script)
        prefix = torch.tensor([[v.mod_out["topo"]]], dtype=torch.long)
        out = batched_ar_decode(
            model, v, prefix, ["topo", "track"],
            output_layout="interleaved", argmax=True, structural_decode=False,
            max_new_tokens=40,
        )
        expected_topo = [100, 200, 300, 900, 1500, 2500, 150, 250, 350, 950, 1550, 2550]
        expected_track = [3850, 3900, 3950, 4700, 5700, 6700, 3855, 3905, 3955, 4705, 5705, 6705]
        assert out["topo"][0].tolist() == expected_topo
        assert out["track"][0].tolist() == expected_track


class TestBucketByPrefixLength:

    def test_groups_by_length(self):
        from hep4m.models.nano_hep.decode import bucket_by_prefix_length

        ps = [
            torch.zeros(5, dtype=torch.long),
            torch.zeros(7, dtype=torch.long),
            torch.zeros(5, dtype=torch.long),
            torch.zeros(7, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
        ]
        buckets = bucket_by_prefix_length(ps)
        assert set(buckets.keys()) == {5, 7, 10}
        assert sorted(buckets[5]) == [0, 2]
        assert sorted(buckets[7]) == [1, 3]
        assert buckets[10] == [4]
