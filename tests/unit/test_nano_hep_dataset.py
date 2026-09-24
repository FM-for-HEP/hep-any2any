"""Tests for ``TokenizedMemmapDataset`` over the tokenised fixture: the autoregressive
token sequence of the nanoHEP ``GPT``, parameterised by ``sampling_dict`` and
``output_layout: grouped | interleaved``.
"""
from pathlib import Path

import numpy as np
import torch


ALL_MODALITIES = ["topo", "track", "truthjet", "truthpart"]


class TestNanoHepViewSequence:
    """The dataset builds the token sequence from per-modality
    memmaps + a sampling_dict. Verifies sequence structure for both
    layouts."""

    BLOCK_SIZE = 1024

    def _build_ds(self, val_root: str, output_layout: str, output_mods=None, input_mods=None):
        from hep4m.datasets.tokenized_memmap import TokenizedMemmapDataset
        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(modalities=ALL_MODALITIES)
        return TokenizedMemmapDataset(
            tokenized_root=str(Path(val_root).parent),  # parent of val/
            split="val",
            all_modalities=ALL_MODALITIES,
            sampling_dict={
                "type": "inference",
                "fixed_input_output_modalities": {
                    "input": input_mods or ["track", "topo"],
                    "output": output_mods or ["truthpart"],
                },
                "output_layout": output_layout,
                "input_layout": "grouped",
            },
            block_size=self.BLOCK_SIZE,
            vocab=v,
            max_events=10,
        )

    def test_grouped_layout_returns_seq_and_loss_mask(self, tokenized_fixture_val):
        ds = self._build_ds(tokenized_fixture_val, output_layout="grouped")
        sample = ds[0]
        assert "seq" in sample and "loss_mask" in sample
        assert sample["seq"].shape == (self.BLOCK_SIZE,)
        assert sample["loss_mask"].shape == (self.BLOCK_SIZE,)
        assert sample["seq"].dtype == torch.int64
        assert sample["loss_mask"].dtype == torch.int64

    def test_grouped_has_header_then_task_sep(self, tokenized_fixture_val):
        ds = self._build_ds(tokenized_fixture_val, output_layout="grouped",
                            output_mods=["truthpart"])
        seq = ds[0]["seq"].numpy()
        v = ds.vocab
        # Header: <MOD_OUT:truthpart> at pos 0, <TASK_SEP> at pos 1
        assert seq[0] == v.mod_out["truthpart"]
        assert seq[1] == v.task_sep

    def test_grouped_multi_output_has_internal_boundary(self, tokenized_fixture_val):
        """For OUT=[topo, track] grouped, the model would emit
        <MOD_OUT:topo>...<MOD_OUT:track>... -- the dataset emits both
        cues in the truth sequence so the model learns the modality switch."""
        ds = self._build_ds(
            tokenized_fixture_val, output_layout="grouped",
            input_mods=["truthpart"], output_mods=["topo", "track"],
        )
        seq = ds[0]["seq"].numpy()
        v = ds.vocab
        # Find positions of MOD_OUT tokens in the body (after task_sep).
        task_sep_pos = int(np.where(seq == v.task_sep)[0][0])
        body = seq[task_sep_pos + 1:]
        mod_out_positions = [i for i, t in enumerate(body) if t in set(v.mod_out.values())]
        # At least: helper-supplied first cue + internal boundary between
        # modalities. So >= 2 MOD_OUT tokens in body.
        assert len(mod_out_positions) >= 2, (
            f"grouped multi-output should have >= 2 MOD_OUT in body; got {len(mod_out_positions)}"
        )

    def test_interleaved_has_no_internal_boundaries(self, tokenized_fixture_val):
        """In interleaved layout, OUT body has exactly ONE <MOD_OUT:m> (the
        helper-supplied 'begin generation' cue); everything after is content
        identifiable by vocab range."""
        ds = self._build_ds(
            tokenized_fixture_val, output_layout="interleaved",
            input_mods=["truthpart"], output_mods=["topo", "track"],
        )
        seq = ds[0]["seq"].numpy()
        v = ds.vocab
        task_sep_pos = int(np.where(seq == v.task_sep)[0][0])
        body = seq[task_sep_pos + 1:]
        eos_locs = np.where(body == v.eos)[0]
        body_to_eos = body[: int(eos_locs[0]) if len(eos_locs) else len(body)]
        mod_out_positions = [i for i, t in enumerate(body_to_eos) if t in set(v.mod_out.values())]
        # interleaved: only the very first MOD_OUT cue
        assert len(mod_out_positions) == 1, (
            f"interleaved should have exactly 1 MOD_OUT in body; got {len(mod_out_positions)} at {mod_out_positions}"
        )


# ===========================================================================
# Loss-mask rule (derivable from token IDs alone)
# ===========================================================================

class TestLossMask:

    def test_loss_zero_on_header_and_inputs(self, tokenized_fixture_val):
        from hep4m.datasets.tokenized_memmap import TokenizedMemmapDataset
        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(modalities=ALL_MODALITIES)
        ds = TokenizedMemmapDataset(
            tokenized_root=str(Path(tokenized_fixture_val).parent),
            split="val", all_modalities=ALL_MODALITIES,
            sampling_dict={
                "type": "inference",
                "fixed_input_output_modalities": {"input": ["track", "topo"], "output": ["truthpart"]},
                "output_layout": "grouped", "input_layout": "grouped",
            },
            block_size=1024, vocab=v, max_events=5,
        )
        sample = ds[0]
        seq = sample["seq"].numpy()
        lm = sample["loss_mask"].numpy()
        task_sep_pos = int(np.where(seq == v.task_sep)[0][0])
        # Header + TASK_SEP must have loss=0.
        assert lm[:task_sep_pos + 1].sum() == 0
        # IN regions (between TASK_SEP and first OUT) must be 0.
        body = seq[task_sep_pos + 1:]
        first_out_local = int(np.where(np.isin(body, list(v.mod_out.values())))[0][0])
        gen_start = task_sep_pos + 1 + first_out_local
        assert lm[task_sep_pos + 1: gen_start + 1].sum() == 0

    def test_loss_one_on_output_content_and_eos(self, tokenized_fixture_val):
        from hep4m.datasets.tokenized_memmap import TokenizedMemmapDataset
        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(modalities=ALL_MODALITIES)
        ds = TokenizedMemmapDataset(
            tokenized_root=str(Path(tokenized_fixture_val).parent),
            split="val", all_modalities=ALL_MODALITIES,
            sampling_dict={
                "type": "inference",
                "fixed_input_output_modalities": {"input": ["track", "topo"], "output": ["truthpart"]},
                "output_layout": "grouped", "input_layout": "grouped",
            },
            block_size=1024, vocab=v, max_events=5,
        )
        sample = ds[0]
        seq = sample["seq"].numpy()
        lm = sample["loss_mask"].numpy()
        # EOS must have loss=1
        eos_pos = int(np.where(seq == v.eos)[0][0])
        assert lm[eos_pos] == 1
        # Position right before EOS (last content token) must be 1
        assert lm[eos_pos - 1] == 1

    def test_loss_zero_on_pad(self, tokenized_fixture_val):
        from hep4m.datasets.tokenized_memmap import TokenizedMemmapDataset
        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(modalities=ALL_MODALITIES)
        ds = TokenizedMemmapDataset(
            tokenized_root=str(Path(tokenized_fixture_val).parent),
            split="val", all_modalities=ALL_MODALITIES,
            sampling_dict={
                "type": "inference",
                "fixed_input_output_modalities": {"input": ["track", "topo"], "output": ["truthpart"]},
                "output_layout": "interleaved", "input_layout": "interleaved",
            },
            block_size=1024, vocab=v, max_events=5,
        )
        sample = ds[0]
        seq = sample["seq"].numpy()
        lm = sample["loss_mask"].numpy()
        # Past the first EOS, everything (PAD) must be 0.
        eos_pos = int(np.where(seq == v.eos)[0][0])
        assert lm[eos_pos + 1:].sum() == 0


# ===========================================================================
# Interleaved vocab-range partitioning
# ===========================================================================

class TestInterleavedPartition:

    def test_interleaved_out_content_partitions_by_vocab_range(self, tokenized_fixture_val):
        """In interleaved layout, every OUT content token's modality is
        determinable from its global token ID via vocab.modality_of(). Test
        that every OUT-region content token is assigned to one of the
        declared OUT modalities."""
        from hep4m.datasets.tokenized_memmap import TokenizedMemmapDataset
        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(modalities=ALL_MODALITIES)
        ds = TokenizedMemmapDataset(
            tokenized_root=str(Path(tokenized_fixture_val).parent),
            split="val", all_modalities=ALL_MODALITIES,
            sampling_dict={
                "type": "inference",
                "fixed_input_output_modalities": {"input": ["truthpart"], "output": ["topo", "track"]},
                "output_layout": "interleaved", "input_layout": "interleaved",
            },
            block_size=1024, vocab=v, max_events=5,
        )
        sample = ds[0]
        seq = sample["seq"].numpy()
        task_sep_pos = int(np.where(seq == v.task_sep)[0][0])
        body = seq[task_sep_pos + 1:]
        # Find first MOD_OUT in body (the helper-supplied cue) and EOS.
        mod_out_set = set(v.mod_out.values())
        first_out_local = int(np.where(np.isin(body, list(mod_out_set)))[0][0])
        eos_local = int(np.where(body == v.eos)[0][0])
        out_content = body[first_out_local + 1: eos_local]
        # Every OUT content token must self-identify as either topo or track.
        modalities_seen = {v.modality_of(int(t)) for t in out_content if v.modality_of(int(t))}
        assert modalities_seen.issubset({"topo", "track"})
        # Should see both modalities in the partitioned content.
        assert modalities_seen == {"topo", "track"}
