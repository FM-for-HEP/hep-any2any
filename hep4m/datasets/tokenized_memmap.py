"""Token-store dataset for nanoHEP.

Reads the per-modality files of a token store split (``hep4m.build_token_store`` or
``python -m hep4m.data export``):

    {root}/{split}/{modality}_data.npy       int16  (total_elem, n_codebooks + n_pos_codebooks)
    {root}/{split}/{modality}_offsets.npy    int64  (n_events + 1,) cumulative element counts
    {root}/{split}/{modality}_meta.npz       {n_events, n_codebooks, n_pos_codebooks}

and builds, per event, the flat autoregressive token sequence and loss mask of the
nanoHEP GPT for an (input modalities -> output modalities) split, with grouped or
interleaved layouts (see ``TokenizedMemmapDataset._build_sequence``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .memmap_cache import get_memmap


@dataclass
class ModalityMemmap:
    """Lazy memmap view of one modality in one tokenized split."""

    name: str
    n_codebooks: int
    n_pos_codebooks: int
    n_events: int
    offsets: np.memmap  # (n_events + 1,) int64, cumulative element counts
    data: np.memmap     # (total_elements, n_codebooks + n_pos_codebooks) int16

    @classmethod
    def load(cls, root: str | Path, split: str, modality: str) -> "ModalityMemmap":
        """Load a modality's memmap files from ``root/split/{modality}_*``."""
        base = Path(root) / split
        meta = np.load(base / f"{modality}_meta.npz", allow_pickle=True)
        n_events = int(meta["n_events"])
        n_cb = int(meta["n_codebooks"])
        n_pos_cb = int(meta["n_pos_codebooks"]) if "n_pos_codebooks" in meta.files else 0
        width = n_cb + n_pos_cb

        offsets_path = base / f"{modality}_offsets.npy"
        off_bytes = os.path.getsize(offsets_path)
        # convert_to_npy_tokenized.py writes int64 (8 bytes/entry)
        n_offsets = off_bytes // 8
        assert n_offsets == n_events + 1, (
            f"offsets length {n_offsets} != n_events + 1 = {n_events + 1} for {modality}"
        )
        offsets = np.memmap(offsets_path, dtype=np.int64, mode="r", shape=(n_offsets,))

        data_path = base / f"{modality}_data.npy"
        file_bytes = os.path.getsize(data_path)
        row_bytes = width * 2  # int16 = 2 bytes
        assert file_bytes % row_bytes == 0, (
            f"data file size {file_bytes} not divisible by row size {row_bytes} for {modality}"
        )
        total_elem = file_bytes // row_bytes
        data = get_memmap(str(data_path), "int16", total_elem, width)

        return cls(
            name=modality,
            n_codebooks=n_cb,
            n_pos_codebooks=n_pos_cb,
            n_events=n_events,
            offsets=offsets,
            data=data,
        )

    # ----- per-event accessors -----

    def event_codes(self, idx: int, num_q: int) -> np.ndarray:
        """First ``num_q`` content codebook columns for event ``idx``,
        shape ``(n_elems, num_q)`` int64."""
        a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
        if b == a:
            return np.empty((0, num_q), dtype=np.int64)
        assert num_q <= self.n_codebooks, (
            f"requested num_q={num_q} > n_codebooks={self.n_codebooks} for {self.name}"
        )
        return self.data[a:b, :num_q].astype(np.int64)

    def event_pos_codes(self, idx: int, num_q_pos: int) -> np.ndarray:
        """First ``num_q_pos`` pos codebook columns for event ``idx``,
        shape ``(n_elems, num_q_pos)`` int64. Columns are stored AFTER
        the content block in the row layout."""
        a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
        if b == a:
            return np.empty((0, num_q_pos), dtype=np.int64)
        assert num_q_pos <= self.n_pos_codebooks, (
            f"requested num_q_pos={num_q_pos} > n_pos_codebooks={self.n_pos_codebooks} for {self.name}"
        )
        lo = self.n_codebooks
        hi = self.n_codebooks + num_q_pos
        return self.data[a:b, lo:hi].astype(np.int64)

    def event_row(self, idx: int) -> np.ndarray:
        """Raw concatenated (content + pos) row for event ``idx``,
        shape ``(n_elems, n_codebooks + n_pos_codebooks)`` int64."""
        a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
        if b == a:
            return np.empty((0, self.n_codebooks + self.n_pos_codebooks), dtype=np.int64)
        return self.data[a:b].astype(np.int64)


class TokenizedMemmapDataset(Dataset):
    """nanoHEP samples from a token store split.

    Parameters
    ----------
    tokenized_root : str
        Root dir containing ``{split}/{modality}_*`` memmap files.
    split : str
        Subdir name (``"train"``, ``"val"``, ``"test"``).
    all_modalities : list[str]
        All modalities that may appear in any sample (used to size the
        vocab). Each modality must have memmap files under ``tokenized_root/split/``.
    sampling_dict : dict
        Keys:
          - ``type``: ``"inference"`` (fixed split per sample) or
            ``"inference_rand2"`` (randomized; the batch sampler picks the split).
          - ``fixed_input_output_modalities``: ``{"input": [...], "output": [...]}``
            -- required for ``"inference"``; used as default for ``"inference_rand2"``.
          - ``max_inp_modality`` / ``max_out_modality``: limits of the random splits.
          - ``output_layout``: ``"grouped"`` | ``"interleaved"`` (default ``"interleaved"``).
          - ``input_layout``: ``"grouped"`` | ``"interleaved"`` (default ``"interleaved"``).
    block_size : int
        Fixed sequence length (right-padded with PAD).
    vocab : Vocab, optional
        If None, built from ``all_modalities`` with the default codebook sizes.
    max_events : int
        Truncate dataset to first N events (-1 = all).
    """

    def __init__(
        self,
        tokenized_root: str,
        split: str,
        all_modalities: list,
        sampling_dict: dict,
        block_size: int,
        vocab=None,
        max_events: int = -1,
    ):
        from hep4m.models.nano_hep.vocab import Vocab  # lazy: avoids an import cycle

        self.tokenized_root = tokenized_root
        self.split = split
        self.all_modalities = list(all_modalities)
        self.block_size = int(block_size)
        self.sampling_dict = dict(sampling_dict)

        self.output_layout = self.sampling_dict.get("output_layout", "interleaved")
        self.input_layout = self.sampling_dict.get("input_layout", "interleaved")
        if self.output_layout not in ("grouped", "interleaved"):
            raise ValueError(f"output_layout must be grouped|interleaved, got {self.output_layout!r}")
        if self.input_layout not in ("grouped", "interleaved"):
            raise ValueError(f"input_layout must be grouped|interleaved, got {self.input_layout!r}")

        self.sampling_type = self.sampling_dict.get("type", "inference")
        if self.sampling_type not in ("inference", "inference_rand2"):
            raise ValueError(f"sampling_dict.type must be inference|inference_rand2, got {self.sampling_type!r}")

        fmods = self.sampling_dict.get("fixed_input_output_modalities", None)
        if fmods is None and self.sampling_type == "inference":
            raise ValueError("sampling_dict.type='inference' requires fixed_input_output_modalities")
        if fmods is not None:
            self.fixed_input_mods = list(fmods["input"])
            self.fixed_output_mods = list(fmods["output"])
        else:
            self.fixed_input_mods = []
            self.fixed_output_mods = []

        # Load all modality memmaps once. Sub-selection happens at __getitem__.
        self.mms: Dict[str, ModalityMemmap] = {}
        for m in self.all_modalities:
            self.mms[m] = ModalityMemmap.load(tokenized_root, split, m)

        n_events_all = [mm.n_events for mm in self.mms.values()]
        if len(set(n_events_all)) != 1:
            raise ValueError(f"n_events mismatch across modalities: {n_events_all}")
        self.n_events_total = n_events_all[0]
        self.n_events = (
            min(self.n_events_total, max_events) if max_events > 0 else self.n_events_total
        )

        self.vocab = vocab if vocab is not None else Vocab.build(modalities=self.all_modalities)

        # (input_mods, output_mods) splits for randomized sampling, as modality names
        self.max_inp_modality = self.sampling_dict.get("max_inp_modality", None)
        self.max_out_modality = self.sampling_dict.get("max_out_modality", None)
        self.valid_splits = self._compute_valid_splits()

    def _compute_valid_splits(self) -> list:
        """List of ``(input_mods, output_mods)`` tuples (modality-name lists).

        Empty list if ``sampling_type == 'inference'`` (single fixed split).
        """
        if self.sampling_type == "inference":
            return []
        from itertools import combinations
        n = len(self.all_modalities)
        max_n_inp = (
            min(self.max_inp_modality + 1, n)
            if self.max_inp_modality is not None else n
        )
        out = []
        for n_inp in range(1, max_n_inp):
            for inp_combo in combinations(range(n), n_inp):
                remaining = [i for i in range(n) if i not in inp_combo]
                if not remaining:
                    continue
                input_mods = [self.all_modalities[i] for i in inp_combo]
                if self.max_out_modality is None:
                    out.append((input_mods, [self.all_modalities[i] for i in remaining]))
                else:
                    max_n_out = min(self.max_out_modality, len(remaining))
                    for n_out in range(1, max_n_out + 1):
                        for out_combo in combinations(remaining, n_out):
                            output_mods = [self.all_modalities[i] for i in out_combo]
                            out.append((input_mods, output_mods))
        return out

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx):
        """Build one sample.

        ``idx`` may be an ``int`` (default fixed-split path) or a
        ``(event_idx, split_idx)`` tuple yielded by
        ``NanoHepFracBatchSampler``. In the tuple case, ``split_idx``
        indexes into ``self.valid_splits`` to pick (input, output) modality
        names for this sample.
        """
        if isinstance(idx, tuple):
            event_idx, split_idx = idx
            if split_idx is None or split_idx < 0:
                # sentinel for "fixed split"
                return self.build_sample(
                    event_idx, self.fixed_input_mods, self.fixed_output_mods,
                )
            input_mods, output_mods = self.valid_splits[split_idx]
            return self.build_sample(event_idx, input_mods, output_mods)
        return self.build_sample(idx, self.fixed_input_mods, self.fixed_output_mods)

    # -- public sample builder (used by both default and sampler paths) --

    def build_sample(self, idx: int, input_mods: list, output_mods: list):
        """Build one ``(seq, loss_mask)`` sample for the given (input, output)
        modality split.

        Returns
        -------
        dict with keys:
          - ``"seq"``: (block_size,) int64 token sequence
          - ``"loss_mask"``: (block_size,) int64 (1 where loss is taken)
          - ``"input_modalities"``: tuple[str]
          - ``"output_modalities"``: tuple[str]
          - ``"event_idx"``: int
        """
        seq = self._build_sequence(idx, input_mods, output_mods)
        loss_mask = self._build_loss_mask(seq)
        return {
            "seq": torch.from_numpy(seq),
            "loss_mask": torch.from_numpy(loss_mask),
            "input_modalities": tuple(input_mods),
            "output_modalities": tuple(output_mods),
            "event_idx": idx,
        }

    # -- internals: sequence + loss mask --

    def _build_sequence(self, idx: int, input_mods: list, output_mods: list) -> np.ndarray:
        """Build the token sequence of one event:
            HEADER  : <MOD_OUT:m> for m in output_mods (alphabetical) + <TASK_SEP>
            INPUTS  : <MOD_IN:m_first> + IN content (per input_layout)
            OUTPUTS : <MOD_OUT:m_first> + OUT content (per output_layout) + <EOS>
            PAD     : right-fill to block_size
        """
        v = self.vocab
        parts: list[np.ndarray] = []

        # ---- HEADER: declare OUT modalities (alphabetical) ----
        out_mods_canon = sorted(output_mods)
        header = np.array([v.mod_out[m] for m in out_mods_canon], dtype=np.int64)
        parts.append(header)
        parts.append(np.array([v.task_sep], dtype=np.int64))

        # ---- INPUTS ----
        in_mods_canon = sorted(input_mods)
        if self.input_layout == "grouped":
            for m in in_mods_canon:
                parts.append(np.array([v.mod_in[m]], dtype=np.int64))
                parts.append(self._encode_event_modality(idx, m))
        else:  # interleaved
            if in_mods_canon:
                parts.append(np.array([v.mod_in[in_mods_canon[0]]], dtype=np.int64))
                parts.append(self._encode_interleaved(idx, in_mods_canon))

        # ---- OUTPUTS ----
        if out_mods_canon:
            parts.append(np.array([v.mod_out[out_mods_canon[0]]], dtype=np.int64))
            if self.output_layout == "grouped":
                # First modality's content directly after the helper-supplied <MOD_OUT:m_first>
                parts.append(self._encode_event_modality(idx, out_mods_canon[0]))
                # Remaining modalities: <MOD_OUT:m> + content each
                for m in out_mods_canon[1:]:
                    parts.append(np.array([v.mod_out[m]], dtype=np.int64))
                    parts.append(self._encode_event_modality(idx, m))
            else:  # interleaved
                parts.append(self._encode_interleaved(idx, out_mods_canon))

        parts.append(np.array([v.eos], dtype=np.int64))

        seq = np.concatenate(parts)
        if len(seq) > self.block_size:
            seq = seq[: self.block_size]
        else:
            pad_len = self.block_size - len(seq)
            seq = np.concatenate([seq, np.full(pad_len, v.pad, dtype=np.int64)])
        return seq

    def _encode_event_modality(self, idx: int, modality: str) -> np.ndarray:
        """Encode one modality's content + pos tokens for one event,
        flattened in per-element ``[c0..c{nq-1} p0..p{nqp-1}]`` order.

        Within-modality order is whatever's on disk (the tokenizer pre-sorted
        by pT-descending in ``convert_to_npy_tokenized.py`` for sortable modalities).
        """
        v = self.vocab
        mm = self.mms[modality]
        nq = v.num_quantizers[modality]
        nqp = v.num_q_pos[modality]
        cb = v.codebook_sizes[modality]
        cbp = v.pos_codebook_sizes[modality]

        content = mm.event_codes(idx, nq)
        if nqp > 0 and mm.n_pos_codebooks >= nqp:
            pos = mm.event_pos_codes(idx, nqp)
        else:
            pos = np.zeros((content.shape[0], nqp), dtype=np.int64)

        # clip into the codebook range
        content = np.clip(content, 0, cb - 1)
        pos = np.clip(pos, 0, cbp - 1)

        if content.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)
        return v.encode_element_with_pos(modality, content, pos)

    def _encode_interleaved(self, idx: int, modalities: list) -> np.ndarray:
        """Round-robin pop one element from each modality's pT-descending list.

        When one modality runs out, continue with the others. Element layout
        within each pop is ``[c0..c{nq-1} p0..p{nqp-1}]``.
        """
        v = self.vocab
        # Pre-compute the per-modality flat element streams (already encoded
        # into global IDs); each is a numpy array of shape (n_elems * width,).
        streams: dict[str, tuple[np.ndarray, int]] = {}
        for m in modalities:
            enc = self._encode_event_modality(idx, m)
            width = v.element_token_width(m)
            n_elems = (enc.shape[0] // width) if width > 0 else 0
            streams[m] = (enc.reshape(n_elems, width) if n_elems > 0 else enc.reshape(0, max(width, 1)), width)

        out: list[np.ndarray] = []
        cursors = {m: 0 for m in modalities}
        # round-robin
        while True:
            emitted_any = False
            for m in modalities:
                arr, _ = streams[m]
                ci = cursors[m]
                if ci < arr.shape[0]:
                    out.append(arr[ci])
                    cursors[m] = ci + 1
                    emitted_any = True
            if not emitted_any:
                break
        if not out:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(out).astype(np.int64)

    def _build_loss_mask(self, seq: np.ndarray) -> np.ndarray:
        """Loss mask matching the plan's rule (derivable from token IDs only):

        - 0 on header MOD_OUT declarations and TASK_SEP
        - 0 on MOD_IN tokens and all IN content
        - 0 on the FIRST MOD_OUT token after TASK_SEP (helper-supplied "begin gen" cue)
        - 1 on OUT content
        - 1 on internal MOD_OUT boundaries (grouped only -- absent in interleaved)
        - 1 on final EOS
        - 0 on PAD
        """
        v = self.vocab
        mask = np.zeros_like(seq, dtype=np.int64)

        # Locate TASK_SEP -- end of header.
        task_sep_pos = np.where(seq == v.task_sep)[0]
        if len(task_sep_pos) == 0:
            return mask
        body_start = int(task_sep_pos[0]) + 1

        # Find first MOD_OUT in the body -- this is the helper-supplied
        # "begin generation" cue, loss=0.
        mod_out_vals = set(v.mod_out.values())
        body = seq[body_start:]
        in_body_is_mod_out = np.isin(body, list(mod_out_vals))
        first_mod_out_local = np.where(in_body_is_mod_out)[0]
        if len(first_mod_out_local) == 0:
            return mask
        gen_start = body_start + int(first_mod_out_local[0])

        # From gen_start+1 onwards, mark OUT content + internal boundaries + EOS.
        # PAD and anything past the first EOS get 0.
        eos_locs = np.where(seq == v.eos)[0]
        gen_end = int(eos_locs[0]) if len(eos_locs) > 0 else len(seq) - 1
        # loss = 1 on [gen_start + 1 ... gen_end] inclusive
        if gen_end + 1 > gen_start + 1:
            mask[gen_start + 1 : gen_end + 1] = 1
        return mask


# ===========================================================================
# Randomized-split batch sampler for nanoHEP any-to-any training
# ===========================================================================

from torch.utils.data import BatchSampler


class NanoHepFracBatchSampler(BatchSampler):
    """Per-batch random (input_mods, output_mods) sampling for nanoHEP.

    ``__iter__`` yields lists of ``(event_idx, split_idx)`` tuples, one per batch
    slot, which the dataset's ``__getitem__`` consumes. One split is drawn per
    batch (``split_idx=-1``: the fixed split of the sampling_dict).
    """

    def __init__(
        self,
        batch_size: int,
        valid_splits: list,
        sampling_type: str = "inference_rand2",
        sampler=None,
        drop_last: bool = True,
        fixed_frac: float = 0.0,
    ):
        # fixed_frac: probability that a batch uses the FIXED
        # fixed_input_output_modalities split (split_idx=-1) instead of a
        # random one — keeps a benchmark direction (e.g. pflow) in-distribution
        # while the rest of training covers all modalities.
        self.fixed_frac = float(fixed_frac)
        super().__init__(sampler, batch_size, drop_last=drop_last)
        if sampling_type not in ("inference", "inference_rand2"):
            raise ValueError(
                f"sampling_type must be inference|inference_rand2; got {sampling_type!r}"
            )
        self.sampling_type = sampling_type
        self.valid_splits = valid_splits
        if sampling_type == "inference_rand2" and not valid_splits:
            raise ValueError(
                "inference_rand2 requires a non-empty valid_splits list "
                "(call ds._compute_valid_splits() with max_inp_modality / "
                "max_out_modality set)."
            )

    def _pick_split_idx(self) -> int:
        if self.sampling_type == "inference":
            return -1  # sentinel for "use fixed_input_output_modalities"
        if self.fixed_frac > 0 and torch.rand(1).item() < self.fixed_frac:
            return -1
        return int(torch.randint(0, len(self.valid_splits), (1,)).item())

    def __iter__(self):
        for batch_indices in super().__iter__():
            split_idx = self._pick_split_idx()
            yield [(int(idx), split_idx) for idx in batch_indices]
