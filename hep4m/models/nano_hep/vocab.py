"""Joint nanoHEP vocabulary over all modalities, with input and output role tags.

Each modality has disjoint content and position token ranges
(``offsets[m] + q * codebook_size``), so the model's choice of the next content
token is also a choice of modality. Special tokens: ``<MOD_IN:m>`` and
``<MOD_OUT:m>`` per modality, ``<TASK_SEP>`` between the output-modality header
and the body, ``<EOS>`` and ``<PAD>``. ``modality_of`` maps any token id back to its
modality, which the decoder uses to split an interleaved output stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# Defaults match the released tokenisers (config_m.yml under ${HEP4M_TOKENIZERS})
DEFAULT_MODALITY_CODEBOOK_SIZE: Dict[str, int] = {
    "track": 256,
    "topo": 256,
    "truthpart": 128,
    "truthjet": 128,
    "hgpfpart": 128,
    "cell":      2048,
    "celltruth": 2048,
}
DEFAULT_MODALITY_NUM_QUANTIZERS: Dict[str, int] = {
    "track": 3, "topo": 3, "truthpart": 3, "truthjet": 1,
    "hgpfpart": 3, "cell": 3, "celltruth": 3,
}
DEFAULT_POS_CODEBOOK_SIZE: int = 1024
DEFAULT_NUM_Q_POS: int = 3


@dataclass(frozen=True)
class Vocab:
    """Vocabulary with input/output role tags.

    Layout (cursor walk):
        for m in modalities:
            offsets[m]      ..  +num_q * codebook_size   (content)
            pos_offsets[m]  ..  +num_q_pos * pos_codebook_size  (pos)
        mod_in[m] for m in modalities      (one token per modality)
        mod_out[m] for m in modalities     (one token per modality)
        task_sep                            (1 token)
        eos                                 (1 token)
        pad                                 (1 token)

    Total specials = 2 * len(modalities) + 3.
    """

    modalities: List[str]
    codebook_sizes: Dict[str, int]
    num_quantizers: Dict[str, int]
    pos_codebook_sizes: Dict[str, int]
    num_q_pos: Dict[str, int]

    offsets: Dict[str, int]
    pos_offsets: Dict[str, int]

    mod_in: Dict[str, int]
    mod_out: Dict[str, int]
    task_sep: int
    eos: int
    pad: int
    total: int

    # Reverse lookup tables of length ``total`` (built in ``build``): the modality
    # name of a content (position) token, "" for any other token.
    _content_token_to_modality: tuple = field(default=(), repr=False)
    _pos_token_to_modality: tuple = field(default=(), repr=False)

    # ----- derived properties -----

    def element_token_width(self, mod: str) -> int:
        """Tokens emitted per element (content + pos)."""
        return self.num_quantizers[mod] + self.num_q_pos[mod]

    # ----- lookup -----

    def modality_of(self, tok: int) -> str:
        """Best-effort modality lookup for any token.

        Returns the modality name for content/pos tokens, an empty string
        for specials. Used by the ``interleaved``-layout partitioner.
        """
        if 0 <= tok < len(self._content_token_to_modality):
            m = self._content_token_to_modality[tok]
            if m:
                return m
            m = self._pos_token_to_modality[tok]
            if m:
                return m
        return ""

    # ----- construction -----

    @staticmethod
    def build(
        modalities: List[str],
        codebook_sizes: Dict[str, int] | None = None,
        num_quantizers: Dict[str, int] | None = None,
        pos_codebook_sizes: Dict[str, int] | None = None,
        num_q_pos: Dict[str, int] | None = None,
    ) -> "Vocab":
        """Build a vocabulary.

        `modalities` ordered list (canonical = alphabetical, but accept any
        order; cursor walk preserves it).

        Defaults pulled from the COCOA VQ-VAE checkpoints. If you pass a
        custom dict for any of (codebook_sizes, num_quantizers,
        pos_codebook_sizes, num_q_pos), it must cover every modality.
        """
        # Deduplicate while preserving order.
        seen = set()
        mods_ord = []
        for m in modalities:
            if m not in seen:
                seen.add(m); mods_ord.append(m)

        if codebook_sizes is None:
            codebook_sizes = {m: DEFAULT_MODALITY_CODEBOOK_SIZE[m] for m in mods_ord}
        if num_quantizers is None:
            num_quantizers = {m: DEFAULT_MODALITY_NUM_QUANTIZERS[m] for m in mods_ord}
        if pos_codebook_sizes is None:
            pos_codebook_sizes = {m: DEFAULT_POS_CODEBOOK_SIZE for m in mods_ord}
        if num_q_pos is None:
            num_q_pos = {m: DEFAULT_NUM_Q_POS for m in mods_ord}

        for d, name in [(codebook_sizes, "codebook_sizes"),
                        (num_quantizers, "num_quantizers"),
                        (pos_codebook_sizes, "pos_codebook_sizes"),
                        (num_q_pos, "num_q_pos")]:
            missing = [m for m in mods_ord if m not in d]
            if missing:
                raise ValueError(f"{name} missing modalities: {missing}")

        offsets: Dict[str, int] = {}
        pos_offsets: Dict[str, int] = {}
        cursor = 0
        for m in mods_ord:
            offsets[m] = cursor
            cursor += num_quantizers[m] * codebook_sizes[m]
            pos_offsets[m] = cursor
            cursor += num_q_pos[m] * pos_codebook_sizes[m]

        mod_in = {m: cursor + i for i, m in enumerate(mods_ord)}
        cursor += len(mods_ord)
        mod_out = {m: cursor + i for i, m in enumerate(mods_ord)}
        cursor += len(mods_ord)
        task_sep = cursor; cursor += 1
        eos = cursor; cursor += 1
        pad = cursor; cursor += 1

        total = cursor

        # Build reverse-lookup arrays of length total. content_token_to_modality[tid]
        # is "" for non-content tokens, and modality name otherwise.
        content_lut = [""] * total
        pos_lut = [""] * total
        for m in mods_ord:
            cs = num_quantizers[m] * codebook_sizes[m]
            for i in range(cs):
                content_lut[offsets[m] + i] = m
            ps = num_q_pos[m] * pos_codebook_sizes[m]
            for i in range(ps):
                pos_lut[pos_offsets[m] + i] = m

        return Vocab(
            modalities=list(mods_ord),
            codebook_sizes={m: codebook_sizes[m] for m in mods_ord},
            num_quantizers={m: num_quantizers[m] for m in mods_ord},
            pos_codebook_sizes={m: pos_codebook_sizes[m] for m in mods_ord},
            num_q_pos={m: num_q_pos[m] for m in mods_ord},
            offsets=dict(offsets),
            pos_offsets=dict(pos_offsets),
            mod_in=dict(mod_in),
            mod_out=dict(mod_out),
            task_sep=task_sep,
            eos=eos,
            pad=pad,
            total=total,
            _content_token_to_modality=tuple(content_lut),
            _pos_token_to_modality=tuple(pos_lut),
        )

    # ----- encoders -----

    def encode_element_with_pos(
        self,
        mod: str,
        content_codes: np.ndarray,
        pos_codes: np.ndarray,
    ) -> np.ndarray:
        """Encode (content + pos) per element, content-first.

        Inputs:
          content_codes: (N, num_q_content)  local codebook IDs
          pos_codes:     (N, num_q_pos)      local pos codebook IDs

        Returns flat int64 array of length ``N * (num_q + num_q_pos)``
        with per-element layout ``[c0..c{nq_c-1} p0..p{nq_p-1}]``.
        """
        c = np.asarray(content_codes, dtype=np.int64)
        p = np.asarray(pos_codes, dtype=np.int64)
        nq_c = self.num_quantizers[mod]
        nq_p = self.num_q_pos[mod]
        cb_c = self.codebook_sizes[mod]
        cb_p = self.pos_codebook_sizes[mod]
        if c.ndim != 2 or c.shape[1] != nq_c:
            raise ValueError(f"content expected (N, {nq_c}), got {c.shape}")
        if p.ndim != 2 or p.shape[1] != nq_p:
            raise ValueError(f"pos expected (N, {nq_p}), got {p.shape}")
        if c.shape[0] != p.shape[0]:
            raise ValueError(f"content N={c.shape[0]} != pos N={p.shape[0]}")
        q_offs_c = np.arange(nq_c, dtype=np.int64) * cb_c
        q_offs_p = np.arange(nq_p, dtype=np.int64) * cb_p
        c_shifted = c + q_offs_c[None, :] + self.offsets[mod]
        p_shifted = p + q_offs_p[None, :] + self.pos_offsets[mod]
        per_elem = np.concatenate([c_shifted, p_shifted], axis=1)
        return per_elem.reshape(-1)

    # ----- decoders -----

    def decode_element_with_pos(
        self,
        mod: str,
        global_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Strict inverse of ``encode_element_with_pos``.

        Returns ``(content_local (N, num_q), pos_local (N, num_q_pos))``
        of local codebook IDs. Raises ``ValueError`` on out-of-range IDs.
        """
        ids = np.asarray(global_ids, dtype=np.int64)
        nq_c = self.num_quantizers[mod]
        nq_p = self.num_q_pos[mod]
        group = nq_c + nq_p
        if ids.size % group != 0:
            raise ValueError(
                f"global_ids length {ids.size} not a multiple of group={group}"
            )
        n = ids.size // group
        grouped = ids.reshape(n, group)
        c_block = grouped[:, :nq_c]
        p_block = grouped[:, nq_c:]
        cb_c = self.codebook_sizes[mod]
        cb_p = self.pos_codebook_sizes[mod]
        off_c = self.offsets[mod]
        off_p = self.pos_offsets[mod]
        c_lo, c_hi = off_c, off_c + nq_c * cb_c
        p_lo, p_hi = off_p, off_p + nq_p * cb_p
        if not ((c_block >= c_lo) & (c_block < c_hi)).all():
            raise ValueError(f"content ids out of [{c_lo},{c_hi}) for mod={mod}")
        if not ((p_block >= p_lo) & (p_block < p_hi)).all():
            raise ValueError(f"pos ids out of [{p_lo},{p_hi}) for mod={mod}")
        q_offs_c = np.arange(nq_c, dtype=np.int64) * cb_c
        q_offs_p = np.arange(nq_p, dtype=np.int64) * cb_p
        c_local = c_block - off_c - q_offs_c[None, :]
        p_local = p_block - off_p - q_offs_p[None, :]
        return c_local, p_local

    def __repr__(self) -> str:
        parts = []
        for m in self.modalities:
            c_lo = self.offsets[m]
            c_hi = c_lo + self.num_quantizers[m] * self.codebook_sizes[m]
            p_lo = self.pos_offsets[m]
            p_hi = p_lo + self.num_q_pos[m] * self.pos_codebook_sizes[m]
            parts.append(
                f"{m}:content[{c_lo},{c_hi})x{self.num_quantizers[m]}q "
                f"pos[{p_lo},{p_hi})x{self.num_q_pos[m]}q"
            )
        return (
            f"Vocab(total={self.total}, modalities=[{', '.join(parts)}], "
            f"mod_in={self.mod_in}, mod_out={self.mod_out}, "
            f"task_sep={self.task_sep}, eos={self.eos}, pad={self.pad})"
        )
