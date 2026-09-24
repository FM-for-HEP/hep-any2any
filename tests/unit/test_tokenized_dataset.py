"""Tests for ``hep4m.datasets.dataset_tokens_numpy.TokenDatasetNumpy`` over
the committed tokenized fixture.

The fixture is a 100-event slice of a val-split tokenised store for four
COCOA modalities, made by ``tests/unit/fixtures/generate_tokenized_fixture.py``.
"""
from pathlib import Path

import numpy as np
import pytest
import torch


REDUCE_DS = 50  # only load 50 of the 100 events for fast tests
MAX_TOKEN_CARDINALITY = 50


# ---------------------------------------------------------------------------
# Shared utilities -- per-modality codebook config (sourced from the canonical
# COCOA VQ-VAE checkpoint dirs). Inlined as test fixtures so tests have no
# filesystem dependency beyond the tokenized fixture itself.
# ---------------------------------------------------------------------------

_MODALITY_SHAPES = {
    "track":     (3, 256, 3),
    "topo":      (3, 256, 3),
    "truthpart": (3, 128, 3),
    "truthjet":  (1, 128, 3),
}
_POS_CODEBOOK_SIZE = 1024


def _make_config_v(modality: str) -> dict:
    """Minimal config_v sufficient to drive TokenDatasetNumpy + the
    .getitem(idx, is_input=True) path. Skips the raw-data preprocessing fields
    (transformations, branches_to_read, ...) since we're operating
    purely on already-tokenized data."""
    return {
        "modality": modality,
        "max_token_cardinality": MAX_TOKEN_CARDINALITY,
        "data_type": "set",
    }


def _load_shared_event_numbers(val_path: str) -> np.ndarray:
    """The shared event_numbers.npy is a raw int64 memmap (no .npy header).
    Matches ``hep4m.datasets.dataset_hep4m`` line 57."""
    return np.fromfile(Path(val_path) / "event_numbers.npy", dtype=np.int64)


# ---------------------------------------------------------------------------
# Fixture-level sanity (file presence + shape sanity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modality", sorted(_MODALITY_SHAPES))
def test_fixture_files_exist(tokenized_fixture_val, modality):
    base = Path(tokenized_fixture_val)
    for ext in ("meta.npz", "offsets.npy", "data.npy"):
        p = base / f"{modality}_{ext}"
        assert p.exists(), f"Missing {p}"


def test_fixture_event_numbers_present(tokenized_fixture_val):
    assert (Path(tokenized_fixture_val) / "event_numbers.npy").exists()
    evnums = _load_shared_event_numbers(tokenized_fixture_val)
    assert len(evnums) == 100, f"expected 100 event_numbers; got {len(evnums)}"
    assert evnums.dtype == np.int64


@pytest.mark.parametrize("modality", sorted(_MODALITY_SHAPES))
def test_fixture_meta_shape(tokenized_fixture_val, modality):
    """Per-modality meta matches the documented (n_codebooks, n_pos_codebooks)."""
    meta = np.load(Path(tokenized_fixture_val) / f"{modality}_meta.npz", allow_pickle=True)
    assert int(meta["n_events"]) == 100
    n_cb_expected, _cb_size, n_pos_expected = _MODALITY_SHAPES[modality]
    assert int(meta["n_codebooks"]) == n_cb_expected
    assert int(meta["n_pos_codebooks"]) == n_pos_expected


# ---------------------------------------------------------------------------
# TokenDatasetNumpy: per-modality construction + shape/dtype/range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modality", sorted(_MODALITY_SHAPES))
class TestTokenDatasetNumpy:

    def _make(self, val_path: str, modality: str):
        from hep4m.datasets.dataset_tokens_numpy import TokenDatasetNumpy
        evnums = _load_shared_event_numbers(val_path)
        cv = _make_config_v(modality)
        return TokenDatasetNumpy(
            data_path=str(Path(val_path) / f"{modality}_data.npy"),
            config_v=cv,
            start_idx=0,
            reduce_ds=REDUCE_DS,
            shared_event_numbers=evnums,
        )

    def test_instantiation(self, tokenized_fixture_val, modality):
        ds = self._make(tokenized_fixture_val, modality)
        assert len(ds) == REDUCE_DS
        n_cb, _cb_size, n_pos = _MODALITY_SHAPES[modality]
        assert ds.n_codebooks == n_cb
        assert ds.n_pos_codebooks == n_pos
        assert ds.n_cols == n_cb + n_pos

    def test_getitem_keys(self, tokenized_fixture_val, modality):
        ds = self._make(tokenized_fixture_val, modality)
        item = ds.getitem(0, is_input=True)
        assert "inp_dict" in item
        assert "target_dict" in item
        assert "event_number" in item
        assert "getitem_idx" in item
        assert item["getitem_idx"] == 0

    def test_input_dict_shapes_and_dtypes(self, tokenized_fixture_val, modality):
        ds = self._make(tokenized_fixture_val, modality)
        item = ds.getitem(0, is_input=True)
        inp = item["inp_dict"]
        n_cb, _cb_size, n_pos = _MODALITY_SHAPES[modality]

        assert "tokens" in inp
        tokens = inp["tokens"]
        assert isinstance(tokens, torch.Tensor)
        assert tokens.dtype == torch.int64
        assert tokens.ndim == 2
        assert tokens.shape[1] == n_cb, (
            f"{modality}: tokens.shape[1]={tokens.shape[1]} != n_codebooks={n_cb}"
        )

        if n_pos > 0:
            assert "pos_tokens" in inp
            pos = inp["pos_tokens"]
            assert pos.dtype == torch.int64
            assert pos.shape[1] == n_pos

    def test_token_values_in_codebook_range(self, tokenized_fixture_val, modality):
        """Per-codebook indices must fall in [0, codebook_size). This catches
        tokenization corruption or codebook-size mismatches between the
        VQ-VAE checkpoint and the on-disk memmap."""
        ds = self._make(tokenized_fixture_val, modality)
        _, codebook_size, _ = _MODALITY_SHAPES[modality]
        oob_count = 0
        for idx in range(min(REDUCE_DS, 20)):
            item = ds.getitem(idx, is_input=True)
            tokens = item["inp_dict"]["tokens"]
            if tokens.numel() == 0:
                continue
            if int(tokens.min()) < 0 or int(tokens.max()) >= codebook_size:
                oob_count += 1
        assert oob_count == 0, (
            f"{modality}: {oob_count}/{min(REDUCE_DS, 20)} events have content "
            f"tokens out of [0, {codebook_size})"
        )

    def test_pos_token_values_in_pos_codebook_range(self, tokenized_fixture_val, modality):
        """Per-quantizer pos indices must fall in [0, pos_codebook_size). The pos
        tokenizer is universal across modalities (3x1024)."""
        n_cb, _cb_size, n_pos = _MODALITY_SHAPES[modality]
        if n_pos == 0:
            pytest.skip(f"{modality} has no pos codebooks")
        ds = self._make(tokenized_fixture_val, modality)
        oob_count = 0
        for idx in range(min(REDUCE_DS, 20)):
            item = ds.getitem(idx, is_input=True)
            pos = item["inp_dict"]["pos_tokens"]
            if pos.numel() == 0:
                continue
            if int(pos.min()) < 0 or int(pos.max()) >= _POS_CODEBOOK_SIZE:
                oob_count += 1
        assert oob_count == 0, (
            f"{modality}: {oob_count}/{min(REDUCE_DS, 20)} events have pos "
            f"tokens out of [0, {_POS_CODEBOOK_SIZE})"
        )

    def test_cardinality_matches_inp_length(self, tokenized_fixture_val, modality):
        """inp_dict.cardinality is the real (non-padded) element count; with
        is_input=True it should equal inp_dict.tokens.shape[0]."""
        ds = self._make(tokenized_fixture_val, modality)
        for idx in [0, 1, 5, 10]:
            item = ds.getitem(idx, is_input=True)
            inp_n = int(item["inp_dict"]["tokens"].shape[0])
            card = int(item["inp_dict"]["cardinality"])
            assert inp_n == card, f"{modality} event {idx}: inp_n={inp_n} != card={card}"

    def test_event_number_aligned(self, tokenized_fixture_val, modality):
        """event_number passed through shared_event_numbers should match
        what the dataset reports per event."""
        ds = self._make(tokenized_fixture_val, modality)
        evnums = _load_shared_event_numbers(tokenized_fixture_val)
        for idx in [0, 7, 23, 49]:
            item = ds.getitem(idx, is_input=True)
            assert int(item["event_number"]) == int(evnums[idx])


# ---------------------------------------------------------------------------
# Cross-modality consistency: all four modalities have the same n_events
# (so a multi-modality COCOADatasetHEP4M can zip them together).
# ---------------------------------------------------------------------------

def test_all_modalities_same_n_events(tokenized_fixture_val):
    from hep4m.datasets.dataset_tokens_numpy import TokenDatasetNumpy
    evnums = _load_shared_event_numbers(tokenized_fixture_val)
    ns = {}
    for m in sorted(_MODALITY_SHAPES):
        ds = TokenDatasetNumpy(
            data_path=str(Path(tokenized_fixture_val) / f"{m}_data.npy"),
            config_v=_make_config_v(m),
            start_idx=0, reduce_ds=-1,
            shared_event_numbers=evnums,
        )
        ns[m] = len(ds)
    assert len(set(ns.values())) == 1, f"n_events varies across modalities: {ns}"
    assert next(iter(ns.values())) == 100
