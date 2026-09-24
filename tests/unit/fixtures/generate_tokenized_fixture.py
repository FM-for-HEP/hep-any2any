"""Generate a tiny tokenized fixture for HEP4M tests.

Slices the first 100 events of each modality from a val-split tokenised
store and writes the result to ``tests/unit/fixtures/tokenized_100events/val/``.

    python tests/unit/fixtures/generate_tokenized_fixture.py

On-disk layout produced (per modality m):

    tests/fixtures/tokenized_100events/val/{m}_meta.npz
    tests/fixtures/tokenized_100events/val/{m}_offsets.npy    int64
    tests/fixtures/tokenized_100events/val/{m}_data.npy       int16
    tests/fixtures/tokenized_100events/val/{m}_is_empty.npy   int64
    tests/fixtures/tokenized_100events/val/event_numbers.npy  int64

Format matches what ``hep4m.pre_processing.convert_to_npy_tokenized.py``
writes -- ``int64`` offsets, raw memmap (no .npy header) for offsets and
data, ``.npz`` for the metadata bundle.
"""
from __future__ import annotations

import hep4m.paths as _HP
import os
from pathlib import Path

import numpy as np


SRC = Path(f"{_HP.DATA}/Tokenized_89M/val")
DST = Path(__file__).parent / "tokenized_100events" / "val"
N_KEEP = 100

MODALITIES = ["topo", "track", "truthjet", "truthpart"]


def _read_offsets(path: Path) -> np.ndarray:
    """Read a raw int64 offsets memmap (no .npy header)."""
    n_bytes = os.path.getsize(path)
    n_entries = n_bytes // 8
    return np.memmap(path, dtype=np.int64, mode="r", shape=(n_entries,))


def _read_data(path: Path, n_cols: int, total_elem: int) -> np.ndarray:
    """Read raw int16 data memmap (no .npy header)."""
    return np.memmap(path, dtype=np.int16, mode="r", shape=(total_elem, n_cols))


def _slice_modality(modality: str) -> None:
    """Slice this modality's val tokens to the first N_KEEP events and write to DST."""
    meta = np.load(SRC / f"{modality}_meta.npz", allow_pickle=True)
    n_events_full = int(meta["n_events"])
    n_codebooks = int(meta["n_codebooks"])
    n_pos_codebooks = int(meta["n_pos_codebooks"]) if "n_pos_codebooks" in meta.files else 0
    n_cols = n_codebooks + n_pos_codebooks

    if N_KEEP > n_events_full:
        raise ValueError(f"{modality}: N_KEEP={N_KEEP} > n_events_full={n_events_full}")

    offsets_full = _read_offsets(SRC / f"{modality}_offsets.npy")
    assert offsets_full.shape[0] == n_events_full + 1, (
        f"{modality}: offsets length {offsets_full.shape[0]} != n_events+1={n_events_full + 1}"
    )

    offsets_keep = np.array(offsets_full[: N_KEEP + 1], dtype=np.int64)
    total_elem_keep = int(offsets_keep[-1])

    file_bytes = os.path.getsize(SRC / f"{modality}_data.npy")
    row_bytes = n_cols * 2  # int16
    total_elem_full = file_bytes // row_bytes
    data_full = _read_data(SRC / f"{modality}_data.npy", n_cols, total_elem_full)
    data_keep = np.array(data_full[:total_elem_keep], dtype=np.int16)

    # Filter is_empty to indices < N_KEEP.
    is_empty_path = SRC / f"{modality}_is_empty.npy"
    if is_empty_path.exists():
        is_empty = np.load(is_empty_path)
        is_empty_keep = is_empty[is_empty < N_KEEP].astype(np.int64)
    else:
        is_empty_keep = np.empty((0,), dtype=np.int64)

    # Write
    DST.mkdir(parents=True, exist_ok=True)
    np.savez(
        DST / f"{modality}_meta.npz",
        n_events=np.int64(N_KEEP),
        n_codebooks=np.int64(n_codebooks),
        n_pos_codebooks=np.int64(n_pos_codebooks),
    )
    offsets_keep.tofile(DST / f"{modality}_offsets.npy")
    data_keep.tofile(DST / f"{modality}_data.npy")
    np.save(DST / f"{modality}_is_empty.npy", is_empty_keep)

    print(
        f"  {modality}: n_events={N_KEEP} n_codebooks={n_codebooks} "
        f"n_pos_codebooks={n_pos_codebooks} total_elem={total_elem_keep} "
        f"data={data_keep.nbytes / 1024:.1f} KB offsets={offsets_keep.nbytes / 1024:.1f} KB"
    )


def _slice_event_numbers() -> None:
    """Truncate the shared event_numbers.npy (raw int64, no .npy header) to N_KEEP."""
    src = SRC / "event_numbers.npy"
    evnums = np.fromfile(src, dtype=np.int64)
    keep = evnums[:N_KEEP].astype(np.int64)
    keep.tofile(DST / "event_numbers.npy")
    print(f"  event_numbers: kept {len(keep)} entries")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source not found: {SRC} (set HEP4M_DATA)")
    print(f"Source:      {SRC}")
    print(f"Destination: {DST}")
    print(f"N_KEEP:      {N_KEEP}")
    print()
    for m in MODALITIES:
        _slice_modality(m)
    _slice_event_numbers()
    print()
    print("Total fixture size:")
    total = 0
    for p in sorted(DST.glob("*")):
        sz = os.path.getsize(p)
        total += sz
        print(f"  {p.name:40s} {sz / 1024:>8.1f} KB")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':40s} {total / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
