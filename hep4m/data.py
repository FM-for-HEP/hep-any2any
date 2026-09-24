"""Loader, packer and checker for the HEP4M tokenised COCOA single-jet release.

Dependencies: numpy and zstandard (``python -m pip install numpy zstandard``).
No torch and no HEP4M package import, so the file can be copied on its own.

Two on-disk layouts
-------------------

1. **Original layout**, read by ``hep4m/datasets/tokenized_memmap.py`` (one directory
   per split)::

       {mod}_data.npy       headerless int16, shape (n_rows, n_codebooks + n_pos_codebooks)
       {mod}_offsets.npy    headerless int64, shape (n_events + 1,)
       {mod}_meta.npz       n_events, n_codebooks, n_pos_codebooks
       {mod}_is_empty.npy   .npy with header, int64 indices of events with zero rows
       event_numbers.npy    headerless int64, shape (n_events,)

2. **Release layout** (the Zenodo record). Zenodo has no folders, so names are flat and
   carry the split as a prefix. Every ``.npy.zst`` is a normal ``.npy`` (with header)
   compressed with zstd, so ``zstd -d f.npy.zst`` followed by ``np.load("f.npy")`` works::

       schema.json                               describes everything below
       SHA256SUMS                                ``sha256sum -c SHA256SUMS`` compatible
       cell_window_lut.npy                       int16 (n_windows, 156, 3): cell position
                                                 tokens of each calorimeter window
       {split}.event_numbers.npy.zst             int64 (n_events,)
       {split}.{mod}_data.npy.zst                int16 (n_rows, n_cb + n_pos_cb), original
                                                 columns; mod = track, topo, truthpart,
                                                 hgpfpart, truthjet
       {split}.{mod}_counts.npy.zst              uint8 (n_events,) rows per event;
                                                 mod = track, topo, truthpart, hgpfpart
       {split}.{mod}_content.partNN.npy.zst      int16 (156 * n_events_part, 3), content
                                                 tokens only; mod = cell, celltruth; parts
                                                 are cut at event boundaries
       {split}.cell_window_index.npy.zst         uint16 (n_events,) row of the window table
                                                 for each event (cell and celltruth share it)
       {split}.original_meta.tar                 the original {mod}_meta.npz and
                                                 {mod}_is_empty.npy files, byte for byte

What is rebuilt
---------------

- offsets: ``concatenate([[0], cumsum(counts)])`` for track/topo/truthpart/hgpfpart,
  ``156 * arange(n + 1)`` for cell/celltruth and ``8 * arange(n + 1)`` for truthjet;
- cell and celltruth position tokens: ``cell_window_lut[window_index[event]]`` (156 x 3).

``export_original_layout`` writes the original layout back and, when ``schema.json``
carries the sha256 of the original files, checks every written file against it.

CLI
---

::

    python -m hep4m.data download [DST] [--record ID] [--include 'val.*' 'test.*']
                                  [--checkpoints [ROW ...]] [--tokenizers] [--raw-test]
                                  [--modalities M ...] [--splits S ...]
    python -m hep4m.data unpack [DIR] [--ckpt-dir D] [--tokenizers-dir D]
    python -m hep4m.data verify DIR [--partial]
    python -m hep4m.data info DIR
    python -m hep4m.data export DIR DST [--splits val test] [--modalities M ...]
                                [--max-events N] [--no-check]
    python -m hep4m.data decompress DIR [--splits val]   # plain .npy for memmap

``DST``/``DIR`` default to ``$HEP4M_RELEASE`` (``$HEP4M_DATA/release``). The trained models
are ``ckpt.<row>.tar.zst`` files in the same record (rows in ``CHECKPOINT_ROWS``); each
unpacks to ``ckpt.<row>/`` with ``model.ckpt``, ``config_t.yml``, ``config_m.yml`` (HEP4M
only), ``modality_dict.yml``, ``inference_test*.yml`` and ``README.json``. ``download
--checkpoints`` and ``unpack`` extract them into ``$HEP4M_CKPT`` and the tokenisers
(``tokenizers.tar.zst``) into ``$HEP4M_TOKENIZERS``.

Packing (how the record was built; the round-trip tests use it): ``build_window_lut``,
``pack_split``, ``build_schema``, ``write_sha256sums``, and ``pack_release`` which runs all four.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # only needed for the compressed files
    import zstandard
except ImportError:  # pragma: no cover
    zstandard = None

FORMAT_NAME = "hep4m-tokenised-release"
FORMAT_VERSION = 1

#: order used in the schema and in ``original_file_names``
MODALITIES: Tuple[str, ...] = ("cell", "celltruth", "track", "topo", "truthpart", "truthjet", "hgpfpart")
#: fixed rows per event (offsets are k * arange)
FIXED_ROWS: Dict[str, int] = {"cell": 156, "celltruth": 156, "truthjet": 8}
#: stored content-only; positions come from the window table
CELL_MODALITIES: Tuple[str, ...] = ("cell", "celltruth")
#: stored with uint8 per-event counts
COUNT_MODALITIES: Tuple[str, ...] = ("track", "topo", "truthpart", "hgpfpart")

PATCHES = 156
PART_EVENTS = 10_000_000
LUT_FILE = "cell_window_lut.npy"
DATA_DTYPE = np.dtype("<i2")
OFFSET_DTYPE = np.dtype("<i8")
COUNT_DTYPE = np.dtype("u1")
WINDOW_DTYPE = np.dtype("<u2")

#: Zenodo record of this release (DOI 10.5281/zenodo.22917591); --record / HEP4M_ZENODO_RECORD override it.
ZENODO_RECORD_ID: Optional[str] = "22917591"
ZENODO_API = "https://zenodo.org/api/records/{record}"

_BLOCK = 1 << 24

#: trained models in the record, one ``ckpt.<row>.tar.zst`` each
CHECKPOINT_ROWS: Tuple[str, ...] = ("nanoHEP-pflow", "nanoHEP-pflow-matched30k", "nanoHEP-sim", "nanoHEP-multi",
                                    "HEP4M-pflow", "HEP4M-sim", "HEP4M-multi")
TOKENIZERS_FILE = "tokenizers.tar.zst"
RAW_TEST_FILE = "eval.raw_test_singlejet_0_100kseg_bw100.0.root"
HGPFLOW_TEST_FILE = "eval.hgpflow_pred_singlejet_0_100kseg_bw100.0_merged.root"


def checkpoint_file(row: str) -> str:
    """Record file name of one trained model, e.g. ``ckpt.HEP4M-pflow.tar.zst``."""
    return f"ckpt.{row}.tar.zst"


def _site_dir(var: str, fallback: str) -> Path:
    """A location from ``hep4m.paths`` when the package is importable, else the environment."""
    try:
        import hep4m.paths  # noqa: F401  (sets the defaults in os.environ)
    except ImportError:
        pass
    return Path(os.environ.get(var) or fallback)


def default_release_dir() -> Path:
    return _site_dir("HEP4M_RELEASE", "release")


# ---------------------------------------------------------------------------
# .npy / zstd streaming helpers
# ---------------------------------------------------------------------------

def _require_zstd():
    if zstandard is None:
        raise ImportError("this needs the 'zstandard' package: python -m pip install zstandard")


def npy_header(dtype, shape) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
              "fortran_order": False, "shape": tuple(int(s) for s in shape)})
    return buf.getvalue()


def _read_npy_header(fh) -> Tuple[np.dtype, Tuple[int, ...]]:
    major, _ = np.lib.format.read_magic(fh)
    reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(fh)
    if fortran:
        raise ValueError("fortran-ordered arrays are not used in this release")
    return np.dtype(dtype), tuple(shape)


def _read_exact(fh, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        b = fh.read(min(n - len(out), _BLOCK))
        if not b:
            raise EOFError(f"stream ended after {len(out)} of {n} bytes")
        out += b
    return bytes(out)


class NpyZstStream:
    """Sequential row reader for a ``.npy.zst`` file (``.npy`` files work too)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._raw = open(self.path, "rb")
        if self.path.suffix == ".zst":
            _require_zstd()
            self._r = zstandard.ZstdDecompressor().stream_reader(self._raw, read_size=1 << 22)
        else:
            self._r = self._raw
        self.dtype, self.shape = _read_npy_header(self._r)
        self.n_rows = int(self.shape[0])
        self.row_bytes = int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize
        self.pos = 0

    def read_rows(self, n: int) -> np.ndarray:
        n = max(0, min(n, self.n_rows - self.pos))
        buf = _read_exact(self._r, n * self.row_bytes) if n else b""
        self.pos += n
        return np.frombuffer(buf, self.dtype).reshape((n,) + self.shape[1:])

    def skip_rows(self, n: int) -> None:
        step = max(1, _BLOCK // max(1, self.row_bytes))
        while n > 0:
            k = min(n, step)
            self.read_rows(k)
            n -= k

    def close(self):
        if self._r is not self._raw:
            self._r.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_npy_zst(path: Path) -> np.ndarray:
    """Load a whole ``.npy.zst`` (or ``.npy``) into memory."""
    with NpyZstStream(path) as s:
        return s.read_rows(s.n_rows).copy()


def npy_zst_header(path: Path) -> Tuple[np.dtype, Tuple[int, ...]]:
    with NpyZstStream(path) as s:
        return s.dtype, s.shape


class NpyZstWriter:
    """Write header + payload to ``path`` (``.npy.zst``) through a tmp file, checking that
    exactly the declared number of bytes was written."""

    def __init__(self, path: Path, dtype, shape, level: int = 19, threads: int = -1):
        _require_zstd()
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.dtype = np.dtype(dtype)
        hdr = npy_header(self.dtype, shape)
        self.expected = len(hdr) + int(np.prod(shape, dtype=np.int64)) * self.dtype.itemsize
        self._fh = open(self.tmp, "wb")
        cctx = zstandard.ZstdCompressor(level=level, threads=threads, write_checksum=True)
        self._w = cctx.stream_writer(self._fh, size=self.expected, closefd=False)
        self.n = 0
        self._write(hdr)

    def _write(self, b) -> None:
        mv = memoryview(b).cast("B")
        self._w.write(mv)
        self.n += len(mv)

    def write(self, arr: np.ndarray) -> None:
        self._write(np.ascontiguousarray(arr, dtype=self.dtype))

    def close(self) -> int:
        if self.n != self.expected:
            self._fh.close()
            self.tmp.unlink()
            raise ValueError(f"{self.path}: wrote {self.n} bytes, expected {self.expected}")
        self._w.close()
        self._fh.close()
        os.replace(self.tmp, self.path)
        return os.path.getsize(self.path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_BLOCK), b""):
            h.update(b)
    return h.hexdigest()


def _event_chunks(n: int, step: int) -> Iterator[Tuple[int, int]]:
    for a in range(0, n, step):
        yield a, min(n, a + step)


# ---------------------------------------------------------------------------
# Packing: original layout -> release layout
# ---------------------------------------------------------------------------

def _orig_meta(src: Path, mod: str) -> Tuple[int, int, int]:
    m = np.load(Path(src) / f"{mod}_meta.npz")
    return int(m["n_events"]), int(m["n_codebooks"]), int(m["n_pos_codebooks"])


def _orig_data(src: Path, mod: str):
    n, ncb, npc = _orig_meta(src, mod)
    w = ncb + npc
    size = os.path.getsize(Path(src) / f"{mod}_data.npy")
    if size % (2 * w):
        raise ValueError(f"{src}/{mod}_data.npy: {size} B is not a multiple of {2 * w}")
    data = np.memmap(Path(src) / f"{mod}_data.npy", dtype=DATA_DTYPE, mode="r", shape=(size // (2 * w), w))
    off = np.fromfile(Path(src) / f"{mod}_offsets.npy", dtype=OFFSET_DTYPE)
    if off.size != n + 1 or off[0] != 0 or off[-1] != data.shape[0]:
        raise ValueError(f"{src}/{mod}: offsets do not match n_events={n} / rows={data.shape[0]}")
    return (n, ncb, npc), off, data


def _pos_blocks(data, a: int, b: int, ncb: int) -> np.ndarray:
    return np.ascontiguousarray(data[a * PATCHES:b * PATCHES, ncb:ncb + 3]).reshape(b - a, PATCHES, 3)


def _block_keys(blocks: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(blocks, dtype=DATA_DTYPE).reshape(blocks.shape[0], -1)
    return flat.view(np.dtype((np.void, flat.shape[1] * 2))).ravel()


def sort_window_lut(blocks: np.ndarray) -> np.ndarray:
    """Canonical window order: by the eta token of patch 0, then the azimuth of patch 0
    (from its cos/sin tokens), then the full block. Same order as the release build."""
    blocks = np.asarray(blocks, dtype=DATA_DTYPE)
    r0 = blocks[:, 0, :].astype(np.float64)
    ang = np.arctan2(r0[:, 2] - 511.5, r0[:, 1] - 511.5)
    flat = blocks.reshape(blocks.shape[0], -1)
    keys = [flat[:, j] for j in range(flat.shape[1] - 1, -1, -1)] + [ang, r0[:, 0]]
    return blocks[np.lexsort(keys)]


def build_window_lut(src_dirs: Sequence[Path], chunk_events: int = 1_000_000) -> np.ndarray:
    """Distinct cell/celltruth position blocks over all splits, int16 (n_windows, 156, 3)."""
    seen: Dict[bytes, np.ndarray] = {}
    for src in src_dirs:
        for mod in CELL_MODALITIES:
            (n, ncb, _), _, data = _orig_data(Path(src), mod)
            for a, b in _event_chunks(n, chunk_events):
                blocks = _pos_blocks(data, a, b, ncb)
                keys, first = np.unique(_block_keys(blocks), return_index=True)
                for k, i in zip(keys, first):
                    seen.setdefault(k.tobytes(), blocks[i].copy())
    if not seen:
        raise ValueError("no cell position blocks found")
    if len(seen) > np.iinfo(WINDOW_DTYPE).max:
        raise ValueError(f"{len(seen)} windows do not fit uint16")
    return sort_window_lut(np.stack(list(seen.values())))


def window_index_for_blocks(blocks: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Row of ``lut`` for each (156, 3) block. Raises if a block is missing, so a successful
    call proves the rebuild is exact for these events."""
    lut_keys = {k.tobytes(): i for i, k in enumerate(_block_keys(lut))}
    keys, inverse = np.unique(_block_keys(blocks), return_inverse=True)
    try:
        ids = np.array([lut_keys[k.tobytes()] for k in keys], dtype=WINDOW_DTYPE)
    except KeyError:
        raise ValueError("a cell position block is not in the window lookup table") from None
    return ids[inverse.ravel()]


def _write_original_meta_tar(src: Path, out: Path, modalities: Sequence[str]) -> None:
    tmp = out.with_name(out.name + ".tmp")
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as t:
        for mod in modalities:
            for suf in ("_meta.npz", "_is_empty.npy"):
                p = Path(src) / f"{mod}{suf}"
                if not p.exists():
                    continue
                ti = t.gettarinfo(str(p), arcname=f"{mod}{suf}")
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mode = 0o644
                with open(p, "rb") as f:
                    t.addfile(ti, f)
    os.replace(tmp, out)


def pack_split(src: Path, dst: Path, split: str, window_lut: np.ndarray, *,
               modalities: Sequence[str] = MODALITIES, level: int = 19, threads: int = -1,
               part_events: int = PART_EVENTS, chunk_events: int = 100_000, log=print) -> dict:
    """Write one split of the original layout in ``src`` as release files ``dst/{split}.*``.

    Raises on: offsets that are not ``k * arange`` for fixed-row modalities, counts above
    255, a cell position block missing from ``window_lut``, celltruth positions that differ
    from cell positions, and modalities whose n_events differ from ``event_numbers``.
    Returns ``{original file name: sha256}`` of the source files (for ``build_schema``).
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {"event_numbers.npy": sha256_file(src / "event_numbers.npy")}
    ev = np.fromfile(src / "event_numbers.npy", dtype=OFFSET_DTYPE)
    n = int(ev.size)
    w = NpyZstWriter(dst / f"{split}.event_numbers.npy.zst", OFFSET_DTYPE, ev.shape, level, threads)
    w.write(ev)
    w.close()

    widx: Optional[np.ndarray] = None
    for mod in modalities:
        (nm, ncb, npc), off, data = _orig_data(src, mod)
        log(f"[pack] {split}.{mod}: {nm} events, {data.shape[0]} rows")
        if nm != n:
            raise ValueError(f"{split}.{mod}: {nm} events, event_numbers has {n}")
        for name in (f"{mod}_data.npy", f"{mod}_offsets.npy", f"{mod}_meta.npz", f"{mod}_is_empty.npy"):
            if (src / name).exists():
                hashes[name] = sha256_file(src / name)
        if mod in FIXED_ROWS:
            k = FIXED_ROWS[mod]
            if not np.array_equal(off, k * np.arange(n + 1, dtype=np.int64)):
                raise ValueError(f"{split}.{mod}: offsets are not {k}*arange")
        else:
            counts = np.diff(off)
            if counts.min() < 0 or counts.max() > np.iinfo(COUNT_DTYPE).max:
                raise ValueError(f"{split}.{mod}: counts outside 0..255")
            w = NpyZstWriter(dst / f"{split}.{mod}_counts.npy.zst", COUNT_DTYPE, counts.shape, level, threads)
            w.write(counts)
            w.close()

        if mod in CELL_MODALITIES:
            if (ncb, npc) != (3, 3):
                raise ValueError(f"{split}.{mod}: expected 3+3 columns, got {ncb}+{npc}")
            this = np.empty(n, dtype=WINDOW_DTYPE)
            for p, (a, b) in enumerate(_event_chunks(n, part_events)):
                w = NpyZstWriter(dst / f"{split}.{mod}_content.part{p:02d}.npy.zst", DATA_DTYPE,
                                 ((b - a) * PATCHES, 3), level, threads)
                for ca, cb in _event_chunks(b - a, chunk_events):
                    rows = np.asarray(data[(a + ca) * PATCHES:(a + cb) * PATCHES])
                    w.write(rows[:, :3])
                    this[a + ca:a + cb] = window_index_for_blocks(
                        rows[:, 3:6].reshape(cb - ca, PATCHES, 3), window_lut)
                w.close()
            if widx is None:
                widx = this
                w = NpyZstWriter(dst / f"{split}.cell_window_index.npy.zst", WINDOW_DTYPE, widx.shape, level, threads)
                w.write(widx)
                w.close()
            elif not np.array_equal(widx, this):
                raise ValueError(f"{split}.{mod}: position tokens differ from cell in "
                                 f"{int(np.count_nonzero(widx != this))} events")
        else:
            w = NpyZstWriter(dst / f"{split}.{mod}_data.npy.zst", DATA_DTYPE, data.shape, level, threads)
            step = max(1, _BLOCK // (2 * data.shape[1]))
            for r in range(0, data.shape[0], step):
                w.write(data[r:r + step])
            w.close()
    _write_original_meta_tar(src, dst / f"{split}.original_meta.tar", modalities)
    return hashes


def build_schema(root: Path, splits: Optional[Sequence[str]] = None,
                 original_sha256: Optional[Dict[str, Dict[str, str]]] = None,
                 extra_files: Optional[Dict[str, str]] = None, write: bool = True) -> dict:
    """Build ``schema.json`` from the release files in ``root`` (reads only .npy headers).

    ``original_sha256`` is ``{split: {original file name: sha256}}`` (from ``pack_split`` or
    a separate hashing pass); ``extra_files`` maps other record files to a description.
    """
    root = Path(root)
    names = sorted(p.name for p in root.iterdir() if p.is_file())
    if splits is None:
        found = {n.split(".", 1)[0] for n in names if n.endswith(".event_numbers.npy.zst")}
        splits = [s for s in ("train", "val", "test") if s in found] + sorted(found - {"train", "val", "test"})
    schema: dict = {"format": FORMAT_NAME, "format_version": FORMAT_VERSION,
                    "compression": {"codec": "zstd", "container": "npy"},
                    "modalities": {}, "splits": {}}
    lut_shape = None
    if (root / LUT_FILE).exists():
        lut_shape = list(np.load(root / LUT_FILE, mmap_mode="r").shape)
        schema["window_lut"] = {"file": LUT_FILE, "shape": lut_shape, "dtype": "int16"}
    for split in splits:
        _, (n,) = npy_zst_header(root / f"{split}.event_numbers.npy.zst")
        files: dict = {"event_numbers": {"file": f"{split}.event_numbers.npy.zst"}}
        for mod in MODALITIES:
            pat = re.compile(rf"{re.escape(split)}\.{mod}_content\.part\d+\.npy\.zst")
            parts = [x for x in names if pat.fullmatch(x)]
            if mod in CELL_MODALITIES and parts:
                entries, e = [], 0
                for f in parts:
                    _, shp = npy_zst_header(root / f)
                    k = shp[0] // PATCHES
                    entries.append({"file": f, "event_start": e, "event_stop": e + k})
                    e += k
                if e != n:
                    raise ValueError(f"{split}.{mod}: parts cover {e} of {n} events")
                files[f"{mod}_content"] = {"parts": entries, "n_rows": n * PATCHES}
                files["cell_window_index"] = {"file": f"{split}.cell_window_index.npy.zst"}
                ncb, npc = shp[1], (lut_shape[2] if lut_shape else 3)
            elif f"{split}.{mod}_data.npy.zst" in names:
                _, shp = npy_zst_header(root / f"{split}.{mod}_data.npy.zst")
                files[f"{mod}_data"] = {"file": f"{split}.{mod}_data.npy.zst", "n_rows": shp[0]}
                if mod in COUNT_MODALITIES:
                    files[f"{mod}_counts"] = {"file": f"{split}.{mod}_counts.npy.zst"}
                ncb, npc = shp[1] - 3, 3
            else:
                continue
            spec = {"n_codebooks": int(ncb), "n_pos_codebooks": int(npc),
                    "rows": "fixed" if mod in FIXED_ROWS else "counts",
                    "pos_source": "window_lut" if mod in CELL_MODALITIES else "inline"}
            if mod in FIXED_ROWS:
                spec["rows_per_event"] = FIXED_ROWS[mod]
            if schema["modalities"].setdefault(mod, spec) != spec:
                raise ValueError(f"{mod}: column layout differs between splits")
        if f"{split}.original_meta.tar" in names:
            files["original_meta"] = {"file": f"{split}.original_meta.tar"}
        sp = {"n_events": int(n), "files": files}
        if original_sha256 and split in original_sha256:
            sp["original_sha256"] = dict(sorted(original_sha256[split].items()))
        schema["splits"][split] = sp
    if extra_files:
        schema["extra_files"] = extra_files
    if write:
        (root / "schema.json").write_text(json.dumps(schema, indent=1) + "\n")
    return schema


def write_sha256sums(root: Path, name: str = "SHA256SUMS") -> Path:
    """Hash every top-level file of ``root`` (the record is flat) except ``name``,
    ``*.tmp``/``*.partial`` and names starting with ``_`` or ``.``."""
    root = Path(root)
    lines = []
    for p in sorted(root.iterdir()):
        if (not p.is_file() or p.name == name or p.name[0] in "_."
                or p.name.endswith((".tmp", ".partial"))):
            continue
        lines.append(f"{sha256_file(p)}  {p.name}")
    out = root / name
    out.write_text("\n".join(lines) + "\n")
    return out


def pack_release(src_dirs: Dict[str, Path], dst: Path, *, window_lut: Optional[np.ndarray] = None,
                 extra_files: Optional[Dict[str, str]] = None, write_sums: bool = True, **kw) -> dict:
    """Pack ``{"train": dir, "val": dir, "test": dir}`` into the flat release in ``dst``:
    window table, split files, ``schema.json`` and ``SHA256SUMS``."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if window_lut is None:
        window_lut = build_window_lut(list(src_dirs.values()))
    np.save(dst / LUT_FILE, np.asarray(window_lut, dtype=DATA_DTYPE))
    hashes = {s: pack_split(Path(p), dst, s, window_lut, **kw) for s, p in src_dirs.items()}
    schema = build_schema(dst, list(src_dirs), original_sha256=hashes, extra_files=extra_files)
    if write_sums:
        write_sha256sums(dst)
    return schema


# ---------------------------------------------------------------------------
# Checks, download, decompress
# ---------------------------------------------------------------------------

def read_sha256sums(path: Path) -> Dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = h.lower()
    return out


def verify_checksums(root: Path, sums: str = "SHA256SUMS", only_present: bool = False, log=print) -> bool:
    """Check files against ``SHA256SUMS``; with ``only_present`` missing files do not fail."""
    root = Path(root)
    ok = True
    for name, h in read_sha256sums(root / sums).items():
        p = root / name
        if not p.exists():
            log(f"MISSING {name}")
            ok = ok and only_present
            continue
        good = sha256_file(p) == h
        log(f"{'OK     ' if good else 'FAILED '} {name}")
        ok = ok and good
    return ok


def _zenodo_files(record: str) -> List[dict]:
    with urllib.request.urlopen(ZENODO_API.format(record=record)) as r:
        rec = json.load(r)
    files = rec.get("files", [])
    if isinstance(files, dict):
        files = files.get("entries", files)
        files = list(files.values()) if isinstance(files, dict) else files
    out = []
    for f in files:
        key = f.get("key") or f.get("filename")
        links = f.get("links", {})
        url = links.get("content") or links.get("download") or \
            f"https://zenodo.org/records/{record}/files/{urllib.parse.quote(key)}?download=1"
        out.append({"key": key, "url": url, "size": f.get("size"), "checksum": f.get("checksum")})
    return out


def download(dst: Path, record: Optional[str] = None, include: Sequence[str] = ("*",),
             verify: bool = True, log=print) -> Path:
    """Download the record files matching ``include`` into ``dst`` (``schema.json``,
    ``SHA256SUMS``, the window table and this loader always), resuming partial files, and
    check the Zenodo md5 and then ``SHA256SUMS``."""
    record = record or os.environ.get("HEP4M_ZENODO_RECORD") or ZENODO_RECORD_ID
    if not record:
        raise ValueError("no Zenodo record id: pass --record or set HEP4M_ZENODO_RECORD")
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    always = {"schema.json", "SHA256SUMS", LUT_FILE, "hep4m_data.py"}
    for f in _zenodo_files(record):
        key = f["key"]
        if key not in always and not any(fnmatch.fnmatch(key, g) for g in include):
            continue
        out = dst / key
        have = out.stat().st_size if out.exists() else 0
        if f["size"] is not None and have == f["size"]:
            log(f"have {key}")
        else:
            req = urllib.request.Request(f["url"])
            if have:
                req.add_header("Range", f"bytes={have}-")
            log(f"get  {key} ({f['size']} B)")
            with urllib.request.urlopen(req) as r:
                resumed = bool(have) and getattr(r, "status", None) == 206
                with open(out, "ab" if resumed else "wb") as w:
                    for b in iter(lambda: r.read(1 << 22), b""):
                        w.write(b)
        ck = f.get("checksum") or ""
        if ck.startswith("md5:"):
            h = hashlib.md5()
            with open(out, "rb") as fh:
                for b in iter(lambda: fh.read(_BLOCK), b""):
                    h.update(b)
            if h.hexdigest() != ck[4:]:
                raise IOError(f"{key}: md5 mismatch; delete the file and download again")
    if verify and (dst / "SHA256SUMS").exists() and not verify_checksums(dst, only_present=True, log=log):
        raise IOError("sha256 check failed")
    return dst


def split_file_patterns(splits: Sequence[str], modalities: Sequence[str]) -> List[str]:
    """Record-file globs for some modalities of some splits: their data/counts (or cell
    content) files, the cell window index for cell modalities, ``event_numbers`` and
    ``original_meta.tar``. ``export --modalities`` needs exactly these files."""
    unknown = [m for m in modalities if m not in MODALITIES]
    if unknown:
        raise ValueError(f"unknown modality {', '.join(unknown)} (choose from {', '.join(MODALITIES)})")
    out = []
    for s in splits:
        out += [f"{s}.event_numbers.npy.zst", f"{s}.original_meta.tar"]
        out += [f"{s}.{m}_*" for m in modalities]
        if any(m in CELL_MODALITIES for m in modalities):
            out.append(f"{s}.cell_window_index.npy.zst")
    return list(dict.fromkeys(out))


def extract_tar_zst(path: Path, dst: Path, log=print) -> List[str]:
    """Extract a ``.tar.zst`` archive into ``dst`` (streamed; member paths must stay inside
    ``dst``). Returns the top-level names written."""
    _require_zstd()
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    root = dst.resolve()
    tops = set()
    with open(path, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr, \
            tarfile.open(fileobj=zr, mode="r|") as tar:
        for m in tar:
            target = (root / m.name).resolve()
            if not (m.isfile() or m.isdir()) or (target != root and root not in target.parents):
                raise IOError(f"{path}: refusing to extract member {m.name!r}")
            tar.extract(m, root, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
            tops.add(m.name.split("/", 1)[0])
    log(f"unpacked {Path(path).name} -> {dst}")
    return sorted(tops)


def unpack(root: Path, ckpt_dir: Optional[Path] = None, tokenizers_dir: Optional[Path] = None,
           rows: Optional[Sequence[str]] = None, force: bool = False, log=print) -> Dict[str, Path]:
    """Extract downloaded model and tokeniser archives found in ``root``.

    ``ckpt.<row>.tar.zst`` -> ``ckpt_dir/ckpt.<row>/`` (default ``$HEP4M_CKPT``) and
    ``tokenizers.tar.zst`` -> ``tokenizers_dir/<tokeniser>/`` (default ``$HEP4M_TOKENIZERS``).
    A model already unpacked (``model.ckpt`` present) is skipped unless ``force``.
    Returns {row or "tokenizers": directory}.
    """
    root = Path(root)
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else _site_dir("HEP4M_CKPT", "checkpoints")
    tokenizers_dir = Path(tokenizers_dir) if tokenizers_dir else _site_dir("HEP4M_TOKENIZERS", "Checkpoints")
    out: Dict[str, Path] = {}
    for row in rows if rows else CHECKPOINT_ROWS:
        arc = root / checkpoint_file(row)
        if not arc.exists():
            if rows:
                raise FileNotFoundError(arc)
            continue
        target = ckpt_dir / f"ckpt.{row}"
        if (target / "model.ckpt").exists() and not force:
            log(f"have {target}")
        else:
            extract_tar_zst(arc, ckpt_dir, log=log)
        out[row] = target
    arc = root / TOKENIZERS_FILE
    if arc.exists():
        extract_tar_zst(arc, tokenizers_dir, log=log)
        out["tokenizers"] = tokenizers_dir
    return out


def _plain_name(name: str) -> str:
    """Decompressed file name: drop ``.partNN`` and ``.zst``."""
    return re.sub(r"\.part\d+(?=\.npy\.zst$)", "", name)[:-len(".zst")]


def decompress(root: Path, dst: Optional[Path] = None, splits: Optional[Sequence[str]] = None,
               log=print) -> Path:
    """Decompress to plain ``.npy`` files (cell parts merged into one
    ``{split}.{mod}_content.npy``). ``HEP4MData`` then memory-maps them for random access.
    Needs ~203 GB for train and ~0.5 GB each for val and test."""
    root = Path(root)
    dst = Path(dst) if dst is not None else root
    dst.mkdir(parents=True, exist_ok=True)
    schema = json.loads((root / "schema.json").read_text())
    for split, sp in schema["splits"].items():
        if splits and split not in splits:
            continue
        for key, entry in sp["files"].items():
            if key == "original_meta":
                continue
            srcs = [p["file"] for p in entry["parts"]] if "parts" in entry else [entry["file"]]
            out = dst / _plain_name(srcs[0])
            if out.exists():
                log(f"exists {out.name}")
                continue
            heads = [npy_zst_header(root / f) for f in srcs]
            shape = (sum(s[0] for _, s in heads),) + heads[0][1][1:]
            log(f"write  {out.name} {shape}")
            tmp = out.with_name(out.name + ".partial")
            with open(tmp, "wb") as w:
                w.write(npy_header(heads[0][0], shape))
                for f in srcs:
                    with NpyZstStream(root / f) as s:
                        step = max(1, _BLOCK // max(1, s.row_bytes))
                        while s.pos < s.n_rows:
                            w.write(s.read_rows(step).tobytes())
            os.replace(tmp, out)
    return dst


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _with_cell_pos(content: np.ndarray, lut: np.ndarray, widx: np.ndarray) -> np.ndarray:
    k = len(widx)
    out = np.empty((k, PATCHES, 6), dtype=DATA_DTYPE)
    out[:, :, :3] = np.asarray(content).reshape(k, PATCHES, 3)
    out[:, :, 3:] = lut[np.asarray(widx)]
    return out.reshape(k * PATCHES, 6)


class ModalityView:
    """Per-event access to one modality of one split. Same methods as
    ``hep4m.datasets.tokenized_memmap.ModalityMemmap`` (``event_codes``, ``event_pos_codes``,
    ``event_row``, ``event_cardinality``, ``offsets``, ``n_codebooks``, ``n_pos_codebooks``).

    Rows are ``n_codebooks`` content tokens followed by ``n_pos_codebooks`` position tokens,
    as in the original ``{mod}_data.npy``. Cell and celltruth positions come from the
    window table.
    """

    def __init__(self, ds: "HEP4MData", split: str, name: str):
        self.ds, self.split, self.name = ds, split, name
        self.spec = ds.schema["modalities"][name]
        self.n_codebooks = int(self.spec["n_codebooks"])
        self.n_pos_codebooks = int(self.spec["n_pos_codebooks"])
        self.n_events = ds.n_events(split)
        self.is_cell = name in CELL_MODALITIES
        self._offsets: Optional[np.ndarray] = None
        self._stored: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return self.n_codebooks + self.n_pos_codebooks

    @property
    def offsets(self) -> np.ndarray:
        """int64 (n_events + 1,), as in ``{mod}_offsets.npy``."""
        if self._offsets is None:
            if self.name in FIXED_ROWS:
                self._offsets = FIXED_ROWS[self.name] * np.arange(self.n_events + 1, dtype=np.int64)
            else:
                o = np.zeros(self.n_events + 1, dtype=np.int64)
                np.cumsum(self.ds._small(self.split, f"{self.name}_counts"), dtype=np.int64, out=o[1:])
                self._offsets = o
        return self._offsets

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def stored(self) -> np.ndarray:
        """Stored rows (content only for cells): memory-mapped from the decompressed file,
        or read into memory when the split is small."""
        if self._stored is None:
            self._stored = self.ds._rows_array(self.split, self.name)
            if self._stored.shape[0] != int(self.offsets[-1]):
                raise ValueError(f"{self.split}.{self.name}: {self._stored.shape[0]} rows, "
                                 f"offsets end at {int(self.offsets[-1])}")
        return self._stored

    def _span(self, idx: int) -> Tuple[int, int]:
        if not 0 <= idx < self.n_events:
            raise IndexError(idx)
        o = self.offsets
        return int(o[idx]), int(o[idx + 1])

    def rows(self, a_ev: int, b_ev: int) -> np.ndarray:
        """int16 rows in the original column layout for events [a_ev, b_ev)."""
        o = self.offsets
        st = np.asarray(self.stored[int(o[a_ev]):int(o[b_ev])])
        if not self.is_cell:
            return st
        return _with_cell_pos(st, self.ds.window_lut, self.ds.window_index(self.split)[a_ev:b_ev])

    def event_row(self, idx: int) -> np.ndarray:
        """int64 (n_objects, n_codebooks + n_pos_codebooks) for event ``idx``."""
        self._span(idx)
        return self.rows(idx, idx + 1).astype(np.int64)

    def event_codes(self, idx: int, num_q: Optional[int] = None) -> np.ndarray:
        num_q = self.n_codebooks if num_q is None else num_q
        if num_q > self.n_codebooks:
            raise ValueError(f"num_q={num_q} > n_codebooks={self.n_codebooks}")
        a, b = self._span(idx)
        return np.asarray(self.stored[a:b, :num_q]).astype(np.int64)

    def event_pos_codes(self, idx: int, num_q_pos: Optional[int] = None) -> np.ndarray:
        num_q_pos = self.n_pos_codebooks if num_q_pos is None else num_q_pos
        if num_q_pos > self.n_pos_codebooks:
            raise ValueError(f"num_q_pos={num_q_pos} > n_pos_codebooks={self.n_pos_codebooks}")
        a, b = self._span(idx)
        if self.is_cell:
            w = int(self.ds.window_index(self.split)[idx])
            return self.ds.window_lut[w][:, :num_q_pos].astype(np.int64)
        return np.asarray(self.stored[a:b, self.n_codebooks:self.n_codebooks + num_q_pos]).astype(np.int64)

    def event_cardinality(self, idx: int) -> int:
        a, b = self._span(idx)
        return b - a


class HEP4MData:
    """Read the release directory (compressed files, optionally decompressed next to them).

    >>> ds = HEP4MData("hep4m_release")
    >>> ds.splits                                 # ['train', 'val', 'test']
    >>> ds["val"]["track"].event_row(0)           # (n_tracks, 6) int64 tokens
    >>> ev = ds.event("val", 0)                   # {modality: rows, 'event_number': int}
    >>> for rows in ds.iter_rows("train", "topo"):  # streamed from the .zst files
    ...     pass
    >>> ds.export_original_layout("Tokenized_7mod")

    Random access (``event_row``) reads the decompressed ``.npy`` when present (see
    ``decompress``); otherwise a split/modality is read into memory if it is below
    ``max_in_memory_bytes`` (default 4 GB, enough for val and test).
    """

    def __init__(self, root: Path, schema: Optional[dict] = None, max_in_memory_bytes: float = 4e9):
        self.root = Path(root)
        self.schema = schema if schema is not None else json.loads((self.root / "schema.json").read_text())
        self.max_in_memory_bytes = max_in_memory_bytes
        self._lut: Optional[np.ndarray] = None
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._views: Dict[Tuple[str, str], ModalityView] = {}

    # ----- schema -----
    @property
    def splits(self) -> List[str]:
        return list(self.schema["splits"])

    def modalities(self, split: Optional[str] = None) -> List[str]:
        mods = list(self.schema["modalities"])
        if split is None:
            return mods
        files = self.schema["splits"][split]["files"]
        return [m for m in mods if f"{m}_data" in files or f"{m}_content" in files]

    def n_events(self, split: str) -> int:
        return int(self.schema["splits"][split]["n_events"])

    # ----- small arrays -----
    @property
    def window_lut(self) -> np.ndarray:
        if self._lut is None:
            self._lut = np.load(self.root / self.schema["window_lut"]["file"])
        return self._lut

    def _small(self, split: str, key: str) -> np.ndarray:
        ck = (split, key)
        if ck not in self._cache:
            f = self.schema["splits"][split]["files"][key]["file"]
            plain = self.root / _plain_name(f)
            self._cache[ck] = np.load(plain, mmap_mode="r") if plain.exists() else read_npy_zst(self.root / f)
        return self._cache[ck]

    def event_numbers(self, split: str) -> np.ndarray:
        return self._small(split, "event_numbers")

    def window_index(self, split: str) -> np.ndarray:
        return self._small(split, "cell_window_index")

    def _files_of(self, split: str, mod: str) -> List[dict]:
        files = self.schema["splits"][split]["files"]
        if mod in CELL_MODALITIES:
            return files[f"{mod}_content"]["parts"]
        return [{"file": files[f"{mod}_data"]["file"], "event_start": 0, "event_stop": self.n_events(split)}]

    def _rows_array(self, split: str, mod: str) -> np.ndarray:
        parts = self._files_of(split, mod)
        plain = self.root / _plain_name(parts[0]["file"])
        if plain.exists():
            return np.load(plain, mmap_mode="r")
        heads = [npy_zst_header(self.root / p["file"]) for p in parts]
        nbytes = sum(int(np.prod(s, dtype=np.int64)) * d.itemsize for d, s in heads)
        if nbytes > self.max_in_memory_bytes:
            raise MemoryError(
                f"{split}.{mod} is {nbytes / 1e9:.1f} GB uncompressed. Run "
                f"`python -m hep4m.data decompress {self.root} --splits {split}` "
                f"for random access, or stream with iter_rows().")
        return np.concatenate([read_npy_zst(self.root / p["file"]) for p in parts])

    # ----- per-event access -----
    def modality(self, split: str, mod: str) -> ModalityView:
        if (split, mod) not in self._views:
            self._views[(split, mod)] = ModalityView(self, split, mod)
        return self._views[(split, mod)]

    def __getitem__(self, split: str) -> Dict[str, ModalityView]:
        return {m: self.modality(split, m) for m in self.modalities(split)}

    def event(self, split: str, idx: int, modalities: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """All modalities of one event: {name: int64 (n, n_cb + n_pos_cb)}, plus
        ``event_number``."""
        out: Dict[str, np.ndarray] = {"event_number": int(self.event_numbers(split)[idx])}
        for m in modalities or self.modalities(split):
            out[m] = self.modality(split, m).event_row(idx)
        return out

    # ----- streaming (works on the compressed files, any size) -----
    def iter_rows(self, split: str, mod: str, event_start: int = 0, event_stop: Optional[int] = None,
                  chunk_events: int = 100_000) -> Iterator[np.ndarray]:
        """Yield int16 row blocks in the original column layout for events
        [event_start, event_stop), reading the compressed files sequentially."""
        n = self.n_events(split)
        event_stop = n if event_stop is None else min(event_stop, n)
        off = self.modality(split, mod).offsets
        for part in self._files_of(split, mod):
            a, b = max(event_start, part["event_start"]), min(event_stop, part["event_stop"])
            if a >= b:
                continue
            with NpyZstStream(self.root / part["file"]) as s:
                s.skip_rows(int(off[a]) - int(off[part["event_start"]]))
                for ca, cb in _event_chunks(b - a, chunk_events):
                    ea, eb = a + ca, a + cb
                    rows = s.read_rows(int(off[eb]) - int(off[ea]))
                    if mod in CELL_MODALITIES:
                        rows = _with_cell_pos(rows, self.window_lut, self.window_index(split)[ea:eb])
                    yield rows

    # ----- original layout -----
    def original_small_files(self, split: str) -> Dict[str, bytes]:
        """The original ``{mod}_meta.npz`` / ``{mod}_is_empty.npy`` bytes, from
        ``{split}.original_meta.tar`` (rebuilt when the tar is absent)."""
        out: Dict[str, bytes] = {}
        entry = self.schema["splits"][split]["files"].get("original_meta")
        if entry and (self.root / entry["file"]).exists():
            with tarfile.open(self.root / entry["file"]) as t:
                for m in t.getmembers():
                    if m.isfile():
                        out[m.name] = t.extractfile(m).read()
        for mod in self.modalities(split):
            v = self.modality(split, mod)
            if f"{mod}_meta.npz" not in out:
                buf = io.BytesIO()
                np.savez(buf, n_events=v.n_events, n_codebooks=v.n_codebooks, n_pos_codebooks=v.n_pos_codebooks)
                out[f"{mod}_meta.npz"] = buf.getvalue()
            if f"{mod}_is_empty.npy" not in out:
                buf = io.BytesIO()
                np.save(buf, np.flatnonzero(v.counts() == 0).astype(np.int64))
                out[f"{mod}_is_empty.npy"] = buf.getvalue()
        return out

    def original_file_names(self, split: str, modalities: Optional[Sequence[str]] = None) -> List[str]:
        names = []
        for mod in modalities or self.modalities(split):
            names += [f"{mod}_data.npy", f"{mod}_offsets.npy", f"{mod}_meta.npz", f"{mod}_is_empty.npy"]
        return names + ["event_numbers.npy"]

    def iter_original_bytes(self, split: str, name: str, chunk_events: int = 100_000,
                            max_events: Optional[int] = None) -> Iterator[bytes]:
        """Bytes of one original-layout file, e.g. ``topo_data.npy`` or ``cell_offsets.npy``.
        With ``max_events`` (below the split size) the file holds only the first
        ``max_events`` events."""
        n_all = self.n_events(split)
        n = n_all if max_events is None else min(int(max_events), n_all)
        if name == "event_numbers.npy":
            yield np.asarray(self.event_numbers(split)[:n], dtype=OFFSET_DTYPE).tobytes()
        elif name.endswith(("_meta.npz", "_is_empty.npy")) and n == n_all:
            yield self.original_small_files(split)[name]
        elif name.endswith("_meta.npz"):
            with np.load(io.BytesIO(self.original_small_files(split)[name])) as z:
                meta = {k: z[k] for k in z.files}
            meta["n_events"] = np.asarray(n, dtype=np.asarray(meta["n_events"]).dtype)
            buf = io.BytesIO()
            np.savez(buf, **meta)
            yield buf.getvalue()
        elif name.endswith("_is_empty.npy"):
            counts = self.modality(split, name[:-len("_is_empty.npy")]).counts()[:n]
            buf = io.BytesIO()
            np.save(buf, np.flatnonzero(counts == 0).astype(np.int64))
            yield buf.getvalue()
        elif name.endswith("_data.npy"):
            for rows in self.iter_rows(split, name[:-len("_data.npy")], event_stop=n,
                                       chunk_events=chunk_events):
                yield np.ascontiguousarray(rows, dtype=DATA_DTYPE).tobytes()
        elif name.endswith("_offsets.npy"):
            off = self.modality(split, name[:-len("_offsets.npy")]).offsets[:n + 1]
            yield off.astype(OFFSET_DTYPE).tobytes()
        else:
            raise KeyError(name)

    def export_original_layout(self, dst: Path, splits: Optional[Sequence[str]] = None,
                               check: bool = True, chunk_events: int = 100_000, log=print,
                               modalities: Optional[Sequence[str]] = None,
                               max_events: Optional[int] = None) -> List[str]:
        """Write ``dst/{split}/`` in the original training layout, streaming from the
        compressed files. ``hep4m.datasets.tokenized_memmap`` reads the result unchanged
        (point ``HEP4M_TOKENIZED`` at ``dst``). With ``check`` each file
        is compared with the ``original_sha256`` in ``schema.json`` where present; a mismatch
        raises. Returns the files that were checked.

        ``modalities`` writes only these modalities (plus ``event_numbers.npy``), so only
        their files need to be downloaded. ``max_events`` writes only the first
        ``max_events`` events of each split; those files are not the original files, so
        they are not checked."""
        dst = Path(dst)
        checked, bad = [], []
        for split in splits or self.splits:
            if split not in self.schema["splits"]:
                raise KeyError(f"no split {split!r} in the record (splits: {', '.join(self.splits)})")
            have = self.modalities(split)
            mods = list(modalities) if modalities else have
            unknown = [m for m in mods if m not in have]
            if unknown:
                raise KeyError(f"{split}: no modality {', '.join(unknown)} (modalities: {', '.join(have)})")
            n_all = self.n_events(split)
            truncated = max_events is not None and int(max_events) < n_all
            if truncated:
                log(f"{split}: writing the first {int(max_events):,} of {n_all:,} events "
                    f"(no sha256 check for a subset)")
            d = dst / split
            d.mkdir(parents=True, exist_ok=True)
            want = self.schema["splits"][split].get("original_sha256", {})
            for name in self.original_file_names(split, mods):
                tmp = d / (name + ".partial")
                h = hashlib.sha256()
                with open(tmp, "wb") as f:
                    for b in self.iter_original_bytes(split, name, chunk_events,
                                                      max_events if truncated else None):
                        f.write(b)
                        h.update(b)
                os.replace(tmp, d / name)
                log(f"wrote {d / name}")
                if check and not truncated and name in want:
                    checked.append(f"{split}/{name}")
                    if h.hexdigest() != want[name]:
                        bad.append(f"{split}/{name}")
        if bad:
            raise IOError("exported files differ from the original sha256: " + ", ".join(bad))
        return checked

    def info(self) -> str:
        s = self.schema
        lines = [f"{self.root}  ({s.get('format', '?')} v{s.get('format_version', '?')})"]
        if "window_lut" in s:
            lines.append(f"window table: {s['window_lut']['file']} {s['window_lut'].get('shape', '')}")
        for m, spec in s["modalities"].items():
            lines.append(f"  {m:10s} {spec['n_codebooks']} content + {spec['n_pos_codebooks']} position "
                         f"tokens per row, rows {spec.get('rows', '?')}, positions {spec.get('pos_source', '?')}")
        for sp in self.splits:
            files = s["splits"][sp]["files"]
            n_files = sum(len(e["parts"]) if "parts" in e else 1 for e in files.values())
            lines.append(f"{sp}: {self.n_events(sp):,} events, {n_files} files, "
                         f"modalities {', '.join(self.modalities(sp))}")
        for k, v in (s.get("extra_files") or {}).items():
            lines.append(f"extra: {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hep4m.data", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="download the Zenodo record")
    p.add_argument("dst", nargs="?", default=None, help="download directory (default $HEP4M_RELEASE)")
    p.add_argument("--record", default=None)
    p.add_argument("--include", nargs="*", default=None,
                   help="globs, e.g. 'val.*' 'test.*' (default: everything, about 135 GB, or only the "
                        "files selected by --checkpoints/--tokenizers/--raw-test/--modalities when one "
                        "of them is given)")
    p.add_argument("--checkpoints", nargs="*", default=None, metavar="ROW",
                   help=f"trained models to download and unpack into $HEP4M_CKPT (no ROW: all of "
                        f"{', '.join(CHECKPOINT_ROWS)})")
    p.add_argument("--tokenizers", action="store_true",
                   help=f"download {TOKENIZERS_FILE} and unpack it into $HEP4M_TOKENIZERS")
    p.add_argument("--raw-test", action="store_true",
                   help=f"download the raw test events {RAW_TEST_FILE} ($HEP4M_RAW_TEST) and the HGPflow "
                        f"predictions on them, {HGPFLOW_TEST_FILE}")
    p.add_argument("--modalities", nargs="+", default=None, metavar="MOD",
                   help="download the files of these modalities (plus event numbers and metadata) "
                        "for the splits in --splits, e.g. --modalities track topo truthpart")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="splits for --modalities (default: train val test)")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--tokenizers-dir", default=None)
    p = sub.add_parser("unpack", help="unpack downloaded ckpt.<row>.tar.zst and tokenizers.tar.zst")
    p.add_argument("dir", nargs="?", default=None, help="download directory (default $HEP4M_RELEASE)")
    p.add_argument("--ckpt-dir", default=None, help="default $HEP4M_CKPT")
    p.add_argument("--tokenizers-dir", default=None, help="default $HEP4M_TOKENIZERS")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("verify", help="check files against SHA256SUMS")
    p.add_argument("dir")
    p.add_argument("--partial", action="store_true", help="missing files do not fail the check")
    p = sub.add_parser("info", help="summarise schema.json")
    p.add_argument("dir")
    p = sub.add_parser("export", help="write the original training layout")
    p.add_argument("dir")
    p.add_argument("dst")
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--modalities", nargs="*", default=None,
                   help=f"write only these modalities (default: all; choose from {', '.join(MODALITIES)})")
    p.add_argument("--max-events", type=int, default=None,
                   help="write only the first N events of each split (no sha256 check)")
    p.add_argument("--no-check", action="store_true")
    p = sub.add_parser("decompress", help="decompress to plain .npy for memory-mapped access")
    p.add_argument("dir")
    p.add_argument("--dst", default=None)
    p.add_argument("--splits", nargs="*", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "download":
        dst = Path(a.dst) if a.dst else default_release_dir()
        for row in a.checkpoints or []:
            if row not in CHECKPOINT_ROWS:
                ap.error(f"unknown checkpoint {row!r}; choose from {', '.join(CHECKPOINT_ROWS)}")
        extra = []
        if a.checkpoints is not None:
            extra += [checkpoint_file(r) for r in (a.checkpoints or CHECKPOINT_ROWS)]
        if a.tokenizers:
            extra.append(TOKENIZERS_FILE)
        if a.raw_test:
            extra += [RAW_TEST_FILE, HGPFLOW_TEST_FILE]
        if a.modalities:
            try:
                extra += split_file_patterns(a.splits, a.modalities)
            except ValueError as e:
                ap.error(str(e))
        include = list(a.include) if a.include is not None else ([] if extra else ["*"])
        download(dst, a.record, include + extra)
        if a.checkpoints is not None or a.tokenizers:
            unpack(dst, a.ckpt_dir, a.tokenizers_dir, rows=a.checkpoints or None)
    elif a.cmd == "unpack":
        unpack(Path(a.dir) if a.dir else default_release_dir(), a.ckpt_dir, a.tokenizers_dir, force=a.force)
    elif a.cmd == "verify":
        if not (Path(a.dir) / "SHA256SUMS").exists():
            print(f"no SHA256SUMS in {a.dir}; download it with "
                  f"`python -m hep4m.data download {a.dir} --include SHA256SUMS`", file=sys.stderr)
            return 1
        return 0 if verify_checksums(Path(a.dir), only_present=a.partial) else 1
    elif a.cmd == "info":
        print(HEP4MData(Path(a.dir)).info())
    elif a.cmd == "export":
        if a.max_events is not None and a.max_events < 1:
            ap.error("--max-events must be at least 1")
        bad = [m for m in a.modalities or [] if m not in MODALITIES]
        if bad:
            ap.error(f"unknown modality {', '.join(bad)} (choose from {', '.join(MODALITIES)})")
        checked = HEP4MData(Path(a.dir)).export_original_layout(
            Path(a.dst), a.splits, check=not a.no_check, modalities=a.modalities, max_events=a.max_events)
        if not a.no_check and checked:
            print(f"sha256 check passed: {len(checked)} exported files identical to the original layout")
    elif a.cmd == "decompress":
        decompress(Path(a.dir), Path(a.dst) if a.dst else None, a.splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
