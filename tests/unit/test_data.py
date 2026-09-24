"""Tests for hep4m/data.py on a tiny synthetic store.

The synthetic store is written in the original training layout (headerless int16 data,
headerless int64 offsets, meta.npz, is_empty.npy, headerless event_numbers) with the same
column counts and conventions as Tokenized_7mod, then packed into the flat release format,
read back, and exported.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zstandard")

import hep4m.data as hd  # noqa: E402

QUIET = dict(log=lambda *a, **k: None)
N_WINDOWS = 7
SPLIT_SIZES = {"train": 257, "val": 64, "test": 33}
CB = {"cell": 3, "celltruth": 3, "track": 3, "topo": 3, "truthpart": 3, "truthjet": 1, "hgpfpart": 3}
MAXC = {"track": 15, "topo": 55, "truthpart": 27, "hgpfpart": 22}
KMAX = {"track": 256, "topo": 256, "truthpart": 128, "hgpfpart": 128, "truthjet": 128,
        "cell": 2048, "celltruth": 2048}


def _write_original_split(d: Path, n: int, lut: np.ndarray, rng, ev0: int = 1_000_000,
                          celltruth_pos_override=None) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    ev = np.sort(rng.choice(np.arange(ev0, ev0 + 10 * n), size=n, replace=False)).astype("<i8")
    ev.tofile(d / "event_numbers.npy")
    widx = rng.integers(0, lut.shape[0], size=n)
    arrays = {"event_numbers": ev}
    for mod, ncb in CB.items():
        if mod in hd.FIXED_ROWS:
            counts = np.full(n, hd.FIXED_ROWS[mod], dtype=np.int64)
        else:
            counts = rng.integers(0, MAXC[mod] + 1, size=n)
            counts[rng.random(n) < 0.1] = 0
        rows = int(counts.sum())
        content = rng.integers(0, KMAX[mod], size=(rows, ncb))
        if mod in hd.CELL_MODALITIES:
            pos = lut[widx].reshape(-1, 3)
            if mod == "celltruth" and celltruth_pos_override is not None:
                pos = celltruth_pos_override(pos.copy())
        elif mod == "truthjet":
            pos = np.repeat(rng.integers(0, 1024, size=(n, 3)), 8, axis=0)
        else:
            pos = rng.integers(0, 1024, size=(rows, 3))
        data = np.concatenate([content, pos], axis=1).astype("<i2")
        data.tofile(d / f"{mod}_data.npy")
        off = np.zeros(n + 1, dtype="<i8")
        np.cumsum(counts, out=off[1:])
        off.tofile(d / f"{mod}_offsets.npy")
        np.savez(d / f"{mod}_meta.npz", n_events=n, n_codebooks=ncb, n_pos_codebooks=3)
        np.save(d / f"{mod}_is_empty.npy", np.flatnonzero(counts == 0).astype(np.int64))
        arrays[mod] = (data, off)
    return arrays


def _make_lut(rng) -> np.ndarray:
    return rng.integers(0, 1024, size=(N_WINDOWS, 156, 3)).astype(np.int16)


@pytest.fixture()
def store(tmp_path):
    rng = np.random.default_rng(1234)
    lut = _make_lut(rng)
    src = {s: tmp_path / "orig" / s for s in SPLIT_SIZES}
    arrays = {s: _write_original_split(src[s], n, lut, rng, ev0=1_000_000 * (i + 1))
              for i, (s, n) in enumerate(SPLIT_SIZES.items())}
    return {"src": src, "arrays": arrays, "lut": lut, "root": tmp_path}


def _pack(store, **kw):
    rel = store["root"] / "release"
    kw.setdefault("part_events", 100)   # train (257 events) -> 3 cell parts
    kw.setdefault("chunk_events", 40)
    kw.setdefault("level", 3)
    schema = hd.pack_release(store["src"], rel, **kw, **QUIET)
    return rel, schema


def _files(d: Path):
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------

def test_round_trip_byte_identical(store):
    """original layout -> release -> original layout gives the same bytes for every file,
    and export checks each file against the source sha256 stored in schema.json."""
    rel, schema = _pack(store)
    assert len(schema["splits"]["train"]["files"]["cell_content"]["parts"]) == 3
    out = store["root"] / "exported"
    checked = hd.HEP4MData(rel).export_original_layout(out, chunk_events=50, **QUIET)
    n_files = sum(len(_files(s)) for s in store["src"].values())
    assert len(checked) == n_files
    for split, src in store["src"].items():
        assert _files(src) == _files(out / split)
        for name in _files(src):
            assert (src / name).read_bytes() == (out / split / name).read_bytes(), f"{split}/{name}"


def test_loader_matches_original_rows(store):
    rel, _ = _pack(store)
    for decompressed in (False, True):
        if decompressed:
            hd.decompress(rel, **QUIET)
        ds = hd.HEP4MData(rel)
        assert ds.splits == ["train", "val", "test"]
        for split, arrs in store["arrays"].items():
            assert np.array_equal(ds.event_numbers(split), arrs["event_numbers"])
            for mod in CB:
                data, off = arrs[mod]
                v = ds.modality(split, mod)
                assert isinstance(v.stored, np.memmap) == decompressed
                assert np.array_equal(v.offsets, off)
                for i in range(ds.n_events(split)):
                    want = data[off[i]:off[i + 1]].astype(np.int64)
                    assert np.array_equal(v.event_row(i), want)
                    assert np.array_equal(v.event_codes(i, CB[mod]), want[:, :CB[mod]])
                    assert np.array_equal(v.event_pos_codes(i, 3), want[:, CB[mod]:])
                    assert v.event_cardinality(i) == off[i + 1] - off[i]
            ev = ds.event(split, 5)
            assert ev["event_number"] == arrs["event_numbers"][5]
            assert set(ev) == set(CB) | {"event_number"}


def test_iter_rows_subranges_cross_part_boundaries(store):
    rel, _ = _pack(store)
    ds = hd.HEP4MData(rel)
    for mod in ("cell", "celltruth", "topo", "truthjet"):
        data, off = store["arrays"]["train"][mod]
        for a, b in [(0, 257), (95, 205), (100, 101), (199, 257), (3, 3)]:
            got = list(ds.iter_rows("train", mod, a, b, chunk_events=17))
            got = np.concatenate(got) if got else np.empty((0, data.shape[1]), np.int16)
            assert np.array_equal(got, data[off[a]:off[b]]), (mod, a, b)


def test_release_file_names_and_schema(store):
    rel, schema = _pack(store)
    files = set(_files(rel))
    assert all("/" not in f for f in files)            # flat, as on Zenodo
    assert {"schema.json", "SHA256SUMS", "cell_window_lut.npy"} <= files
    for split in SPLIT_SIZES:
        want = {f"{split}.event_numbers.npy.zst", f"{split}.cell_window_index.npy.zst",
                f"{split}.original_meta.tar", f"{split}.truthjet_data.npy.zst"}
        want |= {f"{split}.{m}_data.npy.zst" for m in hd.COUNT_MODALITIES}
        want |= {f"{split}.{m}_counts.npy.zst" for m in hd.COUNT_MODALITIES}
        want |= {f"{split}.{m}_content.part00.npy.zst" for m in hd.CELL_MODALITIES}
        assert want <= files
        assert f"{split}.truthjet_counts.npy.zst" not in files
        assert not any(f.startswith(f"{split}.cell_data") for f in files)
    assert hd.read_npy_zst(rel / "train.track_counts.npy.zst").dtype == np.uint8
    assert hd.read_npy_zst(rel / "train.cell_window_index.npy.zst").dtype == np.uint16
    c = hd.read_npy_zst(rel / "train.cell_content.part01.npy.zst")
    assert c.dtype == np.int16 and c.shape == (100 * 156, 3)
    assert np.load(rel / "cell_window_lut.npy").shape == (N_WINDOWS, 156, 3)
    assert schema["modalities"]["truthjet"] == {"n_codebooks": 1, "n_pos_codebooks": 3, "rows": "fixed",
                                                "pos_source": "inline", "rows_per_event": 8}
    assert schema["modalities"]["cell"]["pos_source"] == "window_lut"
    parts = schema["splits"]["train"]["files"]["celltruth_content"]["parts"]
    assert [(p["event_start"], p["event_stop"]) for p in parts] == [(0, 100), (100, 200), (200, 257)]
    assert json.loads((rel / "schema.json").read_text()) == schema
    # build_schema reproduces the schema from the files alone (plus the hashes)
    again = hd.build_schema(rel, original_sha256={s: schema["splits"][s]["original_sha256"] for s in SPLIT_SIZES},
                            write=False)
    assert again == schema


def test_window_lut_is_order_independent(store):
    a = hd.build_window_lut([store["src"]["train"], store["src"]["val"]])
    b = hd.build_window_lut([store["src"]["val"], store["src"]["train"]], chunk_events=7)
    assert np.array_equal(a, b)
    assert a.shape[0] == N_WINDOWS
    assert {x.tobytes() for x in a} == {x.tobytes() for x in store["lut"]}
    assert np.array_equal(hd.sort_window_lut(a[::-1]), a)


def test_pack_rejects_celltruth_position_mismatch(tmp_path):
    rng = np.random.default_rng(7)
    lut = _make_lut(rng)

    def swap_first_event(pos):   # celltruth event 0 gets another window's positions
        pos[:156] = lut[(np.flatnonzero((lut == pos[:156].reshape(1, 156, 3)).all(axis=(1, 2)))[0] + 1) % N_WINDOWS]
        return pos
    src = tmp_path / "bad"
    _write_original_split(src, 20, lut, rng, celltruth_pos_override=swap_first_event)
    with pytest.raises(ValueError, match="differ from cell in 1 events"):
        hd.pack_release({"val": src}, tmp_path / "rel", level=1, window_lut=lut, **QUIET)


def test_pack_rejects_block_missing_from_lut(tmp_path):
    rng = np.random.default_rng(8)
    lut = _make_lut(rng)
    src = tmp_path / "s"
    _write_original_split(src, 20, lut, rng)
    with pytest.raises(ValueError, match="not in the window lookup table"):
        hd.pack_release({"val": src}, tmp_path / "rel", level=1, window_lut=lut[:1], **QUIET)


def test_verify_checksums_detects_corruption(store):
    rel, _ = _pack(store)
    assert hd.verify_checksums(rel, **QUIET)
    p = rel / "val.track_counts.npy.zst"
    b = bytearray(p.read_bytes())
    b[-1] ^= 0xFF
    p.write_bytes(bytes(b))
    assert not hd.verify_checksums(rel, **QUIET)


def test_export_check_fails_on_wrong_hash(store):
    rel, _ = _pack(store)
    sch = json.loads((rel / "schema.json").read_text())
    sch["splits"]["val"]["original_sha256"]["topo_data.npy"] = "0" * 64
    with pytest.raises(IOError, match="val/topo_data.npy"):
        hd.HEP4MData(rel, schema=sch).export_original_layout(store["root"] / "x", splits=["val"], **QUIET)


def test_meta_rebuilt_without_tar(store):
    rel, schema = _pack(store)
    for s in SPLIT_SIZES:
        del schema["splits"][s]["files"]["original_meta"]
    small = hd.HEP4MData(rel, schema=schema).original_small_files("val")
    src = store["src"]["val"]
    for mod in CB:
        assert small[f"{mod}_is_empty.npy"] == (src / f"{mod}_is_empty.npy").read_bytes()
        a, b = np.load(io.BytesIO(small[f"{mod}_meta.npz"])), np.load(src / f"{mod}_meta.npz")
        assert {k: int(a[k]) for k in a.files} == {k: int(b[k]) for k in b.files}


def test_large_split_needs_decompress(store):
    rel, _ = _pack(store)
    ds = hd.HEP4MData(rel, max_in_memory_bytes=1000)
    with pytest.raises(MemoryError, match="decompress"):
        ds.modality("train", "cell").event_row(0)
    assert sum(r.shape[0] for r in ds.iter_rows("train", "cell")) == 257 * 156   # streaming still works


def test_cli(store, capsys):
    rel, _ = _pack(store)
    assert hd.main(["verify", str(rel)]) == 0
    assert hd.main(["info", str(rel)]) == 0
    assert "val: 64 events" in capsys.readouterr().out
    exp = store["root"] / "cli_export"
    assert hd.main(["export", str(rel), str(exp), "--splits", "test"]) == 0
    for name in _files(store["src"]["test"]):
        assert (exp / "test" / name).read_bytes() == (store["src"]["test"] / name).read_bytes()
    assert hd.main(["decompress", str(rel), "--splits", "val"]) == 0
    assert (rel / "val.cell_content.npy").exists() and not (rel / "train.cell_content.npy").exists()


def test_download_from_file_urls(store, tmp_path, monkeypatch):
    rel, _ = _pack(store)
    entries = []
    for name in _files(rel):
        b = (rel / name).read_bytes()
        entries.append({"key": name, "url": (rel / name).resolve().as_uri(), "size": len(b),
                        "checksum": "md5:" + hashlib.md5(b).hexdigest()})
    monkeypatch.setattr(hd, "_zenodo_files", lambda record: entries)
    dst = tmp_path / "dl"
    hd.download(dst, record="123", include=["val.*"], **QUIET)
    got = set(_files(dst))
    assert {"SHA256SUMS", "schema.json", "cell_window_lut.npy", "val.cell_window_index.npy.zst"} <= got
    assert not any(g.startswith(("train.", "test.")) for g in got)
    ds = hd.HEP4MData(dst)
    assert np.array_equal(ds.modality("val", "topo").event_row(3), store["arrays"]["val"]["topo"][0][
        store["arrays"]["val"]["topo"][1][3]:store["arrays"]["val"]["topo"][1][4]])


def test_view_matches_training_memmap_reader(store):
    """The exported layout, read by the training code's ModalityMemmap, gives the same
    per-event tokens as the release loader."""
    pytest.importorskip("torch")
    tm = pytest.importorskip("hep4m.datasets.tokenized_memmap")
    rel, _ = _pack(store)
    ds = hd.HEP4MData(rel)
    out = store["root"] / "exp"
    ds.export_original_layout(out, splits=["val"], **QUIET)
    for mod in CB:
        mm = tm.ModalityMemmap.load(out, "val", mod)
        v = ds.modality("val", mod)
        assert (mm.n_codebooks, mm.n_pos_codebooks, mm.n_events) == (v.n_codebooks, v.n_pos_codebooks, v.n_events)
        for i in range(v.n_events):
            assert np.array_equal(mm.event_row(i), v.event_row(i))
            assert np.array_equal(mm.event_codes(i, CB[mod]), v.event_codes(i, CB[mod]))
            assert np.array_equal(mm.event_pos_codes(i, 3), v.event_pos_codes(i, 3))


def test_export_modality_and_event_subset(store):
    """export --modalities/--max-events: only the named modalities, and the first N events
    of each split, identical to the head of the full export (read by the training reader)."""
    rel, _ = _pack(store)
    out = store["root"] / "subset"
    n = 40   # train has 3 cell parts of 100 events; 40 stays inside the first
    checked = hd.HEP4MData(rel).export_original_layout(out, splits=["train", "val"], modalities=["track", "truthpart"],
                                                       max_events=n, **QUIET)
    assert checked == []   # a subset is not the original file, so nothing is checked
    for split in ("train", "val"):
        names = _files(out / split)
        assert names == sorted(["event_numbers.npy"] + [f"{m}_{s}" for m in ("track", "truthpart")
                                                        for s in ("data.npy", "offsets.npy", "meta.npz",
                                                                  "is_empty.npy")])
        ev = np.fromfile(out / split / "event_numbers.npy", dtype="<i8")
        assert np.array_equal(ev, store["arrays"][split]["event_numbers"][:n])
        for mod in ("track", "truthpart"):
            data, off = store["arrays"][split][mod]
            o = np.fromfile(out / split / f"{mod}_offsets.npy", dtype="<i8")
            assert np.array_equal(o, off[:n + 1])
            d = np.fromfile(out / split / f"{mod}_data.npy", dtype="<i2").reshape(-1, data.shape[1])
            assert np.array_equal(d, data[:off[n]])
            assert int(np.load(out / split / f"{mod}_meta.npz")["n_events"]) == n
            empty = np.load(out / split / f"{mod}_is_empty.npy")
            assert np.array_equal(empty, np.flatnonzero(np.diff(off[:n + 1]) == 0))
    # the full split is still checked when --max-events is not below its size
    checked = hd.HEP4MData(rel).export_original_layout(store["root"] / "subset2", splits=["test"],
                                                       modalities=["topo"], max_events=10_000, **QUIET)
    assert sorted(checked) == sorted(f"test/{x}" for x in ("topo_data.npy", "topo_offsets.npy", "topo_meta.npz",
                                                           "topo_is_empty.npy", "event_numbers.npy"))
    with pytest.raises(KeyError, match="no modality"):
        hd.HEP4MData(rel).export_original_layout(out, splits=["val"], modalities=["nope"], **QUIET)


def test_cli_export_subset_and_verify_without_sums(store, tmp_path, capsys):
    rel, _ = _pack(store)
    exp = tmp_path / "cli_subset"
    assert hd.main(["export", str(rel), str(exp), "--splits", "val", "--modalities", "topo",
                    "--max-events", "5"]) == 0
    assert int(np.load(exp / "val" / "topo_meta.npz")["n_events"]) == 5
    with pytest.raises(SystemExit):
        hd.main(["export", str(rel), str(exp), "--modalities", "nope"])
    empty = tmp_path / "no_sums"
    empty.mkdir()
    assert hd.main(["verify", str(empty)]) == 1
    assert "download it with" in capsys.readouterr().err


def test_split_file_patterns(store, tmp_path, monkeypatch):
    pats = hd.split_file_patterns(["train"], ["track", "celltruth"])
    assert pats == ["train.event_numbers.npy.zst", "train.original_meta.tar", "train.track_*",
                    "train.celltruth_*", "train.cell_window_index.npy.zst"]
    with pytest.raises(ValueError):
        hd.split_file_patterns(["train"], ["nope"])
    rel, _ = _pack(store)
    entries = []
    for name in _files(rel):
        b = (rel / name).read_bytes()
        entries.append({"key": name, "url": (rel / name).resolve().as_uri(), "size": len(b),
                        "checksum": "md5:" + hashlib.md5(b).hexdigest()})
    monkeypatch.setattr(hd, "_zenodo_files", lambda record: entries)
    dst = tmp_path / "dl"
    assert hd.main(["download", str(dst), "--record", "1", "--modalities", "track", "topo",
                    "--splits", "train", "val"]) == 0
    got = set(_files(dst))
    assert {"train.track_data.npy.zst", "train.topo_counts.npy.zst", "val.event_numbers.npy.zst",
            "val.original_meta.tar"} <= got
    assert not any(g.startswith(("test.", "train.cell", "val.cell")) or "truthpart" in g for g in got)
    out = tmp_path / "store"
    hd.HEP4MData(dst).export_original_layout(out, splits=["train", "val"], modalities=["track", "topo"], **QUIET)
