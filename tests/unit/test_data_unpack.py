"""Model and tokeniser archives of the Zenodo record: names, unpacking and the
download selection (no network: the download itself is replaced by a stub)."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

import hep4m.data as D
from hep4m.paths import safe_load_expanded

zstandard = pytest.importorskip("zstandard")
REPO = Path(__file__).resolve().parents[2]


def _tar_zst(path: Path, members: dict) -> Path:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    return path


def _fake_ckpt(root: Path, row: str) -> Path:
    d = f"ckpt.{row}/"
    return _tar_zst(root / D.checkpoint_file(row), {d + "model.ckpt": b"weights", d + "config_t.yml": b"a: 1\n",
                                                   d + "modality_dict.yml": b"{}\n", d + "README.json": b"{}"})


def test_checkpoint_names():
    assert D.checkpoint_file("HEP4M-pflow") == "ckpt.HEP4M-pflow.tar.zst"
    assert D.ZENODO_RECORD_ID == "22917591"


def test_unpack_layout(tmp_path):
    rel, ck, tok = tmp_path / "rel", tmp_path / "ckpt", tmp_path / "tok"
    rel.mkdir()
    _fake_ckpt(rel, "HEP4M-pflow")
    _fake_ckpt(rel, "nanoHEP-sim")
    _tar_zst(rel / D.TOKENIZERS_FILE, {"vqxtopo/config_m.yml": b"x: 1\n", "vqxtopo/checkpoints/a.ckpt": b"w"})
    out = D.unpack(rel, ck, tok, log=lambda *_: None)
    assert set(out) == {"HEP4M-pflow", "nanoHEP-sim", "tokenizers"}
    assert (ck / "ckpt.HEP4M-pflow" / "model.ckpt").read_bytes() == b"weights"
    assert (ck / "ckpt.nanoHEP-sim" / "modality_dict.yml").exists()
    assert (tok / "vqxtopo" / "checkpoints" / "a.ckpt").exists()
    # a second call keeps what is there
    (ck / "ckpt.HEP4M-pflow" / "model.ckpt").write_bytes(b"kept")
    D.unpack(rel, ck, tok, rows=["HEP4M-pflow"], log=lambda *_: None)
    assert (ck / "ckpt.HEP4M-pflow" / "model.ckpt").read_bytes() == b"kept"


def test_unpack_missing_requested_row(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.unpack(tmp_path, tmp_path / "c", tmp_path / "t", rows=["HEP4M-sim"], log=lambda *_: None)


def test_extract_refuses_paths_outside(tmp_path):
    arc = _tar_zst(tmp_path / "bad.tar.zst", {"../escape.txt": b"x"})
    with pytest.raises(IOError):
        D.extract_tar_zst(arc, tmp_path / "out", log=lambda *_: None)
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("argv, want", [
    (["--checkpoints", "HEP4M-pflow"], ["ckpt.HEP4M-pflow.tar.zst"]),
    (["--checkpoints"], [f"ckpt.{r}.tar.zst" for r in D.CHECKPOINT_ROWS]),
    (["--tokenizers", "--raw-test"], [D.TOKENIZERS_FILE, D.RAW_TEST_FILE, D.HGPFLOW_TEST_FILE]),
    (["--include", "test.*", "--checkpoints", "nanoHEP-sim"], ["test.*", "ckpt.nanoHEP-sim.tar.zst"]),
    ([], ["*"]),
])
def test_download_selection(monkeypatch, tmp_path, argv, want):
    calls = {}
    monkeypatch.setattr(D, "download", lambda dst, record, include: calls.update(dst=dst, include=include))
    monkeypatch.setattr(D, "unpack", lambda *a, **k: calls.update(unpack=k.get("rows")))
    assert D.main(["download", str(tmp_path), *argv]) == 0
    assert calls["include"] == want
    assert ("unpack" in calls) == ("--checkpoints" in argv or "--tokenizers" in argv)


def test_download_rejects_unknown_row(tmp_path):
    with pytest.raises(SystemExit):
        D.main(["download", str(tmp_path), "--checkpoints", "HEP4M-7mod"])


@pytest.mark.parametrize("row", D.CHECKPOINT_ROWS)
def test_inference_config_per_checkpoint(row):
    """configs/infer/<row>.yml reads the unpacked layout $HEP4M_CKPT/ckpt.<row>/ (the modality
    dict may instead be the identical copy in configs/modality_dicts/)."""
    with open(REPO / "configs" / "infer" / f"{row}.yml") as fh:
        cfg = safe_load_expanded(fh)
    model = cfg["init"]["model"]
    for key in ("checkpoint_path", "modality_dict_path", "config_path_m"):
        if key in model:
            ok = {f"ckpt.{row}"} | ({"modality_dicts"} if key == "modality_dict_path" else set())
            assert Path(model[key]).parent.name in ok, (key, model[key])
    assert Path(model["checkpoint_path"]).name == "model.ckpt"
