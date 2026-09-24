"""(e) Tokenise raw ROOT -> tokenised ROOT -> memory-mapped store, through the two CLIs.

Runs ``python -m hep4m.eval_tokenizer`` with the configs in configs/tokenize/ on the first
events of $HEP4M_TEST_ROOT_FILE (CPU), then ``python -m hep4m.build_token_store`` on the
outputs, and reads the new store back with the loader used for training.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import TEST_ROOT_FILE, dump_yaml, load_cfg, run_cli, CFG

pytestmark = pytest.mark.workflow
N_EV = 8
MODS = ["topo", "track", "truthpart", "truthjet"]


def test_tokenise_and_build_store(modality_dict_7mod, tmp_path):
    if not TEST_ROOT_FILE.exists():
        pytest.skip(f"raw ROOT test file not found: {TEST_ROOT_FILE} (set HEP4M_TEST_ROOT_FILE)")
    from hep4m.datasets.tokenized_memmap import ModalityMemmap

    inputs = []
    for mod in MODS:
        cfg = load_cfg(f"tokenize/{mod}.yml")
        cfg["init"].update(gpu=-1, batch_size=N_EV, chunk_size=N_EV, output_dir=str(tmp_path / "tok" / mod))
        cfg["items"] = [dict(info="test", input_path=str(TEST_ROOT_FILE), dir_flag="t", reduce_ds=N_EV)]
        p = dump_yaml(cfg, tmp_path / f"tok_{mod}.yml")
        res = run_cli(["-m", "hep4m.eval_tokenizer", "-i", str(p)])
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        out = tmp_path / "tok" / mod / "t" / TEST_ROOT_FILE.name
        assert out.exists(), out
        inputs += ["--input", f"{mod}={out}"]

    store = tmp_path / "store" / "test"
    res = run_cli(["-m", "hep4m.build_token_store", "--out", str(store),
                   "--modality-dict", str(CFG / "modality_dicts/7mod.yml"),
                   "--expect-events", str(N_EV), *inputs])
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    ev = np.fromfile(store / "event_numbers.npy", dtype=np.int64)
    assert len(ev) == N_EV
    for mod in MODS:
        mm = ModalityMemmap.load(store.parent, "test", mod)
        assert mm.n_events == N_EV, (mod, mm.n_events)
        codes = mm.event_codes(0, mm.n_codebooks)
        assert codes.ndim == 2 and codes.shape[1] == mm.n_codebooks
        assert (store / f"{mod}_is_empty.npy").exists()
