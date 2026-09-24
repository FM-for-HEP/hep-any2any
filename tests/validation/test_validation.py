"""Validate the installation against the downloaded Zenodo record (see hep4m/validate.py).

    HEP4M_VALIDATE_DATA=<record dir> python -m pytest tests/validation -v

Same checks as ``python -m hep4m.validate --data <record dir>``: the seven released
checkpoints reproduce the reference tokens on 32 test events, a tiny nanoHEP and HEP4M
model train for 20 steps on released events, and the released tokenisers reproduce the
released test tokens. CPU only, a few minutes. Skipped when HEP4M_VALIDATE_DATA is not
set. Unpacked archives go to HEP4M_VALIDATE_WORK (default $HEP4M_WORK/validate).
A WARN result (sampled decoding on a different CPU or software stack, or a
vector-quantize-pytorch version other than 1.22.0) passes with a warning.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DATA = os.environ.get("HEP4M_VALIDATE_DATA")
pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not DATA or not Path(DATA).is_dir(),
                                 reason="set HEP4M_VALIDATE_DATA to the downloaded record directory")]

ROWS = ("nanoHEP-pflow", "nanoHEP-pflow-matched30k", "nanoHEP-sim", "nanoHEP-multi",
        "HEP4M-pflow", "HEP4M-sim", "HEP4M-multi")
MODS = ("topo", "track", "truthpart", "truthjet", "hgpfpart", "cell", "celltruth")


@pytest.fixture(scope="session")
def inputs():
    import torch

    import hep4m.paths as P
    from hep4m import validate as V

    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("highest")
    work = Path(os.environ.get("HEP4M_VALIDATE_WORK", Path(P.WORK) / "validate"))
    return V.Inputs(Path(DATA), work)


def _assert(results):
    assert results
    for r in results:
        print(r.line())
        if r.status == "SKIP":
            pytest.skip(r.message)
        if r.status == "WARN":
            warnings.warn(f"{r.name}: {r.message}")
        assert r.status in ("PASS", "WARN"), f"{r.name}: {r.message}"


@pytest.mark.parametrize("mod", MODS)
def test_tokenisation(inputs, mod):
    from hep4m import validate as V

    _assert(V.check_tokenisation(inputs, mods=[mod]))


@pytest.mark.parametrize("row", ROWS)
def test_checkpoint(inputs, row):
    from hep4m import validate as V

    _assert(V.check_checkpoints(inputs, [row], inputs.data / V.REFERENCE_FILE))


def test_training(inputs):
    from hep4m import validate as V

    _assert(V.check_training(inputs))
