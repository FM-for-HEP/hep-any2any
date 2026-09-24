"""Every command-line entry point starts and prints its help (no data needed)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

MODULES = [
    "hep4m.paths",
    "hep4m.data",
    "hep4m.eval_tokenizer",
    "hep4m.build_token_store",
    "hep4m.train_tokenizer",
    "hep4m.train_nano_hep",
    "hep4m.train_hep4m",
    "hep4m.eval_hep4m",
    "hep4m.performance.evaluate",
    "hep4m.judge.build_input_cache",
    "hep4m.judge.c2st_reco",
    "hep4m.judge.c2st_sim",
    "hep4m.validate",
]


@pytest.mark.parametrize("module", MODULES)
def test_help(module, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "usage" in r.stdout.lower()
