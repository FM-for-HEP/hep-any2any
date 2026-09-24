"""Shared helpers for the CPU workflow tests (tokenise, train, infer, judge).

These tests run on CPU in a few minutes. They use
  * the configs in configs/, shrunk to a tiny model size,
  * the released tokeniser checkpoints ($HEP4M_TOKENIZERS, default $HEP4M_DATA/Checkpoints),
  * a tokenised store split ($HEP4M_TEST_STORE / $HEP4M_TEST_STORE_SPLIT, default
    $HEP4M_TOKENIZED / test); only the first few events are read,
  * one raw COCOA ROOT file for the parallel engine and the tokeniser
    ($HEP4M_TEST_ROOT_FILE, default $HEP4M_RAW_TEST) and, for hgpfpart, the HGPflow
    predictions on those events ($HEP4M_TEST_HGPFLOW_FILE).
Tests that need a missing input are skipped with the reason, never passed.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import hep4m.paths as P

REPO = Path(P.REPO)
CFG = REPO / "configs"

TEST_STORE = Path(os.environ.get("HEP4M_TEST_STORE", P.TOKENIZED))
TEST_STORE_SPLIT = os.environ.get("HEP4M_TEST_STORE_SPLIT", "test")
TEST_ROOT_FILE = Path(os.environ.get("HEP4M_TEST_ROOT_FILE", os.environ["HEP4M_RAW_TEST"]))
# HGPflow predictions on the same events (the hgpfpart modality of the raw inputs)
TEST_HGPFLOW_FILE = Path(os.environ.get(
    "HEP4M_TEST_HGPFLOW_FILE", f"{P.RELEASE}/eval.hgpflow_pred_singlejet_0_100kseg_bw100.0_merged.root"))

ALL7 = ["topo", "track", "truthpart", "truthjet", "hgpfpart", "cell", "celltruth"]


def load_cfg(rel: str) -> dict:
    """Load a repo config with ${VAR} expansion."""
    with open(CFG / rel) as fh:
        return P.safe_load_expanded(fh)


def modality_dict_available(md: dict, mods) -> list[str]:
    """Return the missing checkpoint/config paths for the given modalities."""
    miss = []
    for m in mods:
        for k in ("config_path_v", "config_path_m", "checkpoint_path"):
            if not Path(md[m][k]).exists():
                miss.append(md[m][k])
    return miss


@pytest.fixture(scope="session")
def modality_dict_7mod():
    md = load_cfg("modality_dicts/7mod.yml")
    miss = modality_dict_available(md, ALL7)
    if miss:
        pytest.skip(f"tokeniser checkpoints not found (set HEP4M_DATA or HEP4M_TOKENIZERS): {miss[0]}")
    return md


@pytest.fixture(scope="session")
def store_split_dir():
    d = TEST_STORE / TEST_STORE_SPLIT
    if not (d / "topo_data.npy").exists():
        pytest.skip(f"tokenised test store not found: {d} (set HEP4M_TEST_STORE)")
    return d


@pytest.fixture(scope="session")
def train_val_store(tmp_path_factory, store_split_dir):
    """A store root whose train/ and val/ are symlinks to the small test split.

    The training entry points expect train/ and val/ splits; linking avoids
    copying any data.
    """
    root = tmp_path_factory.mktemp("store")
    for split in ("train", "val", "test"):
        (root / split).symlink_to(store_split_dir, target_is_directory=True)
    return root


def shrink_gpt(gpt_config: dict) -> dict:
    g = dict(gpt_config)
    g.update(n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    return g


_EMB_KEYS = {"embed_dim", "embedding_dim", "out_dim", "input_size"}


def shrink_hep4m(config_m: dict, emb: int = 64) -> dict:
    """Shrink a HEP4M model config in place of its anchors: width, depth, heads."""
    cfg = copy.deepcopy(config_m)
    old = cfg["embedding_dim"]

    def walk(o):
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k in _EMB_KEYS and v == old:
                    o[k] = emb
                elif k == "hidden_dim":
                    o[k] = emb
                elif k == "num_layers":
                    o[k] = 1
                elif k == "num_heads":
                    o[k] = 4
                elif k in ("enable_flash_attn", "enable_flex_attn"):
                    o[k] = False
                elif k == "hidden_layers":
                    o[k] = [32]
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(cfg)
    return cfg


def run_cli(args, cwd=None, timeout=900, env_extra=None):
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["CUDA_VISIBLE_DEVICES"] = ""
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([sys.executable, *args], cwd=cwd or REPO, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return r


def dump_yaml(obj, path: Path) -> Path:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))
    return path
