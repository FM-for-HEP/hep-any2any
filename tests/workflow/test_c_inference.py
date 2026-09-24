"""(c) Inference through the entry point (python -m hep4m.eval_hep4m -i <cfg>) for both
engines, on a handful of events, for every item of the configs in configs/infer/.

The configs are loaded unchanged except for: checkpoint, model config and modality dict
-> the tiny model trained one step in test_b's helper and the repository modality dict
(the released ones live in $HEP4M_CKPT/ckpt.<row>/), device -> CPU, output_dir -> a temp
dir, number of events -> 4, max_new_tokens -> 64 (nanoHEP), input files ->
$HEP4M_TEST_ROOT_FILE (HEP4M).
The test checks that the ROOT output is written under init.output_dir, has one entry
per event and has the {modality}_truth / {modality}_reco branches for every output
modality.
"""
from __future__ import annotations

import copy

import pytest
import uproot

from _tiny_models import hep4m_one_step, nano_one_step
from conftest import (CFG, TEST_HGPFLOW_FILE, TEST_ROOT_FILE, TEST_STORE, TEST_STORE_SPLIT, dump_yaml, load_cfg,
                      run_cli)

pytestmark = pytest.mark.workflow
N_EV = 4


def _check_root(path, out_mods):
    assert path.exists(), f"no output {path}"
    with uproot.open(path) as f:
        t = f["event_tree"]
        assert type(t).__name__.startswith("Model_TTree"), type(t).__name__
        assert t.num_entries == N_EV, (path, t.num_entries)
        keys = set(t.keys())
    for m in out_mods:
        assert any(k.startswith(f"{m}_reco") for k in keys), (m, sorted(keys)[:20])
        assert any(k.startswith(f"{m}_truth") for k in keys), (m, sorted(keys)[:20])


def _nano_cfg(infer_rel, ckpt, md_rel, outdir):
    cfg = load_cfg(infer_rel)
    cfg["init"].update(device="cpu", gpu=-1, chunk_size=N_EV, batch_size=N_EV, num_workers=0,
                       output_dir=str(outdir))
    cfg["init"]["model"].update(checkpoint_path=str(ckpt), modality_dict_path=str(CFG / md_rel))
    cfg["init"]["nano_hep"].update(tokenized_root=str(TEST_STORE), split=TEST_STORE_SPLIT,
                                   max_events=N_EV)
    for it in cfg["items"]:
        it["max_new_tokens"] = 64
    return cfg


@pytest.mark.parametrize("engine_cfg", [
    ("train/nanohep_pflow.yml", "infer/nanoHEP-pflow.yml", "modality_dicts/3mod.yml"),
    ("train/nanohep_pflow_matched30k.yml", "infer/nanoHEP-pflow-matched30k.yml", "modality_dicts/3mod.yml"),
    ("train/nanohep_multi.yml", "infer/nanoHEP-multi.yml", "modality_dicts/7mod.yml"),
    ("train/nanohep_sim.yml", "infer/nanoHEP-sim.yml", "modality_dicts/3mod.yml"),
], ids=["nanoHEP-pflow", "nanoHEP-pflow-matched30k", "nanoHEP-multi", "nanoHEP-sim"])
def test_nanohep_inference_cli(engine_cfg, train_val_store, modality_dict_7mod, tmp_path):
    train_rel, infer_rel, md_rel = engine_cfg
    r = nano_one_step(train_rel, train_val_store, tmp_path)
    cfg = _nano_cfg(infer_rel, r["ckpt"], md_rel, tmp_path / "out")
    p = dump_yaml(cfg, tmp_path / "infer.yml")
    res = run_cli(["-m", "hep4m.eval_hep4m", "-i", str(p)])
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    for it in cfg["items"]:
        out_mods = it["sampling_dict"]["fixed_input_output_modalities"]["output"]
        suffix = it.get("suffix", "prediction")
        _check_root(tmp_path / "out" / it["dir_flag"] / f"{suffix}.root", out_mods)


def _hep4m_item(it, root_file):
    it = copy.deepcopy(it)
    it["filepath_dict"] = {m: f"['{TEST_HGPFLOW_FILE if m == 'hgpfpart' else root_file}']"
                           for m in it["filepath_dict"]}
    it["reduce_ds"] = N_EV
    return it


_PFLOW = ("train/hep4m_pflow/config_t.yml", "train/hep4m_pflow/config_m.yml", "train/hep4m_pflow/modality_dict.yml")
_SIM = ("train/hep4m_sim/config_t.yml", "train/hep4m_sim/config_m.yml", "train/hep4m_sim/modality_dict.yml")
_MULTI = ("train/hep4m_multi/config_t.yml", "train/hep4m_multi/config_m.yml", "train/hep4m_multi/modality_dict.yml")


@pytest.mark.parametrize("engine_cfg", [
    (*_PFLOW, "infer/HEP4M-pflow.yml"),
    (*_SIM, "infer/HEP4M-sim.yml"),
    (*_MULTI, "infer/HEP4M-multi.yml"),
    (*_MULTI, "infer/HEP4M-multi-tasks.yml"),
], ids=["HEP4M-pflow", "HEP4M-sim", "HEP4M-multi", "HEP4M-multi-tasks"])
def test_hep4m_inference_cli(engine_cfg, train_val_store, modality_dict_7mod, tmp_path):
    if not TEST_ROOT_FILE.exists():
        pytest.skip(f"raw ROOT test file not found: {TEST_ROOT_FILE} (set HEP4M_TEST_ROOT_FILE)")
    cfg_t, cfg_m, md, infer_rel = engine_cfg
    r = hep4m_one_step(cfg_t, cfg_m, md, train_val_store, tmp_path)
    cfg = load_cfg(infer_rel)
    if any("hgpfpart" in it["filepath_dict"] for it in cfg["items"]) and not TEST_HGPFLOW_FILE.exists():
        pytest.skip(f"HGPflow test file not found: {TEST_HGPFLOW_FILE} (set HEP4M_TEST_HGPFLOW_FILE)")
    cfg["init"].update(device="cpu", gpu=-1, chunk_size=N_EV, batch_size=N_EV, num_workers=0,
                       output_dir=str(tmp_path / "out"))
    cfg["init"]["model"].update(config_path_m=str(r["config_m_path"]), checkpoint_path=str(r["ckpt"]),
                                modality_dict_path=str(CFG / md))
    cfg["items"] = [_hep4m_item(it, TEST_ROOT_FILE) for it in cfg["items"]]
    p = dump_yaml(cfg, tmp_path / "infer.yml")
    res = run_cli(["-m", "hep4m.eval_hep4m", "-i", str(p)], timeout=1800)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    # predictions go to init.output_dir, not next to config_m.yml
    assert not (r["config_m_path"].parent / "inference").exists()
    outs = sorted((tmp_path / "out").rglob("*.root"))
    assert len(outs) == len(cfg["items"]), outs
    for it in cfg["items"]:
        found = [o for o in outs if o.parent.name == it["dir_flag"]]
        assert found, (it["dir_flag"], outs)
        _check_root(found[0], it["output_modalities"])
