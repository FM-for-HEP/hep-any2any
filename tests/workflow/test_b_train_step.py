"""(b) Build nanoHEP and HEP4M from the training configs (tiny width/depth,
all other settings as in the config: vocab, modalities, sampling, block
size, optimiser, schedule) and run one Lightning training step on CPU."""
from __future__ import annotations

import math

import pytest

from _tiny_models import hep4m_one_step, nano_one_step

pytestmark = pytest.mark.workflow

NANO_CONFIGS = [
    "train/nanohep_pflow.yml",
    "train/nanohep_pflow_matched30k.yml",
    "train/nanohep_multi.yml",
    "train/nanohep_sim.yml",
]
HEP4M_CONFIGS = [
    # (config_t, config_m, modality_dict)
    ("train/hep4m_pflow/config_t.yml", "train/hep4m_pflow/config_m.yml", "train/hep4m_pflow/modality_dict.yml"),
    ("train/hep4m_multi/config_t.yml", "train/hep4m_multi/config_m.yml", "train/hep4m_multi/modality_dict.yml"),
    ("train/hep4m_sim/config_t.yml", "train/hep4m_sim/config_m.yml", "train/hep4m_sim/modality_dict.yml"),
]


@pytest.mark.parametrize("cfg_rel", NANO_CONFIGS)
def test_nanohep_one_training_step(cfg_rel, train_val_store, tmp_path_factory):
    r = nano_one_step(cfg_rel, train_val_store, tmp_path_factory.mktemp("nano"))
    print(f"[train-step] loss={r['loss']:.4f} params_changed={r['n_changed']}/{r['n_trainable']}")
    assert r["steps"] == 1
    assert math.isfinite(r["loss"]) and r["loss"] > 0
    assert r["n_changed"] > 0, "optimizer step did not update any parameter"
    assert r["ckpt"].exists()


@pytest.mark.parametrize("cfgs", HEP4M_CONFIGS, ids=lambda c: c[0].split("/", 1)[1])
def test_hep4m_one_training_step(cfgs, train_val_store, modality_dict_7mod, tmp_path_factory):
    r = hep4m_one_step(*cfgs, train_val_store, tmp_path_factory.mktemp("hep4m"))
    print(f"[train-step] loss={r['loss']:.4f} params_changed={r['n_changed']}/{r['n_trainable']}")
    assert r["steps"] == 1
    assert math.isfinite(r["loss"]) and r["loss"] > 0
    assert r["n_changed"] > 0, "optimizer step did not update any parameter"
    # frozen tokenisers must stay frozen
    for m, tok in r["lit"].model.tokenizers.items():
        assert not any(p.requires_grad for p in tok.parameters()), m


def test_hep4m_finetune_from_checkpoint(train_val_store, modality_dict_7mod, tmp_path_factory):
    """Weights-only initialisation (init_weights_path) from a (tiny) HEP4M-pflow checkpoint:
    every weight is loaded from the checkpoint, then one training step runs."""
    import torch

    base = hep4m_one_step(*HEP4M_CONFIGS[0], train_val_store, tmp_path_factory.mktemp("hep4m"))
    r = hep4m_one_step(*HEP4M_CONFIGS[0], train_val_store, tmp_path_factory.mktemp("hep4m_ft"),
                       init_weights_path=base["ckpt"])
    trained = base["lit"].state_dict()
    for n, v in r["init"].items():
        assert torch.equal(v, trained[n]), f"{n} not loaded from the checkpoint"
    assert r["steps"] == 1
    assert math.isfinite(r["loss"]) and r["loss"] > 0
    assert r["n_changed"] > 0
