"""Build nanoHEP / HEP4M from the training configs at tiny size and run
one real Lightning training step on the small store. Shared by test_b and test_c."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from conftest import load_cfg, shrink_gpt, shrink_hep4m

_CACHE: dict = {}


def _trainer(tmpdir: Path):
    from lightning.pytorch import Trainer

    return Trainer(max_steps=1, accelerator="cpu", devices=1, logger=False,
                   enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False, limit_val_batches=0,
                   num_sanity_val_steps=0, default_root_dir=str(tmpdir))


def _snapshot(module):
    return {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}


def _changed(before, module):
    return sum(int(not torch.equal(before[n], p.detach())) for n, p in module.named_parameters()
               if n in before)


def nano_one_step(cfg_rel: str, store_root: Path, tmpdir: Path) -> dict:
    """Returns dict(lit, ckpt, n_changed, n_trainable, loss)."""
    key = ("nano", cfg_rel)
    if key in _CACHE:
        return _CACHE[key]
    from hep4m.lightnings.nano_hep_lightning import NanoHepDataModule, NanoHepLightning

    cfg = load_cfg(cfg_rel)
    pflow_eval = dict(cfg.get("pflow_eval", {}) or {})
    pflow_eval["enabled"] = False  # the in-training physics eval needs a GPU-sized budget
    torch.manual_seed(0)
    lit = NanoHepLightning(
        gpt_config=shrink_gpt(cfg["gpt_config"]),
        vocab_args=dict(cfg["vocab_args"]),
        optimizer_args=dict(cfg.get("optimizer_args", {}) or {}),
        pflow_eval=pflow_eval,
    )
    dm = NanoHepDataModule(
        tokenized_root=str(store_root),
        all_modalities=list(cfg["all_modalities"]),
        sampling_dict=dict(cfg["sampling_dict"]),
        block_size=int(cfg["block_size"]),
        vocab_args=dict(cfg["vocab_args"]),
        batch_size=4, num_workers=0, max_events_train=32, max_events_val=8,
    )
    before = _snapshot(lit)
    losses = []

    orig = lit.training_step

    def training_step(batch, batch_idx):
        loss = orig(batch, batch_idx)
        losses.append(float(loss.detach()))
        return loss

    lit.training_step = training_step
    tr = _trainer(tmpdir)
    tr.fit(lit, datamodule=dm)
    lit.training_step = orig
    ckpt = tmpdir / f"nano_{Path(cfg_rel).stem}_tiny.ckpt"
    tr.save_checkpoint(str(ckpt))
    out = dict(lit=lit, ckpt=ckpt, n_changed=_changed(before, lit), n_trainable=len(before),
               loss=losses[0] if losses else float("nan"), steps=tr.global_step)
    _CACHE[key] = out
    return out


def hep4m_one_step(cfg_t_rel: str, cfg_m_rel: str, md_rel: str, store_root: Path, tmpdir: Path,
                   init_weights_path=None) -> dict:
    """Returns dict(lit, ckpt, config_m_path, n_changed, n_trainable, loss, steps, modality_dict).

    With ``init_weights_path`` the model starts from that checkpoint as in
    ``train_hep4m`` (hep4m.finetune.apply_init_weights); ``init`` holds the weights
    right after the initialisation."""
    key = ("hep4m", cfg_t_rel, str(init_weights_path))
    if key in _CACHE:
        return _CACHE[key]
    from hep4m.lightnings.hep4m_lightning import HEP4MDataModule, HEP4MLightning

    cfg_t = load_cfg(cfg_t_rel)
    cfg_m = shrink_hep4m(load_cfg(cfg_m_rel))
    md = load_cfg(md_rel)
    cfg_t = copy.deepcopy(cfg_t)
    cfg_t.update(preprocessed_dir=str(store_root),
                 batchsize_train=4, batchsize_val=4, reduce_ds_train=32, reduce_ds_val=8,
                 num_workers=0, persistent_workers=False, device="cpu",
                 base_root_dir=str(tmpdir))
    torch.manual_seed(0)
    lit = HEP4MLightning(config_m=cfg_m, modality_dict=md, config_t=cfg_t, device="cpu")
    init = None
    if init_weights_path is not None:
        from hep4m.finetune import apply_init_weights

        assert apply_init_weights(lit, cfg_t, path=str(init_weights_path))
        init = {k: v.detach().clone() for k, v in lit.state_dict().items()}
    dm = HEP4MDataModule(config_t=cfg_t, config_v_dict=lit.config_v_dict, modality_dict=md)
    before = _snapshot(lit)
    losses = []
    orig = lit.training_step

    def training_step(batch, batch_idx):
        loss = orig(batch, batch_idx)
        losses.append(float(loss.detach()))
        return loss

    lit.training_step = training_step
    tr = _trainer(tmpdir)
    tr.fit(lit, datamodule=dm)
    lit.training_step = orig
    tag = Path(cfg_t_rel).parent.name + "_" + Path(cfg_t_rel).stem + ("_ft" if init_weights_path else "")
    ckpt = tmpdir / f"hep4m_{tag}_tiny.ckpt"
    tr.save_checkpoint(str(ckpt))
    cfg_m_path = tmpdir / f"hep4m_{tag}_tiny" / "config_m.yml"
    cfg_m_path.parent.mkdir(exist_ok=True)
    import yaml

    cfg_m_path.write_text(yaml.safe_dump(cfg_m, sort_keys=False))
    out = dict(lit=lit, ckpt=ckpt, config_m_path=cfg_m_path, n_changed=_changed(before, lit),
               n_trainable=len(before), loss=losses[0] if losses else float("nan"),
               steps=tr.global_step, modality_dict=md, init=init, config_t=cfg_t)
    _CACHE[key] = out
    return out
