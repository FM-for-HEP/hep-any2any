"""Train a nanoHEP model.

    python -m hep4m.train_nano_hep -ct configs/train/nanohep_pflow.yml [-g all|0,1|cpu] [-d]
    python -m hep4m.train_nano_hep -edir <run directory>      # resume from its last.ckpt

The run directory is ``<base_root_dir>/<project_name>/<run_name>-<key>``, where the key
is the SLURM job id when present and otherwise a hash of the run name and config path.
A run started again with the same config (or with ``-edir``) resumes from
``checkpoints/last.ckpt`` of that directory.
"""
from hep4m.paths import safe_load_expanded as _hp_safe_load
try:  # optional logger backend; comet-ml asks to be imported before torch
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
from hep4m import log_to_stderr
import hashlib
import os
import shutil
from pathlib import Path



argparser = argparse.ArgumentParser(description="Train a nanoHEP model.")
argparser.add_argument('--config_path_train', '-ct', type=str, required=False,
                       help='training config (configs/train/nanohep_*.yml)')
argparser.add_argument('--exp_dir', '-edir', type=str, required=False,
                       help='resume the run in this directory from checkpoints/last.ckpt')
argparser.add_argument('--debug_mode', '-d', action='store_true', help='no logger')
argparser.add_argument('--gpu', '-g', type=str, required=False, default='all',
                       help="GPUs to use: 'all', a list such as '0,1', or 'cpu'")


def main():
    log_to_stderr()
    args = argparser.parse_args()
    exp_dir = args.exp_dir
    if exp_dir is None and args.config_path_train is None:
        argparser.error("give -ct (new run) or -edir (resume)")

    if args.gpu == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu != 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if exp_dir is not None:
        exp_dir = exp_dir.rstrip('/')
        exp_key = os.path.basename(exp_dir)
        config_path_t = f'{exp_dir}/config_t.yml'
    else:
        config_path_t = args.config_path_train

    with open(config_path_t, 'r') as fp:
        config_t = _hp_safe_load(fp)

    if exp_dir is None:
        # deterministic, so that every DDP rank (and a restarted job) finds the same directory
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            exp_key = f'{config_t["run_name"]}-{slurm_job_id}'
        else:
            seed = f'{config_t["run_name"]}::{config_path_t}'
            exp_key = f'{config_t["run_name"]}-local-{hashlib.md5(seed.encode()).hexdigest()[:8]}'
        exp_dir = f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}'
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path_t, os.path.join(exp_dir, 'config_t.yml'))

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, 'last.ckpt')
    ckpt_path = last if os.path.exists(last) else None

    # late imports (after CUDA_VISIBLE_DEVICES is set)
    import torch
    import torch.multiprocessing as mp

    from .lightnings.nano_hep_lightning import NanoHepLightning, NanoHepDataModule
    from lightning.pytorch.callbacks import (
        EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar,
    )
    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DDPStrategy
    import warnings

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

    torch.set_float32_matmul_precision('medium')
    torch.set_autocast_dtype('cuda', torch.bfloat16)

    if 'mp_share_strategy' in config_t:
        mp.set_sharing_strategy(config_t['mp_share_strategy'])

    lightning_model = NanoHepLightning(
        gpt_config=dict(config_t['gpt_config']),
        vocab_args=dict(config_t['vocab_args']),
        optimizer_args=dict(config_t.get('optimizer_args', {}) or {}),
        pflow_eval=dict(config_t.get('pflow_eval', {}) or {}),
    )

    datamodule = NanoHepDataModule(
        tokenized_root=config_t['tokenized_root'],
        all_modalities=list(config_t['all_modalities']),
        sampling_dict=dict(config_t['sampling_dict']),
        block_size=int(config_t['block_size']),
        vocab_args=dict(config_t['vocab_args']),
        batch_size=int(config_t['batch_size']),
        num_workers=int(config_t.get('num_workers', 2)),
        max_events_train=int(config_t.get('max_events_train', -1)),
        max_events_val=int(config_t.get('max_events_val', -1)),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_loss',
        mode='min',
        every_n_train_steps=0,
        every_n_epochs=1,
        save_top_k=3,
        filename='{epoch}-{val_loss:.4f}',
    )
    checkpoint_callback_periodic = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=config_t.get('periodic_checkpoint_interval', 25),
        save_top_k=-1,
        filename='periodic-{epoch}',
    )
    # Best checkpoints by jet pT IQR of the fixed-direction evaluation, next to the
    # best by val_loss. Only when that evaluation is on: Lightning raises if the
    # monitored key is never logged.
    _physics_cbs = []
    if bool((config_t.get('pflow_eval', {}) or {}).get('enabled', True)):
        _physics_cbs = [ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor='val_pflow_iqr_jet_pt_response',
            mode='min',
            save_top_k=2,
            save_on_train_epoch_end=False,  # the key is logged at validation end
            filename='best_iqr-{epoch}-{val_pflow_iqr_jet_pt_response:.4f}',
        )]
    # the two most recent step checkpoints, and last.ckpt, every step_checkpoint_interval steps
    checkpoint_callback_steps = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=config_t.get('step_checkpoint_interval', 1000),
        save_top_k=2,
        monitor='step',
        mode='max',
        save_last=True,
        filename='step-{step}',
    )

    devices = config_t['num_devices']
    if "SLURM_NTASKS" in os.environ:
        num_nodes = int(os.environ["SLURM_NNODES"])
        # static_graph is incompatible with gradient accumulation
        accum = int(config_t.get('accumulate_grad_batches', 1))
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=(accum == 1),
            gradient_as_bucket_view=True,
        )
    else:
        num_nodes = 1
        strategy = 'ddp' if devices > 1 else "auto"

    max_steps = int(config_t.get('max_steps', -1))
    common = dict(
        max_epochs=config_t['num_epochs'] if max_steps == -1 else -1,
        accelerator='cpu' if args.gpu == 'cpu' else config_t['device'],
        devices=devices,
        default_root_dir=exp_dir,
        check_val_every_n_epoch=(None if config_t.get('val_check_interval') else config_t['eval_every_n_epoch']),
        val_check_interval=config_t.get('val_check_interval', 1.0),
        precision=config_t['trainer_precision'],
        strategy=strategy,
        num_nodes=num_nodes,
        accumulate_grad_batches=config_t.get('accumulate_grad_batches', 1),
        max_steps=max_steps,
    )

    if args.debug_mode:
        trainer = Trainer(callbacks=[checkpoint_callback, checkpoint_callback_steps], **common)
    else:
        backend = os.environ.get("HEP4M_LOGGER", "csv").lower().strip()
        if backend == "comet":
            from lightning.pytorch.loggers import CometLogger
            if comet_ml is None or not os.environ.get("COMET_API_KEY") or not os.environ.get("COMET_WORKSPACE"):
                raise RuntimeError(
                    "HEP4M_LOGGER=comet requires comet-ml, COMET_API_KEY and COMET_WORKSPACE. "
                    "Unset HEP4M_LOGGER (defaults to csv) or pass --debug_mode."
                )
            active_logger = CometLogger(
                api_key=os.environ["COMET_API_KEY"],
                project=config_t["project_name"],
                workspace=os.environ["COMET_WORKSPACE"],
                name=config_t["run_name"],
                offline_directory=config_t["base_root_dir"],
                experiment_key=exp_key,
            )
            active_logger.experiment.log_asset(config_path_t, file_name='config_t')
        elif backend == "csv":
            from lightning.pytorch.loggers import CSVLogger
            active_logger = CSVLogger(save_dir=f'{config_t["base_root_dir"]}/{config_t["project_name"]}', name=exp_key)
        else:  # wandb (optional extra)
            import importlib.util
            if importlib.util.find_spec("wandb") is None:
                raise RuntimeError('HEP4M_LOGGER=wandb needs the wandb package: python -m pip install -e ".[wandb]"; '
                                   'or use HEP4M_LOGGER=csv (the default)')
            from lightning.pytorch.loggers import WandbLogger
            active_logger = WandbLogger(
                project=config_t["project_name"],
                name=config_t["run_name"],
                save_dir=config_t["base_root_dir"],
                id=exp_key,
                resume="allow",
            )

        extra_callbacks = [LearningRateMonitor(logging_interval='step')]
        es_cfg = config_t.get('early_stopping') or {}
        if es_cfg.get('enabled', False):
            extra_callbacks.append(EarlyStopping(
                monitor=es_cfg.get('monitor', 'val_loss'),
                patience=int(es_cfg.get('patience', 20)),
                min_delta=float(es_cfg.get('min_delta', 0.001)),
                mode='min',
            ))

        trainer = Trainer(
            callbacks=[checkpoint_callback, checkpoint_callback_periodic,
                       *_physics_cbs, checkpoint_callback_steps,
                       TQDMProgressBar(refresh_rate=100), *extra_callbacks],
            log_every_n_steps=int(config_t.get('log_every_n_steps', 200)),
            logger=active_logger,
            **common,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*sync_dist=True.*when logging on epoch level in distributed setting.*",
            category=UserWarning,
        )
        trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
