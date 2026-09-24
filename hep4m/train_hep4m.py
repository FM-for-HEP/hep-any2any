"""Train a HEP4M model (encoder-decoder, parallel decoding).

A new run writes to ``<base_root_dir>/<project_name>/<run_name>-local-<hash>/`` (copies of
the three configs and ``checkpoints/``); a run directory that already has
``checkpoints/last.ckpt`` resumes from it, as does ``--exp_dir <run directory>``.

Usage:
    python -m hep4m.train_hep4m -ct config_t.yml -cm config_m.yml -md modality_dict.yml [-g all|0,1|cpu] [-d]
"""
import argparse
from hep4m import log_to_stderr
import hashlib
import os
import shutil
from pathlib import Path

from hep4m.paths import safe_load_expanded

argparser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
argparser.add_argument('--modality_dict_path', '-md', type=str, required=False)
argparser.add_argument('--config_path_model', '-cm', type=str, required=False)
argparser.add_argument('--config_path_train', '-ct', type=str, required=False)
argparser.add_argument('--exp_dir', '-edir', type=str, required=False,
                       help='resume the run in this directory (reads its configs and checkpoints/last.ckpt)')
argparser.add_argument('--debug_mode', '-d', action='store_true', help='no logger, best-val checkpoint only')
argparser.add_argument('--gpu', '-g', type=str, required=False, default='all',
                       help="GPUs to use: 'all', a comma-separated list, or 'cpu'")
argparser.add_argument('--init_weights_path', '-iw', type=str, required=False, default=None,
                       help='Fine-tune: load the model weights of this checkpoint (no optimiser '
                            'state, epoch 0). Overrides config_t init_weights_path; ignored on resume.')


def main():
    log_to_stderr()
    args = argparser.parse_args()
    modality_dict_path = args.modality_dict_path
    config_path_m = args.config_path_model
    config_path_t = args.config_path_train
    exp_dir = args.exp_dir

    # must be set before torch initialises CUDA
    if args.gpu == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if exp_dir is not None:
        exp_dir = exp_dir.rstrip('/')
        modality_dict_path = f'{exp_dir}/modality_dict.yml'
        config_path_m = f'{exp_dir}/config_m.yml'
        config_path_t = f'{exp_dir}/config_t.yml'

    with open(config_path_t, 'r') as fp:
        config_t = safe_load_expanded(fp)
    with open(config_path_m, 'r') as fp:
        config_m = safe_load_expanded(fp)
    with open(modality_dict_path, 'r') as fp:
        modality_dict = safe_load_expanded(fp)
    if args.gpu == 'cpu':
        config_t.update(device='cpu', num_devices=1)

    if exp_dir is None:
        # deterministic name, so every DDP rank agrees on it
        seed = f'{config_t["run_name"]}::{config_path_t}'
        exp_key = f'{config_t["run_name"]}-local-{hashlib.md5(seed.encode()).hexdigest()[:8]}'
        exp_dir = f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}'
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path_t, os.path.join(exp_dir, 'config_t.yml'))
        shutil.copyfile(config_path_m, os.path.join(exp_dir, 'config_m.yml'))
        shutil.copyfile(modality_dict_path, os.path.join(exp_dir, 'modality_dict.yml'))
    exp_key = os.path.basename(exp_dir)

    ckpt_path = os.path.join(exp_dir, 'checkpoints', 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None

    import resource
    import warnings
    from datetime import timedelta

    import torch
    import torch.multiprocessing as mp
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

    from .lightnings.hep4m_lightning import HEP4MLightning, HEP4MDataModule

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

    torch.set_float32_matmul_precision('medium')
    torch.set_autocast_dtype('cuda', torch.bfloat16)
    if 'mp_share_strategy' in config_t:
        mp.set_sharing_strategy(config_t['mp_share_strategy'])

    lightning_model = HEP4MLightning(
        config_m=config_m, modality_dict=modality_dict,
        config_t=config_t, comet_logger=None, device=config_t['device'])
    if ckpt_path is None:
        from .finetune import apply_init_weights
        apply_init_weights(lightning_model, config_t, path=args.init_weights_path)
    else:
        print(f"resuming from {ckpt_path}", flush=True)

    datamodule = HEP4MDataModule(
        config_t=config_t, config_v_dict=lightning_model.config_v_dict,
        modality_dict=modality_dict)

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    # best validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, monitor='val_total_loss', mode='min',
        every_n_epochs=1, save_top_k=3, save_last=True,
        filename='{epoch}-{val_total_loss:.4f}')
    # every periodic_checkpoint_interval epochs
    checkpoint_callback_periodic = ModelCheckpoint(
        dirpath=ckpt_dir, every_n_epochs=config_t.get('periodic_checkpoint_interval', 100),
        save_top_k=-1, filename='periodic-{epoch}')
    # best jet pT IQR of the particle-flow evaluation (checked at validation end only)
    checkpoint_callback_physics = ModelCheckpoint(
        dirpath=ckpt_dir, monitor='val_pflow_iqr_jet_pt_response', mode='min',
        save_top_k=2, save_on_train_epoch_end=False,
        filename='best_iqr-{epoch}-{val_pflow_iqr_jet_pt_response:.4f}')
    # every 15 minutes (updates last.ckpt, the resume point, within long epochs)
    checkpoint_callback_time = ModelCheckpoint(
        dirpath=ckpt_dir, train_time_interval=timedelta(minutes=15),
        save_top_k=2, monitor='step', mode='max', save_last=True,
        filename='time-{step}')

    callbacks = [checkpoint_callback]
    es_cfg = config_t.get('early_stopping', {}) or {}
    if es_cfg.get('enabled', False):
        callbacks.append(EarlyStopping(
            monitor=es_cfg.get('monitor', 'val_total_loss'),
            mode=es_cfg.get('mode', 'min'),
            patience=int(es_cfg.get('patience', 10)),
            min_delta=float(es_cfg.get('min_delta', 0.0)),
            check_finite=True,
            verbose=True,
        ))

    devices = config_t['num_devices']
    if "SLURM_NTASKS" in os.environ:
        # launched by srun: devices must match --ntasks-per-node
        from lightning.pytorch.strategies import DDPStrategy
        num_nodes = int(os.environ["SLURM_NNODES"])
        # static_graph=False: the frozen tokenisers have no gradients
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=False,
                               gradient_as_bucket_view=True)
    else:
        num_nodes = 1
        strategy = 'ddp' if devices > 1 else 'auto'

    logger = False
    if not args.debug_mode:
        callbacks += [checkpoint_callback_periodic, checkpoint_callback_physics,
                      checkpoint_callback_time, TQDMProgressBar(refresh_rate=100)]
        backend = os.environ.get("HEP4M_LOGGER", "csv").lower().strip()
        if backend == "comet":
            if not os.environ.get("COMET_API_KEY") or not os.environ.get("COMET_WORKSPACE"):
                raise RuntimeError("HEP4M_LOGGER=comet requires COMET_API_KEY and COMET_WORKSPACE.")
            import comet_ml  # noqa: F401  (optional dependency: pip install ".[comet]")
            from lightning.pytorch.loggers import CometLogger
            logger = CometLogger(
                api_key=os.environ["COMET_API_KEY"],
                project=config_t["project_name"],
                workspace=os.environ["COMET_WORKSPACE"],
                name=config_t["run_name"],
                offline_directory=config_t["base_root_dir"],
                experiment_key=exp_key,
            )
            lightning_model.set_comet_logger(logger)
            logger.experiment.log_asset(config_path_t, file_name='config_t')
            logger.experiment.log_asset(config_path_m, file_name='config_m')
        elif backend == "csv":
            from lightning.pytorch.loggers import CSVLogger
            logger = CSVLogger(save_dir=f'{config_t["base_root_dir"]}/{config_t["project_name"]}', name=exp_key)
        else:
            import importlib.util
            if importlib.util.find_spec("wandb") is None:
                raise RuntimeError('HEP4M_LOGGER=wandb needs the wandb package: python -m pip install -e ".[wandb]"; '
                                   'or use HEP4M_LOGGER=csv (the default)')
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(
                project=config_t["project_name"],
                name=config_t["run_name"],
                save_dir=config_t["base_root_dir"],
                id=exp_key,
                resume="allow",
            )
            lightning_model.wandb_logger = logger

    trainer = Trainer(
        max_epochs=config_t['num_epochs'],
        accelerator=config_t['device'],
        devices=devices,
        default_root_dir=exp_dir,
        callbacks=callbacks,
        check_val_every_n_epoch=config_t['eval_every_n_epoch'],
        log_every_n_steps=1,
        logger=logger,
        precision=config_t['trainer_precision'],
        strategy=strategy,
        num_nodes=num_nodes,
        accumulate_grad_batches=config_t.get('accumulate_grad_batches', 1),
    )

    # the lightning module reduces the validation metrics across ranks itself
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*sync_dist=True.*when logging on epoch level in distributed setting.*",
            category=UserWarning,
        )
        trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
