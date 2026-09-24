from hep4m.paths import safe_load_expanded as _hp_safe_load
try:  # optional logger backend; comet-ml asks to be imported before torch
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
from hep4m import log_to_stderr
import os
from pathlib import Path


argparser = argparse.ArgumentParser(description="Train one VQ-VAE tokeniser")
argparser.add_argument('--config_path_var', '-cv', type=str, required=False)
argparser.add_argument('--config_path_model', '-cm', type=str, required=False)
argparser.add_argument('--config_path_train', '-ct', type=str, required=False)
argparser.add_argument('--exp_dir', '-edir', type=str, required=False,
                       help='resume an existing run directory (from its checkpoints/last.ckpt)')
argparser.add_argument('--debug_mode', '-d', action='store_true', help='no logger, no run directory')
argparser.add_argument('--gpu', '-g', type=str, required=False, default='all',
                       help="GPUs to use: 'all', a list such as '0,1', or 'cpu'")


def main():
    log_to_stderr()

    args = argparser.parse_args()
    config_path_v = args.config_path_var
    config_path_m = args.config_path_model
    config_path_t = args.config_path_train
    debug_mode = args.debug_mode
    exp_dir = args.exp_dir

    # CUDA_VISIBLE_DEVICES must be set before torch initialises CUDA
    if args.gpu not in ('all', 'cpu'):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ckpt_path = None
    if exp_dir is not None:
        exp_key = exp_dir.rstrip('/').split('/')[-1]
        config_path_m = f'{exp_dir}/config_m.yml'
        config_path_v = f'{exp_dir}/config_v.yml'
        config_path_t = f'{exp_dir}/config_t.yml'
        if os.path.exists(f'{exp_dir}/checkpoints/last.ckpt'):
            ckpt_path = f'{exp_dir}/checkpoints/last.ckpt'

    import random
    import resource
    import shutil
    import string

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    from .lightnings.tokenize_lightning import TokenizeLightning
    from lightning.pytorch.loggers import CometLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
    from lightning.pytorch import Trainer

    with open(config_path_t, 'r') as fp:
        config_t = _hp_safe_load(fp)
    with open(config_path_v, 'r') as fp:
        config_v = _hp_safe_load(fp)
    with open(config_path_m, 'r') as fp:
        config_m = _hp_safe_load(fp)
    if args.gpu == 'cpu':
        config_t['device'], config_t['num_devices'] = 'cpu', 1

    lightning_model = TokenizeLightning(config_v, config_m, config_t)

    # keep the best 3 checkpoints (validation loss) and the last one
    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_loss',
        mode='min',
        every_n_train_steps=0,
        every_n_epochs=1,
        train_time_interval=None,
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{val_total_loss:.4f}')

    devices = config_t['num_devices']
    if "SLURM_NTASKS" in os.environ:
        # multi-node: Lightning needs devices == tasks per node, num_nodes from SLURM
        from lightning.pytorch.strategies import DDPStrategy
        num_nodes = int(os.environ["SLURM_NNODES"])
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True,
                               gradient_as_bucket_view=True)
    else:
        num_nodes = 1
        strategy = 'ddp' if devices > 1 else "auto"

    if debug_mode:
        trainer = Trainer(
            max_epochs = config_t['num_epochs'],
            accelerator = config_t['device'],
            devices = devices,
            default_root_dir = config_t["base_root_dir"],
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch = config_t['eval_every_n_epoch'],
            gradient_clip_val=1.0 if lightning_model.automatic_optimization else None,
            use_distributed_sampler=False
        )

    else:
        if exp_dir is None:
            exp_key  = f'{config_t["run_name"]}xxx'
            exp_key += ''.join(random.choices(string.ascii_lowercase + string.digits, k=32-len(exp_key)))

            dst = f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}'
            Path(dst).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config_path_t, os.path.join(dst, 'config_t.yml'))
            shutil.copyfile(config_path_v, os.path.join(dst, 'config_v.yml'))
            shutil.copyfile(config_path_m, os.path.join(dst, 'config_m.yml'))

        backend = os.environ.get("HEP4M_LOGGER", "csv").lower().strip()
        if backend == "comet":
            if comet_ml is None or not os.environ.get("COMET_API_KEY") or not os.environ.get("COMET_WORKSPACE"):
                raise RuntimeError("HEP4M_LOGGER=comet needs the comet-ml package, COMET_API_KEY and "
                                   "COMET_WORKSPACE. Use HEP4M_LOGGER=csv or wandb, or pass --debug_mode.")
            active_logger = CometLogger(
                api_key = os.environ["COMET_API_KEY"],
                project = config_t["project_name"],
                workspace = os.environ["COMET_WORKSPACE"],
                name = config_t["run_name"],
                offline_directory = config_t["base_root_dir"],
                experiment_key = exp_key
            )
            lightning_model.set_comet_logger(active_logger)
            active_logger.experiment.log_asset(config_path_v, file_name='config_v')
            active_logger.experiment.log_asset(config_path_t, file_name='config_t')
            active_logger.experiment.log_asset(config_path_m, file_name='config_m')
            active_logger.experiment.log_parameter('ekey', exp_key)
            active_logger.experiment.log_parameter(
                'experiment_path', f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}')
        elif backend == "csv":
            from lightning.pytorch.loggers import CSVLogger
            active_logger = CSVLogger(save_dir=f'{config_t["base_root_dir"]}/{config_t["project_name"]}',
                                      name=exp_key)
        else:  # wandb (optional extra)
            import importlib.util
            if importlib.util.find_spec("wandb") is None:
                raise RuntimeError('HEP4M_LOGGER=wandb needs the wandb package: python -m pip install -e ".[wandb]"; '
                                   'or use HEP4M_LOGGER=csv (the default)')
            from lightning.pytorch.loggers import WandbLogger
            active_logger = WandbLogger(
                project=config_t["project_name"], name=config_t["run_name"], id=exp_key,
                save_dir=config_t["base_root_dir"], resume="allow")

        trainer = Trainer(
            max_epochs = config_t['num_epochs'],
            accelerator = config_t['device'],
            devices = devices,
            default_root_dir = config_t["base_root_dir"],
            callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate=100)],
            check_val_every_n_epoch = config_t['eval_every_n_epoch'],
            log_every_n_steps = 1,
            logger = active_logger,
            gradient_clip_val=1.0 if lightning_model.automatic_optimization else None,
            strategy=strategy,
            num_nodes=num_nodes,
            use_distributed_sampler=False
        )

    trainer.fit(lightning_model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
