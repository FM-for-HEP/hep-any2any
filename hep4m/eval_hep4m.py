"""Run inference (generation) with a trained nanoHEP or HEP4M model.

The inference config has two parts:

  init   model and run settings: ``model_type`` (``nano_hep`` or ``hep4m``, default
         ``hep4m``), ``gpu`` (-1 for CPU), ``precision``, ``batch_size``,
         ``chunk_size``, ``num_workers``, ``output_dir`` and ``model`` (checkpoint
         and config paths).
  items  one entry per task: input and output modalities, sampling settings,
         ``dir_flag`` (output sub-directory) and ``suffix`` (output file suffix).

Any input -> output modality direction can be requested per item; the model must
have been trained on (or be able to compose) that direction. Predictions are written
as ROOT files under ``<output_dir>/<dir_flag>/``. See ``configs/infer/`` for examples.

Usage:
    python -m hep4m.eval_hep4m -i configs/infer/nanoHEP-pflow.yml [--gpu -1] [--output-dir DIR]
        [--checkpoint model.ckpt] [--max-events N] [--items 0,1]
"""
from __future__ import annotations

import argparse
from hep4m import log_to_stderr
import logging
import os
import sys

from hep4m.paths import safe_load_expanded


def build_helper(init: dict):
    """Return the inference helper for ``init['model_type']``."""
    model_type = init.get('model_type', 'hep4m')
    if model_type == 'nano_hep':
        from .evaluations.nano_hep_inference_helper import NanoHepInferenceHelper
        return NanoHepInferenceHelper(init)
    if model_type == 'hep4m':
        from .evaluations.hep4m_inference_helper import HEP4MInferenceHelper
        return HEP4MInferenceHelper(init)
    raise ValueError(f"Unknown model_type: {model_type!r}; expected 'hep4m' or 'nano_hep'")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inference_input_path', '-i', required=True, help='inference config (YAML)')
    ap.add_argument('--gpu', type=int, default=None, help='override init.gpu (-1: CPU)')
    ap.add_argument('--output-dir', default=None, help='override init.output_dir')
    ap.add_argument('--checkpoint', default=None, help='override init.model.checkpoint_path')
    ap.add_argument('--max-events', type=int, default=None,
                    help='number of events per item (nanoHEP: init.nano_hep.max_events, HEP4M: reduce_ds)')
    ap.add_argument('--items', default=None,
                    help='comma-separated indices of the config items to run (default: all)')
    args = ap.parse_args(argv)
    log_to_stderr()

    with open(args.inference_input_path, 'r') as fp:
        config = safe_load_expanded(fp)
    init = config['init']
    if args.gpu is not None:
        init['gpu'] = args.gpu
    if args.output_dir is not None:
        init['output_dir'] = args.output_dir
    if args.checkpoint is not None:
        init['model']['checkpoint_path'] = args.checkpoint
    if init.get('gpu', 0) == -1:
        init['device'] = 'cpu'
    if args.max_events is not None:
        if 'nano_hep' in init:
            init['nano_hep']['max_events'] = args.max_events
        for item in config['items']:
            item['reduce_ds'] = args.max_events

    # must be set before torch initialises CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = str(init.get('gpu', 0))
    import torch
    torch.set_float32_matmul_precision(init.get('precision', 'highest'))

    helper = build_helper(init)
    items = config['items']
    if args.items is not None:
        items = [items[int(i)] for i in args.items.split(',')]
    for inf_dict in items:
        logging.getLogger("hep4m.eval_hep4m").info("running item: %s", inf_dict.get("info", ""))
        helper.run_inference(inf_dict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
