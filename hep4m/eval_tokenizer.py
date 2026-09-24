"""Tokenise raw COCOA ROOT files with a trained (frozen) VQ-VAE tokeniser.

Reads the objects of one modality from raw ROOT files, encodes them into
residual codebook tokens plus position tokens, and writes a tokenised ROOT file
per input file (``<output_dir>/<dir_flag>/<input file name>``). The tokenised ROOT
files are turned into a memory-mapped token store with ``hep4m.build_token_store``.

Config (see ``configs/tokenize/``):

  init   ``model`` (``config_path_v``, ``config_path_m``, ``checkpoint_path`` of the
         tokeniser), ``gpu`` (-1 for CPU), ``precision``, ``batch_size``,
         ``chunk_size``, ``num_workers``, ``output_dir``.
  items  ``input_path`` (a path, or a Python list expression of paths),
         ``dir_flag``, ``reduce_ds`` (number of events, -1 for all).

Usage:
    python -m hep4m.eval_tokenizer -i configs/tokenize/topo.yml [--gpu -1]
    python -m hep4m.eval_tokenizer -i configs/tokenize/topo.yml --input-file f.root --dir-flag test
"""
from __future__ import annotations

import argparse
from hep4m import log_to_stderr
import os
import sys

from hep4m.paths import safe_load_expanded


def _as_list(input_path):
    """``input_path`` may be a path or a Python list expression of paths."""
    if isinstance(input_path, list):
        return input_path
    s = str(input_path).strip()
    if s.startswith('['):
        return list(eval(s))  # config-provided list comprehension, e.g. over file indices
    return [s]


def main(argv=None) -> int:
    log_to_stderr()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inference_input_config_path', '-i', required=True, help='tokenisation config (YAML)')
    ap.add_argument('--input-file', '-ifp', default=None, help='tokenise this one file instead of the config items')
    ap.add_argument('--dir-flag', '-df', default=None, help='output sub-directory (with --input-file)')
    ap.add_argument('--reduce-ds', '-rd', type=int, default=-1, help='number of events (with --input-file)')
    ap.add_argument('--gpu', type=int, default=None, help='override init.gpu (-1: CPU)')
    ap.add_argument('--output-dir', default=None, help='override init.output_dir')
    args = ap.parse_args(argv)

    with open(args.inference_input_config_path, 'r') as fp:
        config = safe_load_expanded(fp)
    init = config['init']
    if args.gpu is not None:
        init['gpu'] = args.gpu
    if args.output_dir is not None:
        init['output_dir'] = args.output_dir

    # must be set before torch initialises CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = str(init.get('gpu', 0))
    import torch
    from .evaluations.tokenizer_inference_helper import TokenizerInferenceHelper
    torch.set_float32_matmul_precision(init.get('precision', 'highest'))

    helper = TokenizerInferenceHelper(init)
    if args.input_file is not None:
        if args.dir_flag is None:
            ap.error('--dir-flag is required with --input-file')
        helper.run_inference({'input_path': args.input_file, 'dir_flag': args.dir_flag,
                              'reduce_ds': args.reduce_ds})
        return 0
    for item in config['items']:
        for path in _as_list(item['input_path']):
            inf_dict = dict(item, input_path=path)
            print(f'\ntokenising {path}', flush=True)
            helper.run_inference(inf_dict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
