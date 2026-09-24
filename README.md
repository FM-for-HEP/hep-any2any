# hep-any2any: code for "Prompting Particle Physics: Tokenized Multi-modal Foundation Models for Combinatorially Many Tasks"

This repository holds the code for two models that map any set of collider-event
modalities to any other set. Every object in a jet (tracks, calorimeter cells and
clusters, truth and reconstructed particles, jets) is turned into discrete tokens
by a frozen per-modality VQ-VAE tokeniser. **nanoHEP** is a decoder-only
transformer that generates the output tokens autoregressively; **HEP4M** is an
encoder-decoder transformer that predicts them in parallel. A task, for example
particle flow or detector simulation, is a choice of input and output modalities
for the same trained weights. The Python package is imported as `hep4m`.

Contents: [Install](#install) · [Modalities](#modalities) · [Glossary](#glossary) ·
[Data](#data) · [Checkpoints](#checkpoints) ·
[Validate your installation](#validate-your-installation) · [Quickstart](#quickstart) ·
[Training](#training) · [Inference for any task](#inference-for-any-task) ·
[Reproducing the paper's particle flow numbers](#reproducing-the-papers-particle-flow-numbers) ·
[Reproducing the released outputs](#reproducing-the-released-outputs) ·
[Evaluation](#evaluation) · [Tests](#tests) · [Citation](#citation)

## Install

Python 3.11 or 3.12.

```bash
git clone https://github.com/FM-for-HEP/hep-any2any.git
cd hep-any2any
python -m venv .venv && source .venv/bin/activate

# CUDA PyTorch (default wheels)
python -m pip install -e ".[dev]"
# or CPU-only PyTorch
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple -e ".[dev]"
```

The install is about 1.6 GB (mostly PyTorch). Optional extras: `.[gpu]` installs
`flash-attn` (install PyTorch with CUDA first), `.[wandb]` and `.[comet]` install the
Weights & Biases and Comet training loggers. `pixi install` (CUDA) or `pixi install -e cpu`
is an alternative (`pixi.toml`). Install with `-e`: the configs are found relative to
the checkout.

Set two locations, in the environment or in a `.env` file at the repository root
(`cp .env.example .env`):

| variable | content |
|---|---|
| `HEP4M_DATA` | downloaded data: `release/` (Zenodo files), `Tokenized_7mod/` (token store), `Checkpoints/` (tokenisers) |
| `HEP4M_WORK` | writable: `runs/` (training), `outputs/` (inference, evaluation), `checkpoints/` (released models) |

`python -m hep4m.paths` prints every resolved path. Configs refer to these as
`${HEP4M_DATA}`, `${HEP4M_RELEASE}`, `${HEP4M_RAW_TEST}`, `${HEP4M_TOKENIZED}`,
`${HEP4M_TOKENIZERS}`, `${HEP4M_CKPT}`, `${HEP4M_RUNS}` and `${HEP4M_OUTPUT_ROOT}`; the
defaults are in `hep4m/paths.py`. The shell commands below use these variables; set them
in your shell with

```bash
eval "$(python -m hep4m.paths --export)"
```

## Modalities

The code and the paper use different names for the seven modalities:

| code name | paper | what it is |
|---|---|---|
| `track` | TK | reconstructed tracks |
| `topo` | CL | topoclusters (calorimeter clusters) |
| `truthpart` | TP | truth particles |
| `hgpfpart` | RP | HGPflow particles: the particles reconstructed by the HGPflow reference model |
| `truthjet` | TJ | truth jets |
| `cell` | CE | calorimeter cells, 156 patches per event |
| `celltruth` | CF | per-cell charged-energy fraction (truth), on the same 156 patches as `cell` |

## Glossary

- **direction**: a choice of input and output modalities, e.g. (`track`, `topo`) ->
  `truthpart` is particle flow and `truthpart` -> (`topo`, `track`) is detector
  simulation.
- **row**: one trained model, named as in the paper tables (e.g. `HEP4M-pflow`); the
  checkpoint archives and the configs in `configs/infer/` use these names.
- **argmax vs sampled**: argmax decoding takes the most likely token at every step
  (deterministic; used for the particle-flow tables). Sampled decoding draws from the
  predicted distribution at temperature T (used for simulation).
- **cardinality**: the number of output objects in an event. HEP4M predicts it first and
  then fills that many slots; nanoHEP stops when it generates an end token.
- **token store**: the tokenised data on disk, one directory per split with
  `{modality}_data.npy` and `{modality}_offsets.npy` (`$HEP4M_TOKENIZED`, written by
  `python -m hep4m.data export`). nanoHEP reads its inputs from it; HEP4M tokenises the
  raw ROOT files on the fly.
- **modality dict**: a YAML file (`configs/modality_dicts/`) that names the tokeniser
  checkpoint and variable config of each modality a model uses.

## Data

The simulated single-jet events (COCOA detector), the tokenisers and the trained models
are published as one Zenodo record: [10.5281/zenodo.22917591](https://doi.org/10.5281/zenodo.22917591)
(record id `22917591`). It holds the tokenised train/val/test splits of seven modalities
(`track`, `topo`, `truthpart`, `hgpfpart`, `truthjet`, `cell`, `celltruth`; about 132 GB
compressed, of which the test and val splits are about 0.15 GB each), the seven tokeniser
checkpoints (`tokenizers.tar.zst`), the raw ROOT files of the test and val events and the
HGPflow predictions on the test events (`eval.*.root`), and one archive per trained model
(`ckpt.<row>.tar.zst`, about 3 GB for all seven, see [Checkpoints](#checkpoints)).

`hep4m/data.py` downloads, checks and unpacks the record (it needs only `numpy` and
`zstandard`). Files go to `$HEP4M_RELEASE` (default `$HEP4M_DATA/release`). For inference
and evaluation the test and val splits are enough:

```bash
python -m hep4m.data download --include 'test.*' 'val.*' --tokenizers --raw-test
python -m hep4m.data verify   $HEP4M_RELEASE --partial   # sha256 of the downloaded files
python -m hep4m.data info     $HEP4M_RELEASE
python -m hep4m.data export   $HEP4M_RELEASE $HEP4M_DATA/Tokenized_7mod --splits val test
python -m hep4m.data decompress $HEP4M_RELEASE --splits test   # optional, see below
```

`python -m hep4m.data download` with no selection fetches the whole record (about 135 GB).
`--tokenizers` unpacks the tokenisers into `$HEP4M_TOKENIZERS` (default
`$HEP4M_DATA/Checkpoints/<tokeniser>/`); the modality dicts in `configs/modality_dicts/`
name them. `--raw-test` fetches the raw test file (`$HEP4M_RAW_TEST`) that the HEP4M
inference and the tokenise configs read, and the HGPflow predictions on the same events
(the `hgpfpart` inputs). `export` writes the token store
(`Tokenized_7mod/{train,val,test}/{mod}_data.npy`, `{mod}_offsets.npy`,
`{mod}_meta.npz`, `{mod}_is_empty.npy`, `event_numbers.npy`) and checks each file against
the sha256 of the original store. Inference needs `val` and `test` exported; training
also needs `train` (see [Training](#training)). `--modalities` and `--max-events` export
only some modalities or the first N events of each split (a subset is not checked
against the sha256). The loader can also be used from Python without exporting:

```python
from hep4m.data import HEP4MData
ds = HEP4MData("path/to/release")
ev = ds.event("test", 0)          # {modality: int16 token rows} for one event
```

Random access to a large split needs its files decompressed to plain `.npy` next to the
compressed ones (`decompress`; about 203 GB for the train split, 0.5 GB each for val
and test); smaller splits are read into memory.

## Checkpoints

Each trained model is one file `ckpt.<row>.tar.zst` in the record.
`python -m hep4m.data download --checkpoints [ROW ...]` downloads them (all rows when
none is named, about 3 GB) and unpacks them into `$HEP4M_CKPT` (default
`$HEP4M_WORK/checkpoints`); `python -m hep4m.data unpack` does the unpacking for archives
downloaded by hand. Each unpacks to

```
$HEP4M_CKPT/ckpt.<row>/
  model.ckpt            weights only (loads with torch.load(weights_only=True))
  config_t.yml          training config of the run
  config_m.yml          model config (HEP4M rows)
  modality_dict.yml     tokenisers used by the model (paths under ${HEP4M_TOKENIZERS})
  inference_test*.yml   the inference settings of the released predictions (provenance)
  README.json           provenance: epoch, step, sha256, parameter counts, verification,
                        and the paper tables and figures made with this checkpoint
```

Run a checkpoint with `configs/infer/<row>.yml`. The `inference_test*.yml` and
`config_t.yml` files inside an archive record how the released predictions were made;
they use `${HEP4M_TOKENIZED}` for the token store and `${HEP4M_CKPT_DIR}` / `${HEP4M_OUTPUT}` for the unpacked
directory and the output directory, so use the repository configs to run the models.

```bash
python -m hep4m.data download --checkpoints HEP4M-pflow --tokenizers --raw-test
python -m hep4m.eval_hep4m -i configs/infer/HEP4M-pflow.yml   # all test events; --gpu -1 for CPU
```

That command runs all 99,980 test events. On CPU (16 threads) HEP4M takes about 30 s
per 1,000 events (about 50 min for the test split) and nanoHEP about 5 min per 1,000
events (about 8 h); a GPU is much faster. `--max-events N` runs the first N events.

| row | model | paper | training config |
|---|---|---|---|
| `nanoHEP-pflow` | nanoHEP, particle flow | Table 2 | `configs/train/nanohep_pflow.yml` |
| `nanoHEP-pflow-matched30k` | nanoHEP-pflow recipe, stopped at 29k steps | Table 2, "nanoHEP-pflow (matched, 29k steps)" | `configs/train/nanohep_pflow_matched30k.yml` |
| `nanoHEP-sim` | nanoHEP, detector simulation | Table 3 | `configs/train/nanohep_sim.yml` |
| `nanoHEP-multi` | nanoHEP, all seven modalities | Tables 2, 3 and 4 | `configs/train/nanohep_multi.yml` |
| `HEP4M-pflow` | HEP4M, particle flow | Table 2 | `configs/train/hep4m_pflow/config_t.yml` |
| `HEP4M-sim` | HEP4M, detector simulation | Table 3 | `configs/train/hep4m_sim/config_t.yml` |
| `HEP4M-multi` | HEP4M, all seven modalities | Tables 2 and 3 | `configs/train/hep4m_multi/config_t.yml` |

`README.json` in each archive lists all paper tables and figures made with that
checkpoint (`paper_tables_figures`, by LaTeX label) and the parameter count of the
saved state dict (`n_parameters_state_dict_total`, which includes the embeddings and,
for HEP4M, the frozen tokenisers, so it is larger than the model size quoted in the
paper).

## Validate your installation

One command checks the installed code against the record, on CPU, in about four minutes
(plus the download of about 3.5 GB, mostly the checkpoints):

```bash
python -m hep4m.data download --include 'test.*' 'eval.raw_test_*' 'eval.hgpflow_*' \
    'tokenizers.tar.zst' 'ckpt.*' 'reference_outputs.npz'
python -m hep4m.validate --data $HEP4M_RELEASE
```

A `PASS` means that your install produces the same tokens as ours on the same events
(token-identical outputs for argmax decoding); it does not recompute any paper number
(see [Reproducing the paper's particle flow numbers](#reproducing-the-papers-particle-flow-numbers)).

| check | what it does | pass |
|---|---|---|
| tokenisation | tokenises the first 256 raw test events with the released tokenisers (all seven modalities) and compares them with the released test tokens | >= 99.9% identical rows per modality |
| checkpoints | each model in `reference_outputs.npz` (the seven rows above) generates the first 32 test events, with the decoding of `configs/infer/<row>.yml` (argmax or sampled; the items that have reference tokens); the tokens are compared with the reference | argmax: all tokens identical; sampled: identical with a fixed seed |
| training | 20 steps of a small nanoHEP and a small HEP4M particle-flow model (the training configs at reduced width) on 8 released events | loss finite and going down; the checkpoint is written and reloads with identical weights |

Each line of the output is `PASS`, `WARN`, `FAIL` or `SKIP`; the exit code is 1 if any
check failed. Missing files skip their check with the reason. Unpacked archives (another
3 GB), small token stores and logs go to `--work` (default `$HEP4M_WORK/validate`);
nothing is written to the download directory. `--work` must be on a local disk: the
checks write memory-mapped files, which network file systems such as Lustre or DVS may
not allow. `--checks`, `--rows`, `--n-events` and `--threads` select less;
`python -m hep4m.validate --help` lists them. The training check uses the first
train events when `train.{track,topo,truthpart}_{data,counts}.npy.zst` are downloaded
(about 13 GB), and the first test events otherwise.

The reference outputs were made with the install above (CPU PyTorch wheels, torch 2.14,
`vector-quantize-pytorch==1.22.0`), on CPU in fp32 with torch attention. Things to know:

- With `vector-quantize-pytorch` 1.23 or later the cell tokens change (see
  [Reproducing the released outputs](#reproducing-the-released-outputs)); the command
  prints a warning and the tokenisation and HEP4M results are then expected to differ.
- Sampled decoding uses `torch.manual_seed`, and the random stream changes between torch
  versions. A sampled item that differs is reported as `WARN`, not `FAIL`; for HEP4M the
  number of objects per event (argmax cardinality) must still be identical. Argmax
  outputs were identical with torch 2.5 and 2.14 and with 4 or 16 threads.
- The references use torch attention. The released HEP4M-pflow predictions were made with
  flash attention in the tokenisers, so they are not bit-identical to these references
  (see below).

`HEP4M_VALIDATE_DATA=$HEP4M_RELEASE python -m pytest tests/validation` runs the same
checks as tests.

## Quickstart

A CPU run of every step on 200 events: tokenise one raw ROOT file, build a small token
store, train a tiny nanoHEP particle-flow model for 50 steps, generate particles with it,
compute the metrics and run the C2ST judge. It needs only `HEP4M_DATA` and `HEP4M_WORK`,
the tokenisers and the raw test file (`python -m hep4m.data download --tokenizers --raw-test`),
and takes a few minutes. On 16 events the AUCs of step 5 carry no information, and the
split-half null (8 against 8 events) cannot meet the default tolerance of 0.02, so
`--quick` records the null result as `FAIL (not enforced: --quick)` without aborting; a
real comparison needs thousands of events and the default settings.

```bash
eval "$(python -m hep4m.paths --export)"
RAW=$HEP4M_RAW_TEST
QS=$HEP4M_WORK/quickstart

# 1. tokenise one raw file (one frozen tokeniser per modality), then build a token store
for m in topo track truthpart; do
  python -m hep4m.eval_tokenizer -i configs/tokenize/$m.yml --gpu -1 \
      --input-file $RAW --dir-flag quickstart --reduce-ds 200 --output-dir $QS/tokenized_root/$m
done
python -m hep4m.build_token_store --out $QS/store/train --modality-dict configs/modality_dicts/3mod.yml \
    --input topo=$QS/tokenized_root/topo/quickstart/$(basename $RAW) \
    --input track=$QS/tokenized_root/track/quickstart/$(basename $RAW) \
    --input truthpart=$QS/tokenized_root/truthpart/quickstart/$(basename $RAW)
ln -sfn train $QS/store/val && ln -sfn train $QS/store/test   # smoke test only: same events in every split

# 2. train a tiny model for 50 steps on CPU
HEP4M_TOKENIZED=$QS/store HEP4M_LOGGER=csv \
    python -m hep4m.train_nano_hep -ct configs/train/nanohep_pflow_tiny.yml -g cpu
CKPT=$(ls $HEP4M_RUNS/hep-any2any/nanohep_pflow_tiny-local-*/checkpoints/last.ckpt)

# 3. generate particles for 16 events with that model
HEP4M_TOKENIZED=$QS/store python -m hep4m.eval_hep4m -i configs/infer/nanoHEP-pflow.yml --gpu -1 \
    --checkpoint $CKPT --max-events 16 --output-dir $QS/inference

# 4. jet and particle metrics
python -m hep4m.performance.evaluate metrics --root-path $QS/inference/pflow/argmax.root \
    --engine nano_hep --case pflow --run-id tiny --metrics-out $QS/metrics.json

# 5. C2ST judge (marginal and joint); --quick: 1 bootstrap, 5 epochs, null check not enforced
python -m hep4m.judge.build_input_cache --modalities topo track --split test --store-dir $QS/store \
    --modality-dict configs/modality_dicts/3mod.yml --max-events 16 --out $QS/inputs.npz
python -m hep4m.judge.c2st_reco --cache $QS/inputs.npz --row "tiny:nano_hep:$QS/inference/pflow/argmax.root" \
    --quick --device cpu --out-dir $QS/c2st
```

The prediction file `$QS/inference/pflow/argmax.root` has one entry per event with
`truthpart_truth_*` and `truthpart_reco_*` branches. The training run directory is
`$HEP4M_RUNS/hep-any2any/nanohep_pflow_tiny-local-<hash>/`; running the same command
again resumes from its `checkpoints/last.ckpt`.

## Training

| model | command |
|---|---|
| tokeniser (one per modality) | `python -m hep4m.train_tokenizer -cv configs/tokenizers/variables/<mod>.yml -cm configs/tokenizers/<mod>/model.yml -ct configs/tokenizers/<mod>/train.yml` |
| nanoHEP | `python -m hep4m.train_nano_hep -ct configs/train/nanohep_{pflow,pflow_matched30k,sim,multi}.yml` |
| HEP4M | `python -m hep4m.train_hep4m -ct configs/train/<run>/config_t.yml -cm configs/train/<run>/config_m.yml -md configs/train/<run>/modality_dict.yml` |

`<run>` is `hep4m_pflow`, `hep4m_sim` or `hep4m_multi`. Flags: `-g` GPUs (`all`, `0,1`,
or `cpu`), `-d` debug mode without a logger, `-edir` resume an existing run directory.
Multi-node runs use Lightning DDP and read the node layout from SLURM variables when
present. Logger: `HEP4M_LOGGER=csv` (default, CSV files in the run directory), `wandb`
(install `.[wandb]`) or `comet` (install `.[comet]`).

The paper models were trained on nodes with 4 A100 GPUs: the training configs set 4
devices per node and bf16 precision, and run for up to hundreds of thousands of steps.
They are not meant for a CPU. The only CPU-sized training config is
`configs/train/nanohep_pflow_tiny.yml` (used by the quickstart); there is no tiny HEP4M
config.

The training data is the token store (`tokenized_root` / `preprocessed_dir` in the
configs, default `$HEP4M_TOKENIZED`) with `train` and `val` exported. The full train
split of all seven modalities is about 131 GB to download. A particle-flow model needs
only `track`, `topo` and `truthpart` (about 13 GB for train and val), and
`--max-events` exports the first N training events:

```bash
python -m hep4m.data download --modalities track topo truthpart --splits train val
python -m hep4m.data export $HEP4M_RELEASE $HEP4M_TOKENIZED --splits train val \
    --modalities track topo truthpart --max-events 1000000
```

Which directions a model learns is set by `sampling_dict` (nanoHEP) or `sampling_type` /
`max_inp_modality` (HEP4M): fixed input/output modalities for a single task, or random
subsets for any-to-any training.

The released tokenisers are in the record. Retraining one needs the raw COCOA training
sample, which is not in the record: the tokeniser train configs read its ROOT files from
`$HEP4M_RAW_TRAIN` (default `$HEP4M_DATA/raw_train`), the HGPflow predictions on it
(`hgpflow/`, for `hgpfpart`) and a numpy export of its calorimeter-cell images
(`cells_numpy/`, for `cell` and `celltruth`) that this repository does not produce.

**Fine-tuning from a checkpoint.** `init_weights_path` in a HEP4M `config_t` (or
`--init_weights_path` / `-iw` of `python -m hep4m.train_hep4m`) loads only the model
weights of a checkpoint, for example `$HEP4M_CKPT/ckpt.HEP4M-pflow/model.ckpt`: the run
starts at epoch 0 with a fresh optimiser and its own warm-up (`hep4m/finetune.py`). The
model config must match the checkpoint. A resumed run ignores it.

## Inference for any task

`python -m hep4m.eval_hep4m -i <config>` runs both engines. Each item of the config
names the input and output modalities, so one multi-modal model serves any direction:

```yaml
# nanoHEP (configs/infer/nanoHEP-multi.yml, item 2)
sampling_dict:
  type: inference
  fixed_input_output_modalities:
    input: [truthpart]          # flash simulation: truth particles ->
    output: [hgpfpart]          #   HGPflow particles
argmax: true                    # false: sample at `temperature`
temperature: 1.0

# HEP4M (configs/infer/HEP4M-multi.yml)
input_modalities: [truthpart]
output_modalities: [topo, track]
card_topk_dict: {topo: 1, track: 1}          # cardinality: 1 argmax, -1 sample
top_k_token_dict: {topo: -1, track: -1}      # tokens: 1 argmax, -1 sample, k top-k
```

nanoHEP reads its inputs from the token store; HEP4M tokenises the raw ROOT files
listed in `filepath_dict` on the fly. Predictions are written to
`<init.output_dir>/<dir_flag>/` as ROOT files with `{modality}_truth_*` and
`{modality}_reco_*` branches. Options: `--gpu -1` (CPU), `--output-dir`, `--items 0,2`,
`--max-events N`.

| config | engine | task |
|---|---|---|
| `configs/infer/nanoHEP-pflow.yml` | nanoHEP | (track, topo) -> truth particles |
| `configs/infer/nanoHEP-pflow-matched30k.yml` | nanoHEP | same, 29k-step model |
| `configs/infer/nanoHEP-sim.yml` | nanoHEP | truth particles -> (topo, track) |
| `configs/infer/nanoHEP-multi.yml` | nanoHEP | particle flow, detector simulation and flash simulation from one model |
| `configs/infer/HEP4M-pflow.yml` | HEP4M | (topo, track) -> truth particles |
| `configs/infer/HEP4M-sim.yml` | HEP4M | truth particles -> (topo, track) |
| `configs/infer/HEP4M-multi.yml` | HEP4M | both directions from one model |
| `configs/infer/HEP4M-multi-tasks.yml` | HEP4M | eight directions of HEP4M-multi with per-direction decoding settings |

## Reproducing the paper's particle flow numbers

Table 2 of the paper uses argmax decoding on the test split. Each row is one inference
config, followed by `evaluate metrics` on its prediction file:

| Table 2 row | command | `--engine` |
|---|---|---|
| HGPflow | none: the record has the predictions, `$HEP4M_RELEASE/eval.hgpflow_pred_singlejet_0_100kseg_bw100.0_merged.root` | `hgpflow` |
| nanoHEP-pflow | `configs/infer/nanoHEP-pflow.yml` | `nano_hep` |
| nanoHEP-pflow (matched, 29k steps) | `configs/infer/nanoHEP-pflow-matched30k.yml` | `nano_hep` |
| nanoHEP-multi | `configs/infer/nanoHEP-multi.yml --items 0` | `nano_hep` |
| HEP4M-pflow | `configs/infer/HEP4M-pflow.yml` | `hep4m` |
| HEP4M-multi | `configs/infer/HEP4M-multi.yml --items 0` | `hep4m` |

```bash
python -m hep4m.eval_hep4m -i configs/infer/HEP4M-pflow.yml --max-events 20000
python -m hep4m.performance.evaluate metrics --root-path <prediction .root> \
    --engine hep4m --case pflow --run-id HEP4M-pflow --metrics-out HEP4M-pflow.json \
    --raw-truth $HEP4M_RELEASE/eval.hgpflow_pred_singlejet_0_100kseg_bw100.0_merged.root
```

The prediction file is under `init.output_dir` of the config (`$HEP4M_OUTPUT_ROOT/inference/<row>/`).
The jet pT response median and IQR of Table 2 are `truthpart.median_jet_pt_response` and
`truthpart.iqr_jet_pt_response` in the metrics JSON.

The paper scores every row against the raw truth particles, on the events whose truth
tokenisation kept all particles (the tokeniser drops particles outside its eta-phi
window). These are 95,870 of the 99,980 test events; 95,859 of them have a non-empty
truth jet, and those are the events of Table 2. `--raw-truth` applies exactly this
selection: it reads the raw truth particles (from the HGPflow file of the record, as in
the paper, or from `$HEP4M_RAW_TEST`, which gives the same jet pT response), keeps the
events whose particle count equals their row count in `$HEP4M_TOKENIZED/test`
(`--store` to change it), and writes the counts under `_selection` in the JSON. Without
`--raw-truth`, `evaluate metrics` compares against the tokenised truth in the prediction
file, on all events. On the matched events the tokenised and the raw truth give IQRs
within 0.0001 of each other; the other events make the all-event IQR larger (by about
0.0007 on the full test split for HEP4M-pflow and nanoHEP-pflow). The
prediction file must hold the first N test events in order (as written by `eval_hep4m`,
with or without `--max-events`).

How many events to use: on 2,000 events the IQR varies by about ±0.002 from one set of
events to another, which is enough to swap the order of rows that are close; use at
least 5,000 events, and the full test split to compare with the table. CPU times are in
[Checkpoints](#checkpoints) (HEP4M about 30 s and nanoHEP about 5 min per 1,000 events
on 16 threads); `evaluate metrics` itself takes about 2 minutes on the full split. With
torch attention (no `flash-attn`), HEP4M-pflow predictions are not bit-identical to the
released ones (see [Reproducing the released outputs](#reproducing-the-released-outputs)).

## Reproducing the released outputs

**Tokenisation.** The released token stores were made in fp32 (`precision: highest`),
with torch attention in the tokenisers and cuDNN TF32 off, and with
`vector-quantize-pytorch==1.22.0` (pinned in `pyproject.toml`; 1.23 and later clamp the
codebook distances at 1e-8, which ties entries of the cell tokeniser's small residual
codebooks and changes cell codes at residual levels 2 and 3). The tokeniser configs
request flash attention; it is used only when `flash-attn` is installed and a GPU with
compute capability 8.0 or higher is present, otherwise the code falls back to torch
attention, so an install without the `gpu` extra gives torch attention. The entry points
do not change the cuDNN setting; to switch TF32 off, run them through Python:

```bash
python -c "import runpy, sys, torch; torch.backends.cudnn.allow_tf32 = False; \
sys.argv = ['eval_tokenizer'] + sys.argv[1:]; runpy.run_module('hep4m.eval_tokenizer', run_name='__main__')" \
    -i configs/tokenize/topo.yml
```

**Attention in the HEP4M decode.** HEP4M tokenises its inputs and decodes its outputs
with the same tokenisers at inference time, so the attention path matters there too. The
released HEP4M-pflow predictions were made with flash attention in the tokenisers and are
reproduced bit for bit only with `flash-attn` installed. With torch attention the
per-particle pT agrees within ±0.5% (1st to 99th percentile) and 3 of 2048 events get a
different particle count.
HEP4M-multi, HEP4M-sim and all nanoHEP predictions were made with torch attention.
`README.json` in each checkpoint archive records the path used and the checks run.

## Evaluation

```bash
# jet and particle metrics (jet pT response and IQR, marginals, cardinality) for one prediction file
python -m hep4m.performance.evaluate metrics --root-path pred.root --engine nano_hep \
    --case pflow --run-id my_model --metrics-out metrics.json

# classifier two-sample test (C2ST), marginal and joint (conditioned on the inputs)
python -m hep4m.judge.build_input_cache --modalities topo track --split test --out inputs.npz
python -m hep4m.judge.c2st_reco --cache inputs.npz --row "my_model:nano_hep:pred.root" --out-dir c2st
python -m hep4m.judge.build_input_cache --modalities truthpart --split test --out inputs_tp.npz
python -m hep4m.judge.c2st_sim  --cache inputs_tp.npz --row "my_sim:nano_hep:sampled.root" --out-dir c2st_sim
```

`--case` is `pflow` (particle flow), `sim` (detector simulation) or `flashsim` (truth
particles -> HGPflow particles); `--engine` is `nano_hep`, `hep4m` or `hgpflow`. An AUC
of 0.5 means the classifier cannot tell the generated sample from the reference. The
set-transformer C2ST is also available as a library (`hep4m.judge.c2st_set.c2st_set`).

`c2st_reco` prints four AUCs per model: an event-aggregate MLP and a set transformer,
each marginal and joint. The classifier AUC of Tables 2 and 3 is the `set-trans joint`
line (`joint_set` in `c2st_reco.json`), run with the default settings on the full test
split (about 100k events). That setting needs a GPU; on a CPU, 5,000 events already
take about 15 minutes. The AUC is a lower bound on how separable the two samples are:
with fewer events or a shorter training the classifier separates less, so the AUC comes
out lower. `HEP4M_C2ST_RENORM_PHI=1` puts each generated (cos phi, sin phi) pair back on
the unit circle before the test, so that off-circle angles from the parallel decoder do
not by themselves separate the samples; the HEP4M rows of Table 2 use it.
`gen_quick_marginal_particle_auc` in the `evaluate metrics` JSON is a different, quick
check: a small MLP that separates single particles (not events), so it sees only the
per-particle marginals. It is not the paper's classifier AUC.

## Layout

```
hep4m/                 package
  models/              nanoHEP (nano_hep/), HEP4M, VQ-VAE tokeniser, shared layers
  lightnings/          Lightning modules and data modules for the three model types
  datasets/            raw ROOT readers and the token store (tokenized_memmap.py)
  evaluations/         inference helpers used by eval_hep4m / eval_tokenizer
  decoding/            tokens -> physics objects (Detokenizer)
  performance/         metrics and plots (evaluate.py, pflow_report.py, generative_report.py)
  judge/               C2ST
  data.py              data-record download, check, export and unpacking of the model archives
  finetune.py          weights-only initialisation for fine-tuning (init_weights_path)
  validate.py          checks an installation against the data record (python -m hep4m.validate)
  train_*.py, eval_*.py, build_token_store.py, paths.py   command-line entry points
configs/
  tokenizers/          tokeniser model/train configs and input variables
  tokenize/            tokenise raw ROOT with the frozen tokenisers
  modality_dicts/      which tokeniser checkpoint serves which modality
  train/               nanoHEP and HEP4M training configs
  infer/               inference configs
tests/                 unit tests, CPU workflow tests, CLI and site-path checks
```

## Tests

```bash
python -m pytest tests            # everything; tests that need data are skipped with the reason
```

- `tests/test_cli.py`, `tests/test_site_paths.py` and most of `tests/unit/` need no data.
- `tests/validation/` compares the installation with the Zenodo record (see
  [Validate your installation](#validate-your-installation)); it runs when
  `HEP4M_VALIDATE_DATA` points at the downloaded record.
- `tests/workflow/` runs tokenise, one training step per training config (tiny width),
  a weights-only fine-tune from a checkpoint, inference through the CLI for every config
  in `configs/infer/` (with tiny stand-in models), and the C2ST, on CPU. They need
  `HEP4M_DATA` (tokenisers and token store). Optional: `HEP4M_TEST_STORE` /
  `HEP4M_TEST_STORE_SPLIT` (default `$HEP4M_TOKENIZED` / `test`) and
  `HEP4M_TEST_ROOT_FILE` (default `$HEP4M_RAW_TEST`, for the HEP4M and tokeniser tests).
- Some unit tests use small fixtures that are not in the repository. `fixtures.tar.zst`
  in the Zenodo record unpacks into `tests/unit/fixtures/`
  (`tar --zstd -xf fixtures.tar.zst -C tests/unit/fixtures`), as the CI does; or set
  `HEP4M_FIXTURE_ROOT` (1000-event raw ROOT slice) and `HEP4M_TOKENIZED_FIXTURE`
  (100-event token store, made with `tests/unit/fixtures/generate_tokenized_fixture.py`).

## Citation

If you use this code, please cite the paper:

N. Kakati, D. Murnane, B. Hashemi, S. Klein, J. Krupa, E. Gross, L. Heinrich, M. Kagan,
"Prompting Particle Physics: Tokenized Multi-modal Foundation Models for Combinatorially Many Tasks" (2026).
arXiv identifier to follow.

`CITATION.cff` has the same entry in machine-readable form.

## Licence

The code is released under the Apache License 2.0 (see `LICENSE`). The data record
(10.5281/zenodo.22917591) is released under CC-BY-4.0.
