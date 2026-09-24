"""Validate an installation against the released data record (CPU, a few minutes).

After downloading the Zenodo record (10.5281/zenodo.22917591) into one directory::

    python -m hep4m.validate --data <record dir>

runs three checks and prints one line per result:

  checkpoints   each released model (``ckpt.<row>.tar.zst``) generates the first 32
                test events on CPU, and the generated tokens are compared with
                ``reference_outputs.npz`` from the record. Argmax items must be
                token-identical. Sampled items use a fixed seed (torch.manual_seed)
                and are compared with a reference made with that seed. They are
                token-identical only with the same torch version (the random stream
                changes between versions), so a difference there is a WARN; for HEP4M
                the number of objects per event (argmax cardinality) must still match.
                The ``model.ckpt`` sha256 is checked against the archive's README.json.
  training      20 optimiser steps of a tiny nanoHEP and a tiny HEP4M particle-flow model
                (the training configs at reduced width) on 8 released events (train
                split when its track/topo/truthpart files are present, else test):
                finite loss that goes down, and a checkpoint that is written and reloads
                with identical weights.
  tokenisation  the first 256 raw test events (``eval.raw_test_*.root``; hgpfpart from
                ``eval.hgpflow_pred_*.root``) are tokenised with the released tokenisers
                and compared row by row with the released test tokens; >= 99.9% identical
                rows per modality is a pass.

The reference outputs were made with this package installed as in the README
(``pip install -e .`` with the CPU PyTorch wheels: torch 2.14, vector-quantize-pytorch
1.22.0), on CPU, fp32, torch attention; ``meta`` inside the file records the versions.
Argmax outputs were also identical with torch 2.5, and with 4 or 16 CPU threads.
The tokens depend on the vector-quantize-pytorch version: with 1.23 or later the cell
tokens differ, so every run warns when another version is installed. flash-attn is
never used here (CPU); the released HEP4M-pflow predictions were made with flash-attn in
the tokenisers, so these references are not bit-identical to that prediction file.
Nothing is written to the data directory. Unpacked archives, small token stores and logs
of the package output go to ``--work`` (default ``$HEP4M_WORK/validate``); the work
directory must allow writable memory maps (a local disk, not a network file system that
lacks them).

Options: ``--checks checkpoints,training,tokenisation``, ``--rows HEP4M-pflow,...``,
``--n-events`` (checkpoint check, at most the reference size), ``--n-tok-events``.
Exit code 0 when nothing failed. ``pytest tests/validation`` runs the same checks when
``HEP4M_VALIDATE_DATA`` points at the record directory.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REFERENCE_FILE = "reference_outputs.npz"
VQ_VERSION = "1.22.0"
SEED = 1234
N_EVENTS = 32
N_TOK_EVENTS = 256
TOK_THRESHOLD = 0.999
TRAIN_EVENTS = 8
TRAIN_STEPS = 20
ALL7 = ["topo", "track", "truthpart", "truthjet", "hgpfpart", "cell", "celltruth"]
CHECKS = ("checkpoints", "training", "tokenisation")
REPO = Path(__file__).resolve().parent.parent


@dataclass
class Result:
    check: str
    name: str
    status: str  # PASS, FAIL, WARN, SKIP
    message: str = ""
    details: dict = field(default_factory=dict)

    def line(self) -> str:
        return f"{self.status:<4}  {self.check:<12}  {self.name:<32}  {self.message}"


# ---------------------------------------------------------------------------
# environment and inputs
# ---------------------------------------------------------------------------

def vq_version() -> str:
    import importlib.metadata as md

    try:
        return md.version("vector-quantize-pytorch")
    except md.PackageNotFoundError:
        return "not installed"


def vq_warning() -> Optional[str]:
    v = vq_version()
    if v == VQ_VERSION:
        return None
    return (f"vector-quantize-pytorch {v} is installed, the release pins {VQ_VERSION}. Versions >= 1.23 "
            f"clamp codebook distances and change cell (and a few hgpfpart) tokens, so the tokenisation and "
            f"HEP4M checks are expected to differ. Install with: python -m pip install "
            f"vector-quantize-pytorch=={VQ_VERSION}")


def environment() -> dict:
    import torch

    cpu = platform.processor() or ""
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                cpu = ln.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"python": platform.python_version(), "torch": torch.__version__,
            "vector_quantize_pytorch": vq_version(), "numpy": np.__version__,
            "platform": platform.platform(), "cpu": cpu, "torch_threads": torch.get_num_threads()}


class Inputs:
    """Record files, unpacked archives and small token stores used by the checks."""

    def __init__(self, data: Path, work: Path, log=print):
        self.data = Path(data).resolve()
        self.work = Path(work).resolve()
        self.log = log
        self.work.mkdir(parents=True, exist_ok=True)
        from hep4m import data as hd

        self.hd = hd
        self.raw_test = self.data / hd.RAW_TEST_FILE
        self.hgpflow_test = self.data / hd.HGPFLOW_TEST_FILE
        self.tokenizers = self.work / "tokenizers"
        self.checkpoints = self.work / "checkpoints"
        # every config resolves ${HEP4M_*} through the environment at load time
        os.environ.update(HEP4M_RELEASE=str(self.data), HEP4M_RAW_TEST=str(self.raw_test),
                          HEP4M_TOKENIZERS=str(self.tokenizers),
                          HEP4M_CKPT=str(self.checkpoints), HEP4M_OUTPUT_ROOT=str(self.work / "outputs"),
                          HEP4M_LOGGER="csv")
        self._ds = None

    @property
    def ds(self):
        if self._ds is None:
            if not (self.data / "schema.json").exists():
                raise FileNotFoundError(f"schema.json not found in {self.data}")
            self._ds = self.hd.HEP4MData(self.data)
        return self._ds

    def has_split(self, split: str, mods: Sequence[str]) -> bool:
        files = self.ds.schema["splits"].get(split, {}).get("files", {})
        need = []
        for m in mods:
            need += [p["file"] for p in files[f"{m}_content"]["parts"]] if m in ("cell", "celltruth") else \
                [files[f"{m}_data"]["file"]] + ([files[f"{m}_counts"]["file"]] if f"{m}_counts" in files else [])
        need += [files["event_numbers"]["file"]]
        if any(m in ("cell", "celltruth") for m in mods):
            need += [files["cell_window_index"]["file"], self.ds.schema["window_lut"]["file"]]
        return all((self.data / f).exists() for f in need)

    def tokenizers_ready(self) -> Optional[str]:
        """Unpack tokenizers.tar.zst when needed; returns an error message or None."""
        if (self.tokenizers / "TOKENIZERS.md").exists():
            return None
        arc = self.data / self.hd.TOKENIZERS_FILE
        if not arc.exists():
            return f"{self.hd.TOKENIZERS_FILE} not found in {self.data}"
        self.hd.extract_tar_zst(arc, self.tokenizers, log=self.log)
        return None

    def checkpoint_dir(self, row: str) -> Optional[Path]:
        d = self.checkpoints / f"ckpt.{row}"
        if (d / "model.ckpt").exists():
            return d
        arc = self.data / self.hd.checkpoint_file(row)
        if not arc.exists():
            return None
        self.hd.extract_tar_zst(arc, self.checkpoints, log=self.log)
        return d

    def first_rows(self, split: str, key: str, n: int) -> np.ndarray:
        """The first ``n`` rows of one record array, streamed (a train file is never read whole)."""
        with self.hd.NpyZstStream(self.data / self.ds.schema["splits"][split]["files"][key]["file"]) as s:
            return np.array(s.read_rows(n))

    def first_events(self, split: str, mod: str, n: int):
        """Token rows and offsets of the first ``n`` events of one modality."""
        if mod in ("cell", "celltruth"):
            rows = np.concatenate(list(self.ds.iter_rows(split, mod, 0, n)))
            return rows, 156 * np.arange(n + 1, dtype=np.int64)
        files = self.ds.schema["splits"][split]["files"]
        if f"{mod}_counts" in files:
            counts = self.first_rows(split, f"{mod}_counts", n).astype(np.int64)
        else:  # truthjet: 8 rows per event
            counts = np.full(n, self.ds.schema["modalities"][mod]["rows_per_event"], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(counts)])
        return self.first_rows(split, f"{mod}_data", int(off[-1])), off

    def store(self, split: str, mods: Sequence[str], n_events: int) -> Path:
        """Write the first ``n_events`` of ``split`` in the token-store layout read by the
        training and inference code (``<root>/<split>/{mod}_data.npy`` ...). Returns <root>."""
        root = self.work / f"store_{split}_{n_events}"
        d = root / split
        if (d / "done.json").exists() and set(json.loads((d / "done.json").read_text())) >= set(mods):
            return root
        d.mkdir(parents=True, exist_ok=True)
        ds = self.ds
        np.asarray(self.first_rows(split, "event_numbers", n_events), dtype=np.int64).tofile(d / "event_numbers.npy")
        for m in mods:
            rows, off = self.first_events(split, m, n_events)
            n_cb = ds.schema["modalities"][m]["n_codebooks"]
            n_pos = ds.schema["modalities"][m]["n_pos_codebooks"]
            np.ascontiguousarray(rows, dtype=np.int16).tofile(d / f"{m}_data.npy")
            off.tofile(d / f"{m}_offsets.npy")
            np.savez(d / f"{m}_meta.npz", n_events=n_events, n_codebooks=n_cb, n_pos_codebooks=n_pos)
            np.save(d / f"{m}_is_empty.npy", np.flatnonzero(np.diff(off) == 0).astype(np.int64))
        (d / "done.json").write_text(json.dumps(list(mods)))
        return root


@contextlib.contextmanager
def _quiet(logfile: Path):
    """Send the progress output of the package code to a log file in the work directory."""
    import logging
    import warnings

    logfile.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("lightning.pytorch")
    level = lg.level
    lg.setLevel(logging.WARNING)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        with open(logfile, "a") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh), \
                warnings.catch_warnings():
            # also at the file-descriptor level: some writers hold the original sys.stdout
            os.dup2(fh.fileno(), 1)
            os.dup2(fh.fileno(), 2)
            warnings.simplefilter("ignore")
            yield
            fh.flush()
    finally:
        for stream in (sys.__stdout__, sys.__stderr__):
            if stream is not None:
                stream.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        lg.setLevel(level)


def load_cfg(path) -> dict:
    from hep4m.paths import safe_load_expanded

    with open(path) as fh:
        return safe_load_expanded(fh)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. released checkpoints
# ---------------------------------------------------------------------------

def _is_sampled(engine: str, item: dict) -> bool:
    if engine == "nano_hep":
        return not bool(item.get("argmax", True))
    ks = list(item["top_k_token_dict"].values()) + list(item.get("top_k_gpos_token_dict", {}).values()) + \
        list(item["card_topk_dict"].values())
    return any(int(k) != 1 for k in ks)


def _nano_generate(ckpt_dir: Path, cfg: dict, store_root: Path, n_events: int,
                   items: Optional[Sequence[int]] = None) -> Dict[str, dict]:
    """Tokens generated by a nanoHEP checkpoint for every item of an inference config,
    with the prefix construction and batched decode of NanoHepInferenceHelper."""
    import torch

    from hep4m.datasets.tokenized_memmap import TokenizedMemmapDataset
    from hep4m.models.nano_hep import load_model_from_ckpt
    from hep4m.models.nano_hep.decode import batched_ar_decode, bucket_by_prefix_length

    model, vocab, _, _ = load_model_from_ckpt(ckpt_dir / "model.ckpt", device="cpu")
    model.eval()
    md = load_cfg(ckpt_dir / "modality_dict.yml")
    out = {}
    for i, item in enumerate(cfg["items"]):
        if items is not None and i not in items:
            continue
        sd = item["sampling_dict"]
        in_mods = list(sd["fixed_input_output_modalities"]["input"])
        out_mods = sorted(sd["fixed_input_output_modalities"]["output"])
        ds = TokenizedMemmapDataset(tokenized_root=str(store_root), split="test",
                                    all_modalities=list(md.keys()), sampling_dict=sd,
                                    block_size=model.config.block_size, vocab=vocab, max_events=n_events)
        prefixes = []
        for idx in range(len(ds)):
            seq = ds.build_sample(idx, in_mods, out_mods)["seq"]
            sep = int((seq == vocab.task_sep).nonzero()[0].item())
            mod_out = set(vocab.mod_out.values())
            first = next(j for j, t in enumerate(seq[sep + 1:].tolist()) if t in mod_out)
            prefixes.append(seq[: sep + 1 + first + 1])
        preds: Dict[int, Dict[str, np.ndarray]] = {}
        torch.manual_seed(SEED + i)
        with torch.no_grad():
            for _, idxs in sorted(bucket_by_prefix_length(prefixes).items()):
                pred = batched_ar_decode(model, vocab, torch.stack([prefixes[j] for j in idxs]), out_mods,
                                         output_layout=sd.get("output_layout", "interleaved"),
                                         temperature=float(item.get("temperature", 1.0)),
                                         argmax=bool(item.get("argmax", True)),
                                         max_new_tokens=int(item.get("max_new_tokens", 1024)),
                                         structural_decode=item.get("structural_decode", item.get("argmax", True)),
                                         device="cpu")
                for b, j in enumerate(idxs):
                    preds[j] = {m: pred[m][b].cpu().numpy().reshape(-1, 1) for m in out_mods}
        out[f"{i}"] = {"info": item.get("info", ""), "sampled": _is_sampled("nano_hep", item),
                       "tokens": {m: [preds[j][m] for j in range(len(ds))] for m in out_mods}}
    return out


def _hep4m_generate(ckpt_dir: Path, cfg: dict, inputs: Inputs, n_events: int,
                    items: Optional[Sequence[int]] = None) -> Dict[str, dict]:
    """Tokens (content + position codes of every predicted object) generated by a HEP4M
    checkpoint, with the data path and forward pass of HEP4MInferenceHelper."""
    import torch

    from hep4m.evaluations.hep4m_inference_helper import HEP4MInferenceHelper

    init = copy.deepcopy(cfg["init"])
    init.update(device="cpu", gpu=-1, num_workers=0, batch_size=n_events, chunk_size=n_events,
                output_dir=str(inputs.work / "outputs"))
    init["model"].update(config_path_m=str(ckpt_dir / "config_m.yml"), checkpoint_path=str(ckpt_dir / "model.ckpt"),
                         modality_dict_path=str(ckpt_dir / "modality_dict.yml"))
    helper = HEP4MInferenceHelper(init)
    model = helper.lightning_model.model
    out = {}
    for i, item in enumerate(cfg["items"]):
        if items is not None and i not in items:
            continue
        inf = copy.deepcopy(item)
        files = {m: (str(inputs.hgpflow_test) if m == "hgpfpart" else str(inputs.raw_test)) for m in inf["filepath_dict"]}
        inf.update(filepath_dict=files, reduce_ds=n_events)
        model.set_input_output_modalities(inf["input_modalities"], inf["output_modalities"])
        torch.manual_seed(SEED + i)
        loader = helper.get_dataloader(inf)
        toks: Dict[str, list] = {m: [] for m in inf["output_modalities"]}
        with torch.no_grad():
            for batch in loader:
                input_dict = model.encode_to_tokens(batch, model.input_modalities)
                pred = model.fast_forward(
                    input_dict=input_dict, output_modalities=model.output_modalities, q_mask_dict=None,
                    top_k_token_dict=inf["top_k_token_dict"], top_k_gpos_token_dict=inf["top_k_gpos_token_dict"],
                    get_logits=False, temperature_token_dict=inf["temperature_token_dict"],
                    temperature_gpos_token_dict=inf["temperature_gpos_token_dict"],
                    use_truth_cardinality=inf["use_truth_cardinality"], card_topk_dict=inf["card_topk_dict"],
                    card_temperature_dict=inf["card_temperature_dict"])
                for m in inf["output_modalities"]:
                    t, q = pred[m]["tokens"], pred[m]["q_mask"].bool()
                    g = pred[m].get("gpos_tokens")
                    for b in range(t.shape[0]):
                        rows = t[b][q[b]].reshape(int(q[b].sum()), -1)
                        if g is not None:
                            rows = torch.cat([rows, g[b][q[b]].reshape(rows.shape[0], -1)], dim=1)
                        toks[m].append(rows.cpu().numpy())
        card_argmax = all(int(k) == 1 for k in item["card_topk_dict"].values())
        out[f"{i}"] = {"info": item.get("info", ""), "sampled": _is_sampled("hep4m", item),
                       "card_argmax": card_argmax, "tokens": toks}
    return out


def generate(row: str, inputs: Inputs, n_events: int, items: Optional[Sequence[int]] = None) -> Dict[str, dict]:
    """Tokens for the items of ``configs/infer/<row>.yml`` (only the indices in ``items`` when given)."""
    ckpt_dir = inputs.checkpoint_dir(row)
    cfg = load_cfg(REPO / "configs" / "infer" / f"{row}.yml")
    if cfg["init"].get("model_type", "hep4m") == "nano_hep":
        md = load_cfg(ckpt_dir / "modality_dict.yml")
        store_root = inputs.store("test", list(md.keys()), n_events)
        return _nano_generate(ckpt_dir, cfg, store_root, n_events, items)
    return _hep4m_generate(ckpt_dir, cfg, inputs, n_events, items)


def _pack(gen: Dict[str, dict], row: str) -> Dict[str, np.ndarray]:
    arrs = {}
    for i, it in gen.items():
        for m, per_ev in it["tokens"].items():
            key = f"{row}__{i}__{m}"
            width = per_ev[0].shape[1] if per_ev else 1
            arrs[key + "__tokens"] = np.concatenate([a.reshape(-1, width) for a in per_ev]).astype(np.int32) \
                if per_ev else np.zeros((0, 1), np.int32)
            arrs[key + "__counts"] = np.array([a.shape[0] for a in per_ev], dtype=np.int32)
    return arrs


def _unpack(ref, row: str, item: str, mod: str, n_events: int) -> List[np.ndarray]:
    toks, counts = ref[f"{row}__{item}__{mod}__tokens"], ref[f"{row}__{item}__{mod}__counts"]
    off = np.concatenate([[0], np.cumsum(counts)])
    return [toks[off[j]:off[j + 1]] for j in range(min(n_events, len(counts)))]


def _checkpoint_one(inputs: Inputs, row: str, ref, meta: dict, n: int, reference: Path) -> List[Result]:
    if row not in meta["rows"]:
        return [Result("checkpoints", row, "SKIP", f"no reference for {row} in {reference.name}")]
    ckpt_dir = inputs.checkpoint_dir(row)
    if ckpt_dir is None:
        return [Result("checkpoints", row, "SKIP", f"{inputs.hd.checkpoint_file(row)} not in {inputs.data}")]
    want = json.loads((ckpt_dir / "README.json").read_text()).get("model_ckpt_sha256")
    if want and sha256(ckpt_dir / "model.ckpt") != want:
        return [Result("checkpoints", row, "FAIL", "model.ckpt sha256 differs from README.json")]
    t0 = time.time()
    res = []
    # items added to the config after the reference was made have no reference tokens
    items = [int(k) for k in meta["items"][row]] if row in meta.get("items", {}) else None
    with _quiet(inputs.work / "logs" / f"checkpoint_{row}.log"):
        gen = generate(row, inputs, n, items)
    for i, it in gen.items():
        n_same, n_card, diff_mods = 0, 0, set()
        for j in range(n):
            same = card = True
            for m, per_ev in it["tokens"].items():
                r = _unpack(ref, row, i, m, n)[j]
                card &= per_ev[j].shape == r.shape
                if per_ev[j].shape != r.shape or not np.array_equal(per_ev[j], r):
                    same = False
                    diff_mods.add(m)
            n_same += same
            n_card += card
        name = f"{row} [{i}] {'sampled' if it['sampled'] else 'argmax'}"
        msg = f"{n_same}/{n} events token-identical ({it['info']}; {time.time() - t0:.0f}s)"
        if it.get("card_argmax") and it["sampled"]:
            # HEP4M: the number of objects is the argmax of the cardinality head even when
            # the tokens are sampled, so it must match on any machine
            msg += f"; {n_card}/{n} events with the same number of objects (argmax cardinality)"
        if n_same == n:
            status = "PASS"
        elif it["sampled"] and not (it.get("card_argmax") and n_card < n):
            status = "WARN"
            msg += (f"; differing: {sorted(diff_mods)}. Sampled decoding with a fixed seed is exact only with "
                    f"the same torch version and CPU type (reference: torch {meta['env']['torch']}, "
                    f"{meta['env']['cpu']})")
        else:
            status = "FAIL"
            msg += f"; differing: {sorted(diff_mods)}"
        res.append(Result("checkpoints", name, status, msg, {"identical": n_same, "n": n}))
    return res


def check_checkpoints(inputs: Inputs, rows: Optional[Sequence[str]], reference: Path, n_events: int = N_EVENTS,
                      log=print) -> List[Result]:
    """Check ``rows`` (default: every model in the reference outputs)."""
    res = []
    if not reference.exists():
        return [Result("checkpoints", "reference outputs", "SKIP",
                       f"{reference.name} not found in {reference.parent}; download it from the record")]
    ref = np.load(reference)
    meta = json.loads(str(ref["meta"]))
    rows = list(rows or meta["rows"])
    n = min(n_events, int(meta["n_events"]))
    if err := inputs.tokenizers_ready():
        return [Result("checkpoints", r, "SKIP", err) for r in rows]
    for row in rows:
        try:
            rr = _checkpoint_one(inputs, row, ref, meta, n, reference)
        except Exception as e:  # report and continue with the next model
            rr = [Result("checkpoints", row, "FAIL", f"{type(e).__name__}: {e} (log: {inputs.work}/logs)")]
        for r in rr:
            log(r.line())
        res += rr
    return res


def write_reference(inputs: Inputs, rows: Sequence[str], path: Path, n_events: int = N_EVENTS, log=print) -> Path:
    """Generate the reference tokens (maintainers; run in the release environment on CPU)."""
    if err := inputs.tokenizers_ready():
        raise FileNotFoundError(err)
    arrs, items = {}, {}
    for row in rows:
        t0 = time.time()
        gen = generate(row, inputs, n_events)
        arrs.update(_pack(gen, row))
        items[row] = {i: {"info": it["info"], "sampled": it["sampled"]} for i, it in gen.items()}
        log(f"reference {row}: {len(gen)} item(s), {time.time() - t0:.0f}s")
    meta = {"n_events": n_events, "seed": SEED, "rows": list(rows), "items": items, "env": environment(),
            "device": "cpu", "precision": "fp32", "attention": "torch (flash-attn not used on CPU)",
            "events": "first n_events of the test split / raw test file",
            "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrs)
    log(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB, sha256 {sha256(path)})")
    return path


# ---------------------------------------------------------------------------
# 2. training loop
# ---------------------------------------------------------------------------

def _shrink_hep4m(cfg: dict, emb: int = 64) -> dict:
    cfg = copy.deepcopy(cfg)
    old = cfg["embedding_dim"]

    def walk(o):
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k in ("embed_dim", "embedding_dim", "out_dim", "input_size") and v == old:
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


def _fit(lit, dm, tmpdir: Path, steps: int):
    from lightning.pytorch import Trainer

    losses = []
    orig = lit.training_step

    def training_step(batch, batch_idx):
        loss = orig(batch, batch_idx)
        losses.append(float(loss.detach()))
        return loss

    lit.training_step = training_step
    tr = Trainer(max_steps=steps, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                 enable_progress_bar=False, enable_model_summary=False, limit_val_batches=0,
                 num_sanity_val_steps=0, default_root_dir=str(tmpdir))
    tr.fit(lit, datamodule=dm)
    lit.training_step = orig
    return tr, losses


def _loss_result(name: str, losses: List[float], ckpt_ok: bool, ckpt_msg: str, t0: float) -> Result:
    if not losses:
        return Result("training", name, "FAIL", "no training step ran")
    first, last = losses[0], float(np.mean(losses[-5:]))
    ok = all(math.isfinite(x) for x in losses) and last < first and ckpt_ok
    msg = (f"{len(losses)} steps, loss {first:.3f} -> {last:.3f} (mean of last 5); {ckpt_msg}; "
           f"{time.time() - t0:.0f}s")
    return Result("training", name, "PASS" if ok else "FAIL", msg, {"losses": losses})


def train_nanohep(store_root: Path, tmpdir: Path, steps: int = TRAIN_STEPS) -> Result:
    import torch

    from hep4m.lightnings.nano_hep_lightning import NanoHepDataModule, NanoHepLightning
    from hep4m.models.nano_hep import load_model_from_ckpt

    t0 = time.time()
    cfg = load_cfg(REPO / "configs/train/nanohep_pflow.yml")
    gpt = dict(cfg["gpt_config"], n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    opt = dict(cfg.get("optimizer_args", {}) or {})
    opt["lr_schedule"] = dict(opt.get("lr_schedule") or {}, peak=1e-3, floor=1e-4, warmup_steps=2, decay_steps=steps)
    opt["learning_rate"] = 1e-3
    torch.manual_seed(0)
    lit = NanoHepLightning(gpt_config=gpt, vocab_args=dict(cfg["vocab_args"]), optimizer_args=opt,
                           pflow_eval=dict(cfg.get("pflow_eval", {}) or {}, enabled=False))
    dm = NanoHepDataModule(tokenized_root=str(store_root), all_modalities=list(cfg["all_modalities"]),
                           sampling_dict=dict(cfg["sampling_dict"]), block_size=int(cfg["block_size"]),
                           vocab_args=dict(cfg["vocab_args"]), batch_size=TRAIN_EVENTS, num_workers=0,
                           max_events_train=TRAIN_EVENTS, max_events_val=TRAIN_EVENTS)
    tr, losses = _fit(lit, dm, tmpdir, steps)
    ckpt = tmpdir / "nanohep_tiny.ckpt"
    tr.save_checkpoint(str(ckpt))
    model, _, _, step = load_model_from_ckpt(ckpt, device="cpu")  # the loader used for inference
    trained = {k: v for k, v in lit.model.state_dict().items()}
    same = all(torch.equal(v, trained[k]) for k, v in model.state_dict().items())
    return _loss_result("nanoHEP (nanohep_pflow.yml, tiny)", losses, same and step == steps,
                        f"checkpoint reloads with identical weights: {same}, step {step}", t0)


def train_hep4m(store_root: Path, tmpdir: Path, steps: int = TRAIN_STEPS) -> Result:
    import torch

    from hep4m.lightnings.hep4m_lightning import HEP4MDataModule, HEP4MLightning

    t0 = time.time()
    base = REPO / "configs/train/hep4m_pflow"
    cfg_t = load_cfg(base / "config_t.yml")
    cfg_m = _shrink_hep4m(load_cfg(base / "config_m.yml"))
    md = load_cfg(base / "modality_dict.yml")
    cfg_t.update(preprocessed_dir=str(store_root), batchsize_train=TRAIN_EVENTS,
                 batchsize_val=TRAIN_EVENTS, reduce_ds_train=TRAIN_EVENTS, reduce_ds_val=TRAIN_EVENTS,
                 num_workers=0, persistent_workers=False, device="cpu", base_root_dir=str(tmpdir),
                 warmup_steps=2, learning_rate=1e-3)
    torch.manual_seed(0)
    lit = HEP4MLightning(config_m=cfg_m, modality_dict=md, config_t=cfg_t, device="cpu")
    dm = HEP4MDataModule(config_t=cfg_t, config_v_dict=lit.config_v_dict, modality_dict=md)
    tr, losses = _fit(lit, dm, tmpdir, steps)
    ckpt = tmpdir / "hep4m_tiny.ckpt"
    tr.save_checkpoint(str(ckpt))
    # reload as the inference helper does: fresh module + state_dict from the file
    fresh = HEP4MLightning(config_m=cfg_m, modality_dict=md, device="cpu")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    fresh.load_state_dict(sd, strict=False)
    trained, reloaded = lit.state_dict(), fresh.state_dict()
    same = all(torch.equal(reloaded[k], v) for k, v in trained.items() if k in reloaded)
    n_missing = sum(k not in reloaded for k in trained if not k.startswith("losses."))
    ok = same and n_missing == 0
    return _loss_result("HEP4M (hep4m_pflow config, tiny)", losses, ok,
                        f"checkpoint reloads with identical weights: {ok}", t0)


def check_training(inputs: Inputs, log=print) -> List[Result]:
    mods = ["topo", "track", "truthpart"]
    if err := inputs.tokenizers_ready():
        return [Result("training", "nanoHEP + HEP4M", "SKIP", err)]
    split = "train" if inputs.has_split("train", mods) else "test"
    if not inputs.has_split(split, mods):
        return [Result("training", "nanoHEP + HEP4M", "SKIP", f"no {split}.track/topo/truthpart files in {inputs.data}")]
    log(f"training on the first {TRAIN_EVENTS} events of the {split} split"
        + ("" if split == "train" else " (train.* files not downloaded)"))
    root = inputs.store(split, mods, TRAIN_EVENTS)
    store = inputs.work / f"train_store_{split}"
    for s in ("train", "val", "test"):
        (store / s).parent.mkdir(parents=True, exist_ok=True)
        if not (store / s).exists():
            (store / s).symlink_to(root / split, target_is_directory=True)
    res = []
    for fn, sub in ((train_nanohep, "nano"), (train_hep4m, "hep4m")):
        tmp = inputs.work / "train_runs" / sub
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            with _quiet(inputs.work / "logs" / f"training_{sub}.log"):
                r = fn(store, tmp)
        except Exception as e:  # report and continue with the next check
            r = Result("training", sub, "FAIL", f"{type(e).__name__}: {e}")
        r.details["split"] = split
        res.append(r)
        log(r.line())
    return res


# ---------------------------------------------------------------------------
# 3. tokenisation
# ---------------------------------------------------------------------------

def _tokenise_one(inputs: Inputs, m: str, n_events: int, md: dict, tmp: Path, warn: Optional[str]) -> Result:
    from hep4m.build_token_store import build
    from hep4m.evaluations.tokenizer_inference_helper import TokenizerInferenceHelper

    t0 = time.time()
    src = inputs.hgpflow_test if m == "hgpfpart" else inputs.raw_test
    if not src.exists():
        return Result("tokenisation", m, "SKIP", f"{src.name} not in {inputs.data}")
    if not inputs.has_split("test", [m]):
        return Result("tokenisation", m, "SKIP", f"released test.{m} files not in {inputs.data}")
    cfg = load_cfg(REPO / "configs" / "tokenize" / f"{m}.yml")
    bs = 16 if m in ("cell", "celltruth") else 64  # cell images: keeps the peak memory near 2 GB
    init = dict(cfg["init"], gpu=-1, batch_size=bs, chunk_size=64, output_dir=str(tmp / m))
    out_root = tmp / m / "validate" / src.name
    store = tmp / "store" / "test"
    with _quiet(inputs.work / "logs" / f"tokenise_{m}.log"):
        TokenizerInferenceHelper(init).run_inference({"input_path": str(src), "dir_flag": "validate",
                                                      "reduce_ds": n_events})
        gc.collect()  # close the ROOT files now: uproot's thread pools cannot shut down at interpreter exit
        rc = build(store, {m: str(out_root)}, md, expect_events=n_events, log=print)
        gc.collect()
    if rc:
        return Result("tokenisation", m, "FAIL", f"build_token_store returned {rc}")
    off = np.fromfile(store / f"{m}_offsets.npy", dtype=np.int64)
    new = np.fromfile(store / f"{m}_data.npy", dtype=np.int16).reshape(off[-1], -1)
    view = inputs.ds.modality("test", m)
    rel = np.concatenate(list(inputs.ds.iter_rows("test", m, 0, n_events)))
    rel_counts = np.diff(np.asarray(view.offsets[:n_events + 1]))
    if not np.array_equal(np.diff(off), rel_counts):
        nd = int((np.diff(off) != rel_counts).sum())
        return Result("tokenisation", m, "FAIL", f"objects per event differ in {nd}/{n_events} events")
    ev_file = store / "event_numbers.npy"
    if ev_file.exists() and not np.array_equal(np.fromfile(ev_file, dtype=np.int64)[:n_events],
                                               inputs.ds.event_numbers("test")[:n_events]):
        return Result("tokenisation", m, "FAIL", "event numbers differ from the released test split")
    eq = new == rel
    rows = float(eq.all(1).mean()) if len(eq) else 1.0
    levels = " / ".join(f"{x:.4%}" for x in eq.mean(0)[: view.n_codebooks])
    status = "PASS" if rows >= TOK_THRESHOLD else ("WARN" if warn else "FAIL")
    msg = (f"{rows:.4%} of {len(eq)} rows identical (content levels {levels}), {n_events} events, "
           f"{time.time() - t0:.0f}s")
    if warn and rows < TOK_THRESHOLD:
        msg += f"; expected with vector-quantize-pytorch {vq_version()}"
    return Result("tokenisation", m, status, msg, {"row_identical": rows, "rows": int(len(eq))})


def check_tokenisation(inputs: Inputs, n_events: int = N_TOK_EVENTS, mods: Sequence[str] = ALL7,
                       log=print) -> List[Result]:
    import tempfile

    if err := inputs.tokenizers_ready():
        return [Result("tokenisation", m, "SKIP", err) for m in mods]
    if not inputs.raw_test.exists():
        return [Result("tokenisation", m, "SKIP", f"{inputs.raw_test.name} not in {inputs.data}") for m in mods]
    warn = vq_warning()
    md = load_cfg(REPO / "configs/modality_dicts/7mod.yml")
    res = []
    with tempfile.TemporaryDirectory(dir=inputs.work, prefix="tokenise_") as tmp:
        for m in mods:
            try:
                r = _tokenise_one(inputs, m, n_events, md, Path(tmp), warn)
            except Exception as e:  # report and continue with the next modality
                r = Result("tokenisation", m, "FAIL", f"{type(e).__name__}: {e} (log: {inputs.work}/logs)")
            res.append(r)
            log(r.line())
    return res


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def run(data: Path, work: Path, checks: Sequence[str] = CHECKS, rows: Optional[Sequence[str]] = None,
        n_events: int = N_EVENTS, n_tok_events: int = N_TOK_EVENTS, reference: Optional[Path] = None,
        log=print) -> List[Result]:
    inputs = Inputs(data, work, log=log)
    reference = Path(reference) if reference else inputs.data / REFERENCE_FILE
    res: List[Result] = []
    if "tokenisation" in checks:
        res += check_tokenisation(inputs, n_tok_events, log=log)
    if "checkpoints" in checks:
        res += check_checkpoints(inputs, rows, reference, n_events, log=log)
    if "training" in checks:
        res += check_training(inputs, log=log)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="directory with the downloaded record files")
    ap.add_argument("--work", type=Path, default=None, help="scratch directory (default $HEP4M_WORK/validate)")
    ap.add_argument("--checks", default=",".join(CHECKS), help=f"comma-separated subset of {','.join(CHECKS)}")
    ap.add_argument("--rows", default=None, help="comma-separated checkpoint rows (default: every model in the reference outputs)")
    ap.add_argument("--n-events", type=int, default=N_EVENTS, help="events per checkpoint item")
    ap.add_argument("--n-tok-events", type=int, default=N_TOK_EVENTS, help="events for the tokenisation check")
    ap.add_argument("--reference", type=Path, default=None, help=f"default <data>/{REFERENCE_FILE}")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: min(16, CPU count))")
    ap.add_argument("--write-reference", type=Path, default=None,
                    help="maintainers: generate the reference outputs into this file instead of checking")
    ap.add_argument("--json", type=Path, default=None, help="also write the results to this JSON file")
    a = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch starts
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(a.threads or min(16, os.cpu_count() or 1))
    import hep4m.paths as P

    work = a.work or Path(P.WORK) / "validate"
    rows = a.rows.split(",") if a.rows else None
    env = environment()
    print("hep4m.validate: " + ", ".join(f"{k} {v}" for k, v in env.items()), flush=True)
    if w := vq_warning():
        print(f"WARNING: {w}", flush=True)
    if not a.data.is_dir():
        print(f"no data directory {a.data}")
        return 2
    t0 = time.time()
    if a.write_reference:
        from hep4m.data import CHECKPOINT_ROWS

        write_reference(Inputs(a.data, work), rows or list(CHECKPOINT_ROWS), a.write_reference, a.n_events)
        return 0
    checks = [c.strip() for c in a.checks.split(",") if c.strip()]
    bad = [c for c in checks if c not in CHECKS]
    if bad:
        ap.error(f"unknown checks {bad}; choose from {CHECKS}")
    res = run(a.data, work, checks, rows, a.n_events, a.n_tok_events, a.reference)
    print(f"\n{'':-<100}\nsummary ({time.time() - t0:.0f}s):")
    for r in res:
        print(r.line())
    n = {s: sum(r.status == s for r in res) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print(", ".join(f"{v} {k}" for k, v in n.items()))
    if a.json:
        a.json.write_text(json.dumps({"env": env, "seconds": time.time() - t0,
                                      "results": [r.__dict__ for r in res]}, indent=1))
    gc.collect()
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
