"""Decode the input modalities of the tokenised store into a cache for the joint C2ST.

The joint (conditional) C2ST needs the model INPUTS (topo, track) per event,
but the prediction ROOTs only carry the output collections. This script reads
the token memmaps from the tokenised store (``$HEP4M_TOKENIZED/{split}``, same
layout as ``TokenDatasetNumpy``: ``{m}_data.npy`` int16 rows of n_codebooks
content + n_pos_codebooks position columns, ``{m}_offsets.npy`` raw int64,
``event_numbers.npy`` raw int64), decodes them with the frozen VQ-VAE
``Detokenizer``, and writes ONE compressed npz keyed by event_number:

    event_numbers          (N,)    int64   raw store event numbers (store order)
    modalities             (M,)    str
    {m}_features           (T, F)  float32 flat per-object physical features
    {m}_offsets            (N+1,)  int64   event i -> rows [off[i], off[i+1])
    {m}_feature_names      (F,)    str

Feature columns mirror ``evaluate._build_c2st_features`` per modality
(topo: e, eta, cosphi, sinphi, em_frac, rho; track: pt, eta, cosphi, sinphi,
d0, z0) except col 0 stays RAW (no log1p) so per-event sums are physical.
Consume with ``evaluate.load_c2st_input_cache``.

Usage (CPU works; ~seconds for a few hundred events):
    python -m hep4m.judge.build_input_cache \
        --modalities topo track --split test --max-events 200 \
        --out /path/to/c2st_inputs_test_200.npz
"""
import hep4m.paths as _HP
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from hep4m.decoding.detokenizer import Detokenizer
from hep4m.performance.evaluate import MODALITY_SCHEMA

DEFAULT_MODALITY_DICT = f"{_HP.REPO}/configs/modality_dicts/7mod.yml"


def _disable_flash_attn(detok: Detokenizer) -> None:
    """Switch the tokenisers' attention to the torch path for CPU decoding
    (the flash-attention kernel is CUDA-only)."""
    for vq in detok.vqs.values():
        for mod in vq.modules():
            if getattr(mod, "enable_flash_attn", False):
                mod.enable_flash_attn = False


def _cache_columns(modality: str, detok: Detokenizer) -> list:
    """Cache column names (short, no modality prefix), _build_c2st_features order."""
    schema = MODALITY_SCHEMA[modality]
    cols = [schema["kinematic"], "eta", "cosphi", "sinphi"]
    if schema["has_class"]:
        cols.append("class")
    cols += list(schema["extras"])
    # every column must exist in the detokenizer's decoded feature names
    decoded_names = {n.removeprefix(f"{modality}_") for n in detok.feature_names(modality)}
    missing = [c for c in cols if c not in decoded_names]
    if missing:
        raise KeyError(f"{modality}: cache columns {missing} not among decoded "
                       f"features {sorted(decoded_names)}")
    return cols


def _decode_chunk(detok, modality, data, offsets, ev_lo, ev_hi, max_card, cols, ncb):
    """Decode store events [ev_lo, ev_hi) -> (flat (T, F) float32, counts (n,) int64)."""
    lens = np.minimum(np.diff(offsets[ev_lo:ev_hi + 1]), max_card).astype(np.int64)
    B = ev_hi - ev_lo
    N = int(lens.max()) if len(lens) and lens.max() > 0 else 1
    ct = torch.zeros(B, N, ncb, dtype=torch.long)
    pt = torch.zeros(B, N, data.shape[1] - ncb, dtype=torch.long)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        n = int(lens[b])
        if n == 0:
            continue
        start = offsets[ev_lo + b]
        rows = torch.from_numpy(np.array(data[start:start + n])).long()  # copy: memmap is read-only
        ct[b, :n] = rows[:, :ncb]
        pt[b, :n] = rows[:, ncb:]
        mask[b, :n] = True
    decoded = detok.decode_tokens(modality, ct, pt, mask)
    m_np = mask.numpy()
    feats = []
    for c in cols:
        v = decoded.get(f"{modality}_{c}", decoded.get(c))
        if v is None:
            raise KeyError(f"{modality}: decoded output missing column {c!r} "
                           f"(got {sorted(decoded)})")
        feats.append(v.numpy()[m_np].astype(np.float32))
    return np.stack(feats, axis=-1), lens


def main():
    ap = argparse.ArgumentParser(description="Build the detokenised-inputs cache for the joint C2ST")
    ap.add_argument("--modalities", nargs="+", default=["topo", "track"],
                    choices=sorted(MODALITY_SCHEMA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-events", type=int, default=-1,
                    help="-1 = all store events")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--store-dir", default=_HP.TOKENIZED,
                    help="tokenised store root (default $HEP4M_TOKENIZED)")
    ap.add_argument("--modality-dict", default=DEFAULT_MODALITY_DICT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="events decoded per batch (bounds memory)")
    args = ap.parse_args()

    if args.device == "cpu":
        # the VQ-VAE attention imports flash_attn at construction time whenever
        # CUDA is visible; hiding CUDA keeps CPU decoding on the torch path
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not args.store_dir:
        raise SystemExit("--store-dir not given and $HEP4M_TOKENIZED unset")
    store = Path(args.store_dir) / args.split
    if not store.is_dir():
        raise SystemExit(f"store split dir not found: {store}")

    detok = Detokenizer(args.modality_dict, args.modalities, device=args.device)
    if args.device == "cpu":
        _disable_flash_attn(detok)

    # raw event numbers are a headerless int64 sidecar (see dataset_hep4m.py)
    event_numbers = np.fromfile(store / "event_numbers.npy", dtype="int64")
    n_events = len(event_numbers)
    if args.max_events > 0:
        n_events = min(n_events, args.max_events)
        event_numbers = event_numbers[:n_events]

    out_arrays = {
        "event_numbers": event_numbers,
        "modalities": np.array(args.modalities),
    }
    for m in args.modalities:
        meta = np.load(store / f"{m}_meta.npz")
        ncb, npb = int(meta["n_codebooks"]), int(meta["n_pos_codebooks"])
        assert int(meta["n_events"]) == len(np.fromfile(store / "event_numbers.npy", dtype="int64")), \
            f"{m}: meta n_events disagrees with event_numbers.npy"
        offsets = np.memmap(store / f"{m}_offsets.npy", dtype="int64", mode="r")
        data = np.memmap(store / f"{m}_data.npy", dtype="int16", mode="r").reshape(-1, ncb + npb)
        # same truncation as TokenDatasetNumpy: max_token_cardinality from the VQ config
        max_card = int(detok.config_v[m]["features"][f"{m}_feat0"][0])
        cols = _cache_columns(m, detok)

        flat_chunks, count_chunks = [], []
        for lo in range(0, n_events, args.chunk_size):
            hi = min(lo + args.chunk_size, n_events)
            flat, counts = _decode_chunk(detok, m, data, offsets, lo, hi, max_card, cols, ncb)
            flat_chunks.append(flat)
            count_chunks.append(counts)
            print(f"  {m}: decoded events [{lo}, {hi})", flush=True)
        flat = np.concatenate(flat_chunks, axis=0) if flat_chunks else np.zeros((0, len(cols)), np.float32)
        counts = np.concatenate(count_chunks) if count_chunks else np.zeros(0, np.int64)
        out_arrays[f"{m}_features"] = flat
        out_arrays[f"{m}_offsets"] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        out_arrays[f"{m}_feature_names"] = np.array(cols)
        print(f"{m}: {flat.shape[0]} objects over {n_events} events "
              f"(features: {cols})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **out_arrays)
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB, {n_events} events, "
          f"modalities: {args.modalities})")


if __name__ == "__main__":
    main()
