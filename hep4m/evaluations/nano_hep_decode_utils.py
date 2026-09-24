"""nanoHEP autoregressive decoding followed by detokenisation, per event.

Used by the fixed-direction evaluation during training
(``NanoHepLightning.on_validation_epoch_end``), which passes the arrays to
``hep4m.performance.pflow_report.run_report_from_arrays``.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..models.nano_hep import Vocab
from ..models.nano_hep.decode import batched_ar_decode, bucket_by_prefix_length
from ..datasets.tokenized_memmap import TokenizedMemmapDataset
from ..decoding import Detokenizer


logger = logging.getLogger(__name__)


def build_ar_prefixes(
    ds: TokenizedMemmapDataset,
    vocab: Vocab,
    idxs: Sequence[int],
    input_mods: Sequence[str],
    output_mods: Sequence[str],
) -> List[Optional[torch.Tensor]]:
    """AR prefix (header + inputs + first ``<MOD_OUT>`` cue) of each event.

    The entry is ``None`` for an event whose sequence has no output cue (the inputs
    fill the whole block).
    """
    mod_out_ids = set(vocab.mod_out.values())
    prefixes: List[Optional[torch.Tensor]] = []
    for idx in idxs:
        seq = ds.build_sample(idx, list(input_mods), list(output_mods))["seq"]
        task_sep_positions = (seq == vocab.task_sep).nonzero()
        if task_sep_positions.numel() == 0:
            prefixes.append(None)
            continue
        task_sep_pos = int(task_sep_positions[0].item())
        body_first_mod_out = next(
            (i for i, t in enumerate(seq[task_sep_pos + 1:].tolist()) if t in mod_out_ids),
            None,
        )
        if body_first_mod_out is None:
            prefixes.append(None)
            continue
        prefixes.append(seq[: task_sep_pos + 1 + body_first_mod_out + 1].clone())
    return prefixes


def _decode_truth_event(
    ds: TokenizedMemmapDataset,
    det: Detokenizer,
    idx: int,
    modality: str,
) -> Dict[str, np.ndarray]:
    """Decode one event's truth tokens for one modality. Returns {short_feat: ndarray}."""
    mm = ds.mms[modality]
    if det.data_type.get(modality) == "cell_image":
        content = torch.from_numpy(mm.event_codes(idx, mm.n_codebooks)).long()
        pos = torch.from_numpy(mm.event_pos_codes(idx, mm.n_pos_codebooks)).long()
        return det.decode_cell_image_event(modality, content, pos)
    content = torch.from_numpy(
        mm.event_codes(idx, mm.n_codebooks)
    ).unsqueeze(0).long()
    pos = torch.from_numpy(
        mm.event_pos_codes(idx, mm.n_pos_codebooks)
    ).unsqueeze(0).long()
    mask = torch.ones(1, content.shape[1], dtype=torch.bool)
    if content.shape[1] == 0:
        return {fn.removeprefix(f"{modality}_"): np.empty((0,), dtype=np.float32)
                for fn in det.feature_names(modality)}
    decoded = det.decode_tokens(modality, content, pos, mask)
    jagged = det.to_jagged(decoded, mask, modality=modality)
    return {fn.removeprefix(f"{modality}_"): per_evt[0]
            for fn, per_evt in jagged.items()}


def _decode_reco_event(
    vocab: Vocab,
    det: Detokenizer,
    pred_tokens: torch.Tensor,
    modality: str,
) -> Dict[str, np.ndarray]:
    """Decode one event's predicted tokens for one modality. Returns {short_feat: ndarray}.

    Tolerates malformed AR emissions (out-of-vocab-range slots) by recording
    an empty-cardinality event — this keeps the schema consistent and pflow
    metrics treat the event as a 0-particle prediction.
    """
    empty = {fn.removeprefix(f"{modality}_"): np.empty((0,), dtype=np.float32)
             for fn in det.feature_names(modality)}
    if det.data_type.get(modality) == "cell_image":
        # cell_image: the stream must contain the FULL fixed grid (all patch
        # elements) to form a decodable image. Short/malformed streams -> empty.
        empty_cells = {k: np.empty((0,), dtype=np.float32)
                       for k in (*[fn.removeprefix(f"{modality}_") for fn in det.feature_names(modality)],
                                 "eta", "phi", "layer", "ind_logit")}
        width = vocab.element_token_width(modality)
        n_grid = det.config_v[modality]["features"][f"{modality}_feat0"][0]
        if pred_tokens.numel() < n_grid * width:
            return empty_cells
        pred_tokens = pred_tokens[: n_grid * width]
        try:
            c_local, p_local = vocab.decode_element_with_pos(modality, pred_tokens.cpu().numpy())
        except ValueError:
            return empty_cells
        try:
            return det.decode_cell_image_event(
                modality,
                torch.from_numpy(c_local).long(),
                torch.from_numpy(p_local).long(),
            )
        except (RuntimeError, IndexError, AssertionError) as e:
            logger.warning("cell-image decode failed for %s: %s; recording an empty event", modality, e)
            return empty_cells
    width = vocab.element_token_width(modality)
    if pred_tokens.numel() == 0 or pred_tokens.numel() % width != 0:
        n_complete = (pred_tokens.numel() // width) * width
        pred_tokens = pred_tokens[:n_complete]
    if pred_tokens.numel() == 0:
        return empty
    try:
        c_local, p_local = vocab.decode_element_with_pos(modality, pred_tokens.cpu().numpy())
    except ValueError:
        return empty
    content = torch.from_numpy(c_local).unsqueeze(0).long()
    pos = torch.from_numpy(p_local).unsqueeze(0).long()
    mask = torch.ones(1, content.shape[1], dtype=torch.bool)
    try:
        decoded = det.decode_tokens(modality, content, pos, mask)
    except (RuntimeError, IndexError) as e:
        # CUDA index-out-of-bounds in vqvae.indices_to_zq if predicted token
        # is somehow outside expected codebook range. Tolerate as empty event.
        logger.warning("decode_tokens failed for %s: %s; recording an empty event", modality, e)
        return empty
    jagged = det.to_jagged(decoded, mask, modality=modality)
    return {fn.removeprefix(f"{modality}_"): per_evt[0]
            for fn, per_evt in jagged.items()}


def decode_events(
    model,
    vocab: Vocab,
    det: Detokenizer,
    ds: TokenizedMemmapDataset,
    idxs: Sequence[int],
    input_mods: Sequence[str],
    output_mods: Sequence[str],
    *,
    output_layout: str = "interleaved",
    argmax: bool = True,
    temperature: float = 1.0,
    structural_decode: Optional[bool] = None,
    max_new_tokens: int = 512,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    """Build prefixes, decode them in batches of equal prefix length, detokenise each event.

    Returns ``(truth_jagged, reco_jagged)``: each maps modality -> short feature name
    (no ``{modality}_`` prefix) -> list of per-event arrays, the input of
    ``hep4m.performance.pflow_report.run_report_from_arrays`` after ``jagged_to_arrays_dict``.
    """
    output_mods = sorted(output_mods)
    prefixes = build_ar_prefixes(ds, vocab, idxs, input_mods, output_mods)
    valid_idxs = [i for i, p in enumerate(prefixes) if p is not None]

    # empty predictions for events without a valid prefix
    pred_per_event: Dict[int, Dict[str, torch.Tensor]] = {
        ci: {m: torch.empty(0, dtype=torch.long) for m in output_mods}
        for ci in range(len(prefixes))
    }
    valid_prefixes = [prefixes[i] for i in valid_idxs]
    for bucket_local_idxs in bucket_by_prefix_length(valid_prefixes).values():
        batched_prefix = torch.stack([valid_prefixes[i] for i in bucket_local_idxs], dim=0)
        for sub_start in range(0, batched_prefix.shape[0], batch_size):
            sub_prefix = batched_prefix[sub_start: sub_start + batch_size]
            pred = batched_ar_decode(
                model, vocab, sub_prefix, output_mods,
                output_layout=output_layout,
                temperature=temperature, argmax=argmax,
                max_new_tokens=max_new_tokens,
                structural_decode=structural_decode,
                device=device,
            )
            for sub_b in range(sub_prefix.shape[0]):
                global_ci = valid_idxs[bucket_local_idxs[sub_start + sub_b]]
                for m in output_mods:
                    pred_per_event[global_ci][m] = pred[m][sub_b]

    truth_jagged: Dict[str, Dict[str, List[np.ndarray]]] = {
        m: {fn.removeprefix(f"{m}_"): [] for fn in det.feature_names(m)}
        for m in output_mods
    }
    reco_jagged: Dict[str, Dict[str, List[np.ndarray]]] = {
        m: {fn.removeprefix(f"{m}_"): [] for fn in det.feature_names(m)}
        for m in output_mods
    }
    for ci, idx in enumerate(idxs):
        for m in output_mods:
            # setdefault: cell_image events carry extra derived features
            # (eta/phi/layer/ind_logit) beyond feature_names(m)
            for fn, arr in _decode_truth_event(ds, det, idx, m).items():
                truth_jagged[m].setdefault(fn, []).append(arr)
            for fn, arr in _decode_reco_event(vocab, det, pred_per_event[ci][m], m).items():
                reco_jagged[m].setdefault(fn, []).append(arr)
    return truth_jagged, reco_jagged


def jagged_to_arrays_dict(
    jagged: Dict[str, List[np.ndarray]],
    add_phi: bool = True,
) -> dict:
    """Convert one modality's jagged dict to the schema run_report_from_arrays wants.

    Returns a dict with keys {pt, eta, phi, class, cosphi, sinphi}
    where phi is derived from (cosphi, sinphi) if missing.
    """
    import awkward as ak

    out = {k: ak.Array(v) for k, v in jagged.items()}
    if add_phi and "phi" not in out and "cosphi" in out and "sinphi" in out:
        out["phi"] = np.arctan2(out["sinphi"], out["cosphi"])
    return out
