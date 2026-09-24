"""nanoHEP inference: GPT -> batched_ar_decode -> Detokenizer -> ROOT file (TreeWriter).

Writes the same ``{modality}_truth_*`` / ``{modality}_reco_*`` branches as the HEP4M
inference helper, so the same evaluation code reads both.
"""
from __future__ import annotations

from hep4m.paths import safe_load_expanded as _hp_safe_load
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from ..datasets.tokenized_memmap import TokenizedMemmapDataset
from ..decoding import Detokenizer
from ..models.nano_hep import load_model_from_ckpt
from ..models.nano_hep.decode import batched_ar_decode, bucket_by_prefix_length
from ..utility.tree_writer import TreeWriter
from .nano_hep_decode_utils import build_ar_prefixes

logger = logging.getLogger(__name__)


class NanoHepInferenceHelper:
    """Runs nanoHEP inference for the items of an inference config and writes ROOT files."""

    def __init__(self, init_config: Dict):
        self.init_config = init_config

        self.modality_dict_path = init_config["model"]["modality_dict_path"]
        with open(self.modality_dict_path) as f:
            self.modality_dict = _hp_safe_load(f)

        self.device = init_config["device"]
        self.gpu = init_config["gpu"]
        if self.gpu == -1:
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.chunk_size = init_config["chunk_size"]
        self.batch_size = init_config["batch_size"]
        self.output_dir = Path(init_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = init_config["model"]["checkpoint_path"]
        self.load_model(self.checkpoint_path)

        # token store to read the inputs (and the truth) from
        self.tokenized_root = init_config["nano_hep"]["tokenized_root"]
        self.split = init_config["nano_hep"].get("split", "val")
        self.max_events = init_config["nano_hep"].get("max_events", -1)

    # ------------------------------------------------------------------
    def load_model(self, checkpoint_path: str):
        """Load model and vocabulary from a nanoHEP checkpoint."""
        self.model, self.vocab, self.cfg, self.step = load_model_from_ckpt(
            checkpoint_path, device=self.device,
        )
        logger.info("checkpoint %s, global step %d", checkpoint_path, self.step)

    # ------------------------------------------------------------------
    def reset_dict_to_write(self, output_modalities: List[str]):
        """Pre-create the per-modality {modality}_truth / {modality}_reco
        slots so TreeWriter sees a stable schema from event 0."""
        self.dict_to_write: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for m in output_modalities:
            self.dict_to_write[f"{m}_truth"] = defaultdict(list)
            self.dict_to_write[f"{m}_reco"] = defaultdict(list)
        # global scalar branches
        self._extra_global: Dict[str, List] = {"event_number": [], "idx": []}

    def _append_decoded(
        self,
        truth_or_reco: str,  # "truth" | "reco"
        modality: str,
        decoded_jagged: Dict[str, List[np.ndarray]],
    ):
        slot = self.dict_to_write[f"{modality}_{truth_or_reco}"]
        for feat_name, per_event in decoded_jagged.items():
            # Drop the "{modality}_" prefix to match HEP4M's branch naming
            # ({modality}_truth_{pt,eta,...} where {pt,...} is the bare feature).
            fn = feat_name.removeprefix(f"{modality}_")
            slot[fn].extend(per_event)

    # ------------------------------------------------------------------
    def prep_output_filepath(self, item: Dict) -> Path:
        """Derive the per-item output ROOT path."""
        dir_flag = item.get("dir_flag", "default")
        suffix = item.get("suffix", "prediction")
        out_dir = self.output_dir / dir_flag
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{suffix}.root"

    # ------------------------------------------------------------------
    def run_inference(self, item: Dict):
        """Run inference for one YAML ``items`` entry.

        Required ``item`` keys:
          - ``sampling_dict``: {input/output modality split + layouts}
          - ``temperature``: float (default 1.0)
          - ``argmax``: bool (default True)
          - ``structural_decode``: bool (default = argmax)
          - ``max_new_tokens``: int (default 1024)
          - ``dir_flag``: subdir under output_dir
          - ``suffix``: filename stem
        """
        sampling_dict = item["sampling_dict"]
        fmods = sampling_dict["fixed_input_output_modalities"]
        input_mods = list(fmods["input"])
        output_mods = sorted(list(fmods["output"]))

        temperature = float(item.get("temperature", 1.0))
        argmax = bool(item.get("argmax", True))
        structural_decode = item.get("structural_decode", argmax)
        max_new_tokens = int(item.get("max_new_tokens", 1024))
        output_layout = sampling_dict.get("output_layout", "interleaved")
        block_size = self.model.config.block_size

        # The dataset gives the AR prefix of each event (header + inputs + first
        # <MOD_OUT> cue) and the truth tokens of the outputs.
        ds = TokenizedMemmapDataset(
            tokenized_root=self.tokenized_root,
            split=self.split,
            all_modalities=list(self.modality_dict.keys()),
            sampling_dict=sampling_dict,
            block_size=block_size,
            vocab=self.vocab,
            max_events=self.max_events,
        )
        logger.info("dataset events: %d", len(ds))

        # Set up detokenizer for all output modalities (and any other modalities
        # we want to decode for truth-side branches; here just output_mods).
        det = Detokenizer(
            self.modality_dict_path,
            modalities=output_mods,
            device=self.device,
        )

        # Set up TreeWriter
        out_path = self.prep_output_filepath(item)
        logger.info("writing %s", out_path)
        tw = TreeWriter(str(out_path), tree_name="event_tree", chunk_size=self.chunk_size,
                        dtype_to_32=True)

        self.reset_dict_to_write(output_mods)

        # Iterate in chunks. Inside each chunk, build prefixes per event, then
        # bucket by prefix length and run batched_ar_decode per bucket.
        n_events = len(ds)
        for chunk_start in tqdm(range(0, n_events, self.chunk_size), desc="chunks"):
            chunk_end = min(chunk_start + self.chunk_size, n_events)
            chunk_idxs = list(range(chunk_start, chunk_end))

            # AR prefix per event; None (no output cue) -> empty prediction.
            all_prefixes = build_ar_prefixes(ds, self.vocab, chunk_idxs, input_mods, output_mods)
            valid = [ci for ci, p in enumerate(all_prefixes) if p is not None]
            prefixes = [all_prefixes[ci] for ci in valid]
            buckets = bucket_by_prefix_length(prefixes)

            # Pre-allocate per-event per-modality result holders.
            chunk_pred_tokens: Dict[int, Dict[str, torch.Tensor]] = {
                ci: {m: torch.empty(0, dtype=torch.long) for m in output_mods}
                for ci in range(len(chunk_idxs))
            }
            for bucket_idxs_within_chunk in buckets.values():
                batched_prefix = torch.stack(
                    [prefixes[i] for i in bucket_idxs_within_chunk], dim=0
                )
                # Iterate the bucket in chunks of self.batch_size.
                for sub_start in range(0, batched_prefix.shape[0], self.batch_size):
                    sub_end = sub_start + self.batch_size
                    sub_prefix = batched_prefix[sub_start:sub_end]
                    pred = batched_ar_decode(
                        self.model, self.vocab, sub_prefix, output_mods,
                        output_layout=output_layout,
                        temperature=temperature, argmax=argmax,
                        max_new_tokens=max_new_tokens,
                        structural_decode=structural_decode,
                        device=self.device,
                    )
                    for sub_b in range(sub_prefix.shape[0]):
                        ci = valid[bucket_idxs_within_chunk[sub_start + sub_b]]
                        for m in output_mods:
                            chunk_pred_tokens[ci][m] = pred[m][sub_b]

            # Decode + write each event in the chunk.
            for ci_within_chunk, idx in enumerate(chunk_idxs):
                pred_per_mod = chunk_pred_tokens[ci_within_chunk]

                # --- TRUTH side: pull from memmap (re-uses detokenizer) ---
                for m in output_mods:
                    mm = ds.mms[m]
                    content = torch.from_numpy(mm.event_codes(idx, mm.n_codebooks)).unsqueeze(0).long()
                    pos = torch.from_numpy(mm.event_pos_codes(idx, mm.n_pos_codebooks)).unsqueeze(0).long()
                    mask = torch.ones(1, content.shape[1], dtype=torch.bool)
                    if content.shape[1] == 0:
                        # Empty: just append empty arrays per feature.
                        feats = det.feature_names(m)
                        for fn in feats:
                            short = fn.removeprefix(f"{m}_")
                            self.dict_to_write[f"{m}_truth"][short].append(np.empty((0,), dtype=np.float32))
                        continue
                    decoded = det.decode_tokens(m, content, pos, mask)
                    jagged = det.to_jagged(decoded, mask)
                    self._append_decoded("truth", m, jagged)

                # --- RECO side: decode the model's predictions ---
                for m in output_mods:
                    tokens = pred_per_mod[m]
                    width = self.vocab.element_token_width(m)
                    if tokens.numel() == 0 or tokens.numel() % width != 0:
                        # Truncate to nearest complete element.
                        n_complete = (tokens.numel() // width) * width
                        tokens = tokens[:n_complete]
                    if tokens.numel() == 0:
                        feats = det.feature_names(m)
                        for fn in feats:
                            short = fn.removeprefix(f"{m}_")
                            self.dict_to_write[f"{m}_reco"][short].append(np.empty((0,), dtype=np.float32))
                        continue
                    # Convert global tokens to local (content, pos) codes.
                    try:
                        c_local, p_local = self.vocab.decode_element_with_pos(
                            m, tokens.cpu().numpy()
                        )
                    except ValueError:
                        # Malformed AR emission (out-of-range token in expected
                        # content/pos slot). Record as empty-cardinality event
                        # so the schema stays consistent; downstream metrics
                        # treat as 0-particle prediction.
                        feats = det.feature_names(m)
                        for fn in feats:
                            short = fn.removeprefix(f"{m}_")
                            self.dict_to_write[f"{m}_reco"][short].append(
                                np.empty((0,), dtype=np.float32)
                            )
                        continue
                    content_t = torch.from_numpy(c_local).unsqueeze(0).long()
                    pos_t = torch.from_numpy(p_local).unsqueeze(0).long()
                    mask = torch.ones(1, content_t.shape[1], dtype=torch.bool)
                    decoded = det.decode_tokens(m, content_t, pos_t, mask)
                    jagged = det.to_jagged(decoded, mask)
                    self._append_decoded("reco", m, jagged)

                # global branches
                self._extra_global["event_number"].append(np.array([idx], dtype=np.int64))
                self._extra_global["idx"].append(np.array([idx], dtype=np.int64))

            # Flush this chunk.
            for k, v in self.dict_to_write.items():
                tw.data[k] = {fn: per_evt for fn, per_evt in v.items()}
            for k, v in self._extra_global.items():
                tw.data[k] = v
            tw.write()
            tw.reset_chunk()
            # Reset accumulators
            self.reset_dict_to_write(output_mods)

        logger.info("done: %s", out_path)
        return str(out_path)
