"""Build a memory-mapped token store from tokenised ROOT files.

Input: one tokenised ROOT file per modality, as written by ``hep4m.eval_tokenizer``.
Output: one store directory (one split) in the layout read by
``hep4m.datasets.tokenized_memmap`` and the training/inference code::

    {mod}_data.npy       headerless int16 (n_rows, n_codebooks + n_pos_codebooks)
    {mod}_offsets.npy    headerless int64 (n_events + 1,)
    {mod}_meta.npz       n_events, n_codebooks, n_pos_codebooks
    {mod}_is_empty.npy   int64 indices of events with no objects
    event_numbers.npy    headerless int64 (n_events,)

Objects are sorted by the tokeniser's ``sort_by_var`` (read from its ``config_v.yml``
through the modality dict), as in the released store. Truth-jet position tokens are
repeated per jet. Existing files are never overwritten: a modality whose
``{mod}_meta.npz`` exists is skipped, and partial files without meta abort the run.

Usage:
    python -m hep4m.build_token_store --out store/test \\
        --modality-dict configs/modality_dicts/7mod.yml \\
        --input topo=tok/topo/test/file.root --input track=tok/track/test/file.root
"""
from __future__ import annotations

import argparse
from hep4m import log_to_stderr
import sys
from pathlib import Path

import numpy as np

from hep4m.paths import safe_load_expanded
from hep4m.pre_processing.convert_to_npy_tokenized import convert

REPEAT_POS_TOKENS = {"truthjet"}


def empty_event_indices(offsets_path: Path) -> np.ndarray:
    """Indices of events with zero rows, from a headerless int64 offsets file."""
    off = np.fromfile(str(offsets_path), dtype=np.int64)
    return np.where(np.diff(off) == 0)[0].astype(np.int64)


def build(out: Path, inputs: dict, modality_dict: dict, expect_events: int = -1, log=print) -> int:
    out.mkdir(parents=True, exist_ok=True)
    ev_path = out / "event_numbers.npy"
    for mod, root_path in inputs.items():
        prefix = out / mod
        meta = Path(f"{prefix}_meta.npz")
        if meta.exists():
            log(f"{mod}: exists, skipped ({meta})")
            continue
        partial = [p for p in (Path(f"{prefix}_data.npy"), Path(f"{prefix}_offsets.npy")) if p.exists()]
        if partial:
            log(f"{mod}: partial files without meta: {partial}; move them away first")
            return 3
        if not Path(root_path).exists():
            log(f"{mod}: tokenised ROOT not found: {root_path}")
            return 4
        if expect_events > 0:
            import uproot
            with uproot.open(root_path) as f:
                n = f["event_tree"].num_entries
            if n != expect_events:
                log(f"{mod}: {root_path} has {n} events, expected {expect_events}")
                return 5
        if mod not in modality_dict:
            log(f"{mod}: not in the modality dict")
            return 2
        with open(modality_dict[mod]["config_path_v"]) as fh:
            config_v = safe_load_expanded(fh)
        ev_num_fp = None if ev_path.exists() else str(ev_path)
        log(f"{mod}: {root_path} -> {prefix}_*.npy (sort_by_var={config_v.get('sort_by_var')})")
        convert(modality=mod, filelist=[str(root_path)], output_file_prefix=str(prefix),
                sort_by_var=config_v.get("sort_by_var"), ev_num_fp=ev_num_fp,
                repeat_pos_tokens=mod in REPEAT_POS_TOKENS)
        empty = empty_event_indices(Path(f"{prefix}_offsets.npy"))
        np.save(Path(f"{prefix}_is_empty.npy"), empty)
        log(f"{mod}: {len(empty)} empty events")
    return 0


def main(argv=None) -> int:
    log_to_stderr()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="store directory for one split (created)")
    ap.add_argument("--modality-dict", required=True, help="modality dict YAML naming the tokenisers")
    ap.add_argument("--input", action="append", default=[], metavar="MOD=PATH",
                    help="tokenised ROOT file of one modality; repeat per modality")
    ap.add_argument("--expect-events", type=int, default=-1,
                    help="abort unless every input has exactly this many events")
    a = ap.parse_args(argv)
    if not a.input:
        ap.error("give at least one --input MOD=PATH")
    inputs = {}
    for spec in a.input:
        mod, _, path = spec.partition("=")
        if not path:
            ap.error(f"--input needs MOD=PATH, got {spec!r}")
        inputs[mod.strip()] = path.strip()
    with open(a.modality_dict) as fh:
        md = safe_load_expanded(fh)
    return build(a.out, inputs, md, a.expect_events)


if __name__ == "__main__":
    sys.exit(main())
