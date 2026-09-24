"""End-to-end smoke test for ``NanoHepInferenceHelper``.

Builds a synthetic random-weights checkpoint, runs inference on a few
fixture events, and verifies the output ROOT has the same branch schema
as HEP4M's inference path.

Requires the per-modality VQ-VAE checkpoints to be accessible (the
``Detokenizer`` loads them from ``configs/modality_dicts/3mod.yml``).
Skipped if the VQ-VAE checkpoints are not available (set HEP4M_DATA).
"""
from hep4m.paths import safe_load_expanded as _hp_safe_load
from pathlib import Path

import pytest
import torch


_MODALITY_DICT_PATH = Path(__file__).parents[2] / "configs" / "modality_dicts" / "3mod.yml"


@pytest.fixture(scope="module")
def vqvae_available():
    """Skip if the frozen VQ-VAE checkpoints listed in modality_dict aren't
    available under $HEP4M_TOKENIZERS."""
    if not _MODALITY_DICT_PATH.exists():
        pytest.skip(f"modality_dict not found: {_MODALITY_DICT_PATH}")
    with open(_MODALITY_DICT_PATH) as f:
        mdict = _hp_safe_load(f)
    truthpart = mdict.get("truthpart", {})
    for k in ("config_path_v", "config_path_m", "checkpoint_path"):
        p = truthpart.get(k)
        if not p or not Path(p).exists():
            pytest.skip(f"VQ-VAE asset missing for truthpart.{k}: {p}")
    return str(_MODALITY_DICT_PATH)


@pytest.fixture
def synthetic_ckpt(tmp_path):
    """Save a random-weights checkpoint that load_model_from_ckpt can load."""
    from hep4m.models.nano_hep import GPT, GPTConfig, Vocab

    vocab_args = {"modalities": ["topo", "track", "truthjet", "truthpart"]}
    v = Vocab.build(**vocab_args)
    gpt_kwargs = dict(
        block_size=1024, vocab_size=v.total,
        n_layer=2, n_head=2, n_embd=32,
        dropout=0.0, bias=False,
    )
    model = GPT(GPTConfig(**gpt_kwargs))
    ckpt_path = tmp_path / "synthetic.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            "gpt_config": gpt_kwargs,
            "vocab_args": vocab_args,
            "global_step": 0,
            "config": {},
        },
        ckpt_path,
    )
    return str(ckpt_path)


def _make_init(ckpt_path, modality_dict_path, output_dir, tokenized_root):
    return {
        "model_type": "nano_hep",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": 0 if torch.cuda.is_available() else -1,
        "precision": "highest",
        "chunk_size": 4,
        "batch_size": 4,
        "output_dir": str(output_dir),
        "model": {
            "checkpoint_path": str(ckpt_path),
            "modality_dict_path": modality_dict_path,
        },
        "nano_hep": {
            "tokenized_root": tokenized_root,
            "split": "val",
            "max_events": 8,
        },
    }


def _pflow_item():
    return {
        "info": "smoke",
        "sampling_dict": {
            "type": "inference",
            "fixed_input_output_modalities": {"input": ["track", "topo"], "output": ["truthpart"]},
            "output_layout": "interleaved", "input_layout": "interleaved",
        },
        "temperature": 1.0,
        "argmax": True,
        "structural_decode": True,
        "max_new_tokens": 64,
        "dir_flag": "smoke",
        "suffix": "synthetic_pred",
    }


class TestEndToEndInference:
    """Verifies the helper builds a ROOT with the HEP4M branch
    schema even from a random-weights model. Decode quality is
    irrelevant -- we only care about the pipeline shape."""

    def test_root_produced_with_expected_branches(
        self, vqvae_available, synthetic_ckpt, tokenized_fixture_root, tmp_path,
    ):
        import uproot
        from hep4m.evaluations.nano_hep_inference_helper import NanoHepInferenceHelper

        out_dir = tmp_path / "inference_smoke"
        init = _make_init(synthetic_ckpt, vqvae_available, out_dir, tokenized_fixture_root)
        helper = NanoHepInferenceHelper(init)
        out_path = helper.run_inference(_pflow_item())

        assert Path(out_path).exists()
        f = uproot.open(out_path)
        tree = f["event_tree"]
        branches = set(tree.keys())

        # Required HEP4M-schema branches
        required = {
            "event_number", "idx",
            "truthpart_truth_pt", "truthpart_truth_eta",
            "truthpart_truth_cosphi", "truthpart_truth_sinphi",
            "truthpart_truth_class",
            "truthpart_reco_pt", "truthpart_reco_eta",
            "truthpart_reco_cosphi", "truthpart_reco_sinphi",
            "truthpart_reco_class",
        }
        missing = required - branches
        assert not missing, f"missing branches: {missing}"

    def test_truth_class_populated(
        self, vqvae_available, synthetic_ckpt, tokenized_fixture_root, tmp_path,
    ):
        """The Detokenizer keeps the categorical head of the tokeniser decoder:
        every event must have a non-empty class array on the truth side."""
        import uproot
        from hep4m.evaluations.nano_hep_inference_helper import NanoHepInferenceHelper

        out_dir = tmp_path / "inference_class"
        init = _make_init(synthetic_ckpt, vqvae_available, out_dir, tokenized_fixture_root)
        helper = NanoHepInferenceHelper(init)
        out_path = helper.run_inference(_pflow_item())

        f = uproot.open(out_path)
        tree = f["event_tree"]
        truth_cls = tree["truthpart_truth_class"].array()
        n_nonempty = sum(1 for arr in truth_cls if len(arr) > 0)
        assert n_nonempty == len(truth_cls), (
            f"truthpart_truth_class should be populated for every event "
            f"(class-drop bug regression); got {n_nonempty}/{len(truth_cls)} non-empty"
        )

    def test_random_model_reco_handled_gracefully(
        self, vqvae_available, synthetic_ckpt, tokenized_fixture_root, tmp_path,
    ):
        """Random-weights models emit invalid token sequences that fail the
        strict vocab decoder. The helper must record empty arrays rather
        than crashing -- so the run completes and the schema stays consistent."""
        import uproot
        from hep4m.evaluations.nano_hep_inference_helper import NanoHepInferenceHelper

        out_dir = tmp_path / "inference_robust"
        init = _make_init(synthetic_ckpt, vqvae_available, out_dir, tokenized_fixture_root)
        helper = NanoHepInferenceHelper(init)
        out_path = helper.run_inference(_pflow_item())

        f = uproot.open(out_path)
        tree = f["event_tree"]
        # Just confirm it loaded without error, and all required branches
        # have one entry per event.
        n = tree.num_entries
        assert n > 0
        for b in ("event_number", "idx",
                  "truthpart_truth_pt", "truthpart_reco_pt",
                  "truthpart_truth_class", "truthpart_reco_class"):
            arr = tree[b].array()
            assert len(arr) == n, f"branch {b}: len {len(arr)} != n_events {n}"
