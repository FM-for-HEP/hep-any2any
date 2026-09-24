"""
Integration test: build HEP4MLightning with the released tokenisers
(configs/train/hep4m_pflow/modality_dict.yml, under $HEP4M_TOKENIZERS) and the
tokenised store ($HEP4M_TOKENIZED, train/ and val/), run a forward+backward pass on CPU.
Skipped when either is absent.
"""

import os
from pathlib import Path

import pytest
import torch

from hep4m.paths import TOKENIZED, safe_load_expanded

pytestmark = pytest.mark.data

MODALITY_DICT = Path(__file__).resolve().parents[2] / "configs/train/hep4m_pflow/modality_dict.yml"


def _build_modality_dict():
    with open(MODALITY_DICT) as fp:
        return safe_load_expanded(fp)


_CKPT_AVAILABLE = all(os.path.isfile(e["checkpoint_path"]) for e in _build_modality_dict().values())
_DATA_AVAILABLE = all(os.path.isfile(os.path.join(TOKENIZED, split, "topo_data.npy"))
                      for split in ("train", "val"))

skip_reason = []
if not _CKPT_AVAILABLE:
    skip_reason.append("tokeniser checkpoints not found (HEP4M_TOKENIZERS)")
if not _DATA_AVAILABLE:
    skip_reason.append(f"needs train/ and val/ (topo_data.npy) in the token store $HEP4M_TOKENIZED={TOKENIZED}; "
                       f"export them with `python -m hep4m.data export ... --splits train val`")

skip_no_fixtures = pytest.mark.skipif(bool(skip_reason), reason="; ".join(skip_reason))


def _tiny_model_config():
    """Small HEP4M model config — runs fast on CPU."""
    emb = 64
    return {
        "embedding_dim": emb,
        "enable_flash_attn": False,
        "gpos_token_info": {
            "num_gpos_quantizers": 3,
            "gpos_vocab_size": 1024,
        },
        "token_encoder": {
            "token_comb_type": "sum",
            "max_sincos_posemb": 256,
            "pos_emb_type": "sincos",
        },
        "transformer_encoder": {
            "embed_dim": emb,
            "num_layers": 1,
            "mha_config": {"enable_flash_attn": False, "num_heads": 4},
            "dense_config": {"embed_dim": emb, "hidden_dim": emb},
            "out_dim": emb,
            "layer_scale": True,
        },
        "transformer_decoder": {
            "embed_dim": emb,
            "num_layers": 1,
            "mha_config_cross": {"enable_flash_attn": False, "num_heads": 4},
            "mha_config_self": {"enable_flex_attn": False, "num_heads": 4},
            "dense_config": {"embed_dim": emb, "hidden_dim": emb},
            "out_dim": emb,
            "layer_scale": True,
            "reverse": True,
        },
        "token_predictor": {"type": "sequential3"},
        "cardinality_predictor": None,
    }


def _train_config():
    """Training config on the tokenised store."""
    return {
        "preprocessed_dir": TOKENIZED,
        "contiguous_batches": False,
        "num_workers": 0,
        "persistent_workers": False,
        "reduce_ds_train": 50,
        "reduce_ds_val": 50,
        "batchsize_train": 8,
        "batchsize_val": 8,
        "accumulate_grad_batches": 1,
        "num_epochs": 1,
        "eval_every_n_epoch": 1,
        "train_log_every_n_steps": 1,
        "train_sampling": {
            "type": "inference",
            "fixed_input_output_modalities": {
                "input": ["track", "topo"],
                "output": ["truthpart"],
            },
        },
        "val_sampling": {
            "type": "inference",
            "fixed_input_output_modalities": {
                "input": ["track", "topo"],
                "output": ["truthpart"],
            },
        },
        "loss_config": {
            "token_wt": 1.0,
            "pos_token_wt": 1.0,
            "card_wt": 0.0,
            "label_smoothing": 0.0,
            "pos_label_smoothing": 0.0,
            "modality_wt": {},
        },
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "scheduler_type": "schedulefree",
        "warmup_steps": 2,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_no_fixtures
class TestHEP4MCheckpointLoading:

    def test_model_construction(self):
        from hep4m.lightnings.hep4m_lightning import HEP4MLightning

        modality_dict = _build_modality_dict()
        config_m = _tiny_model_config()
        model = HEP4MLightning(config_m, modality_dict, device="cpu")
        assert hasattr(model.model, "tokenizers")
        for mod in ["topo", "track", "truthpart"]:
            assert mod in model.model.tokenizers

    def test_tokenizers_are_frozen(self):
        from hep4m.lightnings.hep4m_lightning import HEP4MLightning

        modality_dict = _build_modality_dict()
        config_m = _tiny_model_config()
        model = HEP4MLightning(config_m, modality_dict, device="cpu")
        for mod in ["topo", "track", "truthpart"]:
            for p in model.model.tokenizers[mod].parameters():
                assert not p.requires_grad, f"Tokenizer {mod} param not frozen"


@skip_no_fixtures
class TestHEP4MDataModule:

    def test_datamodule_setup(self):
        from hep4m.lightnings.hep4m_lightning import HEP4MDataModule

        modality_dict = _build_modality_dict()
        config_t = _train_config()

        # get_config_dicts is normally called by HEP4MLightning, but we need
        # config_v_dict for the data module — call it directly
        from hep4m.lightnings.hep4m_lightning import HEP4MLightning
        lightning = HEP4MLightning(
            _tiny_model_config(), modality_dict, device="cpu"
        )
        config_v_dict = lightning.config_v_dict

        dm = HEP4MDataModule(config_t, config_v_dict, modality_dict)
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0

    def test_datamodule_produces_batch(self):
        """Get a batch via the actual dataloader (uses FracBatchSampler)."""
        from hep4m.lightnings.hep4m_lightning import HEP4MDataModule, HEP4MLightning

        modality_dict = _build_modality_dict()
        config_t = _train_config()

        lightning = HEP4MLightning(
            _tiny_model_config(), modality_dict, device="cpu"
        )

        dm = HEP4MDataModule(config_t, lightning.config_v_dict, modality_dict)
        # FracBatchSampler needs dm.trainer.world_size — mock a simple trainer
        class _FakeTrainer:
            world_size = 1
        dm.trainer = _FakeTrainer()
        dm.setup("fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "input" in batch
        assert "target" in batch


@skip_no_fixtures
class TestHEP4MForwardPass:

    def _get_one_batch(self):
        """Get a real data batch from the datamodule via the dataloader."""
        from hep4m.lightnings.hep4m_lightning import HEP4MDataModule, HEP4MLightning

        modality_dict = _build_modality_dict()
        config_t = _train_config()

        lightning = HEP4MLightning(
            _tiny_model_config(), modality_dict, config_t=config_t, device="cpu"
        )

        dm = HEP4MDataModule(config_t, lightning.config_v_dict, modality_dict)
        class _FakeTrainer:
            world_size = 1
        dm.trainer = _FakeTrainer()
        dm.setup("fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        return lightning, batch

    def test_forward_produces_finite_loss(self):
        model, batch = self._get_one_batch()
        model.eval()

        with torch.no_grad():
            loss, log_dict, _ = model.fast_forward(batch, get_accuracy=True)

        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert "loss" in log_dict
        assert log_dict["loss"] > 0

    def test_backward_pass(self):
        model, batch = self._get_one_batch()
        model.train()

        loss, log_dict, _ = model.fast_forward(batch)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

        loss.backward()

        grads = [
            p.grad
            for p in model.model.transformer_encoder.parameters()
            if p.grad is not None
        ]
        assert len(grads) > 0, "No gradients flowed through transformer encoder"
