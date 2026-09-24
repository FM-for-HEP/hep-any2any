"""Tests for the thin Lightning wrapper ``NanoHepLightning`` + the
sibling ``NanoHepDataModule`` over the committed tokenized fixture.

Uses a tiny GPT (n_embd=32, 2 layers) so the test runs in seconds on CPU.
"""
import pytest
import torch


TINY_GPT = dict(
    block_size=256, vocab_size=None,
    n_layer=2, n_head=2, n_embd=32,
    dropout=0.0, bias=False,
)


@pytest.fixture(scope="module")
def vocab_args():
    return {"modalities": ["topo", "track", "truthjet", "truthpart"]}


class TestNanoHepLightning:

    def test_instantiates(self, vocab_args):
        from hep4m.lightnings.nano_hep_lightning import NanoHepLightning
        lit = NanoHepLightning(gpt_config=dict(TINY_GPT), vocab_args=vocab_args)
        assert lit.model is not None
        assert lit.vocab.total > 0
        # vocab_size in the gpt config gets filled in from vocab.total
        assert lit.model.config.vocab_size == lit.vocab.total

    def test_loss_is_finite_scalar(self, vocab_args):
        from hep4m.lightnings.nano_hep_lightning import NanoHepLightning
        lit = NanoHepLightning(gpt_config=dict(TINY_GPT), vocab_args=vocab_args)
        v = lit.vocab
        # Synthesise a tiny batch
        seq = torch.zeros(2, TINY_GPT["block_size"], dtype=torch.long)
        mask = torch.zeros(2, TINY_GPT["block_size"], dtype=torch.long)
        # Pretend events with truthpart content + EOS
        seq[:, 0] = v.mod_out["truthpart"]
        seq[:, 1] = v.task_sep
        seq[:, 2:14] = torch.randint(0, min(v.total, 768), (2, 12))  # OUT content (random in modality range)
        seq[:, 14] = v.eos
        seq[:, 15:] = v.pad
        mask[:, 3:15] = 1
        batch = {"seq": seq, "loss_mask": mask}
        loss = lit._compute_loss(batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss).all()
        assert loss.item() > 0

    def test_backward_produces_gradients(self, vocab_args):
        from hep4m.lightnings.nano_hep_lightning import NanoHepLightning
        lit = NanoHepLightning(gpt_config=dict(TINY_GPT), vocab_args=vocab_args)
        v = lit.vocab
        seq = torch.zeros(2, TINY_GPT["block_size"], dtype=torch.long)
        mask = torch.zeros(2, TINY_GPT["block_size"], dtype=torch.long)
        seq[:, 0] = v.mod_out["truthpart"]
        seq[:, 1] = v.task_sep
        seq[:, 2:14] = torch.randint(0, min(v.total, 768), (2, 12))
        seq[:, 14] = v.eos
        seq[:, 15:] = v.pad
        mask[:, 3:15] = 1
        loss = lit._compute_loss({"seq": seq, "loss_mask": mask})
        loss.backward()
        grads = [p.grad for p in lit.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)
        total_grad = sum(g.abs().sum().item() for g in grads)
        assert total_grad > 0


class TestNanoHepDataModule:
    """End-to-end Lightning batch path: NanoHepDataModule -> DataLoader ->
    NanoHepLightning._compute_loss."""

    def test_setup_builds_train_and_val(self, tokenized_fixture_root, vocab_args):
        from hep4m.lightnings.nano_hep_lightning import NanoHepDataModule
        dm = NanoHepDataModule(
            tokenized_root=tokenized_fixture_root,
            all_modalities=["topo", "track", "truthjet", "truthpart"],
            sampling_dict={
                "type": "inference",
                "fixed_input_output_modalities": {"input": ["track", "topo"], "output": ["truthpart"]},
                "output_layout": "grouped", "input_layout": "grouped",
            },
            block_size=TINY_GPT["block_size"],
            vocab_args=vocab_args,
            batch_size=4,
            num_workers=0,
            max_events_train=10,
            max_events_val=10,
        )
        # The fixture only has val/, so train_dataset construction would fail.
        # Skip if no train/ dir.
        from pathlib import Path
        if not (Path(tokenized_fixture_root) / "train").exists():
            pytest.skip("Fixture has only val/, no train/")
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_val_dataloader_yields_loss(self, tokenized_fixture_root, vocab_args):
        """Use val_dataset only (fixture has just val/) and feed one batch
        through the Lightning training step."""
        from hep4m.datasets.tokenized_memmap import TokenizedMemmapDataset
        from hep4m.lightnings.nano_hep_lightning import NanoHepLightning
        from torch.utils.data import DataLoader

        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(**vocab_args)
        ds = TokenizedMemmapDataset(
            tokenized_root=tokenized_fixture_root,
            split="val",
            all_modalities=["topo", "track", "truthjet", "truthpart"],
            sampling_dict={
                "type": "inference",
                "fixed_input_output_modalities": {"input": ["track", "topo"], "output": ["truthpart"]},
                "output_layout": "grouped", "input_layout": "grouped",
            },
            block_size=TINY_GPT["block_size"],
            vocab=v,
            max_events=10,
        )

        def collate(batch_list):
            keys = batch_list[0].keys()
            out = {}
            for k in keys:
                vals = [b[k] for b in batch_list]
                if isinstance(vals[0], torch.Tensor):
                    out[k] = torch.stack(vals, dim=0)
                else:
                    out[k] = vals
            return out

        loader = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=collate, shuffle=False)
        batch = next(iter(loader))
        assert batch["seq"].shape == (4, TINY_GPT["block_size"])

        lit = NanoHepLightning(gpt_config=dict(TINY_GPT), vocab_args=vocab_args)
        # The dataset built its own vocab; make sure Lightning's vocab matches.
        loss = lit._compute_loss({"seq": batch["seq"], "loss_mask": batch["loss_mask"]})
        assert torch.isfinite(loss).all()
        assert loss.item() > 0
