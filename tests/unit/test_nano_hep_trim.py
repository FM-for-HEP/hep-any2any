"""Regression: dynamic per-batch padding trim (NanoHepDataModule._collate,
trim_batch_padding) must be a NO-OP on the masked-AR loss and its gradients.

Trimming drops only trailing columns that are PAD for every sample in the batch.
Because attention is causal (no kept token attends to a trimmed column) and pad
is never a loss position, the loss and all parameter gradients must match the
full-block-padded computation to floating-point tolerance.
"""
import pytest
import torch

TINY_GPT = dict(block_size=512, vocab_size=None, n_layer=2, n_head=2,
                n_embd=32, dropout=0.0, bias=False)


@pytest.fixture(scope="module")
def vocab_args():
    return {"modalities": ["topo", "track", "truthjet", "truthpart"]}


def _make_datamodule(vocab_args, trim):
    from hep4m.lightnings.nano_hep_lightning import NanoHepDataModule
    dm = NanoHepDataModule(
        tokenized_root="/nonexistent",  # never .setup(); only _collate is exercised
        all_modalities=["topo", "track", "truthjet", "truthpart"],
        sampling_dict={"type": "inference",
                       "fixed_input_output_modalities": {"input": ["track", "topo"],
                                                         "output": ["truthpart"]},
                       "output_layout": "grouped", "input_layout": "grouped"},
        block_size=TINY_GPT["block_size"], vocab_args=vocab_args,
        batch_size=3, trim_batch_padding=trim,
    )
    return dm


def _synthetic_batch_list(v, lengths):
    """One dict per sample, each right-padded to block_size with v.pad, with
    `real_len` real tokens (header + truthpart content + EOS) and loss on output."""
    T = TINY_GPT["block_size"]
    out = []
    g = torch.Generator().manual_seed(123)
    for real_len in lengths:
        seq = torch.full((T,), v.pad, dtype=torch.long)
        mask = torch.zeros(T, dtype=torch.long)
        seq[0] = v.mod_out["truthpart"]
        seq[1] = v.task_sep
        content = real_len - 3              # leave room for header(2) + EOS(1)
        seq[2:2 + content] = torch.randint(0, min(v.total, 700), (content,), generator=g)
        seq[2 + content] = v.eos
        mask[3:3 + content] = 1             # learn the OUT content tokens
        out.append({"seq": seq, "loss_mask": mask})
    return out


class TestTrimEquivalence:

    def test_trim_matches_full_padding(self, vocab_args):
        from hep4m.lightnings.nano_hep_lightning import NanoHepLightning
        torch.manual_seed(0)
        lit = NanoHepLightning(gpt_config=dict(TINY_GPT), vocab_args=vocab_args)
        v = lit.vocab

        batch_list = _synthetic_batch_list(v, lengths=[20, 37, 51])
        full = _make_datamodule(vocab_args, trim=False)._collate(batch_list)
        trim = _make_datamodule(vocab_args, trim=True)._collate(batch_list)

        # Trim must actually shrink (else the test is vacuous): longest real
        # sample is 51 tokens, so keep should be ~51 << 512.
        assert full["seq"].shape[1] == TINY_GPT["block_size"]
        assert trim["seq"].shape[1] < full["seq"].shape[1]
        assert trim["seq"].shape[1] >= 51

        def loss_and_grad(batch):
            lit.zero_grad(set_to_none=True)
            loss = lit._compute_loss({"seq": batch["seq"], "loss_mask": batch["loss_mask"]})
            loss.backward()
            grad = torch.cat([p.grad.flatten() for p in lit.parameters()
                              if p.grad is not None])
            return loss.detach(), grad

        l_full, g_full = loss_and_grad(full)
        l_trim, g_trim = loss_and_grad(trim)

        assert torch.allclose(l_full, l_trim, atol=1e-6, rtol=1e-5), \
            f"loss differs: full={l_full.item()} trim={l_trim.item()}"
        max_gdiff = (g_full - g_trim).abs().max().item()
        assert max_gdiff < 1e-5, f"max grad diff {max_gdiff} too large"

    def test_no_trim_when_full_batch_uses_all_columns(self, vocab_args):
        """If some sample fills the whole block, nothing is trimmed (no all-pad
        trailing columns)."""
        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(**vocab_args)
        T = TINY_GPT["block_size"]
        seq = torch.randint(0, min(v.total, 700), (2, T), dtype=torch.long)  # no pad anywhere
        mask = torch.ones(2, T, dtype=torch.long)
        batch_list = [{"seq": seq[i], "loss_mask": mask[i]} for i in range(2)]
        trimmed = _make_datamodule(vocab_args, trim=True)._collate(batch_list)
        assert trimmed["seq"].shape[1] == T  # nothing to trim

    def test_flag_off_keeps_full_block(self, vocab_args):
        from hep4m.models.nano_hep import Vocab
        v = Vocab.build(**vocab_args)
        batch_list = _synthetic_batch_list(v, lengths=[10, 12, 15])
        out = _make_datamodule(vocab_args, trim=False)._collate(batch_list)
        assert out["seq"].shape[1] == TINY_GPT["block_size"]
