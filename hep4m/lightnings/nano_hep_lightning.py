"""Lightning module and data module for nanoHEP: masked autoregressive cross-entropy on
token-store sequences, AdamW with an optional warm-up + cosine schedule, and an optional
fixed-direction evaluation (decode, detokenise, physics metrics) at validation time.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from lightning.pytorch.core.module import LightningModule
from torch.utils.data import DataLoader

from ..datasets.tokenized_memmap import TokenizedMemmapDataset
from ..models.nano_hep import GPT, GPTConfig, Vocab

logger = logging.getLogger(__name__)


class NanoHepLightning(LightningModule):
    """LightningModule wrapping the nanoHEP ``GPT``.

    Parameters
    ----------
    gpt_config : dict
        Kwargs passed to ``GPTConfig`` -- e.g. ``{n_layer, n_head, n_embd,
        block_size, vocab_size, dropout, bias}``. ``vocab_size`` must
        match ``vocab.total``.
    vocab_args : dict
        Kwargs to ``Vocab.build``. Stored on the module so checkpoints
        can rebuild the vocab.
    optimizer_args : dict
        ``{weight_decay, learning_rate, betas}`` for ``GPT.configure_optimizers``, and
        optionally ``lr_schedule: {peak, floor, warmup_steps, decay_steps}`` (linear
        warm-up then cosine decay; without it the learning rate is constant).
    pflow_eval : dict, optional
        Settings of the fixed-direction evaluation run at validation time.
    """

    def __init__(
        self,
        gpt_config: Dict[str, Any],
        vocab_args: Dict[str, Any],
        optimizer_args: Optional[Dict[str, Any]] = None,
        pflow_eval: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = Vocab.build(**vocab_args)
        if gpt_config.get("vocab_size") is None:
            gpt_config["vocab_size"] = self.vocab.total
        elif gpt_config["vocab_size"] != self.vocab.total:
            raise ValueError(
                f"gpt_config.vocab_size={gpt_config['vocab_size']} != vocab.total={self.vocab.total}"
            )

        self.model = GPT(GPTConfig(**gpt_config))

        self.optimizer_args = optimizer_args or {
            "weight_decay": 0.1,
            "learning_rate": 3e-4,
            "betas": (0.9, 0.95),
        }
        # throughput bookkeeping (rank-local; logged rank-zero only)
        self._tps_last_t: Optional[float] = None
        self._tps_last_step: int = -1  # -1 so the step-0 baseline is recorded

    def forward(self, idx: torch.Tensor):
        logits, _ = self.model(idx)
        return logits

    # ---- training / validation ----

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Masked AR cross-entropy.

        ``batch["seq"]``      : (B, T) long
        ``batch["loss_mask"]``: (B, T) long (1 = target token to learn)
        """
        seq = batch["seq"]
        loss_mask = batch["loss_mask"]
        # AR shift: predict seq[1:] from seq[:-1]; targets aligned with loss_mask[1:].
        x = seq[:, :-1].contiguous()
        targets = seq[:, 1:].contiguous()
        m = loss_mask[:, 1:].bool()

        hidden = self.model.forward_hidden(x)  # (B, T-1, n_embd), no projection

        # Guard: inference_rand2 can pick a direction with 0 loss tokens in the
        # batch -> m all False -> CE over 0 elements is NaN, which poisons the
        # DDP all-reduce. Return a graph-connected 0-loss instead.
        if not m.any():
            return hidden.sum() * 0.0

        hidden_sel = hidden[m]                 # (N_loss, n_embd) — only loss positions
        targets_sel = targets[m]               # (N_loss,)
        logits_sel = self.model.lm_head(hidden_sel)   # (N_loss, V) — project just these
        loss = F.cross_entropy(logits_sel, targets_sel)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self._log_throughput(batch)
        return loss

    def _log_throughput(self, batch, every: int = 100):
        """tokens/s and samples/s across all ranks, every `every` steps."""
        step = self.trainer.global_step
        if step % every != 0 or step == self._tps_last_step:
            return
        now = time.monotonic()
        seq = batch["seq"]
        if self._tps_last_t is not None:
            dsteps = step - self._tps_last_step
            world = self.trainer.world_size
            dt = max(now - self._tps_last_t, 1e-6)
            tokens = dsteps * seq.shape[0] * world * (seq.shape[1] - 1)
            self.log("train/tokens_per_sec", tokens / dt, on_step=True, rank_zero_only=True)
            self.log("train/samples_per_sec", dsteps * seq.shape[0] * world / dt,
                     on_step=True, rank_zero_only=True)
        self._tps_last_t = now
        self._tps_last_step = step

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ---- fixed-direction evaluation (same metric keys as HEP4M) ----

    def on_train_start(self):
        # wandb: plot every metric against the global step, so resumed runs continue the x-axis
        if (self.global_rank == 0 and self.logger is not None
                and hasattr(self.logger, "experiment")
                and hasattr(self.logger.experiment, "define_metric")):
            self.logger.experiment.define_metric("trainer/global_step")
            self.logger.experiment.define_metric("*", step_metric="trainer/global_step")

    def on_validation_epoch_end(self):
        """Run the fixed-direction evaluation on a val subset (rank 0 only).

        Decodes a fixed (input → output) split with AR + Detokenizer, then
        calls the SAME ``run_report_from_arrays`` HEP4M's
        ``_run_pflow_eval_if_requested`` uses, and logs metrics under the
        SAME wandb key namespace HEP4M does
        (``val/cardinality_{acc,mae,bias,std}``, ``val_pflow_*``).
        """
        import torch.distributed as dist
        if getattr(self.trainer, "sanity_checking", False):
            return  # sanity val must not arm the eval time-gate or log metrics
        pflow_cfg = (self.hparams.pflow_eval or {})
        # Even when eval is skipped (disabled, wrong epoch, non-zero rank, etc.),
        # we must hit the barrier at the end so all ranks stay in sync. Decide
        # whether rank 0 should run the eval and remember it for the barrier path.
        do_eval = (
            pflow_cfg.get("enabled", False)
            and self.global_rank == 0
            and self.trainer is not None
            and self.trainer.datamodule is not None
        )
        every = int(pflow_cfg.get("every_n_epochs", 1))
        if every > 0 and self.current_epoch % every != 0:
            do_eval = False
        # Wall-clock gate: when min_interval_min is set, run the evaluation at most
        # once per min_interval_min minutes (overrides every_n_epochs). The decision
        # is rank-0-only; the barrier and broadcast below keep ranks in sync.
        import time as _time
        min_int = float(pflow_cfg.get("min_interval_min", 0) or 0)
        if min_int > 0 and pflow_cfg.get("enabled", False) and self.global_rank == 0 \
                and self.trainer is not None and self.trainer.datamodule is not None:
            last_t = getattr(self, "_last_pflow_eval_t", None)
            do_eval = last_t is None or (_time.monotonic() - last_t) / 60.0 >= min_int
        if do_eval and min_int > 0:
            self._last_pflow_eval_t = _time.monotonic()
        if do_eval:
            try:
                self._run_fixed_direction_eval(pflow_cfg)
            except Exception:
                # a failed evaluation does not stop training
                logger.exception("fixed-direction evaluation failed at epoch %d", self.current_epoch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # raises if the CUDA context is unusable
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Broadcast the headline physics metric (rank 0 computed it during the
        # eval) and self.log it on ALL ranks so the best-IQR ModelCheckpoint
        # sees a consistent monitor everywhere. Only on epochs where the eval
        # ran — ModelCheckpoint skips (with a warning) on other val epochs.
        iqr = getattr(self, "_last_pflow_iqr", None)
        if pflow_cfg.get("enabled", False):
            if dist.is_available() and dist.is_initialized():
                t = torch.tensor(
                    [float(iqr) if (self.global_rank == 0 and iqr is not None) else -1.0],
                    device=self.device, dtype=torch.float32)
                dist.broadcast(t, src=0)
                v = float(t.item())
                iqr = None if v < 0 else v
            if iqr is not None and float(iqr) > 1e-6:
                self._iqr_cache = float(iqr)
            # log 1e9 when the evaluation did not run in this validation
            _fresh = iqr is not None and float(iqr) > 1e-6
            self.log("val_pflow_iqr_jet_pt_response",
                     float(iqr) if _fresh else 1e9,
                     on_step=False, on_epoch=True, sync_dist=False)
            # last evaluated value, for display only (not monitored by any checkpoint)
            self.log("val_pflow_iqr_display",
                     float(self._iqr_cache) if getattr(self, "_iqr_cache", None) is not None else 1e9,
                     on_step=False, on_epoch=True, sync_dist=False)
        self._last_pflow_iqr = None

    def _run_fixed_direction_eval(self, pflow_cfg: Dict[str, Any]):
        """Decode ``pflow_cfg.input_modalities -> output_modality`` for the first
        ``max_events`` val events and log physics metrics."""
        import tempfile
        from pathlib import Path
        from ..decoding import Detokenizer
        from ..performance.pflow_report import run_report_from_arrays
        from ..evaluations.nano_hep_decode_utils import decode_events, jagged_to_arrays_dict

        max_events = int(pflow_cfg.get("max_events", 256))
        input_mods = list(pflow_cfg["input_modalities"])
        output_modality = pflow_cfg["output_modality"]
        output_mods = [output_modality]
        logger.info("fixed-direction evaluation at epoch %d: %d events, %s -> %s",
                    self.current_epoch, max_events, input_mods, output_modality)
        modality_dict_path = pflow_cfg["modality_dict_path"]
        output_layout = pflow_cfg.get("output_layout", "interleaved")
        max_new_tokens = int(pflow_cfg.get("max_new_tokens", 512))
        batch_size = int(pflow_cfg.get("batch_size", 32))

        # the first N val events, the same at every evaluation
        val_ds = self.trainer.datamodule.val_dataset
        n = min(max_events, len(val_ds))
        idxs = list(range(n))

        det = Detokenizer(modality_dict_path, modalities=output_mods, device=str(self.device))

        was_training = self.model.training
        self.model.eval()
        try:
            truth_jagged, reco_jagged = decode_events(
                self.model, self.vocab, det, val_ds, idxs,
                input_mods=input_mods, output_mods=output_mods,
                output_layout=output_layout,
                argmax=True, structural_decode=True,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                device=str(self.device),
            )
        finally:
            if was_training:
                self.model.train()

        truth = jagged_to_arrays_dict(truth_jagged[output_modality])
        reco  = jagged_to_arrays_dict(reco_jagged[output_modality])

        outdir = Path(tempfile.mkdtemp(prefix=f"nano_hep_eval_ep{self.current_epoch}_"))
        is_generative = bool(pflow_cfg.get("generative", False))

        if is_generative:
            # detector simulation: jets are not defined for detector objects, so
            # only the distribution-level report runs
            from ..performance.generative_report import run_generative_report_from_arrays
            metrics = run_generative_report_from_arrays(
                truth=truth, reco=reco,
                outdir=str(outdir), output_modality=output_modality,
                c2st_epochs=int(pflow_cfg.get("c2st_epochs", 30)),
            )
            metrics = {(k if k.startswith("_") else f"gen_{k}"): v for k, v in metrics.items()}
        else:
            metrics = run_report_from_arrays(
                truth=truth, reco=reco, reco_ind=None,
                outdir=str(outdir), output_modality=output_modality,
            )
            # add the distribution-level metrics (KS, Wasserstein, occupancy, C2ST) as gen_*
            from ..performance.generative_report import run_generative_report_from_arrays
            gen_metrics = run_generative_report_from_arrays(
                truth=truth, reco=reco,
                outdir=str(outdir), output_modality=output_modality,
                c2st_epochs=int(pflow_cfg.get("c2st_epochs", 30)),
            )
            for k, v in gen_metrics.items():
                if k == "_histograms":
                    metrics.setdefault("_histograms", {}).update(v)
                elif not k.startswith("_"):
                    metrics[f"gen_{k}"] = v

        # cardinality metrics from the per-event lengths of the decoded arrays
        # (truthpart/track have 'pt', topo has 'e')
        import numpy as np
        _kin_keys = ("pt", "e", "energy")
        _t_jag = truth_jagged[output_modality]
        _r_jag = reco_jagged[output_modality]
        _kin = next((k for k in _kin_keys if k in _t_jag and k in _r_jag), None)
        if _kin is None:
            # No usable per-particle magnitude — fall back to whatever's first.
            _kin = next(iter(_t_jag))
        n_true_arr = np.array([len(a) for a in _t_jag[_kin]], dtype=np.float32)
        n_pred_arr = np.array([len(a) for a in _r_jag[_kin]], dtype=np.float32)
        if len(n_true_arr) > 0:
            resid = n_pred_arr - n_true_arr
            metrics["cardinality_acc"]  = float((n_pred_arr == n_true_arr).mean())
            metrics["cardinality_mae"]  = float(np.abs(resid).mean())
            metrics["cardinality_bias"] = float(resid.mean())
            metrics["cardinality_std"]  = float(resid.std()) if len(resid) > 1 else 0.0
            metrics["n_pred_mean"]      = float(n_pred_arr.mean())
            metrics["n_pred_std"]       = float(n_pred_arr.std()) if len(n_pred_arr) > 1 else 0.0
            metrics["n_true_mean"]      = float(n_true_arr.mean())
            metrics["n_true_std"]       = float(n_true_arr.std()) if len(n_true_arr) > 1 else 0.0

        # Hierarchical wandb keys (hep4m/performance/wandb_keys.py), logged with
        # logger.experiment.log(): self.log() from rank 0 alone would need all ranks.
        from ..performance.wandb_keys import (
            to_wandb_subkey, CARDINALITY_REMAP, PFLOW_IMAGES, gen_image_key,
        )
        scalars_to_log = {}

        # Cardinality scalars under val/cardinality/...
        for short_key, wandb_key in CARDINALITY_REMAP.items():
            if short_key in metrics:
                scalars_to_log[wandb_key] = float(metrics[short_key])

        # All pflow/gen scalars get their hierarchical wandb path
        for k, v in metrics.items():
            if k.startswith("_") or not isinstance(v, (int, float)):
                continue
            if k in CARDINALITY_REMAP:                # already logged above
                continue
            ns = "gen" if k.startswith("gen_") else "pflow"
            base = k[len("gen_"):] if k.startswith("gen_") else k
            scalars_to_log[to_wandb_subkey(base, namespace=ns)] = float(v)

        # jet pT IQR for the best-IQR checkpoint callback (on_validation_epoch_end
        # broadcasts it and logs it with self.log, which callbacks can see)
        self._last_pflow_iqr = metrics.get("iqr_jet_pt_response")
        if self._last_pflow_iqr is not None:
            logger.info("epoch %d: jet pT IQR %.4f", self.current_epoch, self._last_pflow_iqr)

        # scalars, plots (wandb.Image) and histograms, with a wandb logger only
        from lightning.pytorch.loggers import WandbLogger
        if isinstance(self.logger, WandbLogger):
            import wandb
            log = self.logger.experiment.log
            if scalars_to_log:
                log(scalars_to_log)
            for png_name, wandb_key in {**PFLOW_IMAGES, **gen_image_key(output_modality)}.items():
                if (outdir / png_name).exists():
                    log({wandb_key: wandb.Image(str(outdir / png_name))})
            for hkey, (counts, bin_edges) in (metrics.get("_histograms") or {}).items():
                log({f"val/pflow/hist/{hkey}": wandb.Histogram(np_histogram=(counts, bin_edges))})
        logger.info("epoch %d: %d scalars", self.current_epoch, len(scalars_to_log))

    def configure_optimizers(self):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        sched_cfg = self.optimizer_args.get("lr_schedule") or None
        # With a schedule, the optimizer's base LR is the PEAK; LambdaLR scales it.
        base_lr = (sched_cfg["peak"] if sched_cfg
                   else self.optimizer_args["learning_rate"])
        optimizer = self.model.configure_optimizers(
            self.optimizer_args["weight_decay"],
            base_lr,
            tuple(self.optimizer_args["betas"]),
            device_type,
        )
        if not sched_cfg:
            return optimizer  # constant learning rate

        # linear warm-up, then cosine decay to `floor` at `decay_steps`, stepped per
        # optimiser step (the LambdaLR state is in the checkpoint, so a resume continues it)
        warm = int(sched_cfg["warmup_steps"])
        peak = float(sched_cfg["peak"])
        floor = float(sched_cfg["floor"])
        total = int(sched_cfg["decay_steps"])

        def lr_lambda(step: int) -> float:
            if step < warm:
                return (step + 1) / max(warm, 1)
            if step >= total:
                return floor / peak
            progress = (step - warm) / max(total - warm, 1)
            return (floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))) / peak

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ---- checkpoint keys read by models/nano_hep/loader.py ----

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["gpt_config"] = dict(self.hparams.gpt_config)
        checkpoint["vocab_args"] = dict(self.hparams.vocab_args)
        checkpoint["config"] = {"optimizer": dict(self.optimizer_args)}


class NanoHepDataModule(LightningDataModule):
    """Train and val ``TokenizedMemmapDataset``; with ``sampling_dict.type='inference_rand2'``
    each batch gets a random (input, output) modality split (``NanoHepFracBatchSampler``)."""

    def __init__(
        self,
        tokenized_root: str,
        all_modalities: list,
        sampling_dict: Dict[str, Any],
        block_size: int,
        vocab_args: Dict[str, Any],
        batch_size: int = 32,
        num_workers: int = 2,
        max_events_train: int = -1,
        max_events_val: int = -1,
        trim_batch_padding: bool = True,
    ):
        super().__init__()
        self.tokenized_root = tokenized_root
        self.all_modalities = list(all_modalities)
        self.sampling_dict = dict(sampling_dict)
        self.block_size = int(block_size)
        self.vocab_args = dict(vocab_args)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_events_train = int(max_events_train)
        self.max_events_val = int(max_events_val)
        # drop trailing columns that are PAD in every sample of a batch (exact, see _collate)
        self.trim_batch_padding = bool(trim_batch_padding)

        self.train_dataset: Optional[TokenizedMemmapDataset] = None
        self.val_dataset: Optional[TokenizedMemmapDataset] = None
        self.vocab = Vocab.build(**vocab_args)
        self.pad_id = int(self.vocab.pad)

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is None:
            self.train_dataset = TokenizedMemmapDataset(
                tokenized_root=self.tokenized_root,
                split="train",
                all_modalities=self.all_modalities,
                sampling_dict=self.sampling_dict,
                block_size=self.block_size,
                vocab=self.vocab,
                max_events=self.max_events_train,
            )
        if self.val_dataset is None:
            # validation always uses the fixed direction, so val_loss is comparable
            # between validations
            val_sampling = {**self.sampling_dict, "type": "inference"}
            self.val_dataset = TokenizedMemmapDataset(
                tokenized_root=self.tokenized_root,
                split="val",
                all_modalities=self.all_modalities,
                sampling_dict=val_sampling,
                block_size=self.block_size,
                vocab=self.vocab,
                max_events=self.max_events_val,
            )

    def _collate(self, batch_list):
        keys = batch_list[0].keys()
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [b[k] for b in batch_list]
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals

        # Dynamic padding trim (exact): drop trailing columns that are PAD for
        # EVERY sample in the batch. Those columns are pure padding -> their
        # hidden states are never loss positions, and causal attention means no
        # kept token attends to them, so the masked-AR loss is unchanged. seq and loss_mask are trimmed together to
        # keep the AR shift (x=seq[:,:-1], targets=seq[:,1:]) aligned.
        if self.trim_batch_padding and isinstance(out.get("seq"), torch.Tensor):
            nonpad_cols = (out["seq"] != self.pad_id).any(dim=0)  # (T,)
            if bool(nonpad_cols.any()):
                keep = int(nonpad_cols.nonzero().max().item()) + 1
            else:
                keep = 1
            keep = max(keep, 2)  # seq[:,:-1]/seq[:,1:] must stay non-empty
            if keep < out["seq"].shape[1]:
                for k in ("seq", "loss_mask"):
                    if isinstance(out.get(k), torch.Tensor):
                        out[k] = out[k][:, :keep].contiguous()
        return out

    def _make_dataloader(self, dataset, shuffle: bool):
        """Build a DataLoader. If ``sampling_dict.type == 'inference_rand2'``,
        wrap with ``NanoHepFracBatchSampler`` so each batch gets a freshly
        sampled (input_mods, output_mods) split."""
        sampling_type = self.sampling_dict.get("type", "inference")
        if sampling_type == "inference_rand2" and dataset.valid_splits:
            from torch.utils.data import RandomSampler, SequentialSampler
            from ..datasets.tokenized_memmap import NanoHepFracBatchSampler
            base_sampler = (RandomSampler(dataset) if shuffle
                            else SequentialSampler(dataset))
            batch_sampler = NanoHepFracBatchSampler(
                batch_size=self.batch_size,
                valid_splits=dataset.valid_splits,
                sampling_type="inference_rand2",
                sampler=base_sampler,
                drop_last=True,
                fixed_frac=float(self.sampling_dict.get("fixed_frac", 0.0) or 0.0),
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self._collate,
                pin_memory=True,
            )
        # fixed-split fallback
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, shuffle=False)
