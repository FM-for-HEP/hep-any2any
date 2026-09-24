from hep4m.paths import safe_load_expanded as _hp_safe_load
import logging
import random
from collections import OrderedDict
from itertools import combinations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from lightning import LightningDataModule
from lightning.pytorch.core.module import LightningModule
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, DistributedSampler
from tqdm import tqdm

from ..models.helpers.attention import MultiheadAttentionVarLen

from ..datasets.dataset_hep4m import COCOADatasetHEP4M, get_collate_fn, FracBatchSampler
from ..models.hep4m import HEP4M
from ..utility.comet_helper import save_plot
from ..utility.hep4m_loss import Seq2SeqLoss

log = logging.getLogger(__name__)


def _worker_init_fn(_):
    # lightweight workers & deterministic seeds
    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))



class HEP4MDataModule(LightningDataModule):
    def __init__(self, config_t, config_v_dict, modality_dict):
        super().__init__()
        self.config_t = config_t
        self.config_v_dict = config_v_dict
        self.modality_dict = modality_dict
        self.modalities = list(modality_dict.keys())

        self.get_filepath_dicts()

        self.train_dataset = None
        self.val_dataset = None

    def get_filepath_dicts(self):
        # token store written by hep4m.data export / build_token_store
        store = self.config_t['preprocessed_dir']
        self.train_filepath_dict = {m: f"{store}/train/{m}_data.npy" for m in self.modalities}
        self.val_filepath_dict = {m: f"{store}/val/{m}_data.npy" for m in self.modalities}

    def setup(self, stage: str):
        if stage in ('fit', 'validate') or stage is None:
            self.train_dataset = COCOADatasetHEP4M(
                filepath_dict=self.train_filepath_dict, config_v_dict=self.config_v_dict,
                start_idx=self.config_t.get('start_idx_train', 0),
                reduce_ds=self.config_t['reduce_ds_train'],
                sampling_type=self.config_t['train_sampling']['type'], use_tokens=True,
                fixed_input_output_modalities= \
                    self.config_t['train_sampling'].get('fixed_input_output_modalities', None))

            self.val_dataset = COCOADatasetHEP4M(
                filepath_dict=self.val_filepath_dict, config_v_dict=self.config_v_dict,
                start_idx=self.config_t.get('start_idx_val', 0),
                reduce_ds=self.config_t['reduce_ds_val'],
                sampling_type=self.config_t['val_sampling']['type'], use_tokens=True,
                fixed_input_output_modalities= \
                    self.config_t['val_sampling'].get('fixed_input_output_modalities', None))


    def get_valid_splits(self, sampling_dict):
        if sampling_dict['type'] == 'inference':
            return None

        n = len(self.modalities)
        max_inp_modality = sampling_dict.get('max_inp_modality', None)
        max_out_modality = sampling_dict.get('max_out_modality', None)
        max_n_inp = min(max_inp_modality + 1, n) if max_inp_modality is not None else n

        valid_splits = []
        for n_inp in range(1, max_n_inp):
            for inp_combo in combinations(range(n), n_inp):
                is_inp = torch.zeros(n, dtype=torch.bool)
                is_inp[list(inp_combo)] = True

                remaining = [i for i in range(n) if i not in inp_combo]

                if max_out_modality is None:
                    # all remaining become targets
                    is_tgt = torch.zeros(n, dtype=torch.bool)
                    is_tgt[remaining] = True
                    valid_splits.append((is_inp, is_tgt))
                else:
                    # enumerate output subsets up to max_out_modality
                    max_n_out = min(max_out_modality, len(remaining))
                    for n_out in range(1, max_n_out + 1):
                        for out_combo in combinations(remaining, n_out):
                            is_tgt = torch.zeros(n, dtype=torch.bool)
                            is_tgt[list(out_combo)] = True
                            valid_splits.append((is_inp, is_tgt))

        return valid_splits


    def get_dataloader(self, dataset, sampling_dict, batch_size, shuffle):
        common_loader_kwargs = dict(
            num_workers=self.config_t['num_workers'],
            persistent_workers=self.config_t.get('persistent_workers', False),
            pin_memory=True,
            collate_fn=get_collate_fn(sampling_dict['type']),
            worker_init_fn=_worker_init_fn,
            prefetch_factor=2 if self.config_t['num_workers'] > 0 else None,
        )

        if sampling_dict['type'] in ['inference_rand2', 'inference']:
            if self.trainer.world_size > 1:
                base_sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
            else:
                base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

            zero_card_idx_dict = {}
            for mod in dataset.modalities:
                zero_card_idx_dict[mod] = dataset.ds_obj_dict[mod].empty_idxs \
                    if hasattr(dataset.ds_obj_dict[mod], 'empty_idxs') else np.array([])

            # compute valid splits, and store in dataset
            valid_splits = self.get_valid_splits(sampling_dict)
            dataset.set_valid_splits(valid_splits)

            batch_sampler = FracBatchSampler(
                batch_size=batch_size, modalities=self.modalities,
                sampling_dict=sampling_dict,
                drop_last=True, sampler=base_sampler,
                contiguous_batches=self.config_t.get('contiguous_batches', False),
                zero_card_idx_dict=zero_card_idx_dict,
                dataset_len=len(dataset),
                valid_splits=valid_splits
            )
            loader = DataLoader(dataset, batch_sampler=batch_sampler, **common_loader_kwargs)

        else:
            loader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=shuffle, drop_last=True,
                **common_loader_kwargs,
            )
        return loader

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_dataset, self.config_t['train_sampling'],
            self.config_t['batchsize_train'], shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(
            self.val_dataset, self.config_t['val_sampling'],
            self.config_t['batchsize_val'], shuffle=False)





class HEP4MLightning(LightningModule):

    def __init__(self, config_m, modality_dict, config_t=None, 
            comet_logger=None, device='cpu'):

        super().__init__()
        self.save_hyperparameters()
        
        self.config_m = config_m
        self.modality_dict = modality_dict
        self.config_t = config_t

        # for each modality
        self.get_config_dicts()

        self.model = HEP4M(config_m, self.config_m_dict)
        self.load_tokenizer_ckpts()
        self.model.set_use_target_tokens(True)

        # config_t is provided for training
        # do these only for training
        if config_t is not None:
            self.losses = nn.ModuleDict()
            for modality in self.modalities:
                self.losses[modality] = Seq2SeqLoss(
                    self.config_t['loss_config'],
                    self.modality_dict[modality]['codebook_loss_wts'],
                    data_type=self.config_m_dict[modality]['data_type'],
                    wt=self.config_t['loss_config']['modality_wt'].get(modality, 1.0)
                )

            self.comet_logger = comet_logger
            self.save_hyperparameters()

            self.val_step_outputs = [] # store every batch

    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger


    def get_config_dicts(self, return_dicts=False):
        '''
            Optionally update the config_v/ms with info from self.modality_dict
        '''

        self.modalities = list(self.modality_dict.keys())

        self.config_v_dict = {}; self.config_m_dict = {}
        for modality in self.modalities:
            with open(self.modality_dict[modality]['config_path_v'], 'r') as fp:
                config_v = _hp_safe_load(fp)

            with open(self.modality_dict[modality]['config_path_m'], 'r') as fp:
                config_m = _hp_safe_load(fp)

            if config_v.get('data_type', 'set') == 'global':
                config_m['max_token_cardinality'] = config_m['quantizer']['heads']
                config_v['max_token_cardinality'] = config_m['quantizer']['heads']
            else:
                config_m['max_token_cardinality'] = \
                    config_v['features'][f'{modality}_feat0'][0]
                config_v['max_token_cardinality'] = \
                    config_v['features'][f'{modality}_feat0'][0]

            config_m['gpos_quantization'] = False
            if len(config_v['features'][f'{modality}_feat0']) > 3:
                if config_v['features'][f'{modality}_feat0'][3] is not None:
                    if len(config_v['features'][f'{modality}_feat0'][3]) > 0:
                        config_m['gpos_quantization'] = True

            # need this for pos_token prediction for global modality
            config_m['data_type'] = config_v.get('data_type', 'set')

            self.config_v_dict[modality] = config_v
            self.config_m_dict[modality] = config_m
        
        if return_dicts:
            return self.config_v_dict, self.config_m_dict


    def load_tokenizer_ckpts(self):
        for modality in self.modalities:
            ckpt = torch.load(self.modality_dict[modality]['checkpoint_path'], 
                map_location='cpu')

            state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                state_dict[k.removeprefix('model.')] = v

            self.model.tokenizers[modality].load_state_dict(state_dict)
            self.model.tokenizers[modality].eval()
            self.model.tokenizers[modality].freeze()


    def fast_forward(self, batch, get_accuracy=False, get_cardinality=False):

        output_modalities = list(batch['target'].keys())

        pred_tokens_dict = self.model.fast_forward(
            input_dict=batch['input'],
            output_modalities=output_modalities,
            q_mask_dict=batch['q_mask_dict'],
            get_logits=True,
            use_truth_cardinality=True,
            target_dict=batch['target'])

        total_loss = 0; log_dict = {}
        for modality, head_loss in self.losses.items():
            if modality in output_modalities:
                loss, loss_dict = head_loss.forward(
                    pred_dict = pred_tokens_dict[modality],
                    target_dict = batch['target'][modality])
            
                if not torch.isnan(loss):
                    total_loss += loss

                for k, v in loss_dict.items():
                    log_dict[f'{modality}_{k}'] = v

        # in case some modality is not used in input or output
        active_inp = set(batch['input'].keys())
        active_out = set(pred_tokens_dict.keys())
        all_mods = set(self.modalities)

        # 2.1 add zero-scaled dummies for *inactive* predictor heads: small enough
        # to not affect gradients, large enough to keep autograd paths alive (DDP)
        dummy_scale = 1e-8
        for modality in (all_mods - active_out):
            predictors = [self.model.token_predictor]
            if hasattr(self.model, 'cardinality_predictor'):
                predictors.append(self.model.cardinality_predictor)
            for predictor in predictors:
                if modality in predictor:
                    head = predictor[modality]
                    dummy = sum(p.abs().sum() for p in head.parameters() if p.requires_grad)
                    total_loss = total_loss + dummy_scale * dummy

        # 2.2 add zero-scaled dummies for embedders that weren’t used
        # (e.g., a modality that is neither input nor output in this batch)
        for modality in (all_mods - active_inp):
            enc = self.model.embedders[modality] \
                if modality in self.model.embedders else None
            if enc is not None:
                dummy = sum(p.abs().sum() for p in enc.parameters() if p.requires_grad)
                total_loss = total_loss + dummy_scale * dummy

        total_loss = total_loss / len(output_modalities)
        log_dict['loss'] = total_loss.item() if total_loss is not None else 0.0

        # 3. token accuracy logging
        if get_accuracy:
            for modality in output_modalities:
                real_mask = batch['target'][modality]['q_mask']

                pred_tokens = torch.argmax(
                    pred_tokens_dict[modality]['token_logits'], dim=-1)[real_mask]
                target_tokens = batch['target'][modality]['tokens'][real_mask]
                accuracy = (pred_tokens == target_tokens).float().mean(dim=-2)
                for acc_i, acc in enumerate(accuracy):
                    log_dict[f'{modality}_acc_{acc_i}'] = acc.item()

        # 4. cardinality logging (argmax, deterministic)
        data_dict = {}
        if get_cardinality:
            for modality in output_modalities:
                if 'cardinality_logits' in pred_tokens_dict[modality]:
                    logits = pred_tokens_dict[modality]['cardinality_logits']  # (b, max_cardinality)
                    pred_card = logits.argmax(dim=-1)  # (b,) deterministic
                    target_card = batch['target'][modality]['cardinality']

                    data_dict.setdefault(modality, {})
                    data_dict[modality]['pred_cardinality'] = pred_card.detach()
                    data_dict[modality]['target_cardinality'] = target_card.detach()

        return total_loss, log_dict, data_dict


    def training_step(self, batch, batch_idx):    
        # log when ANY logger is attached (comet OR wandb), not only comet
        do_log_metric = (self.logger is not None) and \
            (batch_idx % self.config_t['train_log_every_n_steps'] == 0)

        if do_log_metric:
            for m in self.modules():
                if isinstance(m, MultiheadAttentionVarLen): 
                    m.track_metrics = True

        loss, log_dict, _ = self.fast_forward(batch, get_accuracy=do_log_metric)
        if do_log_metric:
            for k, v in log_dict.items():
                self.log(f'train_{k}', v, sync_dist=True)
            self.log('lr', self.optimizers().param_groups[0]['lr'], sync_dist=True)

            # attention health logging
            entropies = []; max_scores = []; q_norms = []
            for m in self.modules():
                if isinstance(m, MultiheadAttentionVarLen):
                    if m.metrics_buffer:
                        entropies.append(m.metrics_buffer.get('entropy'))
                        max_scores.append(m.metrics_buffer.get('max_score'))
                        q_norms.append(m.metrics_buffer.get('q_norm'))

                    # reset                    
                    m.track_metrics = False
                    m.metrics_buffer = {} 
            
            if entropies:
                self.log("monitor/entropy_min", min(entropies))
                self.log("monitor/entropy_mean", sum(entropies) / len(entropies))

            if max_scores:
                self.log("monitor/score_global_max", max(max_scores))
            
            if q_norms:
                self.log("monitor/q_norm_mean", sum(q_norms) / len(q_norms))
            
        return loss


    def validation_step(self, batch, batch_idx):
        loss, log_dict, data_dict = self.fast_forward(
            batch, get_accuracy=True, get_cardinality=True)
        self.val_step_outputs.append((log_dict, data_dict))


    def log_per_modality_grads(self):
        for name, predictor in self.model.token_predictor.items():
            
            total_norm = 0.0
            for p in predictor.parameters():
                if p.grad is not None:
                    # Sum of squares
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5 # Sqrt to get the final L2 norm
            self.log(f"grad_norm_head/{name}", total_norm, sync_dist=True)


    def on_before_optimizer_step(self, optimizer):
        # log when ANY logger is attached (comet OR wandb), not only comet
        do_log_metric = (self.logger is not None) and \
            (self.global_step % self.config_t['train_log_every_n_steps'] == 0)

        if do_log_metric:
            self.log_per_modality_grads()

        # clip gradients manually (required when using fused optimizer)
        max_grad_norm = self.config_t.get('max_grad_norm', 1.0)
        gnorm_unclipped = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # log gradient norm occasionally
        if do_log_metric:
            self.log("train_grad_norm_unclipped", float(gnorm_unclipped), sync_dist=True)
            gnorm_clipped = torch.nn.utils.clip_grad_norm_(self.model.parameters(), torch.inf)
            self.log("train_grad_norm_clipped", float(gnorm_clipped), sync_dist=True)


    def on_train_start(self):
        # wandb: trainer/global_step as x-axis, so curves stay continuous across resumes
        if self.trainer.global_rank == 0 and getattr(self, 'wandb_logger', None) is not None:
            self.wandb_logger.experiment.define_metric('trainer/global_step')
            self.wandb_logger.experiment.define_metric('*', step_metric='trainer/global_step')


    def _set_optimizer_mode(self, train: bool):
        # the schedule-free optimiser keeps separate train/eval parameter states
        opt = self.optimizers()
        if opt is not None and not (isinstance(opt, list) and len(opt) == 0):
            opt.train() if train else opt.eval()


    def on_train_epoch_start(self):
        self._set_optimizer_mode(True)


    def on_train_batch_start(self, batch, batch_idx):
        self._set_optimizer_mode(True)


    def on_validation_epoch_start(self):
        self._set_optimizer_mode(False)


    def on_save_checkpoint(self, checkpoint):
        self._set_optimizer_mode(False)


    def configure_optimizers(self):
        if self.config_t.get('scheduler_type', 'schedulefree') != 'schedulefree':
            raise ValueError(f"scheduler_type must be 'schedulefree', got {self.config_t['scheduler_type']!r}")
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            self.parameters(),
            lr=self.config_t['learning_rate'],
            weight_decay=self.config_t['weight_decay'],
            warmup_steps=self.config_t['warmup_steps'])
        return {'optimizer': optimizer}


    def on_validation_epoch_end(self):
        # 1) collect per-step outputs per key
        epoch_end_log_dict = {}; epoch_end_data_dict = {}
        for log_dict, data_dict in self.val_step_outputs:
            for k, v in log_dict.items():
                epoch_end_log_dict.setdefault(k, []).append(v)
            for modality, tensors in data_dict.items():
                mod_dict = epoch_end_data_dict.setdefault(modality, {})
                for name, val in tensors.items():
                    mod_dict.setdefault(name, []).append(val)

        # 2) get the union of keys across ranks (so missing keys still participate)
        is_ddp = dist.is_available() and dist.is_initialized()
        if is_ddp:
            obj_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(obj_list, set(epoch_end_log_dict.keys()))
            all_keys = sorted(set().union(*obj_list))

            # make sure every rank reaches this point before starting reductions
            dist.barrier()
        else:
            all_keys = sorted(epoch_end_log_dict.keys())

        # 3) per-key global (sum,count) -> mean
        global_means = {}
        for k in all_keys:
            vals = epoch_end_log_dict.get(k, [])
            if len(vals):
                t = torch.cat([torch.as_tensor(v, device=self.device, dtype=torch.float32).flatten()
                            for v in vals])
                s = t.sum()
                n = torch.tensor(float(t.numel()), device=self.device)
            else:
                s = torch.tensor(0.0, device=self.device)
                n = torch.tensor(0.0, device=self.device)

            pair = torch.stack([s, n])  # shape (2,)
            if is_ddp:
                dist.reduce(pair, dst=0, op=dist.ReduceOp.SUM)

            if (not is_ddp) or dist.get_rank() == 0:
                mean_val = (pair[0] / pair[1]) if pair[1].item() > 0 else torch.tensor(0.0, device=self.device)
            else:
                mean_val = torch.tensor(0.0, device=self.device)

            if is_ddp:
                dist.broadcast(mean_val, src=0)

            global_means[k] = mean_val

        # optional alias
        if 'loss' in global_means and 'total_loss' not in global_means:
            global_means['total_loss'] = global_means['loss']

        # Synthetic per-modality seq_loss = token_loss + pos_token_loss
        # (excludes cardinality_loss). Useful as an EarlyStopping target since
        # jet-response quality is driven by tokens/pos, not by the cardinality head.
        for k in list(global_means.keys()):
            if k.endswith('_token_loss') and not k.startswith('pos_') \
                    and '_pos_token_loss' not in k:
                modality = k[:-len('_token_loss')]
                pos_k = f'{modality}_pos_token_loss'
                if pos_k in global_means:
                    global_means[f'{modality}_seq_loss'] = \
                        global_means[k] + global_means[pos_k]

        # 4) log once on rank 0 (already globally reduced)
        to_log = {f"val_{k}": v for k, v in global_means.items()}
        self.log_dict(to_log, sync_dist=False)


        # 5) after scalar logging, gather per-event data across ranks
        self.val_data_gathered = {}
        for modality in sorted(epoch_end_data_dict.keys()):
            tensor_lists = epoch_end_data_dict[modality]
            self.val_data_gathered[modality] = {}
            
            for name in sorted(tensor_lists.keys()):
                chunks = tensor_lists[name]
                local = torch.cat(chunks, dim=0) if len(chunks) else \
                    torch.tensor([], device=self.device, dtype=torch.long)

                if is_ddp:
                    world_size = dist.get_world_size()
                    local_size = torch.tensor([local.shape[0]], device=self.device, dtype=torch.long)
                    all_sizes = [torch.zeros(1, device=self.device, dtype=torch.long)
                                 for _ in range(world_size)]
                    dist.all_gather(all_sizes, local_size)
                    max_size = int(max(s.item() for s in all_sizes))

                    if local.shape[0] < max_size:
                        padded = torch.cat([local, torch.zeros(
                            max_size - local.shape[0], device=self.device, dtype=local.dtype)])
                    else:
                        padded = local

                    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                    dist.all_gather(gathered, padded)
                    combined = torch.cat([
                        gathered[i][:int(all_sizes[i].item())] for i in range(world_size)])
                else:
                    combined = local

                self.val_data_gathered[modality][name] = combined.cpu()

        # 6) cardinality metrics + plot (same val/cardinality_* keys as nanoHEP),
        # computed on rank 0 from the all-gathered val_data_gathered
        if not is_ddp or dist.get_rank() == 0:
            import matplotlib.pyplot as plt
            for modality in self.val_data_gathered.keys():
                if 'pred_cardinality' not in self.val_data_gathered[modality]:
                    continue
                pred = self.val_data_gathered[modality]['pred_cardinality'].float()
                tgt  = self.val_data_gathered[modality]['target_cardinality'].float()
                res  = pred - tgt

                # logged per modality and, for the primary output modality, unprefixed
                card_acc  = (pred == tgt).float().mean().item()
                card_mae  = res.abs().mean().item()
                card_bias = res.mean().item()
                card_std  = res.std().item() if res.numel() > 1 else 0.0
                np_mean = pred.mean().item(); np_std = pred.std().item() if pred.numel() > 1 else 0.0
                nt_mean = tgt.mean().item();  nt_std = tgt.std().item()  if tgt.numel()  > 1 else 0.0

                scalars = {
                    f'val_{modality}_cardinality_acc':  card_acc,
                    f'val_{modality}_cardinality_mae':  card_mae,
                    f'val_{modality}_cardinality_bias': card_bias,
                    f'val_{modality}_cardinality_std':  card_std,
                    f'val_{modality}_n_pred_mean':      np_mean,
                    f'val_{modality}_n_pred_std':       np_std,
                    f'val_{modality}_n_true_mean':      nt_mean,
                    f'val_{modality}_n_true_std':       nt_std,
                }
                if modality in (self.config_t.get('val_sampling', {})
                                .get('fixed_input_output_modalities', {})
                                .get('output', [])[:1]):
                    scalars.update({
                        'val/cardinality_acc':  card_acc,
                        'val/cardinality_mae':  card_mae,
                        'val/cardinality_bias': card_bias,
                        'val/cardinality_std':  card_std,
                        'val/n_pred_mean':      np_mean,
                        'val/n_pred_std':       np_std,
                        'val/n_true_mean':      nt_mean,
                        'val/n_true_std':       nt_std,
                    })
                self.log_dict(scalars, sync_dist=False, rank_zero_only=True)

                # plot: scatter + density histogram
                try:
                    n_true = tgt.numpy().astype(int)
                    n_pred = pred.numpy().astype(int)
                    mx = int(max(n_true.max(), n_pred.max(), 1)) + 1
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
                    ax1.scatter(n_true, n_pred, alpha=0.3, s=10)
                    ax1.plot([0, mx], [0, mx], 'k--', lw=0.5)
                    ax1.set_xlabel('n_true'); ax1.set_ylabel('n_pred')
                    ax1.set_title(f'cardinality: pred vs true (epoch {self.trainer.current_epoch})')
                    bins = np.arange(0, mx + 1) - 0.5
                    ax2.hist(n_true, bins=bins, alpha=0.5, label='true', density=True)
                    ax2.hist(n_pred, bins=bins, alpha=0.5, label='pred', density=True)
                    ax2.set_xlabel('cardinality'); ax2.legend(fontsize=8)
                    ax2.set_title(f'hist  acc={card_acc:.2f} mae={card_mae:.2f}')
                    fig.tight_layout()
                    if getattr(self, 'wandb_logger', None) is not None:
                        import wandb
                        _img = wandb.Image(fig)
                        _plot_keys = {f'val/{modality}_cardinality_plot': _img,
                                      'trainer/global_step': self.trainer.global_step}
                        if modality in (self.config_t.get('val_sampling', {})
                                        .get('fixed_input_output_modalities', {})
                                        .get('output', [])[:1]):
                            _plot_keys['val/cardinality_plot'] = _img
                        self.wandb_logger.experiment.log(_plot_keys, commit=False)
                    save_plot(fig, f"{modality}_cardinality", self.comet_logger)  # no-op without comet
                    plt.close(fig)
                except Exception as e:
                    log.warning('cardinality plot skipped: %s', e)

        # 7) reset
        self.val_step_outputs = []

        # 8) particle-flow evaluation (jet response, marginals) on rank 0
        self._run_pflow_eval_if_requested()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # broadcast the jet pT IQR from rank 0 and log it on all ranks
        # (monitored by the best-IQR ModelCheckpoint)
        _iqr = getattr(self, '_last_pflow_iqr', None)
        if dist.is_available() and dist.is_initialized():
            _holder = [_iqr if self.global_rank == 0 else None]
            dist.broadcast_object_list(_holder, src=0)
            _iqr = _holder[0]
        if _iqr is not None and float(_iqr) > 1e-6:
            self._iqr_cache = float(_iqr)
        # log 1e9 when the eval did not run this validation
        if not getattr(self.trainer, 'sanity_checking', False):
            _fresh = _iqr is not None and float(_iqr) > 1e-6
            self.log('val_pflow_iqr_jet_pt_response',
                     float(_iqr) if _fresh else 1e9,
                     on_step=False, on_epoch=True, sync_dist=False)
            self.log('val_pflow_iqr_display',
                     float(self._iqr_cache) if getattr(self, '_iqr_cache', None) is not None else 1e9,
                     on_step=False, on_epoch=True, sync_dist=False)
        self._last_pflow_iqr = None

        torch.cuda.empty_cache()
        self._set_optimizer_mode(True)


    def _run_pflow_eval_if_requested(self):
        """Particle-flow evaluation on the validation set (jet response, marginals,
        histograms) every ``pflow_eval_every_n_epoch`` epochs, from epoch 1. Decodes with
        the trained cardinality head (``use_truth_cardinality=False``), argmax tokens."""
        pflow_every = self.config_t.get('pflow_eval_every_n_epoch', 0) if self.config_t else 0
        if not pflow_every or self.trainer.sanity_checking:
            return
        if self.trainer.current_epoch == 0 or self.trainer.current_epoch % pflow_every != 0:
            return
        is_ddp = dist.is_available() and dist.is_initialized()
        if is_ddp and dist.get_rank() != 0:
            return
        datamodule = self.trainer.datamodule
        if datamodule is None:
            return
        fixed = self.config_t.get('val_sampling', {}).get('fixed_input_output_modalities')
        if not fixed:
            return
        output_modalities = fixed.get('output', [])
        if not output_modalities:
            return

        # deferred imports so training without this eval does not pay for them
        import os
        import shutil
        import tempfile
        import awkward as ak
        from ..utility.var_transformation import VarTransformation
        from ..performance.pflow_report import run_report_from_arrays
        from ..performance.generative_report import run_generative_report_from_arrays

        log.info('pflow_eval: starting at epoch %d', self.trainer.current_epoch)

        # feature names + inverse transforms per output modality: the tokens are
        # decoded with the tokenisers, then the continuous features are inverted
        feat_names = {}
        inv_transform = {}
        for mod in output_modalities:
            cfg_v = self.config_v_dict[mod]
            feats = cfg_v['features'][f"{mod}_feat0"]
            feat_names[mod] = feats[1]                      # continuous feature names
            tdict = cfg_v.get('transformation_dict', {}) or {}
            inv_transform[mod] = {
                fn: VarTransformation(tdict[fn]) if fn in tdict else None
                for fn in feat_names[mod]
            }

        max_batches = self.config_t.get('pflow_eval_max_batches')
        loader = datamodule.val_dataloader()

        out_mod = output_modalities[0]
        truth_lists = {'pt': [], 'eta': [], 'phi': []}
        reco_lists = {'pt': [], 'eta': [], 'phi': []}

        def _append_event(lists, x, gpos, sel):
            lists['pt'].append(x[sel, 0])                   # first continuous feature is pt
            if gpos is not None and gpos.shape[-1] >= 3:
                lists['eta'].append(gpos[sel, 0])
                lists['phi'].append(np.arctan2(gpos[sel, 2], gpos[sel, 1]))
            else:
                lists['eta'].append(np.zeros(sel.sum(), dtype=np.float32))
                lists['phi'].append(np.zeros(sel.sum(), dtype=np.float32))

        self.model.eval()
        # predicted cardinality != truth cardinality: no teacher forcing from the targets
        self.model.set_use_target_tokens(False)
        with torch.no_grad():
            for bi, batch in enumerate(tqdm(loader, desc='pflow_eval', leave=False)):
                if max_batches is not None and bi >= max_batches:
                    break
                batch_dev = {}
                for k, v in batch.items():
                    if isinstance(v, dict):
                        batch_dev[k] = {kk: {kkk: vvv.to(self.device)
                                              if torch.is_tensor(vvv) else vvv
                                              for kkk, vvv in vv.items()}
                                         if isinstance(vv, dict) else
                                         (vv.to(self.device) if torch.is_tensor(vv) else vv)
                                         for kk, vv in v.items()}
                    else:
                        batch_dev[k] = v

                pred = self.model.fast_forward(
                    input_dict=batch_dev['input'],
                    output_modalities=output_modalities,
                    q_mask_dict=None,
                    get_logits=True,
                    top_k_token_dict={m: 1 for m in output_modalities},
                    top_k_gpos_token_dict={m: 1 for m in output_modalities},
                    temperature_token_dict={m: 1.0 for m in output_modalities},
                    temperature_gpos_token_dict={m: 1.0 for m in output_modalities},
                    use_truth_cardinality=False,
                    card_topk_dict={m: 1 for m in output_modalities},
                    card_temperature_dict={m: 1.0 for m in output_modalities},
                    target_dict=batch_dev.get('target'),
                )
                for mod in output_modalities:               # argmax tokens
                    pred[mod]['tokens'] = pred[mod]['token_logits'].argmax(dim=-1)
                    if 'gpos_token_logits' in pred[mod]:
                        pred[mod]['gpos_tokens'] = pred[mod]['gpos_token_logits'].argmax(dim=-1)

                # decode_from_tokens iterates model.output_modalities: restrict it to out_mod
                prev_in = getattr(self.model, 'input_modalities', None)
                prev_out = getattr(self.model, 'output_modalities', None)
                self.model.set_input_output_modalities(fixed.get('input', []), [out_mod])

                if pred[out_mod]['tokens'].shape[1] == 0:
                    continue

                pred_dec = self.model.decode_from_tokens(pred)
                p_x = pred_dec[out_mod]['x_hat']
                if isinstance(p_x, (tuple, list)):
                    p_x = p_x[0]
                p_gpos = pred_dec[out_mod].get('x_gpos_hat')
                p_mask = pred_dec[out_mod]['q_mask'].bool()

                # truth tokens go through the same decoder
                tgt = batch_dev['target'][out_mod]
                truth_tok_dict = {out_mod: {'tokens': tgt['tokens'], 'q_mask': tgt['q_mask'].bool()}}
                if 'pos_tokens' in tgt:
                    truth_tok_dict[out_mod]['gpos_tokens'] = tgt['pos_tokens']
                truth_dec = self.model.decode_from_tokens(truth_tok_dict)
                if prev_in is not None and prev_out is not None:
                    self.model.set_input_output_modalities(prev_in, prev_out)
                t_x = truth_dec[out_mod]['x_hat']
                if isinstance(t_x, (tuple, list)):
                    t_x = t_x[0]
                t_gpos = truth_dec[out_mod].get('x_gpos_hat')
                t_mask = truth_dec[out_mod]['q_mask'].bool()

                for feat_i, fn in enumerate(feat_names[out_mod]):
                    inv = inv_transform[out_mod][fn]
                    if inv is not None:
                        p_x[:, :, feat_i] = inv.inverse(p_x[:, :, feat_i])
                        t_x[:, :, feat_i] = inv.inverse(t_x[:, :, feat_i])

                p_x_np, t_x_np = p_x.cpu().numpy(), t_x.cpu().numpy()
                p_mask_np, t_mask_np = p_mask.cpu().numpy(), t_mask.cpu().numpy()
                p_gpos_np = p_gpos.cpu().numpy() if p_gpos is not None else None
                t_gpos_np = t_gpos.cpu().numpy() if t_gpos is not None else None
                for b in range(p_x_np.shape[0]):
                    _append_event(reco_lists, p_x_np[b], None if p_gpos_np is None else p_gpos_np[b],
                                  p_mask_np[b])
                    _append_event(truth_lists, t_x_np[b], None if t_gpos_np is None else t_gpos_np[b],
                                  t_mask_np[b])

        self.model.set_use_target_tokens(True)
        self.model.train()

        if len(truth_lists['pt']) == 0:
            return

        truth = {k: ak.Array(truth_lists[k]) for k in ['pt', 'eta', 'phi']}
        reco = {k: ak.Array(reco_lists[k]) for k in ['pt', 'eta', 'phi']}

        outdir = tempfile.mkdtemp(prefix='hep4m_pflow_eval_') if getattr(self, 'wandb_logger', None) is not None else None
        gen_metrics = run_generative_report_from_arrays(
            truth=truth, reco=reco,
            outdir=outdir, output_modality=out_mod,
            c2st_epochs=int(self.config_t.get('c2st_epochs', 30)),
        )
        if self.config_t.get('generative_eval', False):
            # detector simulation: the jet-based particle-flow report does not apply
            metrics = {k if k.startswith('_') else f'gen_{k}': v for k, v in gen_metrics.items()}
        else:
            metrics = run_report_from_arrays(
                truth=truth, reco=reco, reco_ind=None,
                outdir=outdir, output_modality=out_mod,
                ind_threshold=self.config_t.get('pflow_ind_threshold', 0.45),
                n_event_displays=6,
            )
            for k, v in gen_metrics.items():
                if k == '_histograms':
                    metrics.setdefault('_histograms', {}).update(v)
                elif not k.startswith('_'):
                    metrics[f'gen_{k}'] = v

        histograms = metrics.pop('_histograms', {})
        jet_table_data = metrics.pop('_jet_table', {})

        from ..performance.wandb_keys import to_wandb_subkey, PFLOW_IMAGES, gen_image_key
        to_log = {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            ns = 'gen' if k.startswith('gen_') else 'pflow'
            base = k[len('gen_'):] if k.startswith('gen_') else k
            to_log[to_wandb_subkey(base, namespace=ns)] = v

        log.info('pflow_eval: epoch %d, %d events, %d metrics',
                 self.trainer.current_epoch, len(truth_lists['pt']), len(to_log))
        # logged on all ranks after the barrier in on_validation_epoch_end
        self._last_pflow_iqr = metrics.get('iqr_jet_pt_response')

        # logger.experiment.log directly: self.log_dict in on_validation_epoch_end
        # drops scalars logged from one rank in multi-node DDP
        if getattr(self, 'wandb_logger', None) is not None:
            import wandb
            from PIL import Image
            if to_log:
                self.wandb_logger.experiment.log(to_log)
            try:
                image_map = {**PFLOW_IMAGES, **gen_image_key(out_mod)}
                for fname, wandb_key in image_map.items():
                    path = os.path.join(outdir, fname)
                    if os.path.isfile(path):
                        img = Image.open(path).copy()
                        self.wandb_logger.experiment.log({wandb_key: wandb.Image(img)})
                for key, (counts, bin_edges) in histograms.items():
                    self.wandb_logger.experiment.log({
                        f'val/pflow/hist/{key}': wandb.Histogram(np_histogram=(counts, bin_edges))
                    })
                if jet_table_data:
                    cols = list(jet_table_data.keys())
                    table = wandb.Table(
                        columns=cols,
                        data=list(zip(*(jet_table_data[c].tolist() for c in cols)))
                    )
                    self.wandb_logger.experiment.log({'val/pflow/tables/jets': table})
            finally:
                shutil.rmtree(outdir, ignore_errors=True)

        torch.cuda.empty_cache()
