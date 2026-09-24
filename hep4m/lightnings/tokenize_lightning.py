import logging
import torch
import torch.nn.functional as F
from lightning.pytorch.core.module import LightningModule
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import uproot
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from torch.utils.data import DataLoader
from ..datasets.dataset_modalities import get_dataset_class, ContiguousBatchSampler
from ..models.vqvae import VQVAE
from ..utility.comet_helper import save_plot
from .custom_loss_treatments import custom_loss_dict

log = logging.getLogger(__name__)


class TokenizeLightning(LightningModule):

    def __init__(self, config_v, config_m, config_t=None, comet_logger=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.config_v = config_v
        self.config_m = config_m
        self.config_t = config_t

        self.data_type = self.config_v.get('data_type', 'set')
        self.modality = self.config_v['modality']
        self.model = VQVAE(config_m)
        self.num_quantizers = self.model.vector_quantization.num_quantizers

        self.comet_logger = comet_logger
        self.val_step_outputs = [] # store every batch
        self.val_output_to_mod = {} # update every batch

        self.feat0_name = list(self.config_v['features'].keys())[0]
        self.feat0_mask_name = f'{self.feat0_name}_mask'

        self.save_hyperparameters()


    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger


    def get_dataloader(self, filepath, reduce_ds, batch_size, shuffle):
        use_npy_dataset = self.config_t.get('use_npy_dataset', False)
        COCOADataset = get_dataset_class(
            self.modality, npy=use_npy_dataset)
        dataset = COCOADataset(
            filename=filepath,
            config_v=self.config_v,
            reduce_ds=reduce_ds)

        loader_kwargs = {
            'num_workers': self.config_t['num_workers'],
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
        }
        if use_npy_dataset:
            batch_sampler = ContiguousBatchSampler(
                dataset_len=len(dataset),
                batch_size=batch_size,
                drop_last=True,
                shuffle=shuffle,
            )
            loader_kwargs['batch_sampler'] = batch_sampler
        else:
            loader_kwargs['batch_size'] = batch_size
            loader_kwargs['shuffle'] = shuffle

        loader = DataLoader(dataset, **loader_kwargs)
        return loader


    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            filepath=self.config_t['path_train'],
            reduce_ds=self.config_t['reduce_ds_train'],
            batch_size=self.config_t['batchsize_train'],
            shuffle=True
        )


    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            filepath=self.config_t['path_val'],
            reduce_ds=self.config_t['reduce_ds_val'],
            batch_size=self.config_t['batchsize_val'],
            shuffle=False
        )


    def reconstruction_loss(self, x_hat, x, mask):
        if self.modality in custom_loss_dict.keys():
            return custom_loss_dict[self.modality](
                self.config_v, x_hat, x, mask)

        x_cont, x_cat = x
        x_hat_cont, x_hat_cat = x_hat

        loss_cont = F.mse_loss(x_hat_cont, x_cont, reduction='none')
        loss_cont = loss_cont[mask].mean()

        loss_cat = 0
        for v_name, x_cat_v in x_cat.items():
            x_hat_cat_v = x_hat_cat[v_name]
            loss_cat += F.cross_entropy(
                x_hat_cat_v[mask], x_cat_v[mask], reduction='none')
            loss_cat = loss_cat.mean()

        loss = loss_cont + loss_cat

        loss_dict = {'loss_cont': loss_cont}
        if len(x_cat) > 0:
            loss_dict['loss_cat'] = loss_cat

        return loss, loss_dict


    def forward(self, batch, return_xhat=False):
        mask = batch[self.feat0_mask_name]
        if self.data_type == 'cell_image':
            mask = None

        embedding_loss, x_hat, indices = self.model(
            batch[self.feat0_name], batch[f'{self.feat0_name}_categorical'], mask)
        embedding_loss = embedding_loss.mean()

        reco_loss, reco_loss_dict = self.reconstruction_loss(
            x_hat, 
            (batch[self.feat0_name], batch[f'{self.feat0_name}_categorical']),
            mask)

        reco_loss = self.config_t['reco_loss_wt'] * reco_loss
        embedding_loss = self.config_t['embedding_loss_wt'] * embedding_loss
        loss = reco_loss + embedding_loss

        log_dict = {
            'loss': loss.detach().cpu().numpy(),
            'reco_loss': reco_loss.detach().cpu().numpy(),
            'embedding_loss': embedding_loss.detach().cpu().numpy(),
        }
        for k, v in reco_loss_dict.items():
            log_dict[k] = v.detach().cpu().numpy()

        # vectors used
        unique_vectors = [torch.unique(x) for x in indices.chunk(self.num_quantizers, dim=-1)]
        unique_vector_count = np.array([x.shape[0] for x in unique_vectors])
        for i in range(self.num_quantizers):
            log_dict[f'vectors_used_{i}'] = unique_vector_count[i]
            
        # perplexity
        if self.data_type == 'set':
            indices_one_hot = torch.nn.functional.one_hot(
                indices[batch[self.feat0_mask_name]], 
                num_classes = self.model.vector_quantization.codebook_size).float()
        elif self.data_type == 'cell_image':
            self.tmp_num_tokens_this_batch = indices.shape[0] * indices.shape[1]
            indices_one_hot = torch.nn.functional.one_hot(
                indices.view(self.tmp_num_tokens_this_batch, -1), 
                num_classes = self.model.vector_quantization.codebook_size).float()
        elif self.data_type == 'global':
            self.tmp_num_tokens_this_batch = indices.shape[0] * indices.shape[1]
            indices_one_hot = torch.nn.functional.one_hot(
                indices.view(self.tmp_num_tokens_this_batch, -1),
                num_classes = self.model.vector_quantization.codebook_size).float()

        indices_mean = indices_one_hot.mean(0).detach().cpu().numpy()
        perplexity = np.array([np.exp(-np.sum(x * np.log(x + 1e-10))) for x in indices_mean])
        for i in range(self.num_quantizers):
            log_dict[f'perplexity_{i}'] = perplexity[i]

        return_tuple = (loss, log_dict, unique_vectors, indices_mean)
        if return_xhat:
            return_tuple += (x_hat,)

        return return_tuple


    def training_step(self, batch, batch_idx):
        loss, log_dict, _, _ = self(batch)
        if self.comet_logger is not None and \
                batch_idx % self.config_t['train_log_every_n_steps'] == 0:
            self.comet_logger.log_metrics(
                {f'train_{k}': v for k, v in log_dict.items()})
            self.log('lr', self.optimizers().param_groups[0]['lr'])
            grad_norm = clip_grad_norm_(self.parameters(), max_norm=float('inf'))
            self.log("grad_norm", grad_norm)            
            
        return loss


    def validation_step(self, batch, batch_idx):
        loss, log_dict, unique_vectors, indices_mean, x_hat = \
            self(batch, return_xhat=True)

        if self.data_type == 'set':
            x_hat_cont, x_hat_cat = x_hat
            mask = batch[self.feat0_mask_name]

            data_dict = {
                'batch_idx': batch_idx,
                'x_hat_cont': x_hat_cont[mask].detach().cpu().numpy(),
                'x_cont': batch[self.feat0_name][mask].detach().cpu().numpy(),
                'x_hat_cat': {k: torch.argmax(v, dim=-1)[mask].detach().cpu().numpy() 
                    for k, v in x_hat_cat.items()},
                'x_cat': {k: v[mask].detach().cpu().numpy()
                    for k, v in batch[f'{self.feat0_name}_categorical'].items()},
            }
        elif self.data_type == 'cell_image':
            data_dict = {
                'batch_idx': batch_idx,
            }
        elif self.data_type == 'global':
            x_hat_cont, x_hat_cat = x_hat
            data_dict = {
                'batch_idx': batch_idx,
                'x_hat_cont': x_hat_cont.detach().cpu().numpy(),
                'x_cont': batch[self.feat0_name].detach().cpu().numpy(),
                'x_hat_cat': {k: torch.argmax(v, dim=-1).detach().cpu().numpy() 
                    for k, v in x_hat_cat.items()},
                'x_cat': {k: v.detach().cpu().numpy()
                    for k, v in batch[f'{self.feat0_name}_categorical'].items()}, 
            }

        self.val_step_outputs.append([log_dict, data_dict])

        # vectors used
        for i in range(self.num_quantizers):
            vectors_used = unique_vectors[i].detach().cpu().numpy()
            if f'vectors_used_{i}' not in self.val_output_to_mod:
                self.val_output_to_mod[f'vectors_used_{i}'] = np.array([])
            self.val_output_to_mod[f'vectors_used_{i}'] = \
                np.union1d(self.val_output_to_mod[f'vectors_used_{i}'], vectors_used)

        # perplexity
        if self.data_type == 'set':
            indices_sum = indices_mean * batch[self.feat0_mask_name].sum().item()
            n_valid_total = batch[self.feat0_mask_name].sum().item()
        elif self.data_type == 'cell_image': # no mask; everything is valid
            n_valid_total = self.tmp_num_tokens_this_batch
            indices_sum = indices_mean * self.tmp_num_tokens_this_batch
        elif self.data_type == 'global':
            n_valid_total = self.tmp_num_tokens_this_batch
            indices_sum = indices_mean * self.tmp_num_tokens_this_batch

        if 'indices_sum' not in self.val_output_to_mod:
            self.val_output_to_mod['indices_sum'] = np.zeros_like(indices_sum)
            self.val_output_to_mod['n_valid_total'] = 0
        self.val_output_to_mod['indices_sum'] += indices_sum
        self.val_output_to_mod['n_valid_total'] += n_valid_total


    def estimate_step_count_in_one_epoch(self):
        '''
            In general, this should be quite correct; unless we start doing funky stuff
        '''
        if self.config_t['reduce_ds_train'] > 0:
            n_events = self.config_t['reduce_ds_train']

        else:
            if not self.config_t.get('use_npy_dataset', False):
                fps = eval(self.config_t['path_train'])
                tree_name = uproot.open(fps[0]).keys()[0].split(';')[0]
                n_events = 0
                for fp in tqdm(fps, desc='Estimating step count'):
                    n_events += uproot.open(fp)[tree_name].num_entries
            else:
                filename = self.config_t['path_train']
                meta_path = filename.replace('data.npy', 'meta.npz')
                meta = np.load(meta_path)
                n_events = int(meta['n_events'])

        step_count = n_events / self.config_t['batchsize_train']
        num_devices = self.trainer.num_devices * self.trainer.num_nodes     
        step_count = int(np.ceil(step_count) / num_devices)
        log.info("estimated steps per epoch: %d (%d events, batch size %d, %d devices)",
                 step_count, n_events, self.config_t['batchsize_train'], num_devices)

        return step_count


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=self.config_t['learning_rate'],
            weight_decay=self.config_t['weight_decay'])

        step_in_an_epoch = self.estimate_step_count_in_one_epoch()
        total_steps = self.config_t['num_epochs'] * step_in_an_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config_t['learning_rate'],
            total_steps=total_steps,
            pct_start=self.config_t['warmup_step_frac'],
            div_factor=self.config_t['div_factor'],
            final_div_factor=self.config_t['final_div_factor'])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Call scheduler_step every training step
            }
        }



    def on_validation_epoch_end(self):
        epoch_end_log_dict = {}

        for key in self.val_step_outputs[0][0].keys():
            if 'perplexity' in key or 'vectors_used' in key:
                continue
            epoch_end_log_dict[key] = np.hstack([x[0][key] for x in self.val_step_outputs]).mean().item()
        epoch_end_log_dict['total_loss'] = epoch_end_log_dict['loss']
        self.log_dict({f'val_{k}': v for k, v in epoch_end_log_dict.items()})

        # figure
        if self.data_type == 'set':
            marginal_fig, res_fig, scatter_fig, confusion_fig = self.get_figures()
            save_plot(marginal_fig, 'marginals', self.comet_logger)
            save_plot(res_fig, 'residuals', self.comet_logger)
            save_plot(scatter_fig, 'scatter', self.comet_logger)
            if confusion_fig is not None:
                save_plot(confusion_fig, 'confusion', self.comet_logger)

        # vectors used
        for i in range(self.num_quantizers):
            vectors_used = self.val_output_to_mod[f'vectors_used_{i}']
            self.log(f'val vectors used {i} (%)', 
                len(vectors_used) / self.model.vector_quantization.codebook_size * 100)

        # perplexity
        indices_mean = self.val_output_to_mod['indices_sum'] / self.val_output_to_mod['n_valid_total']
        perplexity = np.array([np.exp(-np.sum(x * np.log(x + 1e-10))) for x in indices_mean])
        for i in range(self.num_quantizers):
            self.log(f'val perplexity {i} (%)', 
                perplexity[i] / self.model.vector_quantization.codebook_size * 100)

        self.val_step_outputs = []
        self.val_output_to_mod = {}


    def get_figures(self):

        def get_bins(feat_name, x, x_hat=None, nbins=50):
            if feat_name in self.config_t.get('plotting', {}).get('max_min', {}):
                _min, _max = self.config_t['plotting']['max_min'][feat_name]
                return np.linspace(_min, _max, nbins)
            _min = x.min(); _max = x.max()
            if x_hat is not None:
                _min = min(_min, x_hat.min())
                _max = max(_max, x_hat.max())
            return np.linspace(_min, _max, nbins)

        x = np.vstack([x[1]['x_cont'] for x in self.val_step_outputs])
        x_hat = np.vstack([x[1]['x_hat_cont'] for x in self.val_step_outputs])
        res = x - x_hat

        _keys = list(self.config_v['features'].keys())
        assert len(_keys) == 1, "Only one feature set is supported"
        feat_names = self.config_v['features'][_keys[0]][1]
        n_feat = len(feat_names)

        n_col = 4; n_row = int(np.ceil(n_feat / n_col))

        marginal_fig = plt.figure(figsize=(n_col * 3, n_row * 3))
        marginal_gs = marginal_fig.add_gridspec(n_row, n_col)
        for i in range(n_feat):
            ax = marginal_fig.add_subplot(marginal_gs[i])
            bins = get_bins(feat_names[i], x[:, i], x_hat[:, i])
            ax.hist(x[:, i], bins=bins, histtype='stepfilled', label='x', color='cornflowerblue', alpha=1.0)
            ax.hist(x_hat[:, i], bins=bins, histtype='step', label='x_hat', color='red', alpha=1.0)
            ax.set_xlabel(f'{feat_names[i]}')
            ax.set_ylabel('Count')
            ax.legend()

        res_fig = plt.figure(figsize=(n_col * 3, n_row * 3))
        res_gs = res_fig.add_gridspec(n_row, n_col, wspace=0.3, hspace=0.3)
        for i in range(n_feat):
            ax = res_fig.add_subplot(res_gs[i])
            bins = np.linspace(np.percentile(res[:, i], 1), np.percentile(res[:, i], 99), 50)
            ax.hist(res[:, i], bins=bins, histtype='stepfilled', color='cornflowerblue', alpha=1.0)
            ax.set_xlabel(f'x - x_hat ({feat_names[i]})')
            ax.set_ylabel('Count')

        scatter_fig = plt.figure(figsize=(n_col * 3, n_row * 3))
        scatter_gs = scatter_fig.add_gridspec(n_row, n_col, wspace=0.4, hspace=0.4)
        for i in range(n_feat):
            ax = scatter_fig.add_subplot(scatter_gs[i])
            bins = get_bins(feat_names[i], x[:, i], x_hat[:, i])
            ax.hist2d(x[:, i], x_hat[:, i], bins=[bins, bins], cmap='cool', cmin=1)
            ax.set_xlabel(f'{feat_names[i]}')
            ax.set_ylabel(f'x_hat {feat_names[i]}')

        # confusion matrix
        n_feat = len(self.config_v['features'][_keys[0]][2])
        if n_feat == 0:
            return marginal_fig, res_fig, scatter_fig, None

        n_col = min(4, n_feat); n_row = int(np.ceil(n_feat / n_col))

        confusion_fig = plt.figure(figsize=(n_col * 5, n_row * 5))
        confusion_gs = confusion_fig.add_gridspec(n_row, n_col, wspace=0.4, hspace=0.4)
        for i, (feat_name, n_class) in enumerate(self.config_v['features'][_keys[0]][2]):
            x = np.hstack([x[1]['x_cat'][feat_name] for x in self.val_step_outputs])
            x_hat = np.hstack([x[1]['x_hat_cat'][feat_name] for x in self.val_step_outputs])
            cm = confusion_matrix(x, x_hat)
            
            ax = confusion_fig.add_subplot(confusion_gs[i])
            annot = np.array([["{:,}".format(value) for value in row] for row in cm])
            sns.heatmap(cm, annot=annot, fmt="", cmap='Blues', ax=ax)
            ax.set_title(f'Confusion matrix {feat_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        return marginal_fig, res_fig, scatter_fig, confusion_fig