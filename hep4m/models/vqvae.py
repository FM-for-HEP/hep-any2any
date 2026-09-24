import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

from .helpers.dense import Dense
from .helpers.transformer import TEncoder
from .helpers.utils import padded_to_packed, packed_to_padded
from .helpers.memcodes import Memcodes
from .helpers.bottleneck_mlp import build_mlp
from .helpers.upsampler import Upsampler


class VQVAE(nn.Module):
    def __init__(self, config):
        '''
        NOTES:
            - indices shape convention is flipped for mlp_bottleneck
                - but it's okay, cause it's used only by 4M
                - Tokenizer training used z_q only
            - when we get z_q from indices, we need to transpose for mlp_bottleneck
        '''

        super().__init__()
        self.config = config
        
        self.vae_type = config.get('vae_type', 'transformer')
        if self.vae_type == 'mlp_bottleneck':
            self.init_mlp_vae(config)
        elif self.vae_type == 'cellimg_transformer':
            self.init_cellimg_transformer_vae()
        elif self.vae_type == 'transformer':
            self.init_transformer_vae(config)        


    def init_mlp_vae(self, config):
        self.categorical_vars = config['categorical_vars']
        assert len(self.categorical_vars) == 0, \
            "MLP VQVAE does not support categorical features"

        self.init_net = build_mlp(
            **self.config['encoder']['init_net_bottleneck'])
        self.vector_quantization = Memcodes(**self.config['quantizer'])
        self.output_net_cont = build_mlp(
            **self.config['decoder']['output_net_cont'])


    def init_cellimg_transformer_vae(self):

        # encoding with conv2ds
        conv2d_config = self.config['encoder']['conv2d'].copy()
        self.pow2_scale = self.config['pow2_scale']
        self.num_cell_layers = len(self.pow2_scale)

        self.conv_2ds = nn.ModuleList()
        for i in range(self.num_cell_layers):
            conv2d_config['kernel_size'] = 2**self.pow2_scale[i]
            conv2d_config['stride'] = 2**self.pow2_scale[i]
            conv2d_config['bias'] = False
            self.conv_2ds.append(nn.Conv2d(**conv2d_config))

        # positional encoding
        if self.config['use_pos_emb']:
            pos_emb_enc = nn.Parameter(torch.randn(
                1, self.config['pos_emb_len'], self.config['common_emb_dim']) * 0.02)
            self.register_parameter('pos_emb_enc', pos_emb_enc)

            pos_emb_dec = nn.Parameter(torch.randn(
                1, self.config['pos_emb_len'], self.config['common_emb_dim']) * 0.02)
            self.register_parameter('pos_emb_dec', pos_emb_dec)

        # encoder
        self.transformer_encoder = TEncoder(**self.config['encoder']['transformer'])

        # quantizer
        quantizer_config = self.config['quantizer'].copy()
        _ = quantizer_config.pop('type')
        self.vector_quantization = ResidualVQ(**quantizer_config)

        # decoder
        self.decoder = TEncoder(**self.config['decoder']['transformer'])

        # num patch in each layer
        self.cumsum_num_tokens = np.cumsum(
            [0] + self.config['decoder']['num_tokens_per_layer'])

        # upsamplers
        self.upsamplers = nn.ModuleList()
        for i in range(self.num_cell_layers):
            ups = Upsampler(
                in_channels=self.config['common_emb_dim'],
                n_upsample=self.pow2_scale[i]
            )
            self.upsamplers.append(ups)

        # predictor
        self.output_nets_cont = nn.ModuleList()
        self.output_net_ind = nn.ModuleList()

        output_config = self.config['decoder']['output_conv2d'].copy()
        for i in range(self.num_cell_layers):
            seq = []; seq_ind = []
            for (ic, oc), k in zip(output_config['in_out_channels'], 
                    output_config['kernel_sizes']):
                seq.append(nn.Conv2d(ic, oc, kernel_size=k, padding='same'))
                seq.append(nn.LeakyReLU())
                seq_ind.append(nn.Conv2d(ic, oc, kernel_size=k, padding='same'))
                seq_ind.append(nn.LeakyReLU())

            seq.append(nn.Conv2d(oc, output_config['out_channels'], kernel_size=1))
            seq_ind.append(nn.Conv2d(oc, 1, kernel_size=1))

            self.output_nets_cont.append(nn.Sequential(*seq))
            self.output_net_ind.append(nn.Sequential(*seq_ind))


    def init_transformer_vae(self, config):

        # encoding the categorical features
        self.categorical_vars = config['categorical_vars'] # list to not mess up order
        self.embedding_dict = nn.ModuleDict()
        for name, (n_class, dim) in config['categorical'].items():
            self.embedding_dict[name] = nn.Embedding(n_class, dim)

        # encdoder
        init_net_config = self.config['encoder']['init_net'].copy()
        init_net_config['input_size'] = config['n_feat_cont'] + \
            sum([dim for _, dim in config['categorical'].values()])
        self.init_net = Dense(**init_net_config)
        self.transformer_encoder = TEncoder(**self.config['encoder']['transformer'])

        # quantizer
        quantizer_config = self.config['quantizer'].copy()
        quantizer_type = quantizer_config.pop('type')
        if quantizer_type == 'VectorQuantize':
            self.vector_quantization = VectorQuantize(**quantizer_config)
            self.vector_quantization.num_quantizers = 1

        elif quantizer_type == 'ResidualVQ':
            self.vector_quantization = ResidualVQ(**quantizer_config)

        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")

        # decoder
        self.decoder = TEncoder(**self.config['decoder']['transformer'])

        # continuous features
        self.output_net_cont = Dense(**self.config['decoder']['output_net_cont'])

        # categorical features
        self.output_nets_categorical = nn.ModuleDict()
        for name, (n_class, _) in config['categorical'].items():
            config_categorical = deepcopy(self.config['decoder']['output_net_categorical'])
            config_categorical['output_size'] = n_class
            config_categorical['final_activation'] = None
            self.output_nets_categorical[name] = Dense(**config_categorical)


    # freeze when training the 4M model
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


    def encode_mlp_bottleneck(self, x_cont, x_mask, return_kvmask=False):
        '''
        Returns
            - z_q: (B, n_tokens, n_emb)
            - indices: (B, n_tokens, nq) | (B, nh, 1) for mlp_bottleneck
            - embedding_loss: (B,)
        '''

        z_e = self.init_net(x_cont)
        z_q, indices, embedding_loss = \
            self.vector_quantization(z_e)
        if return_kvmask:
            kvmask = torch.ones(
                indices.size(0), indices.size(1), dtype=torch.bool, device=indices.device)
            return z_q, indices, embedding_loss, kvmask
        return z_q, indices, embedding_loss  


    def encode_cellimg_transformer(self, x_cont, x_cat, return_kvmask=False):
        input_seq = []
        for i in range(self.num_cell_layers):
            v = x_cont[f'feat0_{i}'].permute(0, 3, 1, 2) # (B, C, H, W)
            x = self.conv_2ds[i](v)
            x = x.flatten(start_dim=2, end_dim=-1) # (B, C, H*W)
            x = x.transpose(1, 2) # (B, H*W, C)
            input_seq.append(x)
        x_init = torch.cat(input_seq, dim=1)


        # add positional encoding
        if hasattr(self, 'pos_emb_enc'):
            x_init = x_init + self.pos_emb_enc

        # encode and quantize
        z_e = self.transformer_encoder(x_init)
        z_q, indices, embedding_loss = self.vector_quantization(z_e)

        if return_kvmask:
            kvmask = torch.ones(
                indices.size(0), indices.size(1), dtype=torch.bool, device=indices.device)
            return z_q, indices, embedding_loss, kvmask
        return z_q, indices, embedding_loss 


    def encode_transformer(self, x_cont, x_categorical, x_mask, return_kvmask=False):

        # categorical features
        if len(self.categorical_vars) != 0:
            categorical_features = []
            for v in self.categorical_vars:
                x_cat_v = self.embedding_dict[v](x_categorical[v])
                categorical_features.append(x_cat_v)
            x_cat = torch.cat(categorical_features, dim=-1)

            # concatenate continuous and categorical features
            x = torch.cat([x_cont, x_cat], dim=-1)
        else:
            x = x_cont

        # encode
        x_init = self.init_net(x)
        z_e = self.transformer_encoder(x_init, q_mask=x_mask)        

        z_e_packed, _, _ = padded_to_packed(z_e, mask=x_mask)

        # add batch dim, vq code may break otherwise (with dropout)
        z_e_packed = z_e_packed.unsqueeze(0)

        z_q_packed, indices_packed, embedding_loss = \
            self.vector_quantization(z_e_packed)

        # resvq returns 2d tensor, vq returns 1d tensor
        # we generalize to 2d tensor; shape (n_valid_total, n_codebook)
        if indices_packed.dim() == 2:
            indices_packed = indices_packed.unsqueeze(-1)

        # take away the batch dim we added
        z_q_packed = z_q_packed.squeeze(0)
        indices_packed = indices_packed.squeeze(0)

        z_q = packed_to_padded(z_q_packed, mask=x_mask)
        indices = packed_to_padded(indices_packed, mask=x_mask)

        if return_kvmask:
            return z_q, indices, embedding_loss, x_mask
        return z_q, indices, embedding_loss  


    def encode(self, x_cont, x_categorical, x_mask, return_kvmask=False):
        '''
            - x_mask is in the physics space, kv_mask is in the token space
            - same in set mode, otherwise not
        '''
        if self.vae_type == 'mlp_bottleneck':
            return self.encode_mlp_bottleneck(x_cont, x_mask, return_kvmask)
        elif self.vae_type == 'cellimg_transformer':
            return self.encode_cellimg_transformer(x_cont, x_categorical, return_kvmask)
        elif self.vae_type == 'transformer':
            return self.encode_transformer(x_cont, x_categorical, x_mask, return_kvmask)


    def decode_cellimg(self, z_q):
        if hasattr(self, 'pos_emb_dec'):
            z_q = z_q + self.pos_emb_dec

        z_q = self.decoder(z_q)
        bs, n_tokens, nc = z_q.shape
        assert n_tokens == self.cumsum_num_tokens[-1], \
            f"Expected {self.cumsum_num_tokens[-1]} tokens, got {n_tokens}"

        x_hat_cont = {}
        for i in range(self.num_cell_layers):
            start = self.cumsum_num_tokens[i]
            end = self.cumsum_num_tokens[i + 1]
            length = end - start
            h = w = int(np.sqrt(length))

            z_q_i = z_q[:, start:end, :].transpose(1, 2) # (B, C, H*W)
            z_q_i = z_q_i.view(bs, nc, h, w) # (B, C, H, W)
            z_up_i = self.upsamplers[i](z_q_i)
            z_pred_i = self.output_nets_cont[i](z_up_i)
            z_pred_ind_i = self.output_net_ind[i](z_up_i)
            z_pred_i = torch.cat([z_pred_i, z_pred_ind_i], dim=1)

            x_hat_cont[f'feat0_{i}'] = z_pred_i.permute(0, 2, 3, 1)  # (B, H, W, C)

        return x_hat_cont


    def decode(self, z_q, x_mask=None):
        if self.vae_type == 'cellimg_transformer':
            return self.decode_cellimg(z_q)

        # optional transformer decoder
        if hasattr(self, 'decoder'):
            z_q = self.decoder(z_q, q_mask=x_mask)

        x_hat_cont = self.output_net_cont(z_q)

        x_hat_categorical = {}
        if len(self.categorical_vars) != 0:
            for name, output_net in self.output_nets_categorical.items():
                x_hat_categorical[name] = output_net(z_q)

        return x_hat_cont, x_hat_categorical


    def forward(self, x_cont, x_categorical, x_mask):
        z_q, indices, embedding_loss = self.encode(
            x_cont, x_categorical, x_mask)
        x_hat = self.decode(z_q, x_mask)
        return embedding_loss, x_hat, indices


    def indices_to_zq(self, indices, mask):
        '''
            - Called by 4M
            - indices: (B, n_tokens, 1) | (B, n_tokens) for mlp_bottleneck
            - mask: (B, n_tokens)
        '''

        # assert that we are inference mode
        assert self.vector_quantization.training is False, \
            "Inference mode only"        

        if self.vae_type == 'mlp_bottleneck':
            z_q = self.vector_quantization.indices_to_embedding(indices)
            return z_q

        indices_packed, _, _ = padded_to_packed(indices, mask=mask)

        # All-empty batch (no predicted tokens for this modality across the whole
        # batch): get_output_from_indices' einx solver fails on n=0
        # ("Axis 'n' has value 0 <= 0"). This happens early in training and
        # for any inference event that emits zero particles. Skip the VQ call and
        # return the all-zeros padded tensor packed_to_padded would produce; the
        # all-False mask discards the values downstream anyway. Derive the output
        # dim from a cheap dummy lookup so we don't assume VQ internals.
        if indices_packed.shape[0] == 0:
            q = indices_packed.shape[-1]
            dummy = torch.zeros(1, 1, q, dtype=indices_packed.dtype,
                                device=indices_packed.device)
            dummy_out = self.vector_quantization.get_output_from_indices(
                dummy).squeeze(0)  # (1, d_out)
            z_q_packed = dummy_out.new_zeros((0, dummy_out.shape[-1]))
            return packed_to_padded(z_q_packed, mask=mask)

        indices_packed = indices_packed.unsqueeze(0)

        z_q_packed = self.vector_quantization.get_output_from_indices(
            indices_packed).squeeze(0)

        z_q = packed_to_padded(z_q_packed, mask=mask)
        return z_q
