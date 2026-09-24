import torch
from torch import Tensor
from torch import nn
from .attention import MultiheadAttentionVarLen




class GLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, 
            activation = "SiLU", dropout = 0.0, bias = True, gated = False):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)



class LayerScale(nn.Module):
    """Applies the LayerScale operation from the Cait vision transformer.

    Effective at improving stability and speed of deep transformers.
    Now the standard for vision transformers
    https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma



class TLayer(nn.Module):
    def __init__(self, embed_dim, mha_config, dense_config=None, layer_scale=False):
        super().__init__()
        self.embed_dim = embed_dim

        self.mha = MultiheadAttentionVarLen(embed_dim, **mha_config)

        if dense_config:
            dense_config['embed_dim'] = embed_dim
            self.dense = GLU(**dense_config)
        else:
            self.register_buffer("dense", None)

        self.norm1 = nn.LayerNorm(embed_dim)

        if layer_scale:
            self.layer_scale_attn = LayerScale(embed_dim)
            self.layer_scale_mlp = LayerScale(embed_dim)


    def forward(self, q, q_mask=None, kv=None, kv_mask=None, attn_mask=None):
        '''
        Args:
            q: shape (bs, n_q, embed_dim)
            q_mask: True for valid, False for fake
            kv: shape (bs, n_kv, embed_dim)
            kv_mask: True for valid, False for fake
            attn_mask: shape (bs, n_q, n_kv)
        '''
        q_attn = self.mha(q=q, q_mask=q_mask, kv=kv, kv_mask=kv_mask, attn_mask=attn_mask)

        if hasattr(self, 'layer_scale_attn'):
            q_attn = self.layer_scale_attn(q_attn)

        q = q + q_attn
        q = self.norm1(q)
        
        if self.dense:
            q_mlp = self.dense(q)
            if hasattr(self, 'layer_scale_mlp'):
                q_mlp = self.layer_scale_mlp(q_mlp)
            q = q + q_mlp

        return q



class TEncoder(nn.Module):
    def __init__(
        self, embed_dim, num_layers, mha_config,
        dense_config=None, out_dim=0, layer_scale=False,
        apply_norm_after_last_proj=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.apply_norm_after_last_proj = apply_norm_after_last_proj

        self.layers = nn.ModuleList(
            [TLayer(
                embed_dim, mha_config, dense_config, layer_scale=layer_scale,
            ) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.out_dim:
            self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

        if self.apply_norm_after_last_proj:
            norm_dim = self.out_dim if self.out_dim else self.embed_dim
            self.norm_after_last_projection = nn.LayerNorm(norm_dim)


    def forward(self, q, q_mask=None, attn_mask=None):
        for layer in self.layers:
            q = layer(q, q_mask=q_mask, attn_mask=attn_mask)
        q = self.final_norm(q)

        # Optinal resizing layer
        if self.out_dim:
            q = self.final_linear(q)

        if self.apply_norm_after_last_proj:
            q = self.norm_after_last_projection(q)
        return q



class TDecoder(nn.Module):
    def __init__(self, 
        embed_dim, 
        num_layers, 
        mha_config_cross,
        mha_config_self,
        dense_config=None, 
        out_dim=0,
        layer_scale=False,
        reverse: bool = False, # if true, CA is applied before self-attention
        apply_norm_after_last_proj: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.reverse = reverse
        self.apply_norm_after_last_proj = apply_norm_after_last_proj

        self.ca_layers = nn.ModuleList(
            [TLayer(
                embed_dim, mha_config_cross, dense_config, layer_scale=layer_scale,
            ) for _ in range(num_layers)]
        )

        self.sa_layers = nn.ModuleList(
            [TLayer(
                embed_dim, mha_config_self, dense_config, layer_scale=layer_scale,
            ) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.out_dim:
            self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

        if self.apply_norm_after_last_proj:
            norm_dim = self.out_dim if self.out_dim else self.embed_dim
            self.norm_after_last_projection = nn.LayerNorm(norm_dim)


    def forward(self, q, kv, q_mask=None, kv_mask=None, 
            cross_attn_mask=None, self_attn_mask=None):

        # apply cross-attention first if reverse is True
        if self.reverse:
            for ca_layer, sa_layer in zip(self.ca_layers, self.sa_layers):
                q = ca_layer(q=q, q_mask=q_mask, kv=kv, kv_mask=kv_mask, attn_mask=cross_attn_mask)
                q = sa_layer(q=q, q_mask=q_mask, attn_mask=self_attn_mask)
        
        else:
            for ca_layer, sa_layer in zip(self.ca_layers, self.sa_layers):
                q = sa_layer(q=q, q_mask=q_mask, attn_mask=self_attn_mask)
                q = ca_layer(q=q, q_mask=q_mask, kv=kv, kv_mask=kv_mask, attn_mask=cross_attn_mask)

        q = self.final_norm(q)

        # Optional resizing layer
        if self.out_dim:
            q = self.final_linear(q)

        if self.apply_norm_after_last_proj:
            q = self.norm_after_last_projection(q)

        return q