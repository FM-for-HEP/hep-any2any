"""nanoHEP decoder-only transformer (GPT).

Adapted from nanoGPT (https://github.com/karpathy/nanoGPT, model.py). Changes: an
incremental key/value cache for autoregressive decoding (``past_kv_list``,
``return_kv_list``) and ``forward_hidden``, which returns the hidden states without
the output projection.
"""
# nanoGPT: Copyright (c) 2022 Andrej Karpathy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import inspect
import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, past_kv=None, return_kv=False):
        # past_kv: optional (k_past, v_past) tensors of shape (B, n_head, T_past, head_dim).
        #   When provided, T must be 1 (single AR step) — the new query attends to
        #   the concatenation of past + current keys without any causal mask needed
        #   (one query, all earlier keys are valid).
        # return_kv: if True, also return the (k, v) tensors for caching across steps.
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if past_kv is not None:
            assert T == 1, "KV-cached forward only supports T=1 new tokens per call"
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=2)   # (B, nh, T_past + 1, hs)
            v = torch.cat([v_past, v], dim=2)
            is_causal = False  # single query naturally attends to all earlier keys
        else:
            is_causal = True

        new_kv = (k, v) if return_kv else None

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, return_kv=False):
        a_out, new_kv = self.attn(self.ln_1(x), past_kv=past_kv, return_kv=return_kv)
        x = x + a_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        logger.info("number of parameters: %.2fM", self.get_num_params() / 1e6)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, idx):
        """Return post-ln_f hidden states (B, T, n_embd) WITHOUT the lm_head
        projection. Lets the caller project only loss-mask positions through the
        36k-vocab lm_head — the all-position projection+CE is ~45% of the
        forward but only output positions contribute gradient. Exact: identical
        to gathering positions from full logits.
        """
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x, _ = block(x)
        return self.transformer.ln_f(x)

    def forward(self, idx, targets=None, past_kv_list=None, return_kv_list=False):
        # past_kv_list: optional list-per-layer of (k, v) tensors; each (B, n_head, T_past, head_dim).
        #   Position embeddings shift to start at T_past so absolute positions are preserved.
        # return_kv_list: if True, also return a list-per-layer of updated (k, v) tensors.
        device = idx.device
        b, t = idx.size()

        if past_kv_list is not None:
            T_past = past_kv_list[0][0].shape[2]
            assert t == 1, "KV-cached forward only supports t=1 new tokens per call"
        else:
            T_past = 0
        assert T_past + t <= self.config.block_size, (
            f"Cannot forward sequence of length {T_past + t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(T_past, T_past + t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_kv_list = [] if return_kv_list else None
        for li, block in enumerate(self.transformer.h):
            past_kv = past_kv_list[li] if past_kv_list is not None else None
            x, new_kv = block(x, past_kv=past_kv, return_kv=return_kv_list)
            if return_kv_list:
                new_kv_list.append(new_kv)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_kv_list:
            return logits, loss, new_kv_list
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Parameters with 2 or more dimensions (matmul weights, embeddings) are weight
        # decayed; biases and layer norms are not.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        logger.info("decayed parameter tensors: %d (%s parameters); non-decayed: %d (%s parameters)",
                    len(decay_params), f"{sum(p.numel() for p in decay_params):,}",
                    len(nodecay_params), f"{sum(p.numel() for p in nodecay_params):,}")
        # fused AdamW when available
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
