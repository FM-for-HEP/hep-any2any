import logging
import torch
from torch import nn
import torch.nn.functional as F
from .utils import padded_to_packed, packed_to_padded

log = logging.getLogger(__name__)


class MultiheadAttentionVarLen(nn.Module):
    def __init__(self, 
        embed_dim,
        num_heads,
        enable_flash_attn=False,
        enable_flex_attn=False,  # accepted for config compatibility; must be False
        bias=False,
        dropout=0.0,
        do_qkv_norm=True,
    ):
        super().__init__()

        # Check that the dimension of each heads makes internal sense
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.enable_flash_attn = enable_flash_attn
        if enable_flex_attn:
            raise ValueError("enable_flex_attn is not supported; set it to false")
        self.bias = bias
        self.dropout = dropout
        self.do_qkv_norm = do_qkv_norm
        self.track_metrics = False
        self.metrics_buffer = {}

        # Better parallelism for self-attention when using parameters directly
        self.kqv_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.kqv_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        if self.do_qkv_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
            self.v_norm = nn.RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

        # to flash or not to flash
        if self.enable_flash_attn:
            if not torch.cuda.is_available():
                log.debug("flash attention requires CUDA; using torch attention")
                self.enable_flash_attn = False
            elif torch.cuda.get_device_capability()[0] < 8:
                log.debug("flash attention requires compute capability >= 8.0; using torch attention")
                self.enable_flash_attn = False
            else:
                try:
                    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
                    self.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
                    self.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
                except ImportError:
                    log.debug("flash_attn not installed; using torch attention")
                    self.enable_flash_attn = False


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kqv_weight)
        if self.bias:
            nn.init.constant_(self.kqv_bias, 0.0)
        self.out_proj.reset_parameters()


    def forward(self, q, q_mask=None, kv=None, kv_mask=None, attn_mask=None):
        '''
        Args:
            q: query tensor
            q_mask: True for valid, False for fake
            kv: key/value tensor (cross attention)
            kv_mask: True for valid, False for fake
            attn_mask: (bs, n_q, n_kv)
        '''
        if self.enable_flash_attn:            
            if attn_mask is None:
                if kv is None:
                    return self.forward_flash_self_attn(q=q, q_mask=q_mask)
                else:
                    return self.forward_flash_cross_attn(
                        q=q, kv=kv, q_mask=q_mask, kv_mask=kv_mask)


        return self.forward_torch(
            q=q, q_mask=q_mask, kv=kv, kv_mask=kv_mask, attn_mask=attn_mask)


    def forward_flash_self_attn(self, q, q_mask=None):
        '''
        Args:
            q: query tensor
            q_mask: True for valid, False for fake
        '''

        if q_mask is None:
            # True for all tokens (all tokens are valid)
            q_mask = torch.full(q.shape[:-1], True, dtype=torch.bool, device=q.device)
        q_packed, culens, maxlen = padded_to_packed(q, q_mask)

        # compute qkv
        qkv = F.linear(q_packed, self.kqv_weight, self.kqv_bias)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

        if self.do_qkv_norm:
            dtype = qkv.dtype
            q, k, v = qkv.unbind(1)
            with torch.amp.autocast('cuda', enabled=False):
                q = self.q_norm(q.float()).to(q.dtype)
                k = self.k_norm(k.float()).to(k.dtype)
                v = self.v_norm(v.float()).to(v.dtype)
            qkv = torch.stack([q, k, v], dim=1).to(dtype)

        # Convert to bfloat16 for the flash-varlen backend
        qkv_orig_dtype = qkv.dtype
        qkv = qkv.to(torch.bfloat16) if qkv_orig_dtype != torch.bfloat16 else qkv

        # Run the flash-varlen backend
        dropout = self.dropout if self.training else 0.0
        a_out = self.flash_attn_varlen_qkvpacked_func(qkv, culens, maxlen, dropout)
        a_out = a_out.reshape(-1, self.embed_dim)

        # Convert back to the original dtype (inference)
        if not torch.is_autocast_enabled() and a_out.dtype != qkv_orig_dtype:
            a_out = a_out.to(qkv_orig_dtype)

        # Mix with final linear layer
        a_out = self.out_proj(a_out)

        # Convert back to the original dtype (training)
        if torch.is_autocast_enabled() and a_out.dtype != qkv_orig_dtype:
            a_out = a_out.to(qkv_orig_dtype)

        # unpack the output
        a_out = packed_to_padded(a_out, q_mask)

        return a_out


    def forward_flash_cross_attn(self, q, kv, q_mask=None, kv_mask=None):
        '''
        Args:
            q: query tensor
            kv: key/value tensor (cross attention)
            q_mask: True for valid, False for fake
            kv_mask: True for valid, False for fake
        '''
        device = q.device
        if q_mask is None:
            q_mask = torch.full(q.shape[:-1], True, dtype=torch.bool, device=device)
        if kv_mask is None:
            kv_mask = torch.full(kv.shape[:-1], True, dtype=torch.bool, device=device)

        # Pack Q and KV independently (varlen)
        q_packed, cu_q, max_q = padded_to_packed(q, q_mask)
        kv_packed, cu_k, max_k = padded_to_packed(kv, kv_mask)

        # projections
        q_proj, k_proj, v_proj = self.get_qkv_projections(
            q_packed, self.kqv_weight, self.kqv_bias, kv=kv_packed)

        # Reshape to heads
        H, D = self.num_heads, self.head_dim
        q_proj = q_proj.view(-1, H, D)   # [Tq, H, D]
        k_proj = k_proj.view(-1, H, D)   # [Tk, H, D]
        v_proj = v_proj.view(-1, H, D)   # [Tk, H, D]
        kv_proj = torch.stack([k_proj, v_proj], dim=1)  # [Tk, 2, H, D]

        # Optional per-head norms (keep dtype consistency)
        if getattr(self, "do_qkv_norm", False):
            dtype = q_proj.dtype
            k_proj, v_proj = kv_proj.unbind(dim=1)
            with torch.amp.autocast('cuda', enabled=False):
                q_proj = self.q_norm(q_proj.float()).to(q_proj.dtype)
                k_proj = self.k_norm(k_proj.float()).to(k_proj.dtype)
                v_proj = self.v_norm(v_proj.float()).to(v_proj.dtype)
            kv_proj = torch.stack([k_proj, v_proj], dim=1).to(dtype)

        q_orig_dtype = q_proj.dtype
        if q_orig_dtype != torch.bfloat16:
            q_proj = q_proj.to(torch.bfloat16)
            kv_proj = kv_proj.to(torch.bfloat16)

        dropout_p = self.dropout if self.training else 0.0
        a_out = self.flash_attn_varlen_kvpacked_func(
            q=q_proj, kv=kv_proj, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k, dropout_p=dropout_p)
        a_out = a_out.reshape(-1, self.num_heads * self.head_dim)

        if not torch.is_autocast_enabled() and a_out.dtype != q_orig_dtype:
            a_out = a_out.to(q_orig_dtype)

        # Output projection to model dim
        a_out = self.out_proj(a_out)

        # Cast back to original dtype (match Q path)
        if torch.is_autocast_enabled() and a_out.dtype != q_orig_dtype:
            a_out = a_out.to(q_orig_dtype)

        # Unpack to padded using the Q mask's shape
        a_out = packed_to_padded(a_out, q_mask)
        
        return a_out


    def get_qkv_projections(self, q, weight, bias=None, kv=None):
        if kv is None: # self attention
            return F.linear(q, self.kqv_weight, self.kqv_bias).chunk(3, dim=-1)
        else: # cross attention
            dim = q.size(-1)
            w_q, w_kv = weight.split([dim, dim * 2])
            b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)

            q_proj = F.linear(q, w_q, b_q)
            k_proj, v_proj = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
            return q_proj, k_proj, v_proj


    def get_attn_mask(self, q_shape, kv_mask=None, attn_mask=None):
        ''' 
        attn_mask for torch.nn.functional.scaled_dot_product_attention()
        padded tensors do not send info, but they can receive them
        Args:
            kv_mask: (bs, n_kv) | None
                in case of self attention, this is the same as q_mask
            q_shape: (bs, n_q, n_feat) | None
            attn_mask: (bs, n_q, n_kv) | None
        Returns:
            mask: (bs, n_q, n_kv)
        '''
        mask = None
        if kv_mask is not None:
            mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)

        if attn_mask is not None:
            mask = attn_mask if mask is None else mask & attn_mask

        # for multihead attention
        if mask is not None:
            mask = mask.unsqueeze(1)

        return mask


    def forward_torch(self, q, q_mask=None, kv=None, kv_mask=None, attn_mask=None):
        '''
        Args:
            q: normalized query tensor
            q_mask: True for valid, False for fake
            kv: key/value tensor (cross attention)
            kv_mask: True for valid, False for fake
            attn_mask: (bs, n_q, n_kv)
        '''

        bs, n_nodes, emb_dim = q.shape

        # compute q, k, v; shape = (bs, n_q|n_kv, emb_dim)
        q, k, v = self.get_qkv_projections(
            q=q, kv=kv, weight=self.kqv_weight, bias=self.kqv_bias)

        # transform tensors to (bs, n_head, n_q|n_kv, head_dim)
        shape = (bs, -1, self.num_heads, self.head_dim)  # Dont use S for cross attn
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        if self.do_qkv_norm:
            with torch.amp.autocast('cuda', enabled=False):
                q = self.q_norm(q.float()).to(q.dtype)
                k = self.k_norm(k.float()).to(k.dtype)
                v = self.v_norm(v.float()).to(v.dtype)

        # debugging logging of attention entropy
        if getattr(self, 'track_metrics', False):
            with torch.no_grad():
                q_d, k_d = q.detach(), k.detach()
                
                # 1. Monitor Input Health (Q Norm)
                # If this keeps going up (e.g. > 20), your LayerNorms are failing
                q_norm = torch.norm(q_d, p=2, dim=-1).mean().item()

                # 2. Monitor Max Score (Hardware Safety)
                scale = q_d.size(-1) ** -0.5
                attn_scores = torch.matmul(q_d, k_d.transpose(-2, -1)) * scale
                                
                if attn_mask is not None:
                    attn_mask_broad = attn_mask.unsqueeze(1)                     
                    attn_scores = attn_scores.masked_fill(attn_mask_broad == 0, float('-inf'))
                
                # The single most dangerous number in BF16 training:
                max_score = attn_scores.max().item()

                # 3. Monitor Entropy (Collapse)
                attn_probs = attn_scores.softmax(dim=-1)
                entropy = -(attn_probs * (attn_probs + 1e-9).log()).sum(dim=-1).mean().item()

                # Save all to buffer
                self.metrics_buffer = {
                    "entropy": entropy,
                    "max_score": max_score,
                    "q_norm": q_norm
                }

        # run attention
        attn_mask = self.get_attn_mask(
            kv_mask=q_mask if kv is None else kv_mask, # who sends messages
            q_shape=q.shape, attn_mask=attn_mask)
        dropout = self.dropout if self.training else 0.0    
        a_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout)

        # recombine heads
        a_out = a_out.transpose(1, 2).contiguous().view(bs, n_nodes, emb_dim)

        # Mix with final linear layer
        a_out = self.out_proj(a_out)
        
        return a_out
