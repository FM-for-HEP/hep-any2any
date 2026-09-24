import torch
from torch.nn.functional import pad



def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d
    taken from the 4M codebase
    Returns positional embedding of shape (1, N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32) # Shape (N,)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/2,)
    omega = 1. / (temperature ** omega)
    out = torch.einsum('n,d->nd', [arange, omega]) # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0) # Shape (1, N, D)
    return pos_emb


def padded_to_packed(seq, mask):
    # mask: True for valid tokens
    seqlens = mask.sum(dim=-1)
    maxlen = seqlens.max() # .item()
    culens = pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seq[mask], culens, maxlen


def packed_to_padded(unpadded_seq, mask):
    # mask: True for valid tokens
    shape = (*mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[mask] = unpadded_seq
    return out
