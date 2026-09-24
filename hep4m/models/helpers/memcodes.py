import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import EinMix as Mix


class Memcodes(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        heads = 1,
        temperature = 1.,
        **kwargs,
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.temperature = temperature
        self.codebook_size = codebook_size

        num_codebooks = heads
        codebook_dim = dim // heads

        self.codes = nn.Parameter(torch.randn(num_codebooks, codebook_size, codebook_dim))
        self.to_k = Mix('h n d -> h n c', weight_shape = 'h d c', h = heads, d = codebook_dim, c = codebook_dim)
        self.to_v = Mix('h n d -> h n c', weight_shape = 'h d c', h = heads, d = codebook_dim, c = codebook_dim)

        # needed for compatibility with other quantizers
        # heads are like codebooks (no they are NOT)
        self.num_quantizers = 1 # self.heads


    def indices_to_embedding(self, indices):
        batch = indices.shape[0]

        values = self.to_v(self.codes)
        values = repeat(values, 'h n d -> b h n d', b = batch)

        indices = repeat(indices, '... -> ... d', d = values.shape[-1]).squeeze(2)

        out = values.gather(2, indices.unsqueeze(2))
        return rearrange(out, 'b h n d -> b n (h d)')


    def forward(self, x):
        '''
        Args:
            x: (b, n, d)
        '''
        assert x.shape[-1] == self.dim

        # split out heads
        q = rearrange(x, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # get key / values of codes
        k, v = self.to_k(self.codes), self.to_v(self.codes)

        # straight through gumbel softmax
        logits = einsum('b h i d, h j d -> b h i j', q, k)

        if self.training:
            attn = F.gumbel_softmax(logits, tau = self.temperature, dim = -1, hard = True)
            codebook_indices = attn.argmax(dim = -1)
        else:
            codebook_indices = logits.argmax(dim = -1)
            attn = F.one_hot(codebook_indices, num_classes = self.codebook_size).float()

        if self.heads == 1:
            codebook_indices = codebook_indices.squeeze(1)

        out = einsum('b h i j, h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Dummy codebook loss for compatibility with other types of quantizers
        codebook_loss = torch.tensor([0.], device=x.device)

        return out, codebook_indices, codebook_loss