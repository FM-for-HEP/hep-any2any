import torch
import torch.nn as nn
from .helpers.utils import build_1d_sincos_posemb


class TokenEncoder4M(nn.Module):
    def __init__(self, modality, num_quantizers, embedding_dim,
            vocab_size, max_sincos_posemb, max_len, token_comb_type='sum',
            num_gpos_quantizers=0, gpos_vocab_size=-1, pos_emb_type='sincos', init_std=0.02):
        '''
        modality: string, name of the modality
        num_quantizers: int, number of quantizers
        num_gpos_quantizers: int, number of geometric position quantizers
        vocab_size: int, size of the vocabulary
        max_len: int, maximum allowed length for the input sequence
        embedding_dim: int, embedding dimension for the 4M model
        token_comb_type: 'sum' (the only option): the codebook embeddings of an element
            are summed
        max_sincos_posemb: int, maximum allowed length for sin-cos positional embedding
        pos_emb_type: 'sincos' (the only option): fixed sin-cos positional embedding
        '''
        super().__init__()
        if token_comb_type != 'sum':
            raise ValueError(f"token_comb_type must be 'sum', got {token_comb_type!r}")
        if pos_emb_type != 'sincos':
            raise ValueError(f"pos_emb_type must be 'sincos', got {pos_emb_type!r}")
        self.modality = modality
        self.num_quantizers = num_quantizers
        self.num_gpos_quantizers = num_gpos_quantizers

        # modality embedding
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.mod_emb, std=init_std)

        # positional embedding (token's pos in the sequence)
        if max_len > max_sincos_posemb:
            raise ValueError(f"max_len ({max_len}) cannot be greater than max_sincos_posemb ({max_sincos_posemb})")
        pos_emb = build_1d_sincos_posemb(
            max_len=max_sincos_posemb, embed_dim=embedding_dim)[:, :max_len]
        self.register_buffer('pos_emb', pos_emb)

        # token embedding
        self.token_emb = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for _ in range(num_quantizers)])

        # positional token embedding
        if num_gpos_quantizers > 0:
            self.pos_token_emb = nn.ModuleList([
                nn.Embedding(gpos_vocab_size, embedding_dim) for _ in range(num_gpos_quantizers)])


    def _seq_pos_emb(self, mask):
        pos = torch.cumsum(mask, dim=-1) - 1 # start from 0
        pos = torch.clamp(pos, min=0, max=None)
        pos_emb_expanded = self.pos_emb.expand(pos.size(0), -1, -1)
        return pos_emb_expanded.gather(
            1, pos.unsqueeze(-1).expand(-1, -1, self.pos_emb.size(-1)))


    def _token_sum(self, tokens):
        return torch.sum(torch.stack(
            [self.token_emb[i](tokens[..., i]) for i in range(self.num_quantizers)], dim=0), dim=0)


    def _pos_token_sum(self, pos_tokens):
        return torch.sum(torch.stack(
            [self.pos_token_emb[i](pos_tokens[..., i]) for i in range(self.num_gpos_quantizers)],
            dim=0), dim=0)


    def encode_input_tokens(self, batch):
        '''
        Args:
            batch: dict with 'tokens', 'kv_mask' and, for modalities with position
                tokens, 'pos_tokens'
        Returns:
            {'x+emb': token embedding + modality embedding (+ position-token embedding)}
        '''
        x = self._token_sum(batch['tokens']) + self.mod_emb
        if self.num_gpos_quantizers > 0:
            x = x + self._pos_token_sum(batch['pos_tokens'])
        return {'x+emb': x}


    def init_output_tokens(self, q_mask):
        '''
        Args:
            q_mask: torch.Tensor, shape (bs, max_len)
        '''
        return self.mod_emb + self._seq_pos_emb(q_mask)
