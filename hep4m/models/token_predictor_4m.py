import torch
import torch.nn as nn


class TokenPredictor4M(nn.Module):
    def __init__(self,
            modality, num_quantizers, vocab_size, input_dim,
            type='sequential3', num_gpos_quantizers=0, gpos_vocab_size=-1,
            data_type='set'):
        '''
        Args:
            modality: string, name of the modality
            num_quantizers: int, number of quantizers
            vocab_size: int, size of the vocabulary
            input_dim: int, embedding dimension
            type: 'sequential3' (the only type): one linear head per quantizer; the
                embedding of the previous quantizers' tokens (target tokens in training,
                sampled tokens otherwise) is added to the input of the next head
        '''
        super().__init__()
        if type != 'sequential3':
            raise ValueError(f"Unknown token predictor type: {type}")
        self.modality = modality
        self.num_quantizers = num_quantizers
        self.num_gpos_quantizers = num_gpos_quantizers
        self.vocab_size = vocab_size
        self.gpos_vocab_size = gpos_vocab_size

        self.input_dim = input_dim
        self.type = type
        self.data_type = data_type

        self.use_target_tokens = False

        self.predictor_list = nn.ModuleList([])
        self.seq3_emb = nn.ModuleList([])
        for i in range(num_quantizers):
            self.predictor_list.append(
                nn.Linear(input_dim, vocab_size, bias=False))
            if i < num_quantizers - 1:
                self.seq3_emb.append(nn.Embedding(vocab_size, input_dim))

        if num_gpos_quantizers > 0 and self.data_type != 'cell_image':
            self.pos_predictor = nn.Linear(
                input_dim, gpos_vocab_size*num_gpos_quantizers, bias=False)


    def set_use_target_tokens(self, use_target_tokens):
        '''
        Set whether the next-quantizer input uses the target tokens (training).
        '''
        self.use_target_tokens = use_target_tokens


    def sample_top_k(self, logits, k=1, temperature=1.0):
        original_shape = logits.shape[:-1]

        if k == 1 and temperature == 0:  # pure argmax
            return logits.argmax(dim=-1)

        flat = logits.view(-1, logits.size(-1))
        flat = flat / max(temperature, 1e-8)

        if k == -1:  # sample from full (tempered) posterior
            probs = torch.softmax(flat, dim=1)
            return torch.multinomial(probs, 1).squeeze(1).view(original_shape)

        top_k_vals, top_k_idx = torch.topk(flat, k, dim=1)
        top_k_probs = torch.softmax(top_k_vals, dim=1)  # softmax over survivors
        sampled = torch.multinomial(top_k_probs, 1).squeeze(1)
        sampled_card = top_k_idx.gather(1, sampled.unsqueeze(1)).squeeze(1)
        return sampled_card.view(original_shape)


    def forward(self, x, tokens=None, top_k_token=1, top_k_gpos_token=1, 
            temperature=1.0, temperature_gpos=1.0, get_logits=True):
        '''
        Args:
            x: torch.Tensor, shape (batch_size, seq_len, num_quantizers, input_dim)
            tokens: torch.Tensor, shape (batch_size, seq_len, num_quantizers)
                used for training with labels. q2 prediction should get q1 as input and so on
                during inference, the predicted q1 will be used as input for q2, and so on
            top_k_token: int, 1=argmax, >1=top-k sampling for token prediction
            top_k_gpos_token: int, 1=argmax, >1=top-k sampling for gpos token prediction
            temperature: float, temperature for sampling
            temperature_gpos: float, temperature for gpos sampling
            get_logits: bool, whether to return the logits or the sampled tokens
        Returns:
            pred_token_logits: torch.Tensor, shape (batch_size, seq_len, num_quantizers, vocab_size)
                or pred_tokens: torch.Tensor, shape (batch_size, seq_len, num_quantizers)
            pred_gpos_token_logits: torch.Tensor, shape (batch_size, seq_len, num_gpos_quantizers, gpos_vocab_size)
                or pred_gpos_tokens: torch.Tensor, shape (batch_size, seq_len, num_gpos_quantizers)
        '''

        # 1. positional token prediction
        pred_gpos_tokens = None; pred_gpos_tokens_logits = None
        if hasattr(self, 'pos_predictor'):

            if self.data_type == 'global':
                x_pos = x.mean(dim=1, keepdim=True)  # (b, 1, dim)
                pred_gpos_tokens_logits = self.pos_predictor(x_pos).view(
                    x_pos.shape[0], x_pos.shape[1], self.num_gpos_quantizers, self.gpos_vocab_size)
            else:
                pred_gpos_tokens_logits = self.pos_predictor(x).view(
                    x.shape[0], x.shape[1], self.num_gpos_quantizers, self.gpos_vocab_size)

            if not get_logits:
                pred_gpos_tokens = self.sample_top_k(
                    pred_gpos_tokens_logits, k=top_k_gpos_token, temperature=temperature_gpos)
                    
        # 2. token prediction
        pred_token_logits = []
        prev_argmaxed_logits_emb = torch.zeros_like(x)
        pred_tokens = []

        for i in range(self.num_quantizers):
            inp = x + prev_argmaxed_logits_emb
            logits = self.predictor_list[i](inp)
            pred_token_logits.append(logits)

            # training with target tokens
            if self.use_target_tokens:
                argmaxed_logits = tokens[:, :, i]
            else:
                argmaxed_logits = self.sample_top_k(logits, k=top_k_token, temperature=temperature)
            pred_tokens.append(argmaxed_logits)

            if i < self.num_quantizers - 1:
                argmaxed_logits_emb = self.seq3_emb[i](argmaxed_logits)
                prev_argmaxed_logits_emb = prev_argmaxed_logits_emb + argmaxed_logits_emb

        if get_logits:
            pred_token_logits = torch.stack(pred_token_logits, dim=-2)
            return pred_token_logits, pred_gpos_tokens_logits
        pred_tokens = torch.stack(pred_tokens, dim=-1)
        return pred_tokens, pred_gpos_tokens
