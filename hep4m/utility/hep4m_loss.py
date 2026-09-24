import torch
import torch.nn as nn


class Seq2SeqLoss(nn.Module):
    def __init__(self, config, codebook_loss_wts, data_type, wt=1.0):
        super().__init__()
        codebook_loss_wts = torch.tensor(codebook_loss_wts)
        self.register_buffer('codebook_loss_wts', codebook_loss_wts)

        self.token_wt = config['token_wt']
        self.pos_token_wt = config['pos_token_wt']
        self.card_wt = config['card_wt']
        self.config = config
        self.data_type = data_type
        self.wt = wt

    def forward(self, pred_dict, target_dict):
        '''
        Args:
            pred_dict: dict
                input_token_logits : (b, nt, n_codebook, codebook_size)
                cardinality_logits: (b, max_cardinality) # optional
                q_mask: (b, nt)
            target_dict:
                target_tokens: (b, nt, n_codebook)
        '''
        input_token_logits = pred_dict['token_logits']
        b, _, nc, cs = input_token_logits.size()

        # pos tokens
        if 'gpos_token_logits' in pred_dict:
            input_gpos_token_logits = pred_dict['gpos_token_logits']
            _, _, npc, pcs = input_gpos_token_logits.size()

        # 1. token loss
        # token loss on the real tokens (q_mask=True)
        # CELoss input.shape (q_mask.sum(), codebook_size)
        # CELoss target.shape (q_mask.sum(),)
        real_mask = target_dict['q_mask']
        token_loss_per_codebook = nn.CrossEntropyLoss(reduction='none',
                label_smoothing=self.config.get('label_smoothing', 0.0))(
            input_token_logits[real_mask].view(-1, cs), 
            target_dict['tokens'][real_mask].view(-1))

        if 'gpos_token_logits' in pred_dict:
            real_pos_mask = target_dict['q_pos_mask']
            pos_token_loss_per_codebook = nn.CrossEntropyLoss(reduction='none',
                    label_smoothing=self.config.get('pos_label_smoothing', 0.0))(
                input_gpos_token_logits[real_pos_mask].view(-1, pcs),
                target_dict['pos_tokens'][real_pos_mask].view(-1))

        token_loss_per_codebook = token_loss_per_codebook.view(-1, nc).mean(dim=0)
        token_loss = self.token_wt * \
            (token_loss_per_codebook * self.codebook_loss_wts).sum() / \
            self.codebook_loss_wts.sum()

        # 3. cardinality loss
        cardinality_loss = 0
        if 'cardinality_logits' in pred_dict:
            cardinality_loss = self.card_wt * nn.CrossEntropyLoss(reduction='mean',
                    label_smoothing=self.config.get('card_label_smoothing', 0.0))(
                pred_dict['cardinality_logits'], target_dict['cardinality'])

        # 4. combine losses
        loss = token_loss + cardinality_loss

        return_dict = {
            'token_loss': token_loss.item()
        }
        if 'cardinality_logits' in pred_dict:
            return_dict['cardinality_loss'] = cardinality_loss.item()

        for i in range(nc):
            return_dict[f'codebook_{i}_loss'] = token_loss_per_codebook[i].item()

        if 'gpos_token_logits' in pred_dict:
            pos_token_loss = self.pos_token_wt * pos_token_loss_per_codebook.mean()
            loss = loss + pos_token_loss
            return_dict['pos_token_loss'] = pos_token_loss.item()
            for i in range(npc):
                return_dict[f'pos_codebook_{i}_loss'] = \
                    self.pos_token_wt * pos_token_loss_per_codebook[i].item()

        if self.wt != 1.0:
            loss = loss * self.wt
            for k in return_dict:
                return_dict[k] = return_dict[k] * self.wt

        return loss, return_dict
