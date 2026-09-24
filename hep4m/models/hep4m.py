import torch
import torch.nn as nn
import numpy as np
import copy

from .helpers.dense import Dense
from .helpers.transformer import TEncoder, TDecoder
from .pos_tokenizer import PosTokenizer
from .vqvae import VQVAE
from .token_encoder_4m import TokenEncoder4M
from .token_predictor_4m import TokenPredictor4M



class HEP4M(nn.Module):
    def __init__(self, config, config_mod_dict):
        '''
        Args:
            config: config for the 4M model
            config_mod_dict: config_ms for the modalities
        '''
        super().__init__()
        self.config = config
        self.config_mod_dict = config_mod_dict

        self.modalities = list(config_mod_dict.keys())
        
        self.n_max_token_dict = {}
        for modality in self.modalities:
            self.n_max_token_dict[modality] = \
                config_mod_dict[modality]['max_token_cardinality']

        # pre-trained VQ-VAE tokenizers
        self.tokenizers = nn.ModuleDict()
        for modality, config_m in config_mod_dict.items():
            self.tokenizers[modality] = VQVAE(config_m)
            self.tokenizers[modality].freeze()

        self.embedding_dim = config['embedding_dim']

        # token -> embedding (for each modality)
        embedder_config_base = config['token_encoder']
        embedder_config_base['embedding_dim'] = self.embedding_dim

        self.embedders = nn.ModuleDict()
        for modality, config_m in config_mod_dict.items():
            embedder_config_mod = copy.deepcopy(embedder_config_base)
            embedder_config_mod['modality'] = modality
            if config_m['gpos_quantization']:
                embedder_config_mod.update(config['gpos_token_info'])

            embedder_config_mod['num_quantizers'] = \
                self.tokenizers[modality].vector_quantization.num_quantizers
            embedder_config_mod['vocab_size'] = \
                self.tokenizers[modality].vector_quantization.codebook_size
            embedder_config_mod['max_len'] = self.n_max_token_dict[modality]

            self.embedders[modality] = TokenEncoder4M(**embedder_config_mod)

        # 4m transformer encoder and decoder
        self.transformer_encoder = TEncoder(**self.config['transformer_encoder'])
        self.transformer_decoder = TDecoder(**self.config['transformer_decoder'])

        # token predictor (for each modality)
        self.token_predictor = nn.ModuleDict()
        for modality in self.modalities:

            gpos_token_info = {}
            if config_mod_dict[modality]['gpos_quantization']:
                gpos_token_info = config['gpos_token_info']

            self.token_predictor[modality] = TokenPredictor4M(
                modality=modality,
                num_quantizers=self.tokenizers[modality].vector_quantization.num_quantizers,
                vocab_size=self.tokenizers[modality].vector_quantization.codebook_size,
                input_dim=self.embedding_dim,
                type=self.config['token_predictor']['type'],
                data_type=config_mod_dict[modality]['data_type'],
                **gpos_token_info,
            )

        # cardinality predictor (for each modality; auxiliary task)
        if self.config.get('cardinality_predictor', None) is not None:
            self.cardinality_predictor = nn.ModuleDict()
            for modality in self.modalities:
                if config_mod_dict[modality]['data_type'] in ['global', 'cell_image']:
                    continue
                card_config = copy.deepcopy(self.config['cardinality_predictor'])
                card_config['output_size'] = self.n_max_token_dict[modality] + 1
                self.cardinality_predictor[modality] = Dense(**card_config)

        self.pos_tokenizer = PosTokenizer()


    def set_use_target_tokens(self, use_target_tokens):
        '''
        Set whether the token predictors use the target tokens (training).
        '''
        for modality in self.modalities:
            self.token_predictor[modality].set_use_target_tokens(use_target_tokens)


    def get_output_modalities_separator_mask(self, output_modalities, q_mask_dict):
        lengths = [q_mask_dict[m].shape[1] for m in output_modalities]
        L_total = sum(lengths)
        
        mask = torch.zeros(L_total, L_total, dtype=torch.bool,
                        device=q_mask_dict[output_modalities[0]].device)
        start = 0
        for length in lengths:
            mask[start:start + length, start:start + length] = True
            start += length
        
        return mask.unsqueeze(0)  # (1, L, L)


    def set_input_output_modalities(self, input_modalities, output_modalities):
        '''
            - Used by encode_to_tokens and decode_from_tokens
            - fast_forward doesn't need this. 
                It figures out the input and output modalities from the tokens_dict
        '''
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities


    def encode_to_tokens(self, batch, modalities):
        ''' 
            - with pre-trained VQ-VAE tokenizers 
            - written for final inference
            - all elemtnts in the modality are either in the input or output
        '''
        # encode the input tokens
        # kv_mask is same as x_mask for set. different for cell images
        input_tokens_dict = {}
        for modality in modalities:
            x_cont = batch[modality][f'{modality}_feat0']
            x_cat = batch[modality][f'{modality}_feat0_categorical']
            x_mask = batch[modality][f'{modality}_feat0_mask']
            _, tokens, _, kv_mask = self.tokenizers[modality].encode(
                x_cont, x_cat, x_mask, return_kvmask=True)
            input_tokens_dict[modality] = {
                'tokens': tokens,
                'kv_mask': kv_mask,
            }

            if f'{modality}_feat0_gpos' in batch[modality].keys():
                pos_tokens = self.pos_tokenizer(
                    batch[modality][f'{modality}_feat0_gpos'])
                input_tokens_dict[modality]['pos_tokens'] = pos_tokens

        return input_tokens_dict


    def decode_from_tokens(self, tokens_dict):
        ''' with pre-trained VQ-VAE tokenizers '''
        return_dict = {}
        for modality in self.output_modalities:
            tokens = tokens_dict[modality]['tokens']

            z_q = self.tokenizers[modality].indices_to_zq(tokens, tokens_dict[modality]['q_mask'])
            x_hat = self.tokenizers[modality].decode(z_q, tokens_dict[modality]['q_mask'])

            return_dict[modality] = {
                'x_hat' : x_hat,
                'q_mask' : tokens_dict[modality]['q_mask'],
            }

            if 'gpos_tokens' in tokens_dict[modality].keys():
                gpos_tokens = tokens_dict[modality]['gpos_tokens']
                x_gpos_hat = self.pos_tokenizer.decode(gpos_tokens)
                return_dict[modality]['x_gpos_hat'] = x_gpos_hat

        return return_dict


    def sample_top_k(self, logits, k=-1, temperature=1.0):
        if k == 1 and temperature == 0:  # pure argmax
            return logits.argmax(dim=1)

        logits = logits / max(temperature, 1e-8)

        if k == -1:  # sample from full (tempered) posterior
            probs = torch.softmax(logits, dim=1)
            return torch.multinomial(probs, 1).squeeze(1)

        top_k_vals, top_k_idx = torch.topk(logits, k, dim=1)
        top_k_probs = torch.softmax(top_k_vals, dim=1)  # softmax over survivors
        sampled = torch.multinomial(top_k_probs, 1).squeeze(1)
        sampled_card = top_k_idx.gather(1, sampled.unsqueeze(1)).squeeze(1)
        return sampled_card
    

    def fast_forward(self, input_dict, output_modalities, q_mask_dict=None,
            get_logits=True, top_k_token_dict={}, top_k_gpos_token_dict={},
            temperature_token_dict={}, temperature_gpos_token_dict={},
            use_truth_cardinality=False, target_dict=None, card_topk_dict={},
            card_temperature_dict={}):
        '''
        Args:
            use_truth_cardinality: ONLY for training
            target_dict: ONLY for training
        '''

        input_modalities = list(input_dict.keys())

        # 1.1 embed the tokens and cocatenate them
        input_4m = torch.cat([
            self.embedders[modality].encode_input_tokens(
                input_dict[modality])['x+emb'] \
                    for modality in input_modalities], dim=1)

        # 1.2 concatenate the masks
        input_mask_4m = torch.cat(
            [input_dict[modality]['kv_mask'] for modality in input_modalities], dim=-1)

        # 2. 4m transformer encoder (on the concatenated tokens)
        enc_4m = self.transformer_encoder(
            q=input_4m, q_mask=input_mask_4m)

        predicted_dict = {m: {} for m in output_modalities}

        # 3. cardinality predictor (for each modality)
        card_modalities = []
        if hasattr(self, 'cardinality_predictor'):
            card_modalities = [m for m in output_modalities if m in self.cardinality_predictor.keys()]
            if len(card_modalities) > 0:

                enc4m_global_numerator = torch.sum(
                    enc_4m * input_mask_4m.unsqueeze(-1).float(), dim=1)
                enc4m_global_denominator = 1e-8 + torch.sum(
                    input_mask_4m.float(), dim=1, keepdim=True)
                enc4m_global = enc4m_global_numerator / enc4m_global_denominator

                for modality in card_modalities:
                    card_pred_inp = enc4m_global + self.embedders[modality].mod_emb.squeeze(0)
                    card_pred = self.cardinality_predictor[modality](card_pred_inp)
                    predicted_dict[modality]['cardinality_logits'] = card_pred

        # 4.1 modify the q_masks based on the cardiniality 
        # (truth during training, predicted during inference)
        if use_truth_cardinality:
            assert q_mask_dict is not None, "q_mask_dict must be provided when use_truth_cardinality is True"
        else:
            q_mask_dict = {}
            for modality in output_modalities:
                if modality not in card_modalities:

                    # if no cardinality predictor for this modality, use max cardinality
                    q_mask_dict[modality] = torch.ones(
                        enc_4m.shape[0], self.n_max_token_dict[modality], dtype=torch.bool, device=enc_4m.device)
                    continue

                logits = predicted_dict[modality]['cardinality_logits']
                cardinality = self.sample_top_k(logits=logits,
                    k=card_topk_dict.get(modality, 1),
                    temperature=card_temperature_dict.get(modality, 1.0))
                max_cardinality = logits.shape[1] - 1
                q_mask_dict[modality] = torch.zeros(
                    logits.shape[0], max_cardinality, dtype=torch.bool, device=logits.device)
                indices = torch.arange(max_cardinality, device=logits.device).unsqueeze(0)  # (1, k)
                q_mask_dict[modality][indices < cardinality.unsqueeze(1)] = 1

        # 4.2 initialize the output tokens and concatenate them
        init_output_4m = torch.cat(
            [self.embedders[modality].init_output_tokens(q_mask_dict[modality])
             for modality in output_modalities], dim=1)

        # 4.3 concatenate the output token masks
        output_mask_4m = torch.cat(
            [q_mask_dict[modality] for modality in output_modalities], dim=-1)

        # 4.4 initialize the decoder self attention mask
        decoder_self_attn_mask = \
            self.get_output_modalities_separator_mask(
                output_modalities, q_mask_dict=q_mask_dict) & \
            output_mask_4m.unsqueeze(1) & output_mask_4m.unsqueeze(2)

        # 5. 4m transformer decoder (on the concatenated tokens)
        dec_4m = self.transformer_decoder(
            q=init_output_4m, q_mask=output_mask_4m,
            kv=enc_4m, kv_mask=input_mask_4m, self_attn_mask=decoder_self_attn_mask)

        # 6. token predictor (for each modality)
        split_indices = np.cumsum(
            [q_mask_dict[modality].shape[1] for modality in output_modalities])
        split_indices = split_indices[:-1].tolist()
        split_dec_4m = torch.tensor_split(dec_4m, split_indices, dim=1)

        for modality, dec_chunk in zip(output_modalities, split_dec_4m):

            # ONLY during training
            if self.token_predictor[modality].use_target_tokens:
                token_predictor_output = self.token_predictor[modality](
                    dec_chunk,
                    target_dict[modality]['tokens'],
                    top_k_token=top_k_token_dict.get(modality, 1),
                    top_k_gpos_token=top_k_gpos_token_dict.get(modality, 1),
                    temperature=temperature_token_dict.get(modality, 1.0),
                    temperature_gpos=temperature_gpos_token_dict.get(modality, 1.0),
                    get_logits=get_logits)
            else:
                token_predictor_output = self.token_predictor[modality](
                    dec_chunk,
                    top_k_token=top_k_token_dict.get(modality, 1),
                    top_k_gpos_token=top_k_gpos_token_dict.get(modality, 1),
                    temperature=temperature_token_dict.get(modality, 1.0),
                    temperature_gpos=temperature_gpos_token_dict.get(modality, 1.0),
                    get_logits=get_logits)

            if get_logits:
                pred_tokens_logits, pred_gpos_tokens_logits = token_predictor_output
                predicted_dict[modality]['token_logits'] = pred_tokens_logits
                if pred_gpos_tokens_logits is not None:
                    predicted_dict[modality]['gpos_token_logits'] = pred_gpos_tokens_logits

            else:
                pred_tokens, pred_gpos_tokens = token_predictor_output
                predicted_dict[modality]['tokens'] = pred_tokens
                if pred_gpos_tokens is not None:
                    predicted_dict[modality]['gpos_tokens'] = pred_gpos_tokens

            predicted_dict[modality]['q_mask'] = q_mask_dict[modality]

        return predicted_dict


    def forward(self, batch, top_k_token_dict, top_k_gpos_token_dict,
            temperature_token_dict, temperature_gpos_token_dict,
            use_truth_cardinality, card_topk_dict, card_temperature_dict):

        input_dict = self.encode_to_tokens(batch, self.input_modalities)

        predicted_token_dict = self.fast_forward(
            input_dict=input_dict,
            output_modalities=self.output_modalities,
            q_mask_dict=None, # inference; can't provide
            top_k_token_dict=top_k_token_dict,
            top_k_gpos_token_dict=top_k_gpos_token_dict, get_logits=False,
            temperature_token_dict=temperature_token_dict,
            temperature_gpos_token_dict=temperature_gpos_token_dict,
            use_truth_cardinality=use_truth_cardinality,
            card_topk_dict=card_topk_dict,
            card_temperature_dict=card_temperature_dict)
        output_dict = self.decode_from_tokens(predicted_token_dict)
        return output_dict
