import gc
import logging
import math
from typing import Union, Dict

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, DistributedSampler

from .dataset_tokens_numpy import TokenDatasetNumpy
from .dataset_modalities import get_dataset_class

log = logging.getLogger(__name__)


class COCOADatasetHEP4M(Dataset):
    def __init__(self, 
            filepath_dict: Dict[str, str],
            config_v_dict: Dict,
            start_idx: int =0,
            reduce_ds: int =-1,
            sampling_type: str ='inference',
            use_tokens: bool =False,
            fixed_input_output_modalities: Union[Dict[str, str], None] = None,
    ) -> None:
        '''
        COCOA dataset for HEP4M.
        Args:
            filepath_dict: {modality: filepath}; with use_tokens the token store file
                ({split}/{modality}_data.npy), otherwise raw ROOT file(s)
            config_v_dict: {modality: variable config}
            reduce_ds: int, -1 for no reduction
            sampling_type: str, sampling type for training
            use_tokens: bool, training: read the tokenised store; otherwise read raw
                values (inference, the model tokenises them)
            fixed_input_output_modalities: if provided, the masking will be done using this
        '''
        super().__init__()
        self.config_v_dict = config_v_dict
        self.reduce_ds = reduce_ds
        self.sampling_type = sampling_type
        self.use_tokens = use_tokens
        self.fixed_input_output_modalities = fixed_input_output_modalities

        self.modalities = list(config_v_dict.keys())

        self.ds_obj_dict = {} # dict of dataset objects
        shared_event_numbers = None

        for mi, (modality, config_v) in enumerate(config_v_dict.items()):

            # common event numbers across modalities (token store)
            if use_tokens and (mi == 0):
                ev_num_path = filepath_dict[modality].replace(
                    f'{modality}_data.npy', 'event_numbers.npy')
                shared_event_numbers = np.fromfile(ev_num_path, dtype='int64')
                if start_idx > 0:
                    shared_event_numbers = shared_event_numbers[start_idx:]
                if reduce_ds > 0:
                    shared_event_numbers = shared_event_numbers[:reduce_ds]

            log.info('loading modality %s', modality)
            if use_tokens:
                self.ds_obj_dict[modality] = TokenDatasetNumpy(
                    filepath_dict[modality], config_v,
                    start_idx=start_idx, reduce_ds=reduce_ds,
                    shared_event_numbers=shared_event_numbers)
            else:
                dataset_class = get_dataset_class(modality)
                self.ds_obj_dict[modality] = dataset_class(
                    filepath_dict[modality], config_v, reduce_ds)

        # raw inputs: every modality must list the same events in the same order
        if not use_tokens:
            ev_num_mod0 = ak.concatenate(self.ds_obj_dict[self.modalities[0]].data_dict['event_number'], axis=0)
            for i in range(1, len(self.modalities)):
                assert np.all(
                    ak.concatenate(self.ds_obj_dict[self.modalities[i]].data_dict['event_number'], axis=0) == \
                    ev_num_mod0), \
                    f"Event number mismatch between {self.modalities[i]} and {self.modalities[0]}"

        if use_tokens:
            if self.sampling_type == 'inference':
                assert self.fixed_input_output_modalities is not None
                self._input_modalities = self.fixed_input_output_modalities['input']
                self._output_modalities = self.fixed_input_output_modalities['output']

                self.fixed_is_inputs = [mod in self._input_modalities for mod in self.modalities]
                self.fixed_is_targets = [mod in self._output_modalities for mod in self.modalities]

            elif self.sampling_type != 'inference_rand2':
                raise ValueError(f"Unknown sampling type: {self.sampling_type}")

        self.n_events = len(self.ds_obj_dict[self.modalities[0]])
        gc.collect()
    

    def __len__(self):
        return self.n_events


    def set_valid_splits(self, valid_splits):
        self.valid_splits = valid_splits


    def __getitem__(self, idx):
        '''
        idx: int or tuple
            tuple if sampling_type is inference_rand2
        Options:
            - "use_tokens": training on tokens (output of VQ-VAE tokens are stored already)
                - "sampling_type":
                    - "inference": it's like generation. 
                            specific modalities are used for input and output
                            all tokens in inp are used; all tokens in target are masked
                    - "inference_rand2": it's like generation, 
                            but the input and output are slected randomly per __getitem__

            - not "use_tokens": training on physical values (4M will do tokenization as well)
        '''
        return_dict = {}

        # 1. if physical values
        if not self.use_tokens:
            for modality in self.modalities:
                return_dict[modality] = self.ds_obj_dict[modality][idx]
            return return_dict

        # 2. if using tokens
        idx, split_idx = idx
        if split_idx == -1:
            is_inputs, is_targets = self.fixed_is_inputs, self.fixed_is_targets
        else:
            is_inputs, is_targets = self.valid_splits[split_idx]

        for mi, modality in enumerate(self.modalities):
            return_dict[modality] = \
                self.ds_obj_dict[modality].getitem(
                    idx, 
                    is_input=is_inputs[mi],
                    is_target=is_targets[mi]
                )

        return  return_dict


def hep4m_collate_fn(samples):
    '''
    Use when input and output are constant for all samples in the batch
    '''

    # 1. discover which modalities are input/target (handles both fixed and variable)
    input_modalities = set()
    target_modalities = set()
    for sample in samples:
        for modality, data in sample.items():
            if data['inp_dict'] is not None:
                input_modalities.add(modality)
            if data['target_dict'] is not None:
                target_modalities.add(modality)
    
    modalities = list(input_modalities | target_modalities)
    input_modalities = list(input_modalities)
    target_modalities = list(target_modalities)
    
    # 2. batch size
    bs = len(samples)

    # 3. looping and collating
    input_dict = {}; target_dict = {}
    for modality in modalities:
        
        # 3.1 check if pos tokens are present for this modality
        has_pos_tokens = False
        for sample in samples:
            if sample[modality]['inp_dict'] is not None:
                has_pos_tokens = 'pos_tokens' in sample[modality]['inp_dict']
                break

        # 3.2 get num_quantizer; avoid zero size
        for sample in samples:
            d = sample[modality]['inp_dict'] or sample[modality]['target_dict']
            if d is not None and d['tokens'].size(0) > 0:
                num_quantizer = d['tokens'].size(1)
                if 'pos_tokens' in d:
                    has_pos_tokens = True
                    num_pos_quantizer = d['pos_tokens'].size(1)
                break

        # 3.3 get cardinalities and max
        # for target, need to compute separately for pos tokens
        # for input, pos tokens will be broadcasted if needed
        batch_input_cards = [x[modality]['inp_dict']['cardinality'] \
            if x[modality]['inp_dict'] is not None else 0 for x in samples]
        max_input_card = max(batch_input_cards)

        batch_target_cards = [x[modality]['target_dict']['cardinality'] \
            if x[modality]['target_dict'] is not None else 0 for x in samples]
        max_target_card = max(batch_target_cards)

        batch_pos_target_cards = [x[modality]['target_dict']['pos_cardinality'] \
            if x[modality]['target_dict'] is not None and has_pos_tokens else 0 for x in samples]
        max_pos_target_card = max(batch_pos_target_cards)

        # 3.4 zero init for the input and target; we will fill in the real values later
        if modality in input_modalities:
            input_dict[modality] = {
                'tokens': torch.zeros((bs, max_input_card, num_quantizer), dtype=torch.long),
                'kv_mask': torch.zeros(bs, max_input_card, dtype=torch.bool),
            }
            if has_pos_tokens:
                input_dict[modality]['pos_tokens'] = torch.zeros(
                    (bs, max_input_card, num_pos_quantizer), dtype=torch.long)

        # 3.5 zero init for the target; we will fill in the real values later
        if modality in target_modalities:
            target_dict[modality] = {
                'tokens': torch.zeros((bs, max_target_card, num_quantizer), dtype=torch.long),
                'q_mask': torch.zeros(bs, max_target_card, dtype=torch.bool),
            }
            if has_pos_tokens:
                target_dict[modality]['pos_tokens'] = torch.zeros(
                    (bs, max_pos_target_card, num_pos_quantizer), dtype=torch.long)
                target_dict[modality]['q_pos_mask'] = torch.zeros(bs, max_pos_target_card, dtype=torch.bool)

        # 3.6 loop through samples and fill in the real values for input and target
        for i, sample in enumerate(samples):
            if sample[modality]['inp_dict'] is not None:
                input_dict[modality]['tokens'][i, :batch_input_cards[i]] = \
                    sample[modality]['inp_dict']['tokens']
                input_dict[modality]['kv_mask'][i, :batch_input_cards[i]] = True
                
                if has_pos_tokens:
                    # for global modality, it will broadcast pos_tokens to all tokens shape
                    # (1, nq_pos) --> (tok_card, nq_pos)
                    input_dict[modality]['pos_tokens'][i, :batch_input_cards[i]] = \
                        sample[modality]['inp_dict']['pos_tokens']

            elif sample[modality]['target_dict'] is not None:
                target_dict[modality]['tokens'][i, :batch_target_cards[i]] = \
                    sample[modality]['target_dict']['tokens']
                target_dict[modality]['q_mask'][i, :batch_target_cards[i]] = True
                if has_pos_tokens:
                    target_dict[modality]['pos_tokens'][i, :batch_pos_target_cards[i]] = \
                        sample[modality]['target_dict']['pos_tokens']
                    target_dict[modality]['q_pos_mask'][i, :batch_pos_target_cards[i]] = True

        # 3.7 stack cardinalities
        if modality in input_modalities:
            input_dict[modality]['cardinality'] = torch.tensor(batch_input_cards).long()
        if modality in target_modalities:
            target_dict[modality]['cardinality'] = torch.tensor(batch_target_cards).long()

    batched_dict = {
        'event_number' : [x[input_modalities[0]]['event_number'] for x in samples],
        'getitem_idx' : [x[input_modalities[0]]['getitem_idx'] for x in samples],
        'input' : input_dict,
        'target' : target_dict,
        'q_mask_dict': {
            modality: target_dict[modality]['q_mask'] for modality in target_modalities
        },
    }

    return batched_dict


def get_collate_fn(sampling_type):
    if sampling_type in ['inference', 'inference_rand2']:
        return hep4m_collate_fn
    else:
        raise NotImplementedError(
            f"Collate fn not implemented for sampling type: {sampling_type}")


class FracBatchSampler(BatchSampler):
    def __init__(self, batch_size, modalities, sampling_dict,
            drop_last=True, sampler=None, contiguous_batches: bool=False,
            zero_card_idx_dict: Dict =None, dataset_len: int =None,
            valid_splits: list =None):
        super().__init__(sampler, batch_size, drop_last=drop_last)

        self.modalities = modalities
        self.sampling_dict = sampling_dict
        self.sampling_type = sampling_dict['type']
        self.contiguous_batches = contiguous_batches
        self.valid_splits = valid_splits

        if self.sampling_type == 'inference':
            self.fixed_split_idx = -1  # sentinel

        # Sparse to Dense Conversion for Empty Checks
        self.check_empty = False
        if zero_card_idx_dict is not None and dataset_len is not None:
            self.check_empty = True
            self.is_empty_mask = np.zeros((dataset_len, len(modalities)), dtype=bool)

            for mi, mod in enumerate(modalities):
                if mod in zero_card_idx_dict:
                    idx_list = np.array(zero_card_idx_dict[mod])
                    idx_list = idx_list[idx_list < dataset_len]
                    self.is_empty_mask[idx_list, mi] = True

            all_empty_counts = np.sum(self.is_empty_mask, axis=1)
            log.info("events with all modalities empty: %d / %d",
                     np.sum(all_empty_counts == len(modalities)), dataset_len)

            # prefilter valid_splits against empty mask
            if self.valid_splits is not None:
                self._precompute_valid_splits_per_event(dataset_len)


    def _precompute_valid_splits_per_event(self, dataset_len):
        '''For each split, precompute which input modalities it selects,
        so the empty check is a fast numpy lookup.'''
        n_splits = len(self.valid_splits)
        # split_inp_mask[s, m] = True if split s uses modality m as input
        self.split_inp_mask = np.zeros((n_splits, len(self.modalities)), dtype=bool)
        for s, (is_inp, _) in enumerate(self.valid_splits):
            self.split_inp_mask[s] = is_inp.numpy()


    def get_modality_sampling(self, getitem_idx=None):
        if self.sampling_type == 'inference':
            return self.fixed_split_idx

        while True:
            split_idx = torch.randint(0, len(self.valid_splits), (1,)).item()

            if self.check_empty and getitem_idx is not None:
                inp_mask = self.split_inp_mask[split_idx]
                if self.is_empty_mask[getitem_idx][inp_mask].all():
                    continue
            break

        return split_idx
    

    def __iter__(self):
        if self.contiguous_batches:
            return self._iter_contiguous()
        else:
            return self._iter_not_contiguous()


    def _iter_not_contiguous(self):
        for original_indices_batch in super().__iter__():
            split_idx = self.get_modality_sampling()

            augmented_batch_to_yield = []
            for original_idx in original_indices_batch:
                if self.check_empty and self.is_empty_mask[original_idx].all():
                    continue

                if self.sampling_type == 'inference_rand2':
                    split_idx = self.get_modality_sampling(getitem_idx=original_idx)

                augmented_batch_to_yield.append((original_idx, split_idx))

            yield augmented_batch_to_yield


    def _iter_contiguous(self):
        base_sampler = self.sampler
        if hasattr(base_sampler, "dataset"):
            dataset_len = len(base_sampler.dataset)
        else:
            dataset_len = len(base_sampler.data_source)
            
        if isinstance(base_sampler, DistributedSampler):
            num_replicas = base_sampler.num_replicas
            rank = base_sampler.rank
            epoch = getattr(base_sampler, "epoch", 0)
        else:
            num_replicas = 1
            rank = 0
            epoch = 0

        # 1) global number of batches (in units of batch_size)
        if self.drop_last:
            global_num_batches = dataset_len // self.batch_size
            # drop global extra so divisible by num_replicas
            global_num_batches = (global_num_batches // num_replicas) * num_replicas
            if global_num_batches == 0:
                return
        else:
            global_num_batches = math.ceil(dataset_len / self.batch_size)
            # pad up so divisible by num_replicas
            pad = (-global_num_batches) % num_replicas
            global_num_batches_padded = global_num_batches + pad

        # 2) global shuffled batch IDs (same across ranks)
        g = torch.Generator()
        g.manual_seed(epoch)  # important so all ranks get same perm *per epoch*
        perm = torch.randperm(global_num_batches, generator=g).tolist()

        if self.drop_last:
            num_batches_per_rank = global_num_batches // num_replicas
            start_b = rank * num_batches_per_rank
            end_b = start_b + num_batches_per_rank
            my_batch_ids = perm[start_b:end_b]
        else:
            # pad by repeating from beginning
            if pad:
                perm = perm + perm[:pad]
            num_batches_per_rank = global_num_batches_padded // num_replicas
            start_b = rank * num_batches_per_rank
            end_b = start_b + num_batches_per_rank
            my_batch_ids = perm[start_b:end_b]

        # 3) build contiguous batches
        for b in my_batch_ids:
            start = b * self.batch_size
            end = min(start + self.batch_size, dataset_len)
            if start >= dataset_len:
                continue

            split_idx = self.get_modality_sampling()

            augmented_batch_to_yield = []
            for original_idx in range(start, end):
                if self.check_empty and self.is_empty_mask[original_idx].all():
                    continue

                if self.sampling_type == 'inference_rand2':
                    split_idx = self.get_modality_sampling(getitem_idx=original_idx)

                augmented_batch_to_yield.append((original_idx, split_idx))

            yield augmented_batch_to_yield
