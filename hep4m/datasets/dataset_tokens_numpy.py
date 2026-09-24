import logging
import os
import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import Dataset
from .memmap_cache import get_memmap

log = logging.getLogger(__name__)


class TokenDatasetNumpy(Dataset):
    def __init__(self, 
            data_path: str, 
            config_v: Dict,
            start_idx: int =0,
            reduce_ds: int = -1,
            shared_event_numbers: Optional[np.ndarray] = None
    ) -> None:
        super().__init__()
        
        self.modality = config_v['modality']
        self.config_v = config_v
        self.start_idx = start_idx
        self.reduce_ds = reduce_ds
        self.data_type = config_v.get('data_type', 'set')
        
        # 1. load Metadata
        # we need to know where tokens end and pos_tokens begin
        meta_path = data_path.replace('data.npy', 'meta.npz')
        try:
            meta = np.load(meta_path)
            self.n_codebooks = int(meta['n_codebooks'])
            self.n_pos_codebooks = int(meta['n_pos_codebooks'])
            self.total_file_events = int(meta['n_events'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata not found at {meta_path}. Did conversion finish?")

        self.n_cols = self.n_codebooks + self.n_pos_codebooks
        self.max_token_cardinality = config_v['max_token_cardinality']
        self.max_pos_token_cardinality = \
            1 if self.data_type == 'global' else self.max_token_cardinality

        # 2. load Offsets (RAM)
        # We map it. The OS will cache the 200MB automatically.
        # We calculate the shape manually because it's a raw file.
        offsets_path = data_path.replace('data.npy', 'offsets.npy')

        off_bytes = os.path.getsize(offsets_path)
        n_offsets = off_bytes // 8 # int64 is 8 bytes
        
        self.offsets = np.memmap(offsets_path, dtype='int64', mode='r', 
                                 shape=(n_offsets,))

        # 3. load Data (MMap)
        # we CANNOT USE fromfile HERE (It would load 50GB into RAM)
        # we must use memmap, but we need to calculate the shape manually
        
        # Calculate shape from file size
        file_bytes = os.path.getsize(data_path)
        row_bytes = self.n_cols * 2 # int16 = 2 bytes
        n_elem = file_bytes // row_bytes
        
        # safety check
        assert file_bytes % row_bytes == 0, \
            f"File size {file_bytes} not divisible by row size {row_bytes}. Corrupt file?"

        # map it (zero RAM usage initially)
        self.data = get_memmap(data_path, 'int16', n_elem, self.n_cols)

        # 4. shared event numbers
        # passed from the main wrapper to avoid loading it 7 times
        self.event_numbers = shared_event_numbers

        # 5. handle dataset reduction
        self.n_events = self.total_file_events - self.start_idx
        if reduce_ds != -1:
            self.n_events = min(self.n_events, reduce_ds)
        self.offsets = self.offsets[self.start_idx : self.start_idx + self.n_events + 1]

        # 6.1 load empty event indices
        empties_path = data_path.replace('data.npy', 'is_empty.npy')
        self.empty_idxs = np.load(empties_path)

        # 6.2 filter to only empty indices in our range [start_idx, start_idx + n_events)
        mask = (self.empty_idxs >= self.start_idx) & (self.empty_idxs < self.start_idx + self.n_events)
        self.empty_idxs = self.empty_idxs[mask] - self.start_idx

        log.info("%s: %d events", self.modality, self.n_events)


    def __len__(self):
        return self.n_events


    def getitem(self, idx, is_input=False, is_target=False):
        '''
        Args:
            idx: int, index of the event, __getitem__(idx)
            is_input: bool, if True, return input dict with all tokens as input tokens
            is_target: bool, if True, return target dict with all tokens as target tokens
        Returns:
            {'event_number', 'getitem_idx', 'inp_dict' (or None), 'target_dict' (or None)}
        '''

        assert not (is_input and is_target), "is_input and is_target are exclusive"

        # 1. intialize return dict with event number and getitem idx
        return_dict = {
            'event_number' : self.event_numbers[idx],
            'getitem_idx' : idx,
            'inp_dict' : None,
            'target_dict' : None
        }

        # 2. check if we need to return anything at all
        if not is_input and not is_target:
            return return_dict

        # 3.1 raw data retrieval
        start = self.offsets[idx]
        end = self.offsets[idx+1]

        # 3.2 slice from disk/cache and copy to RAM as LongTensor
        # row shape: (seq_len, n_cols)
        row_data = torch.tensor(self.data[start:end]).long()
        
        # 4. Split Tokens vs Pos Tokens
        tokens = row_data[:, :self.n_codebooks]

        pos_tokens = None
        if self.n_pos_codebooks > 0:
            pos_tokens = row_data[:, self.n_codebooks:]
            if self.config_v.get('data_type', 'set') == 'global':
                pos_tokens = pos_tokens[:1, :] # all are identical, keep only one

        # 5. make sure to not go above max cardinality
        if tokens.size(0) > self.max_token_cardinality:
            tokens = tokens[:self.max_token_cardinality, :]
        if pos_tokens is not None and pos_tokens.size(0) > self.max_pos_token_cardinality:
            pos_tokens = pos_tokens[:self.max_pos_token_cardinality, :]

        # 6. cardinality of this event
        cardinality_this_event = tokens.size(0)
        cardinality_this_event = min(cardinality_this_event, self.max_token_cardinality)
        if self.n_pos_codebooks > 0:
            pos_cardinality_this_event = cardinality_this_event
            if self.data_type == 'global':
                pos_cardinality_this_event = 1

        # 7. compute input dict
        if is_input:
            inp_dict = {'tokens': tokens}            
            if self.n_pos_codebooks > 0:
                inp_dict['pos_tokens'] = pos_tokens
            inp_dict['cardinality'] = cardinality_this_event
            inp_dict['pos_cardinality'] = pos_cardinality_this_event
            return_dict['inp_dict'] = inp_dict

        # 8. compute target dict
        if is_target:
            target_dict = {'tokens': tokens}
            if self.n_pos_codebooks > 0:
                target_dict['pos_tokens'] = pos_tokens
            target_dict['cardinality'] = cardinality_this_event
            target_dict['pos_cardinality'] = pos_cardinality_this_event
            return_dict['target_dict'] = target_dict

        return return_dict