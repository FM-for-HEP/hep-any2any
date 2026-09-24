import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict
from .cell_image_helper import CellImageHelper

log = logging.getLogger(__name__)


class COCOADatasetBaseNumpy(Dataset):
    def __init__(self, 
            filename: str, 
            config_v: Dict, 
            reduce_ds: int = -1
    ) -> None:
        super().__init__()
        
        self.config_v = config_v
        self.modality = config_v['modality']
        self.data_type = config_v.get('data_type', 'set')

        # 1. Load Metadata
        # filename is e.g. "path/to/data.npy"
        meta_path = filename.replace('data.npy', 'meta.npz')
        
        try:
            meta = np.load(meta_path)
            self.total_file_events = int(meta['n_events'])
            
            # Load column names
            self.cell_vars = list(meta['cell_vars'])
            self.event_vars = list(meta['event_vars'])
            
            # Create mapping: 'cell_eta' -> index 0, etc.
            self.c_idx = {name: i for i, name in enumerate(self.cell_vars)}
            self.e_idx = {name: i for i, name in enumerate(self.event_vars)}
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata not found at {meta_path}. Did conversion finish?")

        # 2. Handle Dataset Reduction
        self.n_events = self.total_file_events
        if reduce_ds != -1:
            self.n_events = min(self.n_events, reduce_ds)
        
        end_idx = self.n_events

        # 3. Load Offsets (RAM)
        offsets_path = filename.replace('data.npy', 'offsets.npy')
        
        # FIX: Use np.load(mmap_mode='r') to handle the .npy header correctly
        # Then slice and copy to RAM
        all_offsets = np.load(offsets_path, mmap_mode='r')
        self.offsets = np.array(all_offsets[: end_idx + 1])

        # 4. Load Event Data (RAM)
        events_path = filename.replace('data.npy', 'events.npy')
        
        # FIX: Use np.load to handle header and shape automatically
        all_events = np.load(events_path, mmap_mode='r')
        self.event_data = np.array(all_events[: end_idx])

        # 5. Load Cell Data (Memmap)
        # FIX: Use np.load(mmap_mode='r'). 
        # This returns a disk-backed array (like memmap) but skips the header safely.
        # No need to manually calculate bytes or shape!
        self.cells_mmap = np.load(filename, mmap_mode='r')

        # 6. Initialize Image Helper
        self.cellimg_helper = CellImageHelper(
            pow2_scale=[3, 3, 3, 3, 3, 2] 
        )

        log.info("%s: loaded from numpy, %d events", self.modality, self.n_events)


    def __len__(self):
        return self.n_events


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.data_type in ['set', 'global']:
            return self._getitem_set_or_global(idx)
        elif self.data_type == 'cell_image':
            return self._getitem_cellimg(idx)


    def _getitem_set_or_global(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f"Set and Global modalities not implemented yet in COCOADatasetBaseNumpy. Requested modality: {self.modality}")


    def _getitem_cellimg(self, idx: int) -> Dict[str, torch.Tensor]:

        # 1. Retrieve Event Global Info
        ev_info = self.event_data[idx]
        ref_eta = ev_info[self.e_idx['ref_eta']]
        ref_phi = ev_info[self.e_idx['ref_phi']]
        event_number = np.int32(ev_info[self.e_idx['event_number']])

        # 2. Retrieve Cell Data
        # Offsets are absolute indices into the data.npy file
        start = self.offsets[idx]
        end   = self.offsets[idx+1]

        # Slice from the memmap-backed array
        cells = self.cells_mmap[start:end]

        # 3. Extract columns
        c_eta   = cells[:, self.c_idx[f'{self.modality}_eta']]
        c_phi   = cells[:, self.c_idx[f'{self.modality}_phi']]

        # Cast back to int32 because they are stored as float32 in data.npy
        c_layer_float = cells[:, self.c_idx[f'{self.modality}_layer']]
        c_layer = np.round(c_layer_float).astype(np.int32)

        # 4. Prepare Feature Dict
        feat_name = list(self.config_v['features'].keys())[0]
        requested_vars = self.config_v['features'][feat_name][1]
        
        var_dict = {}
        for var in requested_vars:
            if var in self.c_idx:
                var_dict[var] = cells[:, self.c_idx[var]]
            else:
                raise KeyError(f"Variable {var} not found in npy file columns: {self.cell_vars}")

        # 5. Generate Image
        cont_dict, cat_dict, etaphi_lbirbi_dict, patch_mids_flattened = \
            self.cellimg_helper.set_to_image(
                ref=(ref_eta, ref_phi),
                coordinates=(c_eta, c_phi, c_layer),
                var_dict=var_dict
            )

        # 6. Formatting Output
        return_dict = {}
        return_dict[feat_name] = cont_dict
        return_dict[f'{feat_name}_categorical'] = cat_dict
        return_dict[f'{feat_name}_mask'] = []

        return_dict['event_number'] = event_number
        return_dict['idx'] = idx 
        return_dict['etaphi_lbirbi'] = etaphi_lbirbi_dict
        return_dict[f'{self.modality}_feat0_gpos'] = patch_mids_flattened

        return return_dict