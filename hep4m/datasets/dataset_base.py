import logging
import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm
import gc

import torch
from torch.utils.data import Dataset
from ..utility.var_transformation import VarTransformation
from .cell_image_helper import CellImageHelper

from typing import Union, List, Dict

log = logging.getLogger(__name__)


class COCOADatasetBase(Dataset):
    def __init__(self, 
            filename: Union[str, List[str]], 
            config_v: Dict,
            reduce_ds: int =-1,
            tree_name: str ='EventTree') -> None:
        '''
        Base class for the COCOA dataset.
        Args:
            filename: str | list of str | str to eval to get list of str
            config_v: dict
            reduce_ds: int, -1 for no reduction
            tree_name: str, name of the tree in the file
        '''
        super().__init__()
	    
        self.config_v = config_v
        self.reduce_ds = reduce_ds
        assert reduce_ds == -1 or (isinstance(reduce_ds, int) and reduce_ds > 0), \
            "reduce_ds must be -1 or a positive integer"
        self.data_type = config_v.get('data_type', 'set')
        self.flag_allzero_tokens = self.config_v.get('flag_allzero_tokens', False)

        self.modality = config_v['modality']

        self.sort_by_var = config_v['sort_by_var']
        if not isinstance(filename, list):
            if '[' in filename and ']' in filename:
                filename = eval(filename)
            else:
                filename = [filename]

        self.data_dict = {}; self.n_events = 0
        self.event_count_per_file = []

        for fn_i, fn in enumerate(filename):
            if reduce_ds == 0:
                break

            f = uproot.open(fn)
            tree = f[tree_name]

            n_events_fni = tree.num_entries
            if (reduce_ds != -1):
                n_events_fni = min(n_events_fni, reduce_ds)
                reduce_ds -= n_events_fni

            self.n_events += n_events_fni
            entry_stop = n_events_fni
            self.event_count_per_file.append(n_events_fni)

            # read the data
            branches_to_read = self.config_v['data_loading']['branches_to_read']
            branches_rename = self.config_v['data_loading']['branches_rename']

            raw_branches = self.config_v['data_loading']['branches_store_raw']
            for var in tqdm(branches_to_read, desc=f'reading file {fn_i}/{len(filename)}'):
                v_name = branches_rename[var] if var in branches_rename else var
                if fn_i == 0:
                    self.data_dict[v_name] = []
                    if v_name in raw_branches: # and self.is_hgpflow:
                        self.data_dict[f'{v_name}_raw'] = []

                branch_array = tree[var].array(library='ak', entry_start=0, entry_stop=entry_stop)
                self.data_dict[v_name].append(branch_array)
                if v_name in raw_branches: # and self.is_hgpflow:
                    self.data_dict[f'{v_name}_raw'].append(branch_array)

        # Cut-free modalities (e.g. truthjet: data_processing.cuts == null) have no
        # eta_var to count arrays with — fall back to any loaded branch.
        # apply_etaphi_cuts() already no-ops when cuts is None.
        _cuts = self.config_v['data_processing']['cuts']
        _ref_var = _cuts['eta_var'] if _cuts else next(iter(self.data_dict))
        self.n_arrays = len(self.data_dict[_ref_var])

        # create the transform dict
        self.transform_dicts = {}
        for k, v in config_v['transformation_dict'].items():
            self.transform_dicts[k] = VarTransformation(v)

        # apply cuts
        self.apply_etaphi_cuts()

        # need to do it before sincos & transformations
        self.additional_processing()

        # process data
        self.compute_sincos()
        self.apply_transformations()

        # granularity info to get cell images
        if self.data_type == 'cell_image':
            self.cellimg_helper = CellImageHelper(
                pow2_scale=[3, 3, 3, 3, 3, 2]
            )

        # pos_vars?
        self.get_gpos_vars = False
        if self.config_v['features'][f'{self.modality}_feat0'][3] is not None:
            if len(self.config_v['features'][f'{self.modality}_feat0'][3]) > 0:
                self.get_gpos_vars = True

        self.event_count_per_file = np.array(self.event_count_per_file)
        self.event_count_per_file_cumsum = np.cumsum(np.insert(self.event_count_per_file, 0, 0))

        gc.collect()
        log.info('dataset loaded: %d events', self.n_events)


    def compute_sincos(self) -> None:
        tqdm_obj = tqdm(self.config_v['data_processing']['vars_sin_cos'], \
            desc='adding sin-cos phi', total=len(self.config_v['data_processing']['vars_sin_cos']))
        for var in tqdm_obj:
            cos_var = var.replace('phi', 'cosphi')
            sin_var = var.replace('phi', 'sinphi')

            self.data_dict[cos_var] = []
            self.data_dict[sin_var] = []

            for arr_i in range(self.n_arrays):
                self.data_dict[cos_var].append(np.cos(self.data_dict[var][arr_i]))
                self.data_dict[sin_var].append(np.sin(self.data_dict[var][arr_i]))

            if var in self.config_v['data_processing']['vars_sin_cos_og_delete']:
                del self.data_dict[var]


    def compute_references_etaphi(self, arr_i, verbose=False):

        ref_eta_var = self.config_v['data_processing']['cuts']['ref_eta_var']
        ref_eta_num = ak.sum(self.data_dict[ref_eta_var][arr_i], axis=1)
        ref_eta_den = ak.num(self.data_dict[ref_eta_var][arr_i], axis=1)
        ref_eta = ref_eta_num / ref_eta_den

        ref_phi_var = self.config_v['data_processing']['cuts']['ref_phi_var']
        ref_cosphi_num = ak.sum(np.cos(self.data_dict[ref_phi_var][arr_i]), axis=1)
        ref_sinphi_num = ak.sum(np.sin(self.data_dict[ref_phi_var][arr_i]), axis=1)
        ref_phi_den = ak.num(self.data_dict[ref_phi_var][arr_i], axis=1)
        ref_phi = np.arctan2(ref_sinphi_num / ref_phi_den, ref_cosphi_num / ref_phi_den)

        if np.any(ref_eta_den == 0):
            if verbose:
                log.info('NaN in the reference direction; recomputing from the second set of variables')
            ref_eta_var2 = self.config_v['data_processing']['cuts']['ref_eta_var2']
            ref_eta_num2 = ak.sum(self.data_dict[ref_eta_var2][arr_i], axis=1)
            ref_eta_den2 = ak.num(self.data_dict[ref_eta_var2][arr_i], axis=1)
            ref_eta = np.where(ref_eta_den == 0, ref_eta_num2 / ref_eta_den2, ref_eta)

            ref_phi_var2 = self.config_v['data_processing']['cuts']['ref_phi_var2']
            ref_cosphi_num2 = ak.sum(np.cos(self.data_dict[ref_phi_var2][arr_i]), axis=1)
            ref_sinphi_num2 = ak.sum(np.sin(self.data_dict[ref_phi_var2][arr_i]), axis=1)
            ref_phi_den2 = ak.num(self.data_dict[ref_phi_var2][arr_i], axis=1)
            ref_phi = np.where(ref_phi_den == 0, 
                np.arctan2(ref_sinphi_num2 / ref_phi_den2, ref_cosphi_num2 / ref_phi_den2), 
                ref_phi)

            if np.any((ref_eta_den == 0) & (ref_eta_den2 == 0)):
                if verbose:
                    log.info('reference direction still NaN; set to 0')
                mask = (ref_eta_den == 0) & (ref_eta_den2 == 0)
                ref_eta = np.where(mask, 0, ref_eta)
                ref_phi = np.where(mask, 0, ref_phi)
    
        if self.config_v['data_processing'].get('store_pos_relatives', False):
            if 'ref_eta' not in self.data_dict:
                self.data_dict['ref_eta'] = []
                self.data_dict['ref_phi'] = []
            self.data_dict['ref_eta'].append(ref_eta)
            self.data_dict['ref_phi'].append(ref_phi)

        return ref_eta, ref_phi


    def apply_etaphi_cuts(self) -> None:
        cuts_dict = self.config_v['data_processing']['cuts']
        if cuts_dict is None:
            log.info('no eta/phi cuts applied')
            return

        eta_var = cuts_dict['eta_var'] 
        phi_var = cuts_dict['phi_var']
        e_pt_var = cuts_dict.get('e_pt_var', None)

        for arr_i in tqdm(range(self.n_arrays), desc='applying eta-phi cuts', total=self.n_arrays):
            ref_eta, ref_phi = self.compute_references_etaphi(arr_i)
            mask = ((self.data_dict[eta_var][arr_i] - ref_eta) < cuts_dict['deta_max']) & \
                ((self.data_dict[eta_var][arr_i] - ref_eta) > cuts_dict['deta_min']) & \
                ((self.data_dict[phi_var][arr_i] - ref_phi) < cuts_dict['dphi_max']) & \
                ((self.data_dict[phi_var][arr_i] - ref_phi) > cuts_dict['dphi_min'])

            if e_pt_var is not None:
                mask = mask & (self.data_dict[e_pt_var][arr_i] > cuts_dict['e_pt_min'])

            for var in self.data_dict.keys():
                if var in cuts_dict['vars_to_delete']:
                    continue
                if isinstance(self.data_dict[var][arr_i], ak.Array):
                    if self.data_dict[var][arr_i].ndim == 2:
                        self.data_dict[var][arr_i] = self.data_dict[var][arr_i][mask] 

        for v in cuts_dict['vars_to_delete']:
            del self.data_dict[v]


    def apply_transformations(self) -> None:
        tqdm_obj = tqdm(self.config_v['data_processing']['vars_to_transform'], desc='transforming vars')
        for var in tqdm_obj:
            for arr_i in range(self.n_arrays):
                self.data_dict[var][arr_i] = self.transform_dicts[var].forward(self.data_dict[var][arr_i])


    def additional_processing(self) -> None:
        # add any additional processing here
        pass


    def __len__(self):
        return self.n_events


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.data_type in ['set', 'global']:
            return self._getitem_set_or_global(idx)
        elif self.data_type == 'cell_image':
            return self._getitem_cellimg(idx)


    def _getitem_set_or_global(self, idx: int):

        # 0. file idx and local idx
        fidx = np.searchsorted(
            self.event_count_per_file_cumsum, idx, side='right') - 1
        lidx = idx - self.event_count_per_file_cumsum[fidx]

        # sorting indices
        sort_idx = None
        if self.sort_by_var is not None:
            sort_by_var = torch.from_numpy(
                ak.to_numpy(self.data_dict[self.sort_by_var][fidx][lidx]))
            sort_idx = torch.argsort(sort_by_var, dim=0, descending=True)

        # features to return
        return_dict = {}
        for feat_name, (max_cardinality, var_list_cont, var_list_cat, var_list_pos) \
                in self.config_v['features'].items():
            
            feat = []
            for var in var_list_cont:
                feat.append(torch.from_numpy(
                    ak.to_numpy(self.data_dict[var][fidx][lidx])))
            return_dict[feat_name] = torch.stack(feat, dim=-1)

            # sorting
            if sort_idx is not None:
                return_dict[feat_name] = return_dict[feat_name][sort_idx]

            # categorical features + sorting
            categorical_feat_dict = {}
            for [var, _] in var_list_cat:
                categorical_feat_dict[var] = torch.from_numpy(
                    ak.to_numpy(self.data_dict[var][fidx][lidx])).long()[sort_idx]

            # gpos variables for tokenization
            if self.get_gpos_vars:
                gpos_feat = []
                for var in var_list_pos:
                    gpos_feat.append(torch.from_numpy(
                        ak.to_numpy(self.data_dict[var][fidx][lidx])))
                gpos_feat = torch.stack(gpos_feat, dim=-1)
                if sort_idx is not None:
                    gpos_feat = gpos_feat[sort_idx]

            # zero padding (for transformer)
            cardinality = len(feat[0])
            if max_cardinality != cardinality:
                return_dict[feat_name] = torch.nn.functional.pad(
                    return_dict[feat_name], (0, 0, 0, max_cardinality - cardinality), value=0)
                for k, v in categorical_feat_dict.items():
                    categorical_feat_dict[k] = torch.nn.functional.pad(
                        v, (0, max_cardinality - cardinality), value=0)
                if self.get_gpos_vars:
                    gpos_feat = torch.nn.functional.pad(
                        gpos_feat, (0, 0, 0, max_cardinality - cardinality), value=0)

            # add it to the return dict
            return_dict[f'{feat_name}_categorical'] = categorical_feat_dict
            if self.get_gpos_vars:
                return_dict[f'{feat_name}_gpos'] = gpos_feat

            # mask
            return_dict[f'{feat_name}_mask'] = torch.zeros(max_cardinality, dtype=torch.bool)
            return_dict[f'{feat_name}_mask'][:cardinality] = 1

        # additional variables to return
        for var in self.config_v['getitem_return']:
            return_dict[var] = torch.from_numpy(
                ak.to_numpy(self.data_dict[var][fidx][lidx]))[sort_idx]
            if max_cardinality != cardinality:
                return_dict[var] = torch.nn.functional.pad(
                    return_dict[var], (0, max_cardinality - cardinality), value=0)

        # keep track of the original event number
        return_dict['event_number'] = self.data_dict['event_number'][fidx][lidx]
        return_dict['idx'] = idx

        return return_dict


    def _getitem_cellimg(self, idx: int):

        # 0. file idx and local idx
        fidx = np.searchsorted(
            self.event_count_per_file_cumsum, idx, side='right') - 1
        lidx = idx - self.event_count_per_file_cumsum[fidx]

        feat_name = list(self.config_v['features'].keys())[0]
        modality = feat_name.replace('_feat0', '')

        cont_dict, cat_dict, etaphi_lbirbi_dict, patch_mids_flattened = \
        self.cellimg_helper.set_to_image(
            ref=(self.data_dict['ref_eta'][fidx][lidx], self.data_dict['ref_phi'][fidx][lidx]),
            coordinates=(
                self.data_dict[f'{modality}_eta'][fidx][lidx], self.data_dict[f'{modality}_phi'][fidx][lidx],
                self.data_dict[f'{modality}_layer'][fidx][lidx]),
            var_dict={k: self.data_dict[k][fidx][lidx] \
                for k in self.config_v['features'][feat_name][1]}
        )

        return_dict = {}
        return_dict[feat_name] = cont_dict
        return_dict[f'{feat_name}_categorical'] = cat_dict
        return_dict[f'{feat_name}_mask'] = []

        # keep track of the original event number
        return_dict['event_number'] = self.data_dict['event_number'][fidx][lidx]
        return_dict['idx'] = idx
        return_dict['etaphi_lbirbi'] = etaphi_lbirbi_dict
        return_dict[f'{modality}_feat0_gpos'] = patch_mids_flattened

        return return_dict


