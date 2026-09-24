import awkward as ak
import torch
from torch.utils.data import BatchSampler
import torch.distributed as dist
from typing import Union, List, Dict
from tqdm import tqdm

from .dataset_base import COCOADatasetBase
from .dataset_base_npy import COCOADatasetBaseNumpy
from ..performance.jet_helper import JetHelper



def get_dataset_class(dataset_type: str, npy=False):
    '''
    Get the dataset class based on the dataset type.
    Args:
        dataset_type: str, 
            'topo' | 'track' | 'truthpart' | 'cell' | 'celltruth' | 'hgpart' | 'truthjet'
        npy: bool, 
    Returns:
        COCOADatasetBase: dataset class
    '''
    dict_classes = {
        'topo': COCOADatasetTopo,
        'track': COCOADatasetBase,
        'truthpart': COCOADatasetTruthPart,
        'cell': COCOADatasetBase,
        'celltruth': COCOADatasetCellTruth,
        'hgpfpart': COCOADatasetHGPflowPart,
        'truthjet': COCOADatasetTruthJet,
    }

    dict_classes_npy = {
        'topo': COCOADatasetBaseNumpy,
        'track': COCOADatasetBaseNumpy,
        'truthpart': COCOADatasetBaseNumpy,
        'cell': COCOADatasetBaseNumpy,
        'celltruth': COCOADatasetBaseNumpy,
        'hgpfpart': COCOADatasetBaseNumpy,
        'truthjet': COCOADatasetBaseNumpy,
    }

    if npy == False:
        if dataset_type not in dict_classes:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        return dict_classes[dataset_type]

    if dataset_type not in dict_classes_npy:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return dict_classes_npy[dataset_type]




class COCOADatasetTopo(COCOADatasetBase):
    def __init__(self, 
            filename: Union[str, List[str]], 
            config_v: Dict, 
            reduce_ds: int =-1) -> None:
        '''
        COCOA dataset with topoclusters
        Args:
            filename: str | list of str | str to eval to get list of str
            config_v: dict
            reduce_ds: int, -1 for no reduction
        '''
        super().__init__(filename, config_v, reduce_ds)

    def additional_processing(self) -> None:
        self.data_dict['topo_em_frac'] = []
        self.data_dict['topo_em_category'] = []

        tqdm_obj = tqdm(range(self.n_arrays), desc="computing em_frac", total=self.n_arrays)
        for arr_i in tqdm_obj:
        
            # em fraction
            em_frac = self.data_dict['topo_ecal_e'][arr_i] / \
                (self.data_dict['topo_ecal_e'][arr_i] + self.data_dict['topo_hcal_e'][arr_i] + 1e-8)
            self.data_dict['topo_em_frac'].append(em_frac)

            # em: 0, hadronic: 1, mixed: 2
            em_cat = ak.where(
                self.data_dict['topo_em_frac'][arr_i] == 1, 0, # em
                ak.where(self.data_dict['topo_em_frac'][arr_i] == 0, 1, 2) # hadronic, mixed
            )
            self.data_dict['topo_em_category'].append(em_cat)

        del self.data_dict['topo_ecal_e'], self.data_dict['topo_hcal_e']


class COCOADatasetTruthPart(COCOADatasetBase):
    def __init__(self, 
            filename: Union[str, List[str]], 
            config_v: Dict, 
            reduce_ds: int =-1) -> None:
        '''
        COCOA dataset with truth particles
        Args:
            filename: str | list of str | str to eval to get list of str
            config_v: dict
            reduce_ds: int, -1 for no reduction
        '''
        super().__init__(filename, config_v, reduce_ds)

    def additional_processing(self) -> None:
        # particle_class
        # 0: charged, 1: neut had, 2: photon

        self.data_dict['truthpart_class'] = []
        tqdm_obj = tqdm(range(self.n_arrays), desc="computing particle classes", total=self.n_arrays)
        for arr_i in tqdm_obj:

            # set charged particles, rest are all neutral hadrons
            flat_particle_has_track = ak.flatten(self.data_dict['particle_track_idx'][arr_i]) >= 0
            flat_particle_class = ak.where(flat_particle_has_track, 0, 1)

            # set photons
            flat_particle_is_photon = ak.flatten(self.data_dict['particle_pdgid'][arr_i]) == 22
            flat_particle_class = ak.where(flat_particle_is_photon, 2, flat_particle_class)

            # Unflatten the arrays back to their original structure
            particle_class = ak.unflatten(flat_particle_class, ak.num(self.data_dict['particle_pdgid'][arr_i]))
            self.data_dict['truthpart_class'].append(particle_class)


class COCOADatasetCellTruth(COCOADatasetBase):
    def __init__(self, 
            filename: Union[str, List[str]], 
            config_v: Dict, 
            reduce_ds: int =-1) -> None:
        '''
        COCOA dataset with cell
        Args:
            filename: str | list of str | str to eval to get list of str
            config_v: dict
            reduce_ds: int, -1 for no reduction
        '''
        super().__init__(filename, config_v, reduce_ds)

    def additional_processing(self) -> None:
        # cell charge energy fraction
        self.data_dict['celltruth_ch_e_frac'] = []
        tqdm_obj = tqdm(range(self.n_arrays), desc="computing ch_e frac", total=self.n_arrays)
        for arr_i in tqdm_obj:
            che_frac = self.data_dict['cell_che'][arr_i] / \
                (self.data_dict['cell_che'][arr_i] + self.data_dict['cell_nue'][arr_i] + 1e-8)
            self.data_dict['celltruth_ch_e_frac'].append(che_frac)
        del self.data_dict['cell_che'], self.data_dict['cell_nue']



class COCOADatasetHGPflowPart(COCOADatasetBase):
    def __init__(self, 
            filename: Union[str, List[str]], 
            config_v: Dict, 
            reduce_ds: int =-1) -> None:
        '''
        COCOA dataset with HGPflow particles
        Args:
            filename: str | list of str | str to eval to get list of str
            config_v: dict
            reduce_ds: int, -1 for no reduction
        '''
        super().__init__(filename, config_v, reduce_ds, tree_name='event_tree')

        _, var_list_cont, _, _ = next(iter(self.config_v['features'].values()))
        for var in var_list_cont:
            self.data_dict[var] = ak.values_astype(self.data_dict[var], 'float32')

    def additional_processing(self) -> None:
        # particle_class
        # 0: charged, 1: neut had, 2: photon
        # 0,1,2 --> 0; 3 --> 1; 4 --> 2

        tqdm_obj = tqdm(range(self.n_arrays), desc="updating particle classes", total=self.n_arrays)
        for arr_i in tqdm_obj:
            self.data_dict['hgpfpart_class'][arr_i] = ak.where(
                self.data_dict['hgpfpart_class'][arr_i] < 3, 0, # charged
                self.data_dict['hgpfpart_class'][arr_i] - 2) # neutral hadrons and photons

            # replace hgpflow eta/phi with proxy eta/phi
            # for charged particles, replace hgpflow pt with proxy pt
            self.data_dict['hgpfpart_eta'][arr_i] = self.data_dict['proxy_eta'][arr_i]
            self.data_dict['hgpfpart_phi'][arr_i] = self.data_dict['proxy_phi'][arr_i]
            ch_mask = self.data_dict['hgpfpart_class'][arr_i] == 0
            self.data_dict['hgpfpart_pt'][arr_i] = ak.where(
                ch_mask, self.data_dict['proxy_pt'][arr_i], self.data_dict['hgpfpart_pt'][arr_i])



class COCOADatasetTruthJet(COCOADatasetBase):
    def __init__(self, 
            filename: Union[str, List[str]], 
            config_v: Dict, 
            reduce_ds: int =-1) -> None:
        '''
        COCOA dataset with HGPflow particles
        Args:
            filename: str | list of str | str to eval to get list of str
            config_v: dict
            reduce_ds: int, -1 for no reduction
        '''
        super().__init__(filename, config_v, reduce_ds)

    def additional_processing(self) -> None:
        # compute jets
        # jet = vector sum of all particles

        jet_helper = JetHelper()
        for v in ['truthjet_pt', 'truthjet_eta', 'truthjet_phi', 'truthjet_e']:
            self.data_dict[v] = []

        tqdm_obj = tqdm(range(self.n_arrays), desc="computing truth jets", total=self.n_arrays)
        for arr_i in tqdm_obj:
            # per-file array, not the list of all files (this path was never
            # exercised before the cuts=None fix in dataset_base)
            _pt, _eta, _phi, _e, _ = jet_helper.compute_jets_PtEtaPhiE(
                pts = self.data_dict['particle_pt'][arr_i],
                etas = self.data_dict['particle_eta'][arr_i],
                phis = self.data_dict['particle_phi'][arr_i],
                es = self.data_dict['particle_e'][arr_i]
            )
            self.data_dict['truthjet_pt'].append(_pt)
            self.data_dict['truthjet_eta'].append(_eta)
            self.data_dict['truthjet_phi'].append(_phi)
            self.data_dict['truthjet_e'].append(_e)

            # 64-bit to 32-bit conversion
            for k in ['truthjet_pt', 'truthjet_eta', 'truthjet_phi', 'truthjet_e']:
                self.data_dict[k][arr_i] = ak.values_astype(self.data_dict[k][arr_i], 'float32')

        # we don't need the individual particles anymore
        for k in ['particle_pt', 'particle_eta', 'particle_phi', 'particle_e']:
            del self.data_dict[k]


class ContiguousBatchSampler(BatchSampler):
    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool = True, 
                 shuffle: bool = True, seed: int = 0, epoch: int = 0,
                 num_replicas: int = None, rank: int = None):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

        # DDP setup
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # Total batches in dataset
        n_batches = self.dataset_len // self.batch_size  # always drop_last at global level for clean sharding

        # Drop batches that don't divide evenly across GPUs
        n_batches = (n_batches // self.num_replicas) * self.num_replicas

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)  # same seed across all ranks -> same permutation
            batch_order = torch.randperm(n_batches, generator=g).tolist()
        else:
            batch_order = list(range(n_batches))

        # Each rank takes every num_replicas-th batch
        my_batches = batch_order[self.rank::self.num_replicas]

        for b in my_batches:
            start = b * self.batch_size
            end = start + self.batch_size
            yield list(range(start, end))

    def __len__(self):
        n_batches = self.dataset_len // self.batch_size
        n_batches = (n_batches // self.num_replicas) * self.num_replicas
        return n_batches // self.num_replicas
