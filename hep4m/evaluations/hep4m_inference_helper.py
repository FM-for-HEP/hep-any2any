from hep4m.paths import safe_load_expanded as _hp_safe_load
import logging
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from ..datasets.dataset_hep4m import COCOADatasetHEP4M
from ..datasets.cell_image_helper import CellImageHelper
from ..lightnings.hep4m_lightning import HEP4MLightning
from ..utility.var_transformation import VarTransformation
from ..utility.tree_writer import TreeWriter

torch.set_autocast_dtype('cuda', torch.bfloat16)
log = logging.getLogger(__name__)


class HEP4MInferenceHelper:

    def __init__(self, init_config):
        self.init_config = init_config

        self.config_path_m = init_config['model']['config_path_m']
        with open(self.config_path_m) as fp:
            self.config_m = _hp_safe_load(fp)

        self.modality_dict_path = init_config['model']['modality_dict_path']
        with open(self.modality_dict_path) as fp:
            self.modality_dict = _hp_safe_load(fp)

        self.device = init_config['device']
        self.gpu = init_config['gpu']
        if self.gpu == -1:
            self.device = 'cpu'

        self.chunk_size = init_config['chunk_size']
        self.batch_size = init_config['batch_size']
        self.num_workers = init_config['num_workers']

        self.checkpoint_path = init_config['model']['checkpoint_path']
        self.load_model(self.checkpoint_path)

        self.config_v_dict, self.config_m_dict = \
            self.lightning_model.get_config_dicts(return_dicts=True)

        self.features_dict_cont = {}
        for mod in self.modality_dict.keys():
            self.features_dict_cont[mod] = \
                self.config_v_dict[mod]['features'][f'{mod}_feat0'][1].copy()
            if self.config_v_dict[mod].get('data_type', 'set') == 'cell_image':
                for v in ['eta', 'phi', 'layer']:
                    self.features_dict_cont[mod].append(f'{mod}_{v}')
                self.features_dict_cont[mod] += ['ind_logit']

        self.features_dict_cat = {}
        for mod in self.modality_dict.keys():
            self.features_dict_cat[mod] = \
                [x[0] for x in self.config_v_dict[mod]['features'][f'{mod}_feat0'][2]]

        self.features_dict_gpos = {}
        for mod in self.modality_dict.keys():
            self.features_dict_gpos[mod] = \
                self.config_v_dict[mod]['features'][f'{mod}_feat0'][3].copy()

        self.transform_dict = {}
        for modality, config_v in self.config_v_dict.items():
            tmp_mod_dict = {}
            for k, v in config_v['transformation_dict'].items():
                tmp_mod_dict[k] = VarTransformation(v)
            self.transform_dict[modality] = tmp_mod_dict

        self.cellimg_helper = CellImageHelper(pow2_scale=[3, 3, 3, 3, 3, 2])
        self.cellimg_helper.to_device(self.device)

    
    def load_model(self, checkpoint_path):
        self.lightning_model = HEP4MLightning(
            self.config_m, self.modality_dict, device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.lightning_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.lightning_model.eval()
        self.lightning_model.model.set_use_target_tokens(False)

        if torch.cuda.is_available() and self.gpu != -1:
            self.lightning_model.model.cuda()
            self.lightning_model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        log.info('running on %s', self.device)


    def get_dataloader(self, inf_dict):

        # we don't need the full config_v_dict, only the modalities involved
        config_v_copy = self.config_v_dict.copy()
        for k in list(config_v_copy.keys()):
            if k not in inf_dict['input_modalities'] and \
                    k not in inf_dict['output_modalities']:
                del config_v_copy[k]

        dataset = COCOADatasetHEP4M(
            filepath_dict=inf_dict['filepath_dict'],
            config_v_dict = config_v_copy,
            reduce_ds = inf_dict['reduce_ds'], 
            sampling_type = 'inference',
            use_tokens = False,  # raw inputs, tokenised by the model
            fixed_input_output_modalities = [
                inf_dict['input_modalities'], inf_dict['output_modalities']],
        )
        loader = DataLoader(dataset,
            batch_size = self.init_config['batch_size'],
            num_workers = self.init_config['num_workers'],
            pin_memory=False, shuffle=False)
        return loader
        

    def reset_dict_to_write(self, force=False):
        self.n_entry_buffer = 0

        if not hasattr(self, 'dict_to_write') or force:
            self.global_branches = ['idx', 'event_number']
            self.modality_branches = {}
            for modality in self.output_modalities:
                features = self.features_dict_cont[modality] + \
                    self.features_dict_cat[modality]
                
                if self.config_v_dict[modality].get('data_type', 'set') != 'cell_image':
                    features += self.features_dict_gpos[modality]

                self.modality_branches[f'{modality}_truth'] = \
                    [f'{x.removeprefix(modality+"_")}' for x in features]                
                self.modality_branches[f'{modality}_reco'] = \
                    [f'{x.removeprefix(modality+"_")}' for x in features]
                
            self.dict_to_write = {mod: {} for mod in self.modality_branches.keys()}

        for var in self.global_branches:
            self.dict_to_write[var] = []
        for br_grp, var_list in self.modality_branches.items():
            self.dict_to_write[br_grp] = {}
            for var in self.modality_branches[br_grp]:
                self.dict_to_write[br_grp][var] = []


    def prep_output_filepath(self, inf_dict):
        # init.output_dir when set; otherwise <run dir of config_m.yml>/inference
        output_dir = self.init_config.get('output_dir') or self.config_path_m.replace('config_m.yml', 'inference')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        last_input = list(inf_dict['filepath_dict'].values())[-1]
        fp_name = f"prediction_{last_input.split('/')[-1]}"
        if inf_dict.get('suffix', '') != '':
            fp_name = fp_name.replace('.root', f'_{inf_dict["suffix"]}.root')
        inf_dict['output_filepath'] = \
            os.path.join(output_dir, inf_dict.get('dir_flag', ''), fp_name)
        Path(inf_dict['output_filepath']).parent.mkdir(parents=True, exist_ok=True)
        log.info('output file: %s', inf_dict['output_filepath'])


    def to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v) for v in batch]
        else:
            return batch


    def run_inference(self, inf_dict):
        self.input_modalities = inf_dict['input_modalities']
        self.output_modalities = inf_dict['output_modalities']
        self.lightning_model.model.set_input_output_modalities(
            self.input_modalities, self.output_modalities)

        fp_dict_og = inf_dict.pop('filepath_dict').copy()
        fp_dict_evaled = {k: eval(v) for k, v in fp_dict_og.items()}
        n_fp = len(fp_dict_evaled[list(fp_dict_evaled.keys())[0]])

        for fpi in range(n_fp):

            inf_dict['filepath_dict'] = {k: v[fpi] for k, v in fp_dict_evaled.items()}

            self.prep_output_filepath(inf_dict)
            loader = self.get_dataloader(inf_dict)
            self.reset_dict_to_write(force=True)

            output_obj = TreeWriter(
                inf_dict['output_filepath'], 'event_tree', chunk_size=self.chunk_size, dtype_to_32=True)

            for batch in tqdm(loader, disable=not sys.stdout.isatty()):

                batch = self.to_device(batch)

                output_dict = self.lightning_model.model(
                    batch,
                    top_k_token_dict=inf_dict['top_k_token_dict'],
                    top_k_gpos_token_dict=inf_dict['top_k_gpos_token_dict'],
                    temperature_token_dict=inf_dict['temperature_token_dict'],
                    temperature_gpos_token_dict=inf_dict['temperature_gpos_token_dict'],
                    use_truth_cardinality=inf_dict['use_truth_cardinality'],
                    card_topk_dict=inf_dict['card_topk_dict'],
                    card_temperature_dict=inf_dict['card_temperature_dict'])

                for modality_i, modality in enumerate(self.output_modalities):
                        
                    # cell-image specific
                    if self.config_v_dict[modality].get('data_type', 'set') == 'cell_image':

                        bs = batch[modality]['idx'].size(0)
                        for bs_idx in range(bs):

                            ind_logit_dict={k: v[bs_idx]*100 + (v[bs_idx]-1)*100 \
                                for k, v in batch[modality][f'{modality}_feat0_categorical'].items()}
                            x_cont_bs = self.cellimg_helper.image_to_set(
                                feat_dict={k: v[bs_idx] for k, v in batch[modality][f'{modality}_feat0'].items()},
                                ind_logit_dict=ind_logit_dict,
                                etaphi_lbirbi_dict={k: v[bs_idx] for k, v in batch[modality]['etaphi_lbirbi'].items()})

                            # if we want cell prediction for given cell
                            if inf_dict.get('use_truth_image_indicator', False):
                                ind_hat_logit_dict = {k: v.clone() for k, v in ind_logit_dict.items()}
                            else:
                                ind_hat_logit_dict = {
                                    k.replace('feat0', 'ind'): v[bs_idx][..., -1] for k, v in output_dict[modality]['x_hat'].items()}

                            x_hat_cont_bs = self.cellimg_helper.image_to_set(
                                feat_dict={k: v[bs_idx][..., :-1] for k, v in output_dict[modality]['x_hat'].items()},
                                ind_logit_dict=ind_hat_logit_dict,
                                etaphi_lbirbi_dict={k: v[bs_idx] for k, v in batch[modality]['etaphi_lbirbi'].items()},
                                ind_logit_th=inf_dict['cellind_logit_threshold_dict'][modality])

                            # apply inverse transformation (on the event)
                            for feat_i, feat_name in enumerate(self.features_dict_cont[modality]):
                                if feat_name in self.transform_dict[modality]:
                                    x_cont_bs[:, feat_i] = \
                                        self.transform_dict[modality][feat_name].inverse(x_cont_bs[:, feat_i])
                                    x_hat_cont_bs[:, feat_i] = \
                                        self.transform_dict[modality][feat_name].inverse(x_hat_cont_bs[:, feat_i])

                            for feat_i, feat_name in enumerate(self.features_dict_cont[modality]):
                                fn = feat_name.removeprefix(modality+"_")
                                self.dict_to_write[f'{modality}_truth'][fn].append(x_cont_bs[:, feat_i])
                                self.dict_to_write[f'{modality}_reco'][fn].append(x_hat_cont_bs[:, feat_i])

                            # add idx and event_number
                            if modality_i == 0:
                                self.dict_to_write['idx'].append(batch[modality]['idx'][bs_idx].item())
                                self.dict_to_write['event_number'].append(batch[modality]['event_number'][bs_idx].item())

                            # torch -> numpy
                            for k, v in self.dict_to_write[f'{modality}_truth'].items():
                                self.dict_to_write[f'{modality}_truth'][k][-1] = v[-1].detach().cpu().numpy()
                            for k, v in self.dict_to_write[f'{modality}_reco'].items():
                                self.dict_to_write[f'{modality}_reco'][k][-1] = v[-1].detach().cpu().numpy()

                    # not cell-image
                    else:

                        x_cont = batch[modality][f'{modality}_feat0']
                        x_cat = batch[modality][f'{modality}_feat0_categorical']
                        x_mask = batch[modality][f'{modality}_feat0_mask']

                        x_hat_cont, x_hat_cat = output_dict[modality]['x_hat']
                        q_mask = output_dict[modality]['q_mask']

                        # apply inverse transformation (on the batch)
                        for feat_i, feat_name in enumerate(self.features_dict_cont[modality]):
                            if feat_name in self.transform_dict[modality]:
                                x_cont[:, :, feat_i] = self.transform_dict[modality][feat_name].inverse(x_cont[:, :, feat_i])
                                x_hat_cont[:, :, feat_i] = \
                                    self.transform_dict[modality][feat_name].inverse(x_hat_cont[:, :, feat_i])

                        # postion variables (no transformation; position tokenizer is binned. doesn't care about range)
                        if len(self.features_dict_gpos[modality]) > 0:
                            x_gpos = batch[modality][f'{modality}_feat0_gpos']
                            x_gpos_hat = output_dict[modality]['x_gpos_hat']

                        bs = x_hat_cont.size(0)
                        for bs_idx in range(bs):
                            inp_mask_bs = x_mask[bs_idx]
                            if self.config_v_dict[modality].get('data_type') == 'global':
                                out_mask_bs = torch.ones_like(inp_mask_bs).bool()
                            else:
                                out_mask_bs = q_mask[bs_idx]

                            x_cont_bs = x_cont[bs_idx][inp_mask_bs]
                            x_hat_cont_bs = x_hat_cont[bs_idx][out_mask_bs]

                            for feat_i, feat_name in enumerate(self.features_dict_cont[modality]):
                                fn = feat_name.removeprefix(modality+"_")
                                self.dict_to_write[f'{modality}_truth'][fn].append(x_cont_bs[:, feat_i])
                                self.dict_to_write[f'{modality}_reco'][fn].append(x_hat_cont_bs[:, feat_i])

                            for feat_name in self.features_dict_cat[modality]:
                                fn = feat_name.removeprefix(modality+"_")
                                x_cat_fi_bs = x_cat[feat_name][bs_idx][inp_mask_bs]
                                x_hat_cat_fi_bs = torch.argmax(
                                    x_hat_cat[feat_name][bs_idx][out_mask_bs], dim=-1)

                                self.dict_to_write[f'{modality}_truth'][fn].append(x_cat_fi_bs)
                                self.dict_to_write[f'{modality}_reco'][fn].append(x_hat_cat_fi_bs)

                            for feat_i, feat_name in enumerate(self.features_dict_gpos[modality]):
                                fn = feat_name.removeprefix(modality+"_")

                                x_gpos_bs = x_gpos[bs_idx][inp_mask_bs]
                                x_gpos_hat_bs = x_gpos_hat[bs_idx][out_mask_bs]

                                self.dict_to_write[f'{modality}_truth'][fn].append(x_gpos_bs[:, feat_i])
                                self.dict_to_write[f'{modality}_reco'][fn].append(x_gpos_hat_bs[:, feat_i])

                            # add idx and event_number
                            if modality_i == 0:
                                self.dict_to_write['idx'].append(batch[modality]['idx'][bs_idx].item())
                                self.dict_to_write['event_number'].append(batch[modality]['event_number'][bs_idx].item())

                            # torch -> numpy
                            for k, v in self.dict_to_write[f'{modality}_truth'].items():
                                self.dict_to_write[f'{modality}_truth'][k][-1] = v[-1].detach().cpu().numpy()
                            for k, v in self.dict_to_write[f'{modality}_reco'].items():
                                self.dict_to_write[f'{modality}_reco'][k][-1] = v[-1].detach().cpu().numpy()

                # time to write
                self.n_entry_buffer += bs
                if self.n_entry_buffer >= self.chunk_size:
                    output_obj.data = self.dict_to_write
                    output_obj.write()
                    self.reset_dict_to_write()

            # writing the last chunk
            if self.n_entry_buffer < self.chunk_size and self.n_entry_buffer > 0:
                output_obj.data = self.dict_to_write
                output_obj.write()

            output_obj.close()
            log.info('wrote %s', inf_dict['output_filepath'])