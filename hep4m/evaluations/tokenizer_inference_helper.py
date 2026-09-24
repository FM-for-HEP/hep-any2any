from hep4m.paths import safe_load_expanded as _hp_safe_load
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from ..datasets.dataset_modalities import get_dataset_class
from ..datasets.cell_image_helper import CellImageHelper
from ..models.pos_tokenizer import PosTokenizer
from ..lightnings.tokenize_lightning import TokenizeLightning
from ..utility.var_transformation import VarTransformation
from ..utility.tree_writer import TreeWriter


class TokenizerInferenceHelper:

    def __init__(self, init_config):
        self.init_config = init_config
        self.config_path_v = init_config['model']['config_path_v']
        self.config_v = _hp_safe_load(open(self.config_path_v, 'r'))

        self.config_path_m = init_config['model']['config_path_m']
        self.config_m = _hp_safe_load(open(self.config_path_m, 'r'))

        self.gpu = init_config['gpu']
        self.chunk_size = init_config['chunk_size']
        self.batch_size = init_config['batch_size']
        self.num_workers = init_config['num_workers']

        self.checkpoint_path = init_config['model']['checkpoint_path']
        self.load_model(self.checkpoint_path)

        self.num_codebooks = self.lightning_model.model.vector_quantization.num_quantizers
        self.feat0_name = list(self.config_v['features'].keys())[0] # name of the collection feat0
        self.feat0_mask_name = f'{self.feat0_name}_mask'

        self.features_cont = self.config_v['features'][self.feat0_name][1].copy()
        if self.config_v.get('data_type') == 'cell_image':
            for v in ['eta', 'phi', 'layer']:
                self.features_cont.append(f'{self.feat0_name.replace("_feat0", "")}_{v}')
            self.features_cont += ['ind_logit']
        self.features_cat = [x[0] for x in self.config_v['features'][self.feat0_name][2]]
        self.features_gpos = [x for x in self.config_v['features'][self.feat0_name][3]]
        self.features = self.features_cont + self.features_cat + self.features_gpos
        
        self.modality = self.config_v['modality']

        self.transform_dict = {}
        for k, v in self.config_v['transformation_dict'].items():
            self.transform_dict[k] = VarTransformation(v)

        self.pos_tokenizer = PosTokenizer()
        self.pos_tokenizer.to(self.device)

        if self.config_v.get('data_type') == 'cell_image':
            self.cellimg_helper = CellImageHelper(pow2_scale=self.config_v['pow2_scale'])
            if self.gpu != -1 and torch.cuda.is_available():
                self.cellimg_helper.to_device(self.device)


    def load_model(self, checkpoint_path):
        self.lightning_model = TokenizeLightning(self.config_v, self.config_m)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.lightning_model.load_state_dict(checkpoint['state_dict'])
        self.lightning_model.eval()

        if torch.cuda.is_available() and self.gpu != -1:
            self.lightning_model.model.cuda()
            self.lightning_model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def get_dataloader(self, inf_dict):
        COCOADataset = get_dataset_class(self.config_v['modality'])
        dataset = COCOADataset(
            filename=inf_dict['input_path'],
            config_v=self.config_v,
            reduce_ds=inf_dict['reduce_ds'])
        loader = DataLoader(
            dataset,
            batch_size=self.init_config['batch_size'],
            num_workers=self.init_config['num_workers'],
            pin_memory=False,
            shuffle=False)
        return loader
        

    def reset_dict_to_write(self):
        self.n_entry_buffer = 0
        if not hasattr(self, 'dict_to_write'):
            self.output_branches = \
                [f'inp_{x.removeprefix(self.modality+"_")}' for x in self.features] + \
                [f'reco_{x.removeprefix(self.modality+"_")}' for x in self.features] + \
                [f'token_{i}' for i in range(self.num_codebooks)]

            self.get_gpos_vars = False
            if self.config_v['features'][f'{self.modality}_feat0'][3] is not None:
                if len(self.config_v['features'][f'{self.modality}_feat0'][3]) > 0:
                    self.get_gpos_vars = True
            if self.config_v.get('data_type') == 'cell_image':
                self.get_gpos_vars = True

            if self.get_gpos_vars:
                self.output_branches += \
                [f'pos_token_{i}' for i in range(self.pos_tokenizer.n_quantizer)]

            self.output_branches += \
                [x.removeprefix(self.modality+"_") for x in self.config_v['getitem_return']] + \
                ['idx', 'event_number']
            
            self.dict_to_write = {}
        for var in self.output_branches:
            self.dict_to_write[var] = []


    def prep_output_filepath(self, inf_dict):
        # init.output_dir when set; otherwise <tokeniser dir of config_v.yml>/inference
        output_dir = self.init_config.get('output_dir') or self.config_path_v.replace('config_v.yml', 'inference')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        inf_dict['output_filepath'] = \
            os.path.join(output_dir, inf_dict.get('dir_flag', ''), inf_dict['input_path'].split('/')[-1])
        Path(inf_dict['output_filepath']).parent.mkdir(parents=True, exist_ok=True)


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
        loader = self.get_dataloader(inf_dict)
        self.reset_dict_to_write()
        self.prep_output_filepath(inf_dict)

        output_obj = TreeWriter(
            inf_dict['output_filepath'], 'event_tree', chunk_size=self.chunk_size, dtype_to_32=True)

        for batch in tqdm(loader):

            # forward
            batch = self.to_device(batch)
            _, x_hat, indices = self.lightning_model.model(
                    batch[self.feat0_name], batch[f'{self.feat0_name}_categorical'], 
                    batch[self.feat0_mask_name])
            
            # position tokens
            pos_tokens = None
            if f'{self.feat0_name}_gpos' in batch:
                pos_tokens = self.pos_tokenizer(batch[f'{self.feat0_name}_gpos'])
                decoded_pos = self.pos_tokenizer.decode(pos_tokens)

            # cell-image specific
            if self.config_v.get('data_type') == 'cell_image':

                bs = batch[self.feat0_name]['feat0_0'].size(0)
                for bs_idx in range(bs):

                    x_cont_bs = self.cellimg_helper.image_to_set(
                        feat_dict={k: v[bs_idx] for k, v in batch[self.feat0_name].items()},
                        ind_logit_dict={k: v[bs_idx]*100 + (v[bs_idx]-1)*100 \
                            for k, v in batch[f'{self.feat0_name}_categorical'].items()},
                        etaphi_lbirbi_dict={k: v[bs_idx] for k, v in batch['etaphi_lbirbi'].items()},
                        ind_logit_th=self.init_config['indicator_logit_threshold'])

                    x_hat_cont_bs = self.cellimg_helper.image_to_set(
                        feat_dict={k: v[bs_idx][..., :-1] for k, v in x_hat.items()},
                        ind_logit_dict=\
                            {k.replace('feat0', 'ind'): v[bs_idx][..., -1] for k, v in x_hat.items()},
                        etaphi_lbirbi_dict={k: v[bs_idx] for k, v in batch['etaphi_lbirbi'].items()},
                        ind_logit_th=self.init_config['indicator_logit_threshold'])
                
                    # apply inverse transformation (on the event)
                    for feat_i, feat_name in enumerate(self.features_cont):
                        if feat_name in self.transform_dict:
                            x_cont_bs[:, feat_i] = \
                                self.transform_dict[feat_name].inverse(x_cont_bs[:, feat_i])
                            x_hat_cont_bs[:, feat_i] = \
                                self.transform_dict[feat_name].inverse(x_hat_cont_bs[:, feat_i])

                    for feat_i, feat_name in enumerate(self.features_cont):
                        fn = feat_name.removeprefix(self.modality+"_")
                        self.dict_to_write[f'inp_{fn}'].append(x_cont_bs[:, feat_i])
                        self.dict_to_write[f'reco_{fn}'].append(x_hat_cont_bs[:, feat_i])

                    # tokens
                    indices_bs = indices[bs_idx]
                    for ci in range(self.num_codebooks):
                        self.dict_to_write[f'token_{ci}'].append(indices_bs[:, ci])

                    # pos tokens
                    if pos_tokens is not None:
                        for pi in range(pos_tokens.size(-1)):
                            self.dict_to_write[f'pos_token_{pi}'].append(pos_tokens[bs_idx][:, pi])

                        for vv_i, vv in enumerate(self.features_gpos):
                            fn = vv.removeprefix(self.modality+"_")
                            self.dict_to_write[f'inp_{fn}'].append(
                                batch[f'{self.feat0_name}_gpos'][bs_idx][:, vv_i])
                            self.dict_to_write[f'reco_{fn}'].append(
                                decoded_pos[bs_idx][:, vv_i])

                    # add idx and event_number
                    self.dict_to_write['idx'].append(batch['idx'][bs_idx])
                    self.dict_to_write['event_number'].append(batch['event_number'][bs_idx])        

                    # torch -> numpy
                    for k, v in self.dict_to_write.items():
                        self.dict_to_write[k][-1] = v[-1].detach().cpu().numpy()

            # not cell-image
            else:

                x_hat_cont, x_hat_cat = x_hat
                x_cont = batch[self.feat0_name]
                x_cat = batch[f'{self.feat0_name}_categorical']

                # apply inverse transformation (on the batch)
                for feat_i, feat_name in enumerate(self.features_cont):
                    if feat_name in self.transform_dict:
                        x_cont[:, :, feat_i] = \
                            self.transform_dict[feat_name].inverse(x_cont[:, :, feat_i])
                        x_hat_cont[:, :, feat_i] = \
                            self.transform_dict[feat_name].inverse(x_hat_cont[:, :, feat_i])

                bs = batch[self.feat0_name].size(0)
                for bs_idx in range(bs):
                    mask_bs = batch[self.feat0_mask_name][bs_idx]

                    x_cont_bs = x_cont[bs_idx][mask_bs]
                    x_hat_cont_bs = x_hat_cont[bs_idx][mask_bs]

                    for feat_i, feat_name in enumerate(self.features_cont):
                        fn = feat_name.removeprefix(self.modality+"_")
                        if self.config_v.get('scalar_features', True):
                            self.dict_to_write[f'inp_{fn}'].append(x_cont_bs[:, feat_i])
                            self.dict_to_write[f'reco_{fn}'].append(x_hat_cont_bs[:, feat_i])
                        
                        # mainly for CLIP
                        else:
                            self.dict_to_write[f'inp_{fn}'].append(x_cont_bs[0])
                            self.dict_to_write[f'reco_{fn}'].append(x_hat_cont_bs[0])


                    for feat_name in self.features_cat:
                        fn = feat_name.removeprefix(self.modality+"_")
                        x_cat_fi_bs = x_cat[feat_name][bs_idx][mask_bs]
                        x_hat_cat_fi_bs = torch.argmax(
                            x_hat_cat[feat_name], dim=-1)[bs_idx][mask_bs]

                        self.dict_to_write[f'inp_{fn}'].append(x_cat_fi_bs)
                        self.dict_to_write[f'reco_{fn}'].append(x_hat_cat_fi_bs)

                    # tokens
                    indices_bs = indices[bs_idx]
                    if self.config_v.get('data_type', 'set') == 'set':
                        indices_bs = indices_bs[mask_bs]

                    for ci in range(self.num_codebooks):
                        self.dict_to_write[f'token_{ci}'].append(indices_bs[:, ci])

                    # pos tokens
                    if pos_tokens is not None:
                        for pi in range(pos_tokens.size(-1)):
                            self.dict_to_write[f'pos_token_{pi}'].append(
                                pos_tokens[bs_idx][mask_bs][:, pi])

                        for vv_i, vv in enumerate(self.features_gpos):
                            fn = vv.removeprefix(self.modality+"_")
                            self.dict_to_write[f'inp_{fn}'].append(
                                batch[f'{self.feat0_name}_gpos'][bs_idx][mask_bs][:, vv_i])
                            self.dict_to_write[f'reco_{fn}'].append(
                                decoded_pos[bs_idx][mask_bs][:, vv_i])

                    # additional variables
                    for v in self.config_v['getitem_return']:
                        self.dict_to_write[v.removeprefix(self.modality+"_")].append(
                            batch[v][bs_idx][mask_bs])

                    # add idx and event_number
                    self.dict_to_write['idx'].append(batch['idx'][bs_idx])
                    self.dict_to_write['event_number'].append(batch['event_number'][bs_idx])        

                    # torch -> numpy
                    for k, v in self.dict_to_write.items():
                        self.dict_to_write[k][-1] = v[-1].detach().cpu().numpy()

            # time to write
            self.n_entry_buffer += bs
            if self.n_entry_buffer >= self.chunk_size:
                output_obj.data = self.get_writable_dict()
                output_obj.write()
                self.reset_dict_to_write()

        # writing the last chunk
        if self.n_entry_buffer < self.chunk_size and self.n_entry_buffer > 0:
            output_obj.data = self.get_writable_dict()
            output_obj.write()

        print("\nPredicted file:")
        print(inf_dict['output_filepath'])
        output_obj.close()


    def get_writable_dict(self):
        popped_idx = self.dict_to_write.pop('idx')
        popped_event_number = self.dict_to_write.pop('event_number')

        inp_keys = [k for k in self.dict_to_write.keys() if k.startswith('inp_') and not 'patch_mid' in k]
        reco_keys = [k for k in self.dict_to_write.keys() if k.startswith('reco_') and not 'patch_mid' in k]

        inp_dict = {k.removeprefix('inp_'): self.dict_to_write[k] for k in inp_keys}
        reco_dict = {k.removeprefix('reco_'): self.dict_to_write[k] for k in reco_keys}

        for k in inp_keys + reco_keys:
            del self.dict_to_write[k]

        to_zip_dict = {
            f'{self.modality}_inp': inp_dict,
            f'{self.modality}_reco': reco_dict, 
            self.modality: self.dict_to_write,
            'idx': popped_idx,
            'event_number': popped_event_number
        }

        if self.config_v.get('data_type') == 'global':
            pos_token_keys = [k for k in self.dict_to_write.keys() if 'pos_token' in k]

            pos_dict = {}
            for pos_k in pos_token_keys:
                pos_dict[pos_k.removeprefix('pos_')] = self.dict_to_write.pop(pos_k)
            to_zip_dict[f'{self.modality}_pos'] = pos_dict
                
        return to_zip_dict