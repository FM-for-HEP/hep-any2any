import numpy as np
import awkward as ak
import torch



class CellImageHelper:

    def __init__(self, pow2_scale):
        self.grans = [256, 256, 128, 64, 64, 32]
        self.bins_to_keep_dict = {0:64, 1:64, 2:32, 3:16, 4:16, 5:8}
        self.n_layers = len(self.grans)
        
        # it's same in both eta and phi directions
        self.num_patches_per_layer_1d = []
        for i in range(self.n_layers):
            patch_size = 2 ** pow2_scale[i]
            n_patches_1d = self.bins_to_keep_dict[i] // patch_size
            self.num_patches_per_layer_1d.append(n_patches_1d)

        self.eta_bin_edges_dict = {}; self.phi_bin_edges_dict = {}
        for i, g in enumerate(self.grans):
            self.eta_bin_edges_dict[i] = np.linspace(-3, 3, g + 1)
            self.phi_bin_edges_dict[i] = np.linspace(-2*np.pi, 2*np.pi, 2*g + 1)

        self.eta_bin_mids_dict = {}; self.phi_bin_mids_dict = {}
        for i in range(self.n_layers):
            self.eta_bin_mids_dict[i] = torch.from_numpy(
                0.5 * (self.eta_bin_edges_dict[i][:-1] + self.eta_bin_edges_dict[i][1:]))
            self.phi_bin_mids_dict[i] = torch.from_numpy(
                0.5 * (self.phi_bin_edges_dict[i][:-1] + self.phi_bin_edges_dict[i][1:]))


    def to_device(self, device):
        for i in range(self.n_layers):
            self.eta_bin_mids_dict[i] = self.eta_bin_mids_dict[i].to(device)
            self.phi_bin_mids_dict[i] = self.phi_bin_mids_dict[i].to(device)


    def get_left_right_edge_idx(self, refbin, i):
        left_bin = refbin - self.bins_to_keep_dict[i] // 2
        if left_bin < 0:
            left_bin = 0
            right_bin = self.bins_to_keep_dict[i]
            return left_bin, right_bin
        
        right_bin = refbin + self.bins_to_keep_dict[i] // 2
        if right_bin > self.grans[i]:
            right_bin = self.grans[i]
            left_bin = right_bin - self.bins_to_keep_dict[i]

        return left_bin, right_bin


    def set_to_image(self, ref, coordinates, var_dict):
        '''
        Works on single event. No batches
        '''

        ref_eta, ref_phi = ref
        eta_ak, phi_ak, layer_ak = coordinates

        ref_eta_bin = np.digitize(ref_eta, self.eta_bin_edges_dict[5]) - 1
        ref_phi_bin = np.digitize(ref_phi, self.phi_bin_edges_dict[5]) - 1

        left_eta_bin5, right_eta_bin5 = self.get_left_right_edge_idx(ref_eta_bin, 5)
        left_phi_bin5 = ref_phi_bin - self.bins_to_keep_dict[5] // 2
        right_phi_bin5 = ref_phi_bin + self.bins_to_keep_dict[5] // 2

        # for patch midpoint calculation
        left_eta, right_eta = self.eta_bin_edges_dict[5][left_eta_bin5], self.eta_bin_edges_dict[5][right_eta_bin5]
        left_phi, right_phi = self.phi_bin_edges_dict[5][left_phi_bin5], self.phi_bin_edges_dict[5][right_phi_bin5]

        eta_np = ak.to_numpy(eta_ak)
        phi_np = ak.to_numpy(phi_ak)
        layer_np = ak.to_numpy(layer_ak)

        im_ind_dict = {}; etaphi_lbirbi_dict = {}; patch_mids_flattened = []
        feat_dict = {f'feat0_{i}': [] for i in range(self.n_layers)}

        for li in range(self.n_layers):
            mask = layer_np == li
            eta_li = eta_np[mask]
            phi_li = phi_np[mask]

            # phi from [-π, π] to [-2π, 2π]
            delphi = phi_li - ref_phi
            phi_li[delphi > np.pi] = phi_li[delphi > np.pi] - 2 * np.pi
            phi_li[delphi < -np.pi] = phi_li[delphi < -np.pi] + 2 * np.pi

            eta_lbi = int(self.grans[li] / self.grans[5] * left_eta_bin5)
            eta_rbi = int(self.grans[li] / self.grans[5] * right_eta_bin5) + 1
            eta_bins = self.eta_bin_edges_dict[li][eta_lbi: eta_rbi]

            phi_lbi = int(self.grans[li] / self.grans[-1] * left_phi_bin5)
            phi_rbi = int(self.grans[li] / self.grans[-1] * right_phi_bin5) + 1
            phi_bins = self.phi_bin_edges_dict[li][phi_lbi: phi_rbi]

            # indicator
            im_ind = np.histogram2d(
                eta_li, phi_li, bins=[eta_bins, phi_bins], density=False)[0] > 0
            im_ind_dict[f'ind_{li}'] = torch.from_numpy(im_ind).float()

            # variables
            for vn, v in var_dict.items():
                im_v, _, _ = np.histogram2d(
                    eta_li, phi_li, bins=[eta_bins, phi_bins], density=False,
                    weights=ak.to_numpy(v[mask]))
                feat_dict[f'feat0_{li}'].append(torch.from_numpy(im_v).float())

            # storing the bin edges for later use
            etaphi_lbirbi_dict[li] = torch.tensor([[eta_lbi, eta_rbi], [phi_lbi, phi_rbi]])

            # patch midpoint calculation
            n_patches = self.num_patches_per_layer_1d[li]
            eta_patch_size = (right_eta - left_eta) / n_patches
            phi_patch_size = (right_phi - left_phi) / n_patches
            eta_patch_mids = torch.arange(left_eta, right_eta, eta_patch_size) + eta_patch_size / 2
            phi_patch_mids = torch.arange(left_phi, right_phi, phi_patch_size) + phi_patch_size / 2

            eta_grid, phi_grid = torch.meshgrid(eta_patch_mids, phi_patch_mids, indexing='ij')
            patch_mids_flattened.append(torch.stack([
                eta_grid.reshape(-1),
                torch.cos(phi_grid).reshape(-1),
                torch.sin(phi_grid).reshape(-1)], dim=-1))
                 
        patch_mids_flattened = torch.cat(patch_mids_flattened, dim=0)

        # stack the continuous features
        for li in range(self.n_layers):
            feat_dict[f'feat0_{li}'] = torch.stack(feat_dict[f'feat0_{li}'], axis=-1)

        return feat_dict, im_ind_dict, etaphi_lbirbi_dict, patch_mids_flattened



    def image_to_set(self, feat_dict, ind_logit_dict, etaphi_lbirbi_dict, ind_logit_th=0.0):
        '''
        Works on single event
        '''
        feat_tensor = []
        for i in range(self.n_layers):
            ind_mask = ind_logit_dict[f'ind_{i}'] > ind_logit_th

            # set of features
            feat_tensor.append(
                feat_dict[f'feat0_{i}'][ind_mask])
            
            # eta, phi, layer
            eta_lbi, eta_rbi = etaphi_lbirbi_dict[i][0]
            phi_lbi, phi_rbi = etaphi_lbirbi_dict[i][1]

            eta_idxs, phi_idxs = torch.where(ind_mask)
            eta = self.eta_bin_mids_dict[i][eta_idxs + eta_lbi]
            phi = self.phi_bin_mids_dict[i][phi_idxs + phi_lbi]
            phi = (phi + np.pi) % (2 * np.pi) - np.pi  # back to [-π, π]
            layer = torch.full((len(eta), ), i, dtype=torch.int64, device=eta.device)

            feat_tensor[-1] = torch.cat([
                feat_tensor[-1], eta.unsqueeze(1), phi.unsqueeze(1), 
                layer.unsqueeze(1), ind_logit_dict[f'ind_{i}'][ind_mask].unsqueeze(1)
            ], dim=1)

        feat_tensor = torch.cat(feat_tensor, dim=0)
        return feat_tensor