import torch
import torch.nn.functional as F



def reconstruction_loss_topo(config_v, x_hat, x, mask):
    '''
        em_frac is computed only when the topo is not full EM or hadronic
    '''
    x_cont, x_cat = x
    x_hat_cont, x_hat_cat = x_hat

    loss_cont = F.mse_loss(x_hat_cont, x_cont, reduction='none')

    em_frac_pos = config_v['features']['topo_feat0'][1].index('topo_em_frac')
    
    nonmix_wt = (x_cat['topo_em_category'] == 2).float()    
    loss_cont[:, :, em_frac_pos] = loss_cont[:, :, em_frac_pos] * nonmix_wt
    loss_cont = loss_cont[mask].mean()

    loss_cat = 0
    for v_name, x_cat_v in x_cat.items():
        x_hat_cat_v = x_hat_cat[v_name]
        loss_cat += F.cross_entropy(
            x_hat_cat_v[mask], x_cat_v[mask], reduction='mean')

    loss = loss_cont + loss_cat
    return loss, {'loss_cont': loss_cont, 'loss_cat': loss_cat}



def reconstruction_loss_track(config_v, x_hat, x, mask):
    '''
        d0, z0 losses are computed only when the normalized values are in [-1,1]
    '''
    x_cont, x_cat = x
    x_hat_cont, x_hat_cat = x_hat

    loss_cont = F.mse_loss(x_hat_cont, x_cont, reduction='none')

    # loss=0 if |x|>1 for transformed d0, z0
    if 'track_d0' in config_v['features']['track_feat0'][1]:
        d0_pos = config_v['features']['track_feat0'][1].index('track_d0')    
        loss_cont[:, :, d0_pos] = loss_cont[:, :, d0_pos] * \
            x_cont[:, :, d0_pos].abs().le(1).float()

        z0_pos = config_v['features']['track_feat0'][1].index('track_z0')
        loss_cont[:, :, z0_pos] = loss_cont[:, :, z0_pos] * \
            x_cont[:, :, z0_pos].abs().le(1).float()
    
    loss_cont = loss_cont[mask].mean()
    ret_dict = {'loss_cont': loss_cont}

    loss_cat = 0
    for v_name, x_cat_v in x_cat.items():
        x_hat_cat_v = x_hat_cat[v_name]
        loss_cat += F.cross_entropy(
            x_hat_cat_v[mask], x_cat_v[mask], reduction='mean')
    if len(x_cat) > 0:
        ret_dict['loss_cat'] = loss_cat

    loss = loss_cont + loss_cat
    return loss, ret_dict


def reconstruction_loss_cellimg(config_v, x_hat, x, mask):
    '''
        cellimg reconstruction loss
        x_hat: {[bs, n_seq, n_cont_vars + 1 (mask)]}
    '''

    # stack the continuous features from all layers
    n_layers = len(x[0].keys())

    x_cont=[]; x_ind=[]; x_hat_cont=[]; x_hat_ind=[]
    for i in range(n_layers):
        x_cont.append(x[0][f'feat0_{i}'].flatten(1, 2))
        x_ind.append(x[1][f'ind_{i}'].flatten(1, 2))

        x_hat_cont.append(x_hat[f'feat0_{i}'][..., :-1].flatten(1, 2))  # remove the indicator
        x_hat_ind.append(x_hat[f'feat0_{i}'][..., -1].flatten(1, 2))  # the indicator

    x_ind = torch.cat(x_ind, dim=1)
    x_hat_ind = torch.cat(x_hat_ind, dim=1)
    loss_ind = F.binary_cross_entropy_with_logits(x_hat_ind, x_ind, reduction='mean')

    x_cont = torch.cat(x_cont, dim=1)
    x_hat_cont = torch.cat(x_hat_cont, dim=1)
    ind_mask = x_ind > 0.5
    loss_cont = F.mse_loss(x_hat_cont[ind_mask], x_cont[ind_mask], reduction='mean')
    loss = loss_cont + loss_ind

    return loss, {
        'loss_ind': loss_ind,
        'one_q01': torch.quantile(x_hat_ind[x_ind>0.5], 0.01),
        'one_q25': torch.quantile(x_hat_ind[x_ind>0.5], 0.25),
        'one_q50': torch.quantile(x_hat_ind[x_ind>0.5], 0.50),
        'one_q75': torch.quantile(x_hat_ind[x_ind>0.5], 0.75),
        'one_q99': torch.quantile(x_hat_ind[x_ind>0.5], 0.99),
        'zero_q01': torch.quantile(x_hat_ind[x_ind<=0.5], 0.01),
        'zero_q25': torch.quantile(x_hat_ind[x_ind<=0.5], 0.25),
        'zero_q50': torch.quantile(x_hat_ind[x_ind<=0.5], 0.50),
        'zero_q75': torch.quantile(x_hat_ind[x_ind<=0.5], 0.75),
        'zero_q99': torch.quantile(x_hat_ind[x_ind<=0.5], 0.99)
    }


custom_loss_dict = {
    'topo': reconstruction_loss_topo,
    'track': reconstruction_loss_track,
    'cell': reconstruction_loss_cellimg,
    'celltruth': reconstruction_loss_cellimg
}
