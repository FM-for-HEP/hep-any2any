import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



UNITS = {
    '$p_T$': 'GeV'
}


MODALITY_LABEL = {
    'track': 'track', 'topo': 'topocluster', 'cell': 'cell',
    'truthpart': 'particle', 'truthjet': 'jet',
    'hgpfpart': 'HGPflow particle', 'celltruth': 'charged-energy fraction',
}
VAR_LABEL = {
    'pt': r'$p_T$ [GeV]', 'e': r'$E$ [GeV]', 'eta': r'$\eta$',
    'phi': r'$\phi$', 'd0': r'$d_0$ [mm]', 'z0': r'$z_0$ [mm]',
    'em_frac': 'EM fraction', 'rho': r'$\rho$',
}


def var_label(var, modality=None):
    """Axis label for a raw field name, optionally prefixed by the modality."""
    label = VAR_LABEL.get(var, var)
    if modality is None:
        return label
    return f"{MODALITY_LABEL.get(modality, modality)} {label}"


def marginal_distributions(data_flat_dict, names, vars, log_var_pos, cosmetic_dict, bin_dict={}, annotate_ks=False, modality=None):
    COLORS = cosmetic_dict['COLORS']
    HISTTYPES = cosmetic_dict.get('HISTTYPES', {})
    ALPHAS = cosmetic_dict.get('ALPHAS', {})
    LINESTYLES = cosmetic_dict.get('LINESTYLES', {})

    n_plots = len(vars)
    n_cols = min(4, n_plots)
    n_rows = np.ceil(n_plots / n_cols).astype(int)

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3), dpi=200)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    for vi, v in enumerate(vars):
        ax = fig.add_subplot(gs[vi])
        if v not in bin_dict.keys():
            # Guard against fully-empty arrays (early in training, AR decode can
            # produce 0-length outputs for every event → reco percentile crashes).
            non_empty = [data_flat_dict[k][v] for k in names if data_flat_dict[k][v].size > 0]
            if not non_empty:
                bins = np.linspace(0, 1, 50)
            else:
                _max = max(float(np.percentile(arr, 99)) for arr in non_empty)
                _min = min(float(np.percentile(arr, 1))  for arr in non_empty)
                if _max <= _min:
                    _max = _min + 1.0
                bins = np.linspace(_min, _max, 50)
        else:
            bins = bin_dict[v]

        for name in names:
            ax.hist(data_flat_dict[name][v], bins=bins, histtype=HISTTYPES.get(name, 'step'),
                color=COLORS[name], label=name, zorder=10, alpha=ALPHAS.get(name, 1.0),
                linestyle=LINESTYLES.get(name, '-'), density=True)
        # Kolmogorov-Smirnov statistic of each model vs the Target series (names[0]),
        # annotated per panel (metric 1: point-cloud reconstruction quality).
        if annotate_ks and names and names[0] == "Target":
            from scipy.stats import ks_2samp
            tgt = data_flat_dict[names[0]][v]
            y0 = 0.97
            for name in names[1:]:
                arr = data_flat_dict[name][v]
                if arr.size and tgt.size:
                    ks = ks_2samp(tgt, arr).statistic
                    ax.text(0.97, y0, f"KS={ks:.3f}", transform=ax.transAxes,
                            ha='right', va='top', fontsize=7, color=COLORS[name])
                    y0 -= 0.075
        ax.set_xlabel(var_label(v, modality))
        ax.set_ylabel('normalized')
        if vi == 0: ax.legend()
        ax.grid(True)
        
        
        if vi in log_var_pos:
            ax.set_yscale('log')
    
    return fig



def plot_jets(jets_dict, ratio_bin_dicts={}, cosmetic_dict=None):

    COLORS = cosmetic_dict['COLORS']
    ALPHAS = cosmetic_dict.get('ALPHAS', {})
    LINESTYLES = cosmetic_dict.get('LINESTYLES', {})

    n_reco = len(jets_dict)

    fig = plt.figure(figsize=(15, 8+n_reco*5), dpi=200)
    gs = fig.add_gridspec(3+n_reco, 3, hspace=0.3, wspace=0.3)

    target_jet0 = list(jets_dict.values())[0]['target']

    for vi, v in enumerate(['$p_T$', '$\\eta$', '$\\phi$']):
        truth_v = ak.to_numpy(target_jet0[v])
        reco_vs = {k: ak.to_numpy(j['reco'][v]) for k, j in jets_dict.items()}

        _max = max([np.percentile(truth_v, 99)] + [np.percentile(rj, 99) for rj in reco_vs.values()])
        _min = min([np.min(truth_v)] + [np.min(rj) for rj in reco_vs.values()])
        bins = np.linspace(_min, _max, 80)

        # marginals
        ax1 = fig.add_subplot(gs[0, vi])
        ax1.hist(truth_v, bins=bins, histtype='stepfilled', color='cornflowerblue', label='target', zorder=10, alpha=0.8)
        for name, reco_v in reco_vs.items():
            ax1.hist(reco_v, bins=bins, histtype='step', color=COLORS[name], label=name, zorder=10,
                linestyle=LINESTYLES.get(name, '-'), alpha=ALPHAS.get(name, 1.0))
        xlabel = v + (f' [{UNITS[v]}]' if v in UNITS.keys() else '')
        ax1.set_xlabel(xlabel)
        ax1.grid(True)
        ax1.legend()

        # ratio marginal
        ax2 = fig.add_subplot(gs[1, vi])
        for name, jdict in jets_dict.items():
            ratio = jdict['reco'][v] / jdict['target'][v] if v == '$p_T$' else (jdict['reco'][v] - jdict['target'][v])
            bins = ratio_bin_dicts.get(v, np.linspace(0.5, 1.5, 30))
            ax2.hist(ratio, bins=bins, histtype='step', color=COLORS[name], label=name, zorder=10,
                linestyle=LINESTYLES.get(name, '-'), alpha=ALPHAS.get(name, 1.0))
        y_max = ax2.get_ylim()[1]
        ax2.set_ylim(0, y_max * 1.5)
        ax2.set_xlabel(f'reco / truth ({v})')
        ax2.set_ylabel('counts')
        ax2.legend()
        ax2.grid(True)

    return fig


def cardinality_plot(reco_ind_array, truth_card, ind_th, title=None):
    """Predicted vs true cardinality (reco indicator above ``ind_th``); used by pflow_report."""

    ind_all_flat = ak.flatten(reco_ind_array)

    reco_card_dict = {}
    thresholds = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999]
    for th in thresholds:
        reco_card_dict[th] = ak.to_numpy(ak.sum(reco_ind_array > th, axis=1))

    del_card_means = []; del_card_stds = []
    for th in thresholds:
        del_card = reco_card_dict[th] - truth_card
        del_card_means.append(np.mean(del_card))
        del_card_stds.append(np.std(del_card))
    del_card_means = np.array(del_card_means)
    del_card_stds = np.array(del_card_stds)

    fig = plt.figure(figsize=(16, 4), dpi=200)
    if title is not None:
        fig.suptitle(title)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(ak.to_numpy(ind_all_flat), bins=40, histtype='stepfilled', color='cornflowerblue', zorder=10)
    ax1.set_xlabel('pred indicator')
    ax1.set_yscale('log')
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(thresholds, np.zeros_like(thresholds), ls='--', color='black', zorder=10)
    ax2.plot(thresholds, del_card_means, marker='o', label='mean', color='cornflowerblue', zorder=10)
    ax2.fill_between(thresholds, del_card_means - del_card_stds, del_card_means + del_card_stds, alpha=0.3, color='cornflowerblue', zorder=10)
    ax2.set_xlabel('threshold')
    ax2.set_ylabel('Reco - Target cardinality')
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[2])
    _max = max(max(truth_card), max(reco_card_dict[ind_th]))
    bins = np.linspace(0, _max, _max + 1)
    ax3.hist2d(truth_card, reco_card_dict[ind_th], bins=bins, norm=LogNorm(), cmap='cool')
    ax3.plot([0, _max], [0, _max], ls='--', color='black', zorder=10)
    ax3.set_xlabel('Target cardinality')
    ax3.set_ylabel('Reco cardinality')

    return fig
