"""Particle-flow performance report: jet and particle metrics and plots from in-memory
truth and predicted particles (``run_report_from_arrays``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import awkward as ak
import numpy as np

# non-interactive backend (batch jobs without a display)
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .helper_dicts import get_class_mass_dict
from .helper_fns import cardinality_plot, marginal_distributions, plot_jets
from .jet_helper import JetHelper

log = logging.getLogger(__name__)


def _flatten(arr: ak.Array) -> np.ndarray:
    """Flatten an awkward jagged array to a 1D numpy array."""
    return ak.to_numpy(ak.flatten(arr, axis=None))


def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Compute numer/denom with NaN/inf filtering."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = numer / denom
    return r[np.isfinite(r)]


def _compute_mass_from_class(class_arr: ak.Array) -> ak.Array:
    """Convert integer particle class labels to per-particle masses."""
    if ak.count(class_arr) == 0:
        return ak.zeros_like(class_arr, dtype=np.float32)

    n_classes = int(ak.max(class_arr)) + 1
    try:
        class_mass = get_class_mass_dict(n_classes)
    except Exception:
        # Unknown class schema; fall back to zero masses.
        return ak.zeros_like(class_arr, dtype=np.float32)
    mass = ak.zeros_like(class_arr, dtype=np.float32)
    for i, m in class_mass.items():
        mass = ak.where(class_arr == i, float(m), mass)
    return mass


def _event_display(
    truth: Dict[str, ak.Array],
    reco: Dict[str, ak.Array],
    outpath: Path,
    reco_alpha: Optional[ak.Array] = None,
    num_events: int = 6,
    seed: int = 0,
) -> None:
    """Write a simple eta-phi event display figure."""
    rng = np.random.default_rng(seed)
    n_evt = len(truth["pt"])
    if n_evt == 0:
        return

    idxs = rng.integers(0, n_evt, size=min(num_events, n_evt))
    n_col = min(3, len(idxs))
    n_row = int(np.ceil(len(idxs) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row), dpi=200)
    axes = np.atleast_1d(axes).reshape(n_row, n_col)

    for ax, idx in zip(axes.flatten(), idxs):
        t_pt = truth["pt"][idx]
        t_eta = truth["eta"][idx]
        t_phi = truth["phi"][idx]

        r_pt = reco["pt"][idx]
        r_eta = reco["eta"][idx]
        r_phi = reco["phi"][idx]

        alpha = None
        if reco_alpha is not None:
            alpha = ak.to_numpy(reco_alpha[idx])

        ax.scatter(
            ak.to_numpy(r_eta),
            ak.to_numpy(r_phi),
            s=np.log(ak.to_numpy(r_pt) + 1.0) * 40.0,
            c="red",
            alpha=alpha if alpha is not None else 0.7,
            label="reco",
        )
        ax.scatter(
            ak.to_numpy(t_eta),
            ak.to_numpy(t_phi),
            s=np.log(ak.to_numpy(t_pt) + 1.0) * 40.0,
            c="cornflowerblue",
            marker="x",
            label="truth",
        )
        ax.set_xlabel("eta")
        ax.set_ylabel("phi")
        ax.set_title(f"event {int(idx)}")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True)
        ax.legend()

    for ax in axes.flatten()[len(idxs) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def build_pflow_jets_from_arrays(
    truth: Dict[str, ak.Array],
    reco: Dict[str, ak.Array],
    reco_ind: Optional[ak.Array] = None,
    truth_class: Optional[ak.Array] = None,
    reco_class: Optional[ak.Array] = None,
) -> tuple[Dict[str, ak.Array], Dict[str, ak.Array], Dict[str, ak.Array], Dict[str, ak.Array], Optional[ak.Array]]:
    """Build particle-level and jet-level truth/reco from in-memory awkward arrays.

    Inputs:
        truth: dict with keys 'pt', 'eta', 'phi' (each ak.Array, jagged per event)
        reco: dict with keys 'pt', 'eta', 'phi'
        reco_ind: optional indicator array (jagged per event)
        truth_class: optional class labels for mass
        reco_class: optional class labels for mass

    Outputs:
        (truth, reco, truth_jets, reco_jets, reco_ind)
    """
    truth_mass = (
        _compute_mass_from_class(truth_class) if truth_class is not None else ak.zeros_like(truth["pt"])
    )
    reco_mass = (
        _compute_mass_from_class(reco_class) if reco_class is not None else ak.zeros_like(reco["pt"])
    )

    jet_helper = JetHelper()
    truth_jets = {}
    reco_jets = {}
    truth_jets["$p_T$"], truth_jets["$\\eta$"], truth_jets["$\\phi$"], truth_jets["$E$"], _ = \
        jet_helper.compute_jets_PtEtaPhiM(truth["pt"], truth["eta"], truth["phi"], truth_mass)
    reco_jets["$p_T$"], reco_jets["$\\eta$"], reco_jets["$\\phi$"], reco_jets["$E$"], _ = \
        jet_helper.compute_jets_PtEtaPhiM(reco["pt"], reco["eta"], reco["phi"], reco_mass)

    return truth, reco, truth_jets, reco_jets, reco_ind


def run_report_from_arrays(
    truth: Dict[str, ak.Array],
    reco: Dict[str, ak.Array],
    reco_ind: Optional[ak.Array] = None,
    outdir: Optional[str | Path] = None,
    output_modality: str = "truthpart",
    ind_threshold: float = 0.45,
    n_event_displays: int = 6,
) -> Dict[str, float]:
    """Generate PF performance plots + metrics from in-memory truth/reco arrays.

    Inputs:
        truth: dict with keys 'pt', 'eta', 'phi' (ak.Array jagged per event)
        reco: dict with keys 'pt', 'eta', 'phi'
        reco_ind: optional indicator array
        outdir: if set, write plots and metrics.json there; otherwise only compute metrics
        output_modality: label for titles
        ind_threshold: threshold for cardinality
        n_event_displays: number of event displays to draw (if outdir set)

    Outputs:
        metrics dictionary
    """
    truth_jets = {}
    reco_jets = {}
    _, _, truth_jets, reco_jets, reco_ind = build_pflow_jets_from_arrays(
        truth=truth, reco=reco, reco_ind=reco_ind
    )

    pt_response = _safe_ratio(ak.to_numpy(reco_jets["$p_T$"]), ak.to_numpy(truth_jets["$p_T$"]))
    metrics: Dict[str, float] = {
        "n_events": float(len(truth["pt"])),
        "mean_jet_pt_response": float(np.mean(pt_response)) if pt_response.size else float("nan"),
        "median_jet_pt_response": float(np.median(pt_response)) if pt_response.size else float("nan"),
        "std_jet_pt_response": float(np.std(pt_response)) if pt_response.size else float("nan"),
        "iqr_jet_pt_response": float(np.percentile(pt_response, 75) - np.percentile(pt_response, 25))
        if pt_response.size
        else float("nan"),
    }

    # ---- per-feature jet-level response stats (E, mass ratios; eta, phi residuals) ----
    truth_E_np = ak.to_numpy(truth_jets["$E$"])
    reco_E_np  = ak.to_numpy(reco_jets["$E$"])
    e_response = _safe_ratio(reco_E_np, truth_E_np)
    truth_eta_np = ak.to_numpy(truth_jets["$\\eta$"])
    reco_eta_np  = ak.to_numpy(reco_jets["$\\eta$"])
    truth_pt_np  = ak.to_numpy(truth_jets["$p_T$"])
    reco_pt_np   = ak.to_numpy(reco_jets["$p_T$"])
    # m^2 = E^2 - (pT cosh(eta))^2 ; clamp at 0 to guard against numerical negatives
    truth_mass_np = np.sqrt(np.maximum(truth_E_np**2 - (truth_pt_np * np.cosh(truth_eta_np))**2, 0.0))
    reco_mass_np  = np.sqrt(np.maximum(reco_E_np**2  - (reco_pt_np  * np.cosh(reco_eta_np))**2,  0.0))
    mass_response = _safe_ratio(reco_mass_np, truth_mass_np)
    truth_phi_np = ak.to_numpy(truth_jets["$\\phi$"])
    reco_phi_np  = ak.to_numpy(reco_jets["$\\phi$"])
    eta_residual = reco_eta_np - truth_eta_np
    phi_residual = np.angle(np.exp(1j * (reco_phi_np - truth_phi_np)))   # wraps to [-pi, pi]

    def _stats(arr, prefix):
        if arr.size == 0:
            for s in ("mean", "median", "std", "iqr"):
                metrics[f"{s}_{prefix}"] = float("nan")
            return
        metrics[f"mean_{prefix}"]   = float(np.mean(arr))
        metrics[f"median_{prefix}"] = float(np.median(arr))
        metrics[f"std_{prefix}"]    = float(np.std(arr))
        metrics[f"iqr_{prefix}"]    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    _stats(e_response,    "jet_E_response")
    _stats(mass_response, "jet_mass_response")
    _stats(eta_residual,  "jet_eta_residual")
    _stats(phi_residual,  "jet_phi_residual")

    # ---- per-class fractions (if truth/reco carry 'class') ----
    if "class" in truth and "class" in reco:
        try:
            flat_truth_class = _flatten(truth["class"])
            flat_reco_class  = _flatten(reco["class"])
            if flat_truth_class.size and flat_reco_class.size:
                classes = np.unique(np.concatenate([flat_truth_class, flat_reco_class]))
                for c in classes:
                    metrics[f"frac_truth_class_{int(c)}"] = float((flat_truth_class == c).mean())
                    metrics[f"frac_reco_class_{int(c)}"]  = float((flat_reco_class  == c).mean())
        except Exception as _e:
            # Don't let per-class stats break the rest of the report.
            log.warning("per-class stats skipped: %s", _e)

    # ---- per-PARTICLE pT-rank-matched residuals + 1D KS/Wasserstein ----
    # pT-rank matching is loss-aligned (the model emits particles in pT order),
    # so residual_i = reco[i] - truth[i] after pT-desc sort on both sides.
    # Truncate to min(n_truth, n_reco) per event.
    def _ks(a, b):
        if a.size == 0 or b.size == 0:
            return float("nan")
        try:
            from scipy.stats import ks_2samp
            return float(ks_2samp(a, b).statistic)
        except Exception:
            return float("nan")
    def _wd(a, b):
        if a.size == 0 or b.size == 0:
            return float("nan")
        try:
            from scipy.stats import wasserstein_distance
            return float(wasserstein_distance(a, b))
        except Exception:
            return float("nan")

    particle_residuals: Dict[str, np.ndarray] = {}
    if "pt" in truth and "pt" in reco:
        truth_pt_list = truth["pt"]; reco_pt_list = reco["pt"]
        truth_eta_list = truth.get("eta"); reco_eta_list = reco.get("eta")
        truth_phi_list = truth.get("phi"); reco_phi_list = reco.get("phi")
        pt_ratio, eta_resid, phi_resid = [], [], []
        for ev in range(len(truth_pt_list)):
            tp = ak.to_numpy(truth_pt_list[ev])
            rp = ak.to_numpy(reco_pt_list[ev])
            if tp.size == 0 or rp.size == 0:
                continue
            # sort both by pT desc, pair by index, truncate to min
            t_idx = np.argsort(-tp)
            r_idx = np.argsort(-rp)
            n = min(tp.size, rp.size)
            t_pair = t_idx[:n]; r_pair = r_idx[:n]
            with np.errstate(divide="ignore", invalid="ignore"):
                pt_ratio.append(rp[r_pair] / tp[t_pair])
            if truth_eta_list is not None and reco_eta_list is not None:
                te = ak.to_numpy(truth_eta_list[ev]); re = ak.to_numpy(reco_eta_list[ev])
                eta_resid.append(re[r_pair] - te[t_pair])
            if truth_phi_list is not None and reco_phi_list is not None:
                tph = ak.to_numpy(truth_phi_list[ev]); rph = ak.to_numpy(reco_phi_list[ev])
                # wrap into [-pi, pi]
                phi_resid.append(np.angle(np.exp(1j * (rph[r_pair] - tph[t_pair]))))
        if pt_ratio:
            arr = np.concatenate(pt_ratio); arr = arr[np.isfinite(arr)]
            particle_residuals["pt_ratio"] = arr
        if eta_resid:
            particle_residuals["eta_residual"] = np.concatenate(eta_resid)
        if phi_resid:
            particle_residuals["phi_residual"] = np.concatenate(phi_resid)
    # Aggregate stats + KS + Wasserstein per particle kinematic
    for name, arr in particle_residuals.items():
        if arr.size == 0:
            continue
        metrics[f"mean_particle_{name}"]   = float(np.mean(arr))
        metrics[f"median_particle_{name}"] = float(np.median(arr))
        metrics[f"std_particle_{name}"]    = float(np.std(arr))
        metrics[f"iqr_particle_{name}"]    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    # 1D KS + Wasserstein on the flattened (unmatched) truth/reco distributions
    for var in ("pt", "eta", "phi"):
        if var not in truth or var not in reco:
            continue
        t = _flatten(truth[var]); r = _flatten(reco[var])
        metrics[f"ks_particle_{var}"]          = _ks(t, r)
        metrics[f"wasserstein_particle_{var}"] = _wd(t, r)
    # Also: KS on per-event cardinality (number of particles per event)
    try:
        n_truth_arr = ak.to_numpy(ak.num(truth["pt"], axis=1)).astype(float)
        n_reco_arr  = ak.to_numpy(ak.num(reco["pt"],  axis=1)).astype(float)
        metrics["ks_cardinality"]          = _ks(n_truth_arr, n_reco_arr)
        metrics["wasserstein_cardinality"] = _wd(n_truth_arr, n_reco_arr)
    except Exception:
        pass
    # Stash residual arrays for plot generation downstream
    metrics["_b6b_particle_residuals"] = particle_residuals

    if reco_ind is not None:
        metrics["mean_reco_cardinality_at_threshold"] = float(
            np.mean(ak.to_numpy(ak.sum(reco_ind > ind_threshold, axis=1)))
        )
    else:
        reco_card = ak.to_numpy(ak.num(reco["pt"], axis=1))
        metrics["mean_reco_cardinality_at_threshold"] = float(np.mean(reco_card))

    # --- Structured data for W&B cross-run comparison ---
    flat = {
        "truth": {k: _flatten(v) for k, v in truth.items()},
        "reco": {k: _flatten(v) for k, v in reco.items()},
    }
    truth_card = ak.to_numpy(ak.num(truth["pt"], axis=1))
    if reco_ind is not None:
        reco_card = ak.to_numpy(ak.sum(reco_ind > ind_threshold, axis=1))
    else:
        reco_card = ak.to_numpy(ak.num(reco["pt"], axis=1))

    histograms = {}
    histograms['jet_pt_response'] = np.histogram(pt_response, bins=50, range=(0, 2))
    for var in ['pt', 'eta', 'phi']:
        if var == 'phi':
            range_ = (-np.pi, np.pi)
        elif var == 'pt':
            range_ = (0, float(np.percentile(flat['truth'][var], 99))) if flat['truth'][var].size else (0, 1)
        else:
            if flat['truth'][var].size:
                lo = float(np.percentile(flat['truth'][var], 1))
                hi = float(np.percentile(flat['truth'][var], 99))
                range_ = (lo, hi)
            else:
                range_ = (-1, 1)
        bins = np.linspace(range_[0], range_[1], 51)
        histograms[f'particle_{var}_truth'] = np.histogram(flat['truth'][var], bins=bins)
        histograms[f'particle_{var}_reco'] = np.histogram(flat['reco'][var], bins=bins)
    if reco_ind is not None:
        histograms['indicator'] = np.histogram(
            ak.to_numpy(ak.flatten(reco_ind)), bins=40, range=(0, 1)
        )
    max_card = int(max(truth_card.max(), reco_card.max())) + 2 if (truth_card.size and reco_card.size) else 2
    histograms['cardinality_truth'] = np.histogram(truth_card, bins=np.arange(0, max_card))
    histograms['cardinality_reco'] = np.histogram(reco_card, bins=np.arange(0, max_card))

    # ---- histograms for E/mass response + eta/phi residuals + per-class fractions ----
    if e_response.size:
        histograms['jet_E_response']    = np.histogram(e_response,    bins=50, range=(0, 2))
    if mass_response.size:
        histograms['jet_mass_response'] = np.histogram(mass_response, bins=50, range=(0, 2))
    if eta_residual.size:
        # jet-level residuals are typically O(0.01) — narrow bins so the
        # distribution shape is visible in the wandb histogram chart.
        histograms['jet_eta_residual']  = np.histogram(eta_residual,  bins=50, range=(-0.05, 0.05))
    if phi_residual.size:
        histograms['jet_phi_residual']  = np.histogram(phi_residual,  bins=50, range=(-0.05, 0.05))
    for jet_var, t_arr, r_arr, hrange in [
        ('jet_pt',   truth_pt_np,   reco_pt_np,   (0, float(np.percentile(truth_pt_np, 99)) if truth_pt_np.size else 1)),
        ('jet_eta',  truth_eta_np,  reco_eta_np,  (-3.5, 3.5)),
        ('jet_phi',  truth_phi_np,  reco_phi_np,  (-np.pi, np.pi)),
        ('jet_E',    truth_E_np,    reco_E_np,    (0, float(np.percentile(truth_E_np, 99)) if truth_E_np.size else 1)),
        ('jet_mass', truth_mass_np, reco_mass_np, (0, float(np.percentile(truth_mass_np, 99)) if truth_mass_np.size else 1)),
    ]:
        if t_arr.size:
            bins = np.linspace(hrange[0], hrange[1], 51)
            histograms[f'{jet_var}_truth'] = np.histogram(t_arr, bins=bins)
            histograms[f'{jet_var}_reco']  = np.histogram(r_arr, bins=bins)
    # per-particle (matched, pT-rank) residual distributions as histograms
    for name, arr in particle_residuals.items():
        if arr.size == 0:
            continue
        if name == "pt_ratio":
            histograms[f'particle_{name}']     = np.histogram(arr, bins=50, range=(0, 2))
        else:  # eta_residual or phi_residual
            histograms[f'particle_{name}']     = np.histogram(arr, bins=50, range=(-0.3, 0.3))

    jet_table_data = {
        'truth_pt': ak.to_numpy(truth_jets['$p_T$']),
        'truth_eta': ak.to_numpy(truth_jets['$\\eta$']),
        'truth_phi': ak.to_numpy(truth_jets['$\\phi$']),
        'truth_E': ak.to_numpy(truth_jets['$E$']),
        'truth_mass': truth_mass_np,
        'reco_pt': ak.to_numpy(reco_jets['$p_T$']),
        'reco_eta': ak.to_numpy(reco_jets['$\\eta$']),
        'reco_phi': ak.to_numpy(reco_jets['$\\phi$']),
        'reco_E': ak.to_numpy(reco_jets['$E$']),
        'reco_mass': reco_mass_np,
        'pt_response': pt_response,
        'E_response': e_response if e_response.size == truth_pt_np.size else np.full_like(truth_pt_np, np.nan),
        'mass_response': mass_response if mass_response.size == truth_pt_np.size else np.full_like(truth_pt_np, np.nan),
        'eta_residual': eta_residual,
        'phi_residual': phi_residual,
    }

    metrics['_histograms'] = histograms
    metrics['_jet_table'] = jet_table_data

    if outdir is None:
        return metrics

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Jet plots
    _cosmetic = {
        "COLORS": {"pflow": "tab:orange"},
        "HISTTYPES": {"pflow": "step"},
        "ALPHAS": {"pflow": 1.0},
        "LINESTYLES": {"pflow": "-"},
    }
    jets_combined = {"pflow": {"target": truth_jets, "reco": reco_jets}}
    fig = plot_jets(jets_combined, cosmetic_dict=_cosmetic)
    fig.savefig(outdir / "jet_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    _marginal_cosmetic = {
        "COLORS": {"truth": "cornflowerblue", "reco": "tab:orange"},
        "HISTTYPES": {"truth": "stepfilled", "reco": "step"},
        "ALPHAS": {"truth": 0.5, "reco": 1.0},
        "LINESTYLES": {"truth": "-", "reco": "-"},
    }
    fig = marginal_distributions(
        data_flat_dict=flat,
        names=["truth", "reco"],
        vars=["pt", "eta", "phi"],
        log_var_pos=[0],
        cosmetic_dict=_marginal_cosmetic,
    )
    fig.savefig(outdir / "marginals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if reco_ind is not None:
        fig = cardinality_plot(reco_ind_array=reco_ind, truth_card=truth_card, ind_th=ind_threshold)
        fig.savefig(outdir / "cardinality.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # jet-level residuals plot (pt response, E response, mass response,
    # eta residual, phi residual). Tight ranges so distributions are visible.
    jet_spec = [
        (pt_response,    "pT response (reco/truth)", (0, 2)),
        (e_response,     "E response (reco/truth)",  (0, 2)),
        (mass_response,  "mass response (reco/truth)", (0, 2)),
        (eta_residual,   "eta residual (reco-truth)", (-0.05, 0.05)),
        (phi_residual,   "phi residual (reco-truth)", (-0.05, 0.05)),
    ]
    n_jet = sum(1 for a, _, _ in jet_spec if a.size > 0)
    if n_jet:
        fig, axes = plt.subplots(1, n_jet, figsize=(4 * n_jet, 3.2), dpi=120)
        if n_jet == 1: axes = [axes]
        ax_iter = iter(axes)
        for arr, lbl, rng in jet_spec:
            if arr.size == 0: continue
            ax = next(ax_iter)
            bins = np.linspace(rng[0], rng[1], 51)
            ax.hist(arr, bins=bins, color="cornflowerblue", alpha=0.8)
            med = float(np.median(arr))
            iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            ax.axvline(med, color="black", linestyle="--", lw=1)
            ax.set_xlabel(lbl); ax.set_ylabel("jets")
            ax.set_title(f"median={med:.4f}  IQR={iqr:.4f}", fontsize=10)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "jet_residuals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # per-particle residual plot (pT-rank-matched)
    if particle_residuals:
        n_panels = sum(1 for _, a in particle_residuals.items() if a.size > 0)
        if n_panels:
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.2), dpi=120)
            if n_panels == 1:
                axes = [axes]
            ax_iter = iter(axes)
            spec = {
                "pt_ratio":     ("pT_reco / pT_truth", (0, 2)),
                "eta_residual": ("eta_reco - eta_truth", (-0.3, 0.3)),
                "phi_residual": ("phi_reco - phi_truth (wrapped)", (-0.3, 0.3)),
            }
            for name, arr in particle_residuals.items():
                if arr.size == 0:
                    continue
                ax = next(ax_iter)
                lbl, rng = spec.get(name, (name, None))
                bins = np.linspace(rng[0], rng[1], 51) if rng is not None else 51
                ax.hist(arr, bins=bins, color="crimson", alpha=0.75)
                med = float(np.median(arr))
                iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
                ax.axvline(med, color="black", linestyle="--", lw=1)
                ax.set_xlabel(lbl); ax.set_ylabel("particles")
                ax.set_title(f"median={med:.3f}  IQR={iqr:.3f}", fontsize=10)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / "particle_residuals.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if n_event_displays > 0:
        _event_display(
            truth=truth,
            reco=reco,
            outpath=outdir / "event_displays.png",
            reco_alpha=reco_ind,
            num_events=n_event_displays,
            seed=0,
        )

    # Strip private keys before JSON serialization (numpy/hist objects aren't JSON-serializable)
    metrics_json = {k: v for k, v in metrics.items() if not k.startswith('_')}
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, sort_keys=True)

    return metrics
