"""Distribution metrics for generated detector objects (detector simulation:
truthpart -> topo, track).

Jet response (``pflow_report``) is not meaningful when the outputs are clusters or
tracks, so this report compares distributions: a classifier two-sample test (C2ST),
per-variable Kolmogorov-Smirnov and Wasserstein distances, per-event energy moments
and a 2D (eta, phi) occupancy chi-squared. The C2ST MLP is small and runs on CPU; a
GPU is used when available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import awkward as ak
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _flatten(arr: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.flatten(arr, axis=None))


def _per_event_sum(arr: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.sum(arr, axis=1))


def _per_event_count(arr: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.num(arr, axis=1))


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance. Returns sup|F_a - F_b|."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        # Fallback: inline two-sample KS to avoid hard dep
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        merged = np.concatenate([a_sorted, b_sorted])
        cdf_a = np.searchsorted(a_sorted, merged, side="right") / a.size
        cdf_b = np.searchsorted(b_sorted, merged, side="right") / b.size
        return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------
# classifier 2-sample test (C2ST)
# ----------------------------------------------------------------------------

class _C2STMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _compute_c2st(
    truth_features: np.ndarray,
    reco_features: np.ndarray,
    epochs: int = 30,
    batch_size: int = 1024,
    hidden: int = 64,
    seed: int = 0,
    device: str = "cpu",
    return_scores: bool = False,
) -> Tuple[float, float]:
    """Train a small MLP to distinguish truth (label 0) from reco (label 1).

    If ``return_scores=True``, also returns the held-out test-set classifier
    probabilities and labels as ``(auc, acc, prob_test, y_test)`` for the
    score-distribution diagnostic plot (truth vs generated score histograms).

    Returns (test_AUC, test_accuracy). AUC near 0.5 means indistinguishable.
    """
    if truth_features.shape[0] < 16 or reco_features.shape[0] < 16:
        if return_scores:
            return float("nan"), float("nan"), np.array([]), np.array([])
        return float("nan"), float("nan")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = np.concatenate([truth_features, reco_features], axis=0).astype(np.float32)
    y = np.concatenate([
        np.zeros(truth_features.shape[0], dtype=np.float32),
        np.ones(reco_features.shape[0], dtype=np.float32),
    ])

    # Standardize (helps small MLP converge)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    X = (X - mu) / sd

    # Stratified 70/30 split
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * 0.7)
    tr_idx, te_idx = idx[:split], idx[split:]

    # Lightning's val cycle runs inside torch.inference_mode() (stricter than
    # no_grad — tensors created here become "inference tensors" that error on
    # .backward() even with enable_grad). The C2ST MLP needs its own backward,
    # so escape inference mode entirely AND re-enable grad, and create all
    # tensors/the model inside that scope.
    with torch.inference_mode(False), torch.enable_grad():
        X_tr = torch.from_numpy(X[tr_idx]).to(device)
        y_tr = torch.from_numpy(y[tr_idx]).to(device)
        X_te = torch.from_numpy(X[te_idx]).to(device)
        y_te = torch.from_numpy(y[te_idx]).to(device)

        model = _C2STMLP(X.shape[1], hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        for ep in range(epochs):
            perm = torch.randperm(X_tr.shape[0], device=device)
            for i in range(0, X_tr.shape[0], batch_size):
                sel = perm[i:i + batch_size]
                logits = model(X_tr[sel])
                loss = loss_fn(logits, y_tr[sel])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        prob = torch.sigmoid(logits).cpu().numpy()
        pred = (prob > 0.5).astype(np.float32)
    y_te_np = y_te.cpu().numpy()
    acc = float((pred == y_te_np).mean())

    # AUC via Mann-Whitney U formulation (no sklearn dep)
    try:
        # average ranks over ties — plain argsort ranks bias AUC away from
        # 0.5 when scores are near-degenerate (indistinguishable samples)
        from scipy.stats import rankdata
        ranks = rankdata(prob)
        n_pos = float((y_te_np == 1).sum())
        n_neg = float((y_te_np == 0).sum())
        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
        else:
            sum_ranks_pos = float(ranks[y_te_np == 1].sum())
            U = sum_ranks_pos - n_pos * (n_pos + 1) / 2
            auc = float(U / (n_pos * n_neg))
    except Exception:
        auc = float("nan")
    if return_scores:
        return auc, acc, prob, y_te_np
    return auc, acc


# ----------------------------------------------------------------------------
# main entrypoint
# ----------------------------------------------------------------------------

def _resolve_kinematic_field(truth: Dict[str, ak.Array], reco: Dict[str, ak.Array]) -> str:
    """Pick the canonical 'magnitude' field — topo uses 'e', tracks/particles use 'pt'."""
    for cand in ("pt", "e", "energy"):
        if cand in truth and cand in reco:
            return cand
    raise KeyError(f"No magnitude field (pt/e/energy) in truth+reco. "
                   f"truth keys: {list(truth.keys())}, reco keys: {list(reco.keys())}")


def run_generative_report_from_arrays(
    truth: Dict[str, ak.Array],
    reco: Dict[str, ak.Array],
    outdir: Optional[str | Path] = None,
    output_modality: str = "topo",
    c2st_epochs: int = 30,
    c2st_hidden: int = 64,
    occupancy_bins: Tuple[int, int] = (32, 32),
    seed: int = 0,
) -> Dict[str, Any]:
    """Generative-quality metrics for ONE output modality.

    Inputs:
        truth, reco: dicts with keys 'pt', 'eta', 'phi' (each ak.Array jagged
            per event). 'class' and 'e' are optional.
        outdir: if set, write side-by-side comparison plots
        output_modality: label for plot titles + key disambiguation

    Returns:
        metrics dict with scalar keys (val_pflow_{k} after the Lightning hook
        wraps it). All metrics report distributional similarity — smaller is
        better for KS and occupancy_chi2; closer-to-0.5 is better for
        quick_marginal_particle_auc. That AUC is a quick check: a small MLP separates
        single particles (not events), so it sees only the per-particle marginals and
        is not the event-level classifier AUC of ``hep4m.judge``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kin_field = _resolve_kinematic_field(truth, reco)  # 'pt' for tracks/particles, 'e' for topo
    metrics: Dict[str, Any] = {"n_events": float(len(truth[kin_field]))}

    # --- 1D KS on per-particle kinematics (uses kin_field for magnitude) ---
    for var in (kin_field, "eta", "phi"):
        if var not in truth or var not in reco:
            continue
        t = _flatten(truth[var]); r = _flatten(reco[var])
        # Key uses canonical "kin" name so dashboards overlay across modalities
        key_var = "kin" if var == kin_field else var
        metrics[f"ks_particle_{key_var}"] = _ks_2samp(t, r)
        metrics[f"wasserstein_particle_{key_var}"] = _wasserstein_1d(t, r)

    # --- per-event scalars: multiplicity + total kinematics ---
    n_truth = _per_event_count(truth[kin_field])
    n_reco  = _per_event_count(reco[kin_field])
    metrics["ks_event_multiplicity"]         = _ks_2samp(n_truth.astype(float), n_reco.astype(float))
    metrics["wasserstein_event_multiplicity"] = _wasserstein_1d(n_truth.astype(float), n_reco.astype(float))
    metrics["mean_event_multiplicity_truth"]  = float(n_truth.mean()) if n_truth.size else float("nan")
    metrics["mean_event_multiplicity_reco"]   = float(n_reco.mean())  if n_reco.size  else float("nan")

    total_kin_truth = _per_event_sum(truth[kin_field])
    total_kin_reco  = _per_event_sum(reco[kin_field])
    metrics["ks_event_total_kin"]    = _ks_2samp(total_kin_truth, total_kin_reco)
    metrics["mean_event_total_kin_truth"] = float(total_kin_truth.mean()) if total_kin_truth.size else float("nan")
    metrics["mean_event_total_kin_reco"]  = float(total_kin_reco.mean())  if total_kin_reco.size  else float("nan")
    metrics["_kin_field_name"] = kin_field   # record which field was the magnitude

    # --- 2D (eta, phi) occupancy chi-squared ---
    eta_t = _flatten(truth["eta"]); phi_t = _flatten(truth["phi"])
    eta_r = _flatten(reco["eta"]);  phi_r = _flatten(reco["phi"])
    if eta_t.size and eta_r.size:
        nbe, nbp = occupancy_bins
        eta_edges = np.linspace(-3.5, 3.5, nbe + 1)
        phi_edges = np.linspace(-np.pi, np.pi, nbp + 1)
        H_t, _, _ = np.histogram2d(eta_t, phi_t, bins=(eta_edges, phi_edges))
        H_r, _, _ = np.histogram2d(eta_r, phi_r, bins=(eta_edges, phi_edges))
        # Normalize to probability mass
        H_t_n = H_t / max(H_t.sum(), 1.0)
        H_r_n = H_r / max(H_r.sum(), 1.0)
        # chi^2 with Poisson-flavored denominator (use H_t_n + eps to avoid div0)
        denom = np.maximum(H_t_n, 1e-6)
        metrics["chi2_occupancy_eta_phi"] = float(np.sum((H_r_n - H_t_n) ** 2 / denom))
        # also report total-variation distance — bounded [0, 1], easier to read
        metrics["tv_occupancy_eta_phi"]   = float(0.5 * np.abs(H_r_n - H_t_n).sum())
    else:
        metrics["chi2_occupancy_eta_phi"] = float("nan")
        metrics["tv_occupancy_eta_phi"]   = float("nan")

    _extra_keys = [k for k in ("d0", "z0", "em_frac", "rho", "mass")
                   if k in truth and k in reco]
    def _feats(d):
        cols = []
        cols.append(np.log1p(np.clip(_flatten(d[kin_field]), 0, None)))
        cols.append(_flatten(d["eta"]))
        # Use the RAW generated cosphi/sinphi when present: recomputing them
        # from phi = atan2(sin, cos) silently projects the pair back onto the
        # unit circle and hides a real failure mode -- parallel decoders emit
        # (cos, sin) pairs with cos^2+sin^2 far from 1 (HEP4M sampled:
        # 0.97 +/- 0.30) while AR decoding learns the identity (1.000 +/- 0.008).
        if "cosphi" in d and "sinphi" in d:
            cols.append(_flatten(d["cosphi"]))
            cols.append(_flatten(d["sinphi"]))
        else:
            cols.append(np.cos(_flatten(d["phi"])))
            cols.append(np.sin(_flatten(d["phi"])))
        if "class" in d:
            cols.append(_flatten(d["class"]).astype(np.float32))
        for k in _extra_keys:
            cols.append(_flatten(d[k]).astype(np.float32))
        return np.stack(cols, axis=1)
    if eta_t.size and eta_r.size:
        Xt = _feats(truth); Xr = _feats(reco)
        auc, acc = _compute_c2st(
            Xt, Xr, epochs=c2st_epochs, hidden=c2st_hidden, seed=seed, device=device,
        )
        metrics["quick_marginal_particle_auc"] = auc
        metrics["quick_marginal_particle_acc"] = acc
    else:
        metrics["quick_marginal_particle_auc"] = float("nan")
        metrics["quick_marginal_particle_acc"] = float("nan")

    # --- Histograms (mirror pflow_report shape: dict[name] -> (counts, edges)) ---
    histograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if n_truth.size and n_reco.size:
        max_n = int(max(n_truth.max(), n_reco.max())) + 2
        histograms[f"{output_modality}_n_particles_truth"] = np.histogram(n_truth, bins=np.arange(0, max_n))
        histograms[f"{output_modality}_n_particles_reco"]  = np.histogram(n_reco,  bins=np.arange(0, max_n))
    if total_kin_truth.size and total_kin_reco.size:
        bins = np.linspace(0, float(np.percentile(np.concatenate([total_kin_truth, total_kin_reco]), 99)) + 1, 51)
        histograms[f"{output_modality}_total_kin_truth"] = np.histogram(total_kin_truth, bins=bins)
        histograms[f"{output_modality}_total_kin_reco"]  = np.histogram(total_kin_reco,  bins=bins)
    metrics["_histograms"] = histograms

    # --- Optional: write side-by-side 2D occupancy plots ---
    if outdir is not None and eta_t.size and eta_r.size:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, H, name in [(axes[0], H_t, "truth"), (axes[1], H_r, "reco")]:
                im = ax.imshow(H.T, origin="lower", aspect="auto",
                               extent=(-3.5, 3.5, -np.pi, np.pi), cmap="viridis")
                ax.set_xlabel("eta"); ax.set_ylabel("phi")
                ax.set_title(f"{output_modality} {name} (n={int(H.sum())})")
                fig.colorbar(im, ax=ax)
            fig.suptitle(f"Occupancy ({output_modality}) — TV={metrics['tv_occupancy_eta_phi']:.3f}")
            fig.tight_layout()
            fig.savefig(outdir / f"occupancy_{output_modality}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.warning("occupancy plot skipped: %s", e)

    return metrics
