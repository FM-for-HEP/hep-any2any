"""Evaluation of generated or reconstructed events: loaders, metrics and C2ST.

One interface over the prediction ROOT files of the three engines (``hep4m``,
``nano_hep`` and HGPflow reference files, ``hgpflow``). It wraps the per-direction
reports (``pflow_report.run_report_from_arrays`` for particle flow,
``generative_report.run_generative_report_from_arrays`` for generated detector
objects) and the classifier two-sample tests (C2ST).

Cases (input -> output direction evaluated):
- pflow    : (topo, track) -> truthpart, particle-flow reconstruction
- sim      : truthpart -> (topo, track), detector simulation
- flashsim : truthpart -> hgpfpart

CLI:
    python -m hep4m.performance.evaluate metrics --root-path <pred.root> --engine <e> --case <c> \
        --run-id <name> --metrics-out <out.json> [--max-events N] \
        [--raw-truth <raw truth ROOT> [--store <token store split dir>]]

By default the prediction file's own target is the reference: the tokenised and decoded
truth. ``--raw-truth`` (particle flow only) scores against the raw truth particles instead,
on the events whose tokenised truth kept every raw particle (``raw_truth_selection``).
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import awkward as ak
import numpy as np
import uproot
from .generative_report import _compute_c2st, run_generative_report_from_arrays
from .helper_dicts import get_class_mass_dict
from .pflow_report import run_report_from_arrays

C2ST_DEFAULTS = dict(epochs=100, hidden=64, batch_size=2048, n_boot=10)

# Input and output modalities of each evaluated direction.
CASE_DIRECTIONS = {
    "pflow": {"inputs": ["topo", "track"], "outputs": ["truthpart"]},
    "sim":   {"inputs": ["truthpart"],     "outputs": ["topo", "track"]},
    # Flash simulation: predict the HGPflow particles directly from the truth particles.
    "flashsim": {"inputs": ["truthpart"], "outputs": ["hgpfpart"]},
}

# Per-modality feature schemas. `extras` are loaded if present but not used in
# the canonical metric battery — they show up in the marginals grid only.
MODALITY_SCHEMA = {
    "truthpart": {"kinematic": "pt", "has_class": True,  "extras": []},
    "topo":      {"kinematic": "e",  "has_class": False, "extras": ["em_frac", "rho"]},
    "track":     {"kinematic": "pt", "has_class": False, "extras": ["d0", "z0"]},
    # HGPflow particles carry the same three classes as truth particles (checked
    # against the flash-simulation roots: class values are 0, 1, 2).
    "hgpfpart":  {"kinematic": "pt", "has_class": True,  "extras": []},
}

ENGINES = ("hep4m", "nano_hep", "hgpflow")

@dataclass
class RunSpec:
    case: str                  # key of CASE_DIRECTIONS
    run_id: str                # label written to the metrics file
    engine: str                # 'hep4m' | 'nano_hep' | 'hgpflow'
    pred_root: str             # path to the prediction .root file
    n_classes: int = 3         # 3 for HEP4M/nanoHEP, 5 for HGPflow

def _add_phi(d: Dict[str, ak.Array]) -> None:
    """In-place: compute phi from cosphi/sinphi if not already present."""
    if "phi" not in d and "sinphi" in d and "cosphi" in d:
        d["phi"] = np.arctan2(d["sinphi"], d["cosphi"])

def _add_mass_from_class(d: Dict[str, ak.Array], n_classes: int) -> None:
    """In-place: compute mass from class index using helper_dicts.get_class_mass_dict."""
    if "class" not in d:
        return
    cmap = get_class_mass_dict(n_classes)
    d["mass"] = ak.where(d["class"] <= 2, cmap[0], 0)
    for i in range(1, n_classes):
        d["mass"] = ak.where(d["class"] == i, cmap.get(i, 0.0), d["mass"])

def load_pred_root(run: RunSpec, entry_stop: Optional[int] = None) -> Dict[str, Dict[str, Dict[str, ak.Array]]]:
    """Load a prediction ROOT into a uniform schema:

        out[modality] = {
            'target': {pt|e, eta, cosphi, sinphi, phi, class?, mass?, ...},
            'reco':   {... same ...},
        }

    Handles three branch-name conventions:
    - HEP4M: ``{modality}_truth_{var}`` + ``{modality}_reco_{var}``
    - nanoHEP: same as HEP4M
    - HGPflow: ``truth_{var}`` + ``hgpflow_{var}`` + ``pred_ind`` mask

    ``entry_stop`` (optional) limits the read to the first N events (smoke tests).
    """
    if not os.path.exists(run.pred_root):
        raise FileNotFoundError(f"pred root not found: {run.pred_root}")
    tree = uproot.open(run.pred_root)["event_tree"]
    keys = set(tree.keys())

    out: Dict[str, Dict[str, Dict[str, ak.Array]]] = {}

    # HGPflow case: only truthpart, branches are 'truth_*' and 'hgpflow_*'.
    if run.engine == "hgpflow":
        mod = "truthpart"
        ind = tree["pred_ind"].array(entry_stop=entry_stop) > 0.5
        target, reco = {}, {}
        for v in ["pt", "eta", "phi", "class"]:
            if f"truth_{v}" in keys:
                target[v] = tree[f"truth_{v}"].array(entry_stop=entry_stop)
            if f"hgpflow_{v}" in keys:
                reco[v] = tree[f"hgpflow_{v}"].array(entry_stop=entry_stop)[ind]
        _add_mass_from_class(target, run.n_classes)
        _add_mass_from_class(reco, run.n_classes)
        out[mod] = {"target": target, "reco": reco}
        return out

    # HEP4M / nanoHEP case.
    case_outs = CASE_DIRECTIONS[run.case]["outputs"]
    for mod in case_outs:
        schema = MODALITY_SCHEMA[mod]
        kin = schema["kinematic"]
        vars_to_load = [kin, "eta", "cosphi", "sinphi"] + list(schema["extras"])
        if schema["has_class"]:
            vars_to_load.append("class")
        target, reco = {}, {}
        for v in vars_to_load:
            if f"{mod}_truth_{v}" in keys:
                target[v] = tree[f"{mod}_truth_{v}"].array(entry_stop=entry_stop)
            if f"{mod}_reco_{v}" in keys:
                reco[v] = tree[f"{mod}_reco_{v}"].array(entry_stop=entry_stop)
        if target.get(kin) is None or reco.get(kin) is None:
            # modality wasn't in this ROOT (engine produced an empty branch)
            continue
        _add_phi(target)
        _add_phi(reco)
        # PFlow report alias: it expects 'pt' key even for topo (kinematic=e).
        if kin != "pt":
            target["pt"] = target[kin]
            reco["pt"] = reco[kin]
        if schema["has_class"]:
            _add_mass_from_class(target, run.n_classes)
            _add_mass_from_class(reco, run.n_classes)
        out[mod] = {"target": target, "reco": reco}
    return out

def load_pred_event_numbers(run: RunSpec, entry_stop: Optional[int] = None) -> np.ndarray:
    """Read the per-event join keys from a prediction ROOT as a flat int64 array.

    nanoHEP roots store ``event_number`` as one-element lists per event;
    HEP4M / HGPflow roots store it flat. Both come back as shape ``(N,)``.
    NB the VALUE convention differs by engine — see ``match_cache_rows``.
    """
    tree = uproot.open(run.pred_root)["event_tree"]
    if "event_number" not in set(tree.keys()):
        raise KeyError(f"no 'event_number' branch in {run.pred_root}")
    arr = tree["event_number"].array(entry_stop=entry_stop)
    if arr.ndim > 1:
        if not ak.all(ak.num(arr) == 1):
            raise ValueError(f"jagged 'event_number' with !=1 entries/event in {run.pred_root}")
        arr = ak.flatten(arr)
    return ak.to_numpy(arr).astype(np.int64)

def load_raw_truth(path: str | Path, entry_stop: Optional[int] = None
                   ) -> Tuple[Dict[str, ak.Array], np.ndarray, int]:
    """Raw (not tokenised) truth particles of the test events, in test-store order.

    Reads either file of the record that holds them:
    - the HGPflow reference file (tree ``event_tree``, branches ``truth_{pt,eta,phi,class}``,
      five particle classes);
    - the raw COCOA file (tree ``EventTree``, branches ``particle_{pt,eta,phi,pdgid,track_idx}``);
      particles with pdgid -999 are dropped (4 in the test file; the HGPflow file does not
      have them), and the class follows the truth-particle tokeniser: photon (pdgid 22),
      charged (has a track), otherwise neutral hadron (three classes).
    Both then hold the same particles, in the same order and with the same float64 values,
    so the jet pT response is identical with either file; jet mass and energy use the class
    masses and differ slightly between the two class schemes.

    Returns ``(truth dict, event numbers, n_classes)``.
    """
    f = uproot.open(path)
    if "event_tree" in f:
        tree = f["event_tree"]
        d = {v: tree[f"truth_{v}"].array(entry_stop=entry_stop) for v in ("pt", "eta", "phi", "class")}
        ev = tree["event_number"].array(entry_stop=entry_stop, library="np")
        n_classes = 5
    elif "EventTree" in f:
        tree = f["EventTree"]
        a = tree.arrays(["particle_pt", "particle_eta", "particle_phi", "particle_pdgid", "particle_track_idx"],
                        entry_stop=entry_stop)
        keep = a["particle_pdgid"] != -999   # no generator record; absent from the HGPflow file
        d = {v: ak.values_astype(a[f"particle_{v}"][keep], np.float64) for v in ("pt", "eta", "phi")}
        cls = ak.where(a["particle_track_idx"][keep] >= 0, 0, 1)
        d["class"] = ak.where(a["particle_pdgid"][keep] == 22, 2, cls)
        ev = tree["eventNumber"].array(entry_stop=entry_stop, library="np")
        n_classes = 3
    else:
        raise KeyError(f"{path}: neither 'event_tree' (truth_*) nor 'EventTree' (particle_*)")
    d["cosphi"] = np.cos(d["phi"])
    d["sinphi"] = np.sin(d["phi"])
    _add_mass_from_class(d, n_classes)
    return d, np.asarray(ev, dtype=np.int64), n_classes


def raw_truth_selection(loaded: Dict[str, Dict[str, Dict[str, ak.Array]]], raw_truth: str | Path,
                        store_dir: str | Path, pred_event_numbers: Optional[np.ndarray] = None,
                        modality: str = "truthpart") -> Dict[str, Any]:
    """Replace the target of ``loaded[modality]`` by the raw truth particles and keep only
    the matched events, in place.

    Matched events are those whose tokenised truth kept every raw particle: the number of
    rows of the event in ``{store_dir}/truthpart_offsets.npy`` equals the number of raw
    particles. The tokeniser drops particles outside its eta-phi window, so in the other
    events the tokenised truth, and the models' inputs, miss part of the jet. Prediction
    row i is event i of the test store (the first N events when the file has N); the
    event numbers are checked when ``pred_event_numbers`` is given. Returns counts.
    """
    d = loaded[modality]
    n = len(d["reco"]["pt"])
    raw, raw_ev, _ = load_raw_truth(raw_truth, entry_stop=n)
    if len(raw_ev) < n:
        raise ValueError(f"{raw_truth} has {len(raw_ev)} events, the prediction file {n}")
    if pred_event_numbers is not None:
        pe = np.asarray(pred_event_numbers, dtype=np.int64)
        # HEP4M and HGPflow files carry the raw event number, nanoHEP files the store row
        if not (np.array_equal(pe, raw_ev) or np.array_equal(pe, np.arange(n))):
            raise ValueError("prediction events are not the first events of the test store in order; "
                             "--raw-truth needs a prediction file made on the test split from event 0")
    off = np.fromfile(Path(store_dir) / "truthpart_offsets.npy", dtype=np.int64)
    if len(off) < n + 1:
        raise ValueError(f"{store_dir}/truthpart_offsets.npy covers {len(off) - 1} events, need {n}")
    n_store = np.diff(off[:n + 1])
    matched = np.asarray(ak.num(raw["pt"])) == n_store
    d["target"] = {k: v[matched] for k, v in raw.items()}
    d["reco"] = {k: v[matched] for k, v in d["reco"].items()}
    tx = np.asarray(ak.sum(d["target"]["pt"] * np.cos(d["target"]["phi"]), axis=1))
    ty = np.asarray(ak.sum(d["target"]["pt"] * np.sin(d["target"]["phi"]), axis=1))
    return {"truth": "raw", "raw_truth_file": str(raw_truth), "store": str(store_dir),
            "n_events": int(n), "n_matched": int(matched.sum()),
            "n_matched_nonempty_truth_jet": int((np.hypot(tx, ty) > 0).sum())}


def compute_metrics(
    loaded: Dict[str, Dict[str, Dict[str, ak.Array]]],
    case: str,
    output_modality: str,
    *,
    c2st_hparams: Optional[Dict] = None,
    save_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run pflow_report + generative_report for ONE (loaded, modality) pair.

    Returns a flat dict of scalars (no nested objects). C2ST is a single AUC
    (training-time hyperparameters) here; ``c2st_bootstrap`` gives the seed bootstrap.
    """
    if output_modality not in loaded:
        return {"_status": "missing", "_modality": output_modality}

    d = loaded[output_modality]
    truth, reco = d["target"], d["reco"]

    metrics: Dict[str, Any] = {"_case": case, "_modality": output_modality}

    # 1. Reconstruction metrics (always run — works for any output modality)
    try:
        pflow_m = run_report_from_arrays(
            truth=truth, reco=reco, reco_ind=None,
            outdir=str(save_dir) if save_dir else None,
            output_modality=output_modality,
        )
        metrics.update({k: v for k, v in pflow_m.items()
                        if isinstance(v, (int, float))})
    except Exception as e:
        metrics["_pflow_error"] = f"{type(e).__name__}: {e}"

    # 2. Generative metrics (occupancy + C2ST + multiplicity)
    c2st_kw = dict(c2st_epochs=30, c2st_hidden=64)
    if c2st_hparams:
        c2st_kw.update({k: v for k, v in c2st_hparams.items()
                        if k in ("c2st_epochs", "c2st_hidden")})
    try:
        gen_m = run_generative_report_from_arrays(
            truth=truth, reco=reco,
            outdir=str(save_dir) if save_dir else None,
            output_modality=output_modality,
            **c2st_kw,
        )
        # Filter to scalars only, prefix with 'gen_' to namespace.
        for k, v in gen_m.items():
            if isinstance(v, (int, float)) and not k.startswith("_"):
                metrics[f"gen_{k}"] = v
    except Exception as e:
        metrics["_gen_error"] = f"{type(e).__name__}: {e}"

    return metrics

def _maybe_renorm_cossin(d):
    """Renormalise (cosphi, sinphi) onto the unit circle when
    HEP4M_C2ST_RENORM_PHI=1. Parallel engines sample the pair independently so
    c^2+s^2 != 1; a classifier fed the raw pair keys on that tell. Applied to
    whichever sample passes through, so switching it on treats both classes
    symmetrically."""
    import os
    if os.environ.get("HEP4M_C2ST_RENORM_PHI", "0") != "1":
        return d
    if "cosphi" in d and "sinphi" in d:
        c, s = d["cosphi"], d["sinphi"]
        r = np.sqrt(c ** 2 + s ** 2)
        try:
            r = ak.where(r > 0, r, 1.0)
        except Exception:
            r = np.where(np.asarray(r) > 0, r, 1.0)
        d = dict(d); d["cosphi"], d["sinphi"] = c / r, s / r
    return d

def _build_c2st_features(d: Dict[str, ak.Array], modality: str) -> np.ndarray:
    """Build the (N, F) feature matrix for C2ST.

    For truthpart: (log-pT, eta, cosphi, sinphi, class).
    For topo:      (log-e,  eta, cosphi, sinphi, em_frac, rho).
    For track:     (log-pT, eta, cosphi, sinphi, d0, z0).
    """
    d = _maybe_renorm_cossin(d)
    kin = MODALITY_SCHEMA[modality]["kinematic"]
    feats = [
        np.log1p(np.maximum(ak.to_numpy(ak.flatten(d[kin])), 0.0)),
        ak.to_numpy(ak.flatten(d["eta"])),
        ak.to_numpy(ak.flatten(d["cosphi"])),
        ak.to_numpy(ak.flatten(d["sinphi"])),
    ]
    if MODALITY_SCHEMA[modality]["has_class"]:
        feats.append(ak.to_numpy(ak.flatten(d["class"])).astype(np.float32))
    for extra in MODALITY_SCHEMA[modality]["extras"]:
        if extra in d:
            feats.append(ak.to_numpy(ak.flatten(d[extra])))
    return np.stack(feats, axis=-1).astype(np.float32)

def c2st_bootstrap(
    truth_features: np.ndarray,
    reco_features: np.ndarray,
    *,
    n_boot: int = C2ST_DEFAULTS["n_boot"],
    epochs: int = C2ST_DEFAULTS["epochs"],
    hidden: int = C2ST_DEFAULTS["hidden"],
    batch_size: int = C2ST_DEFAULTS["batch_size"],
    device: str = "cpu",
) -> Dict[str, float]:
    """Bootstrap N C2ST retrains with different seeds. Returns mean + 5/95 percentiles.

    The "indistinguishability score" is 1 - 2*|AUC - 0.5| ∈ [0, 1]:
        1.0  → AUC = 0.5  → simulation perfectly indistinguishable from truth
        0.0  → AUC = 1.0  → simulation trivially distinguishable

    """
    aucs = []
    accs = []
    for seed in range(n_boot):
        auc, acc = _compute_c2st(
            truth_features=truth_features,
            reco_features=reco_features,
            epochs=epochs,
            batch_size=batch_size,
            hidden=hidden,
            seed=seed,
            device=device,
        )
        # Symmetrize: if a single retrain undershoots 0.5 it just means the
        # classifier flipped labels by chance with no signal. Take max(auc, 1-auc).
        auc_sym = max(auc, 1.0 - auc)
        aucs.append(auc_sym)
        accs.append(acc)
    aucs = np.array(aucs)
    accs = np.array(accs)
    auc_mean = float(aucs.mean())
    auc_p5 = float(np.percentile(aucs, 5))
    auc_p95 = float(np.percentile(aucs, 95))
    indist = 1.0 - 2.0 * abs(auc_mean - 0.5)
    return {
        "c2st_auc_mean": auc_mean,
        "c2st_auc_p5": auc_p5,
        "c2st_auc_p95": auc_p95,
        "c2st_acc_mean": float(accs.mean()),
        "c2st_indist_score": indist,
        "n_boot": int(n_boot),
    }

def _build_event_features(d: Dict[str, ak.Array], modality: str) -> np.ndarray:
    """Build a per-EVENT feature matrix for the event-level C2ST.

    The pooled particle-level C2ST is blind to per-event correlation failures
    (e.g. correct pT spectrum but wrong multiplicity-energy correlation).
    Event-level features per event:
        multiplicity, log1p(total kin), log1p(leading kin),
        kin-weighted mean eta, eta spread,
        kin-weighted mean cos(phi), mean sin(phi), phi spread (circular-safe),
    """
    kin_name = MODALITY_SCHEMA[modality]["kinematic"]
    kin = d[kin_name]
    eta = d["eta"]
    d = _maybe_renorm_cossin(d)
    cosphi = d["cosphi"]
    sinphi = d["sinphi"]

    n = ak.to_numpy(ak.num(kin)).astype(np.float32)
    tot = ak.to_numpy(ak.sum(kin, axis=1)).astype(np.float32)
    lead = ak.to_numpy(ak.max(kin, axis=1, mask_identity=False)).astype(np.float32)
    lead = np.nan_to_num(lead, nan=0.0)

    # kin-weighted moments; guard empty events with tot=0
    w_tot = np.maximum(tot, 1e-8)
    mean_eta = ak.to_numpy(ak.sum(kin * eta, axis=1)).astype(np.float32) / w_tot
    mean_eta2 = ak.to_numpy(ak.sum(kin * eta * eta, axis=1)).astype(np.float32) / w_tot
    eta_spread = np.sqrt(np.maximum(mean_eta2 - mean_eta**2, 0.0))
    mean_cos = ak.to_numpy(ak.sum(kin * cosphi, axis=1)).astype(np.float32) / w_tot
    mean_sin = ak.to_numpy(ak.sum(kin * sinphi, axis=1)).astype(np.float32) / w_tot
    # circular spread: 1 - |resultant|; in [0, 1], 0 = perfectly collimated
    phi_spread = 1.0 - np.sqrt(mean_cos**2 + mean_sin**2)

    return np.stack([
        n,
        np.log1p(np.maximum(tot, 0.0)),
        np.log1p(np.maximum(lead, 0.0)),
        mean_eta,
        eta_spread,
        mean_cos,
        mean_sin,
        phi_spread,
    ], axis=-1).astype(np.float32)

# Default input modalities for the joint C2ST (the pflow eval direction).
JOINT_INPUT_MODALITIES = ("topo", "track")

def joint_input_modalities(case: str) -> Tuple[str, ...]:
    """Input modalities to condition the JOINT C2ST on, per case.

    The joint builders below take ``input_modalities`` explicitly; this maps a
    case name to its eval-direction conditioning side (pflow ->
    ('topo', 'track'); sim -> ('truthpart',)). Any cached modality works.
    """
    return tuple(CASE_DIRECTIONS[case]["inputs"])

def alignment_gate(cache: Dict[str, Any], target: Dict[str, "ak.Array"], modality: str,
                   rows: np.ndarray, keep: np.ndarray, label: str = "",
                   min_corr: float | None = None) -> float:
    """Per-event alignment check between a prediction root's TRUTH side and the
    cache rows it was joined to (join coverage alone can accept a root joined to
    the cache of a different file at high coverage).

    Correlates the per-event leading kinematic of the root's truth objects with
    the same quantity in the cache: same modality if the cache holds it
    (aligned: ~1.0), else the cache's 'track' leading pT (aligned: ~0.86, the
    physical truth-track correlation; misaligned: ~0.0). Raises on failure.
    """
    kin = MODALITY_SCHEMA[modality]["kinematic"]
    lead_root = np.asarray(ak.fill_none(ak.max(target[kin], axis=1), 0.0), dtype=np.float64)
    if modality in cache["modalities"]:
        ref_mod, default_min = modality, 0.95
    elif "track" in cache["modalities"]:
        ref_mod, default_min = "track", 0.60
    else:
        raise ValueError(f"{label}: cache has neither {modality!r} nor 'track'; cannot gate alignment")
    off = cache[ref_mod]["offsets"]; feat = cache[ref_mod]["features"][:, 0]
    lead_cache = np.array([feat[a:b].max() if b > a else 0.0 for a, b in zip(off[:-1], off[1:])])
    r = rows[keep]; x = lead_root[keep] if len(lead_root) == len(keep) else lead_root
    corr = float(np.corrcoef(x, lead_cache[r])[0, 1]) if len(r) > 2 else float("nan")
    thr = default_min if min_corr is None else min_corr
    print(f"  alignment gate [{label}]: corr(root truth {modality} lead {kin}, cache {ref_mod} lead) = {corr:.3f} "
          f"(min {thr:.2f}, {len(r)} events)", flush=True)
    if not (corr >= thr):
        raise ValueError(f"{label}: prediction root is NOT aligned with the input cache "
                         f"(corr {corr:.3f} < {thr:.2f}); wrong file or wrong join convention")
    return corr

def load_c2st_input_cache(path: str | Path) -> Dict[str, Any]:
    """Load the detokenized-inputs cache written by hep4m/judge/build_input_cache.py.

    Returns a dict of arrays + an index (no per-event python dicts):

        {
          "event_numbers": (N,) int64 raw store event numbers, store order,
          "modalities": [m, ...],
          m: {"features": (total_obj, F) float32,   # flat per-object features
              "offsets": (N+1,) int64,              # event i -> rows [off[i], off[i+1])
              "feature_names": [str, ...]},
        }

    Feature columns mirror ``_build_c2st_features`` per modality, except col 0
    keeps the kinematic RAW (e/pt, not log1p) so sums are physical;
    ``np.log1p`` of col 0 reproduces the marginal C2ST features exactly.
    """
    z = np.load(path, allow_pickle=False)
    event_numbers = np.asarray(z["event_numbers"], dtype=np.int64)
    cache: Dict[str, Any] = {
        "event_numbers": event_numbers,
        "modalities": [str(m) for m in z["modalities"]],
    }
    n_ev = len(event_numbers)
    for m in cache["modalities"]:
        offsets = np.asarray(z[f"{m}_offsets"], dtype=np.int64)
        features = np.asarray(z[f"{m}_features"], dtype=np.float32)
        assert len(offsets) == n_ev + 1, \
            f"{m}: offsets length {len(offsets)} != n_events+1 ({n_ev + 1})"
        assert offsets[-1] == features.shape[0], \
            f"{m}: offsets end {offsets[-1]} != n rows {features.shape[0]}"
        cache[m] = {
            "features": features,
            "offsets": offsets,
            "feature_names": [str(s) for s in z[f"{m}_feature_names"]],
        }
    return cache

def match_cache_rows(
    inputs_cache: Dict[str, Any],
    event_numbers: np.ndarray,
    join: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Map per-event join keys from a prediction ROOT onto cache rows.

    Two 'event_number' value conventions exist in the prediction ROOTs:
    - nanoHEP roots: row INDEX into the tokenized val store (0..n_store-1;
      inference didn't pass shared_event_numbers, so the dataset fell back to
      the getitem idx) -> join='index'.
    - HEP4M / HGPflow roots: original test-file event index (0..99999). The
      store's raw numbers are ``offset + index`` (offset = raw.min(), 1000000
      for the 7mod val store) with a few empty events absent -> join='raw_offset'.
    A root carrying true raw store numbers joins via join='raw'.

    'auto' picks the convention with the best coverage ('index' is only
    eligible when max(ev) < n_cache — a row index can never reach n_cache).

    Returns ``(rows, keep, info)``: cache row per event (undefined where
    ``~keep``), bool mask of joinable events, and an info dict with the chosen
    convention + per-convention coverage.
    """
    ev = np.asarray(event_numbers, dtype=np.int64)
    raw = inputs_cache["event_numbers"]
    n_cache = len(raw)
    sorter = np.argsort(raw, kind="stable")
    raw_sorted = raw[sorter]

    def _match_keys(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.clip(np.searchsorted(raw_sorted, keys), 0, n_cache - 1)
        rows = sorter[pos]
        keep = raw[rows] == keys
        return rows, keep

    candidates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    candidates["raw"] = _match_keys(ev)
    candidates["raw_offset"] = _match_keys(ev + int(raw.min()))
    if len(ev) == 0 or (ev.min() >= 0 and ev.max() < n_cache):
        candidates["index"] = (np.clip(ev, 0, max(n_cache - 1, 0)),
                               (ev >= 0) & (ev < n_cache))

    coverages = {k: (float(keep.mean()) if len(ev) else 1.0)
                 for k, (_, keep) in candidates.items()}
    if join == "auto":
        # priority order breaks exact ties (index/raw_offset coincide below
        # the store's first gap, so the tie is harmless there)
        join = max(("raw", "index", "raw_offset"),
                   key=lambda k: coverages.get(k, -1.0))
    if join not in candidates:
        raise ValueError(f"join={join!r} not applicable "
                         f"(available: {sorted(candidates)}, ev.max()={ev.max() if len(ev) else None}, n_cache={n_cache})")
    rows, keep = candidates[join]
    info = {
        "join": join,
        "coverage": coverages[join],
        "coverages": coverages,
        "n_events": int(len(ev)),
        "n_matched": int(keep.sum()),
    }
    return rows, keep, info

def _flat_object_features(d: Dict[str, ak.Array], modality: str) -> Tuple[np.ndarray, np.ndarray]:
    """Flat per-object feature matrix + event offsets, cache column layout.

    Same columns/order as ``_build_c2st_features`` but col 0 is the RAW
    kinematic (e/pt) instead of log1p, so per-event kinematic sums stay
    physical. Returns ``((total_obj, F) float32, (N+1,) int64 offsets)``.
    """
    kin = MODALITY_SCHEMA[modality]["kinematic"]
    feats = [
        ak.to_numpy(ak.flatten(d[kin])).astype(np.float32),
        ak.to_numpy(ak.flatten(d["eta"])),
        ak.to_numpy(ak.flatten(d["cosphi"])),
        ak.to_numpy(ak.flatten(d["sinphi"])),
    ]
    if MODALITY_SCHEMA[modality]["has_class"]:
        feats.append(ak.to_numpy(ak.flatten(d["class"])).astype(np.float32))
    for extra in MODALITY_SCHEMA[modality]["extras"]:
        if extra in d:
            feats.append(ak.to_numpy(ak.flatten(d[extra])))
    flat = np.stack(feats, axis=-1).astype(np.float32)
    counts = ak.to_numpy(ak.num(d[kin])).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return flat, offsets

def _aggregate_object_features(flat_feats: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-event aggregates of a flat object-feature block: ``(N, 2 + 3F)``.

    Columns: [n_objects, log1p(sum raw kin),
              mean of each feature, std of each feature,
              leading-object (max raw kin) feature vector].
    Feature moments use the C2ST feature convention (col 0 -> log1p).
    Empty events get all-zero rows.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    n_ev = len(offsets) - 1
    counts = np.diff(offsets)
    F = flat_feats.shape[1]
    out = np.zeros((n_ev, 2 + 3 * F), dtype=np.float32)
    out[:, 0] = counts
    nonempty = counts > 0
    if flat_feats.shape[0] == 0 or not nonempty.any():
        return out

    raw_kin = flat_feats[:, 0].astype(np.float64)
    cfeats = flat_feats.astype(np.float64).copy()
    cfeats[:, 0] = np.log1p(np.maximum(raw_kin, 0.0))

    # segment sums via reduceat; empty segments give garbage (x[start]) and
    # are zeroed by the nonempty mask afterwards
    starts = np.minimum(offsets[:-1], flat_feats.shape[0] - 1)
    seg_sum = np.add.reduceat(cfeats, starts, axis=0)
    seg_sum2 = np.add.reduceat(cfeats ** 2, starts, axis=0)
    kin_sum = np.add.reduceat(raw_kin, starts)
    seg_sum[~nonempty] = 0.0
    seg_sum2[~nonempty] = 0.0
    kin_sum[~nonempty] = 0.0

    denom = np.maximum(counts, 1).astype(np.float64)[:, None]
    mean = seg_sum / denom
    var = np.maximum(seg_sum2 / denom - mean ** 2, 0.0)
    out[:, 1] = np.log1p(np.maximum(kin_sum, 0.0))
    out[:, 2:2 + F] = mean
    out[:, 2 + F:2 + 2 * F] = np.sqrt(var)

    # leading object = argmax raw kin per segment. lexsort's primary key is
    # the segment id, so segment s occupies order[offsets[s]:offsets[s+1]]
    # and order[offsets[s]] is its max-kin row.
    seg_ids = np.repeat(np.arange(n_ev), counts)
    order = np.lexsort((-raw_kin, seg_ids))
    lead_rows = order[offsets[:-1][nonempty]]
    out[nonempty, 2 + 2 * F:] = cfeats[lead_rows].astype(np.float32)
    return out

def _compute_c2st_paired(
    truth_features: np.ndarray,
    reco_features: np.ndarray,
    epochs: int = C2ST_DEFAULTS["epochs"],
    batch_size: int = C2ST_DEFAULTS["batch_size"],
    hidden: int = C2ST_DEFAULTS["hidden"],
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[float, float]:
    """``_compute_c2st`` for PAIRED samples (row i of both classes = same event).

    The joint C2ST puts the SAME input features on both sides of each event
    pair. With an independent sample split (as in ``_compute_c2st``) the
    classifier can memorize a train twin's shared input features and label its
    held-out twin — the null (identical outputs) then reads as AUC far from
    0.5. Splitting train/test by EVENT (both twins on the same side) removes
    that leakage: within train, each input appears with both labels, so only
    genuine conditional (output-given-input) differences are learnable.

    Same MLP/optimizer/metrics as ``_compute_c2st``. Returns (test_AUC, acc).
    """
    import torch
    from torch import nn
    from .generative_report import _C2STMLP

    n = truth_features.shape[0]
    assert reco_features.shape[0] == n, "paired C2ST needs aligned (same-event) rows"
    if n < 16:
        return float("nan"), float("nan")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = np.concatenate([truth_features, reco_features], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    X = (X - mu) / sd

    # event-paired 70/30 split: twins (i, i + n) stay on the same side
    ev_perm = rng.permutation(n)
    split = int(n * 0.7)
    tr_ev, te_ev = ev_perm[:split], ev_perm[split:]
    tr_idx = np.concatenate([tr_ev, tr_ev + n])
    te_idx = np.concatenate([te_ev, te_ev + n])

    with torch.inference_mode(False), torch.enable_grad():
        X_tr = torch.from_numpy(X[tr_idx]).to(device)
        y_tr = torch.from_numpy(y[tr_idx]).to(device)
        X_te = torch.from_numpy(X[te_idx]).to(device)
        y_te = torch.from_numpy(y[te_idx]).to(device)

        model = _C2STMLP(X.shape[1], hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            perm = torch.randperm(X_tr.shape[0], device=device)
            for i in range(0, X_tr.shape[0], batch_size):
                sel = perm[i:i + batch_size]
                loss = loss_fn(model(X_tr[sel]), y_tr[sel])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(X_te)).cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    acc = float(((prob > 0.5).astype(np.float32) == y_te_np).mean())

    # AUC via Mann-Whitney U with tie-averaged ranks (as in _compute_c2st)
    try:
        from scipy.stats import rankdata
        ranks = rankdata(prob)
        n_pos = float((y_te_np == 1).sum())
        n_neg = float((y_te_np == 0).sum())
        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
        else:
            U = float(ranks[y_te_np == 1].sum()) - n_pos * (n_pos + 1) / 2
            auc = float(U / (n_pos * n_neg))
    except Exception:
        auc = float("nan")
    return auc, acc

def _input_aggregates(inputs_cache: Dict[str, Any], modality: str) -> np.ndarray:
    """Aggregate one cached input modality over ALL store events (memoized)."""
    memo = inputs_cache.setdefault("_agg_memo", {})
    if modality not in memo:
        memo[modality] = _aggregate_object_features(
            inputs_cache[modality]["features"], inputs_cache[modality]["offsets"])
    return memo[modality]

def _build_joint_event_features(
    inputs_cache: Dict[str, Any],
    d_out: Dict[str, ak.Array],
    modality: str,
    event_numbers: np.ndarray,
    input_modalities: Tuple[str, ...] = JOINT_INPUT_MODALITIES,
    join: str = "auto",
) -> np.ndarray:
    """Per-EVENT joint feature matrix ``(N_events, F)``: inputs ⊕ outputs.

    For each cached modality in ``input_modalities`` (pflow: topo, track;
    sim: truthpart — see ``joint_input_modalities``) plus the outputs
    ``d_out``, computes the
    ``_aggregate_object_features`` block — [n_objects, log1p(sum kin),
    mean/std of each feature, leading-object features] — and concatenates.
    Every ``event_numbers`` entry must join to the cache (pre-filter with
    ``match_cache_rows`` if the root has unmatched events).
    """
    rows, keep, info = match_cache_rows(inputs_cache, event_numbers, join=join)
    if not keep.all():
        raise ValueError(
            f"{int((~keep).sum())}/{len(keep)} event_numbers not in the input cache "
            f"(join={info['join']}, coverages={info['coverages']}); "
            "pre-filter events with match_cache_rows()")
    blocks = [_input_aggregates(inputs_cache, m)[rows] for m in input_modalities]
    flat_out, off_out = _flat_object_features(d_out, modality)
    assert len(off_out) - 1 == len(rows), \
        f"d_out has {len(off_out) - 1} events but got {len(rows)} event_numbers"
    blocks.append(_aggregate_object_features(flat_out, off_out))
    return np.concatenate(blocks, axis=1).astype(np.float32)

def c2st_joint_bootstrap(
    inputs_cache: Dict[str, Any],
    loaded_modality: Dict[str, Dict[str, ak.Array]],
    modality: str,
    event_numbers: np.ndarray,
    *,
    input_modalities: Tuple[str, ...] = JOINT_INPUT_MODALITIES,
    join: str = "auto",
    min_coverage: float = 0.995,
    n_boot: int = C2ST_DEFAULTS["n_boot"],
    epochs: int = C2ST_DEFAULTS["epochs"],
    hidden: int = C2ST_DEFAULTS["hidden"],
    batch_size: int = C2ST_DEFAULTS["batch_size"],
    device: str = "cpu",
) -> Dict[str, float]:
    """JOINT C2ST: bootstrap classifier on event-level inputs⊕outputs features.

    Class A = inputs ⊕ truth outputs, class B = inputs ⊕ reco outputs — the
    SAME inputs on both sides, matched by event_number, so AUC > 0.5 measures
    conditional (p(y|x)) mismatch that the marginal C2ST cannot see. Same
    bootstrap protocol / return schema as ``c2st_bootstrap`` (symmetrized AUC,
    5/95 percentiles, indist score), except each retrain uses the event-PAIRED
    train/test split of ``_compute_c2st_paired`` — required for validity, see
    its docstring. Adds join info to the returned dict.
    """
    ev = np.asarray(event_numbers, dtype=np.int64)
    rows, keep, info = match_cache_rows(inputs_cache, ev, join=join)
    if info["coverage"] < min_coverage:
        raise ValueError(
            f"join coverage {info['coverage']:.4f} < min_coverage {min_coverage} "
            f"(join={info['join']}, coverages={info['coverages']})")
    d_t, d_r = loaded_modality["target"], loaded_modality["reco"]
    if not keep.all():
        d_t = {k: v[keep] for k, v in d_t.items()}
        d_r = {k: v[keep] for k, v in d_r.items()}
        ev = ev[keep]
    truth_f = _build_joint_event_features(
        inputs_cache, d_t, modality, ev, input_modalities, join=info["join"])
    reco_f = _build_joint_event_features(
        inputs_cache, d_r, modality, ev, input_modalities, join=info["join"])

    aucs, accs = [], []
    for seed in range(n_boot):
        auc, acc = _compute_c2st_paired(
            truth_f, reco_f, epochs=epochs, batch_size=batch_size,
            hidden=hidden, seed=seed, device=device)
        aucs.append(max(auc, 1.0 - auc))  # symmetrize, as in c2st_bootstrap
        accs.append(acc)
    aucs = np.array(aucs)
    auc_mean = float(aucs.mean())
    result = {
        "c2st_auc_mean": auc_mean,
        "c2st_auc_p5": float(np.percentile(aucs, 5)),
        "c2st_auc_p95": float(np.percentile(aucs, 95)),
        "c2st_acc_mean": float(np.mean(accs)),
        "c2st_indist_score": 1.0 - 2.0 * abs(auc_mean - 0.5),
        "n_boot": int(n_boot),
    }
    result["join"] = info["join"]
    result["join_coverage"] = info["coverage"]
    result["n_events_joined"] = info["n_matched"]
    # width of the joint event-aggregate feature vector (checked by the set C2ST callers)
    result["feature_dim"] = int(truth_f.shape[1])
    return result

def build_joint_event_sets(
    inputs_cache: Dict[str, Any],
    d_out: Dict[str, ak.Array],
    modality: str,
    event_numbers: np.ndarray,
    *,
    input_modalities: Tuple[str, ...] = JOINT_INPUT_MODALITIES,
    join: str = "auto",
    n_max_out: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Union set per event for the set-transformer joint C2ST.

    Each event's set = input objects (union over ``input_modalities`` from the
    cache; pflow: topo ∪ track, sim: truthpart) + output
    objects (``d_out``). Every object feature vector is zero-padded to the
    common width, then appended with flags: [is_input, one-hot(topo, track,
    ..., output)]. Class A = inputs∪truth-outputs, B = inputs∪reco-outputs
    (call once per side; pass a shared ``n_max_out`` so shapes match).

    Returns ``(X (N, n_max_total, F+flags) float32, mask (N, n_max_total) bool)``
    — padding lives at fixed per-source slots, which a key_padding_mask
    transformer is invariant to. Every event_number must join to the cache.
    """
    rows, keep, info = match_cache_rows(inputs_cache, event_numbers, join=join)
    if not keep.all():
        raise ValueError(
            f"{int((~keep).sum())}/{len(keep)} event_numbers not in the input cache "
            f"(join={info['join']}); pre-filter events with match_cache_rows()")
    n_ev = len(rows)

    flat_out, off_out = _flat_object_features(d_out, modality)
    assert len(off_out) - 1 == n_ev, \
        f"d_out has {len(off_out) - 1} events but got {n_ev} event_numbers"

    # per-source (flat, starts, counts) in union order: inputs..., outputs
    sources = []
    for m in input_modalities:
        off = inputs_cache[m]["offsets"]
        sources.append((inputs_cache[m]["features"], off[rows],
                        (off[rows + 1] - off[rows]), None))
    sources.append((flat_out, off_out[:-1], np.diff(off_out), n_max_out))

    width = max(s[0].shape[1] for s in sources)
    n_onehot = len(sources)
    F_tot = width + 1 + n_onehot  # + is_input flag + source one-hot

    X_blocks, M_blocks = [], []
    for si, (flat, starts, counts, n_max) in enumerate(sources):
        is_input = si < len(input_modalities)
        n_max = int(n_max if n_max is not None else (counts.max() if len(counts) else 0))
        X = np.zeros((n_ev, n_max, F_tot), dtype=np.float32)
        M = np.zeros((n_ev, n_max), dtype=bool)
        counts = np.minimum(counts, n_max)
        total = int(counts.sum())
        if total > 0:
            # gather flat rows for each event slice [start, start+count)
            cum = np.concatenate([[0], np.cumsum(counts)[:-1]])
            gather = np.repeat(starts - cum, counts) + np.arange(total)
            block = flat[gather].astype(np.float32)
            block[:, 0] = np.log1p(np.maximum(block[:, 0], 0.0))
            ev_ids = np.repeat(np.arange(n_ev), counts)
            pos = np.arange(total) - np.repeat(cum, counts)
            X[ev_ids, pos, :block.shape[1]] = block
            X[ev_ids, pos, width] = 1.0 if is_input else 0.0
            X[ev_ids, pos, width + 1 + si] = 1.0
            M[ev_ids, pos] = True
        X_blocks.append(X)
        M_blocks.append(M)
    return np.concatenate(X_blocks, axis=1), np.concatenate(M_blocks, axis=1)

def _cli_metrics(args: argparse.Namespace) -> None:
    """Compute the metric battery for one prediction ROOT and write metrics.json."""
    n_classes = args.n_classes if args.n_classes is not None else (5 if args.engine == "hgpflow" else 3)
    run = RunSpec(
        case=args.case, run_id=args.run_id, engine=args.engine,
        pred_root=args.root_path, n_classes=n_classes,
    )
    loaded = load_pred_root(run, entry_stop=args.max_events)
    out_metrics: Dict[str, Any] = {"_run_spec": asdict(run)}
    if args.raw_truth:
        if args.case != "pflow" or "truthpart" not in loaded:
            raise SystemExit("--raw-truth needs a particle-flow prediction file (--case pflow)")
        from hep4m.paths import TOKENIZED
        store = args.store or os.path.join(TOKENIZED, "test")
        sel = raw_truth_selection(loaded, args.raw_truth, store,
                                  load_pred_event_numbers(run, entry_stop=args.max_events))
        out_metrics["_selection"] = sel
        print(f"raw truth: {sel['n_matched']} of {sel['n_events']} events matched "
              f"({sel['n_matched_nonempty_truth_jet']} with a non-empty truth jet)")
    elif loaded:
        first = next(iter(loaded.values()))
        out_metrics["_selection"] = {"truth": "tokenised", "n_events": int(len(first["target"]["pt"]))}
    case_outputs = CASE_DIRECTIONS[args.case]["outputs"]
    for mod in case_outputs:
        m = compute_metrics(loaded, args.case, mod)
        out_metrics[mod] = m
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(out_metrics, indent=2, default=str))
    print(f"wrote {args.metrics_out}")

def main():
    parser = argparse.ArgumentParser(description="Metrics for one prediction ROOT file")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("metrics", help="metric battery for a prediction ROOT")
    p1.add_argument("--root-path", required=True)
    p1.add_argument("--engine", choices=ENGINES, required=True)
    p1.add_argument("--case", choices=list(CASE_DIRECTIONS), required=True)
    p1.add_argument("--run-id", required=True)
    p1.add_argument("--metrics-out", required=True)
    p1.add_argument("--n-classes", type=int, default=None,
                    help="particle classes in the file (default 5 for hgpflow, 3 otherwise)")
    p1.add_argument("--max-events", type=int, default=None, help="use only the first N events")
    p1.add_argument("--raw-truth", default=None, metavar="ROOT",
                    help="particle flow: score against the raw truth particles of this file (the "
                         "HGPflow reference file or the raw test file of the record), on the events "
                         "whose tokenised truth kept every particle")
    p1.add_argument("--store", default=None, metavar="DIR",
                    help="token store split with truthpart_offsets.npy, for --raw-truth "
                         "(default $HEP4M_TOKENIZED/test)")
    p1.set_defaults(func=_cli_metrics)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
