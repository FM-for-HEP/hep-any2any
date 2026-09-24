"""Marginal and joint (conditional) C2ST for the detector-simulation direction.

The models generate the detector response (topo, track) from truth particles. The
joint test therefore conditions on the truth-particle inputs: class A = truth
particles + Geant4 outputs, class B = truth particles + generated outputs, with the
same inputs on both sides matched by event number, so any AUC above 0.5 comes from
p(detector | particles).

For each model and each generated modality: the four AUCs of ``hep4m.judge.c2st_reco``
({marginal, joint} x {event-aggregate MLP, set transformer}) plus a split-half null
that must give AUC near 0.5.

Inputs cache: ``python -m hep4m.judge.build_input_cache --modalities truthpart``.
Models are given as ``--row "label:engine:/path/to/sampled.root"``. Writes
c2st_sim.json and one bar chart per modality to --out-dir.
"""

import argparse
import json
import os
from pathlib import Path
import awkward as ak
import numpy as np
import torch

from hep4m.performance import evaluate as pe
from hep4m.performance.evaluate import (
    _build_event_features,
    build_joint_event_sets,
    c2st_joint_bootstrap,
    joint_input_modalities,
    load_c2st_input_cache,
    match_cache_rows,
)
from hep4m.judge.c2st_set import ensure_cossin, event_sets_pflow
from hep4m.judge.c2st_reco import _boot_set, run_metadata, plot_results

INPUT_MODALITIES = joint_input_modalities("sim")  # ('truthpart',)

def run_model(label, engine, root, cache, args):
    """All four AUCs + null control, per generated modality. Returns a dict."""
    run = pe.RunSpec(case="sim", run_id=label, engine=engine, pred_root=root)
    loaded = pe.load_pred_root(run, entry_stop=args.max_events)
    ev = pe.load_pred_event_numbers(run, entry_stop=args.max_events)

    # ---- alignment guardrails: join + coverage report, drop unmatched ----
    rows, keep, info = match_cache_rows(cache, ev)
    print(f"  join={info['join']} coverage={info['coverage']:.4f} "
          f"({info['n_matched']}/{info['n_events']} events; "
          f"all conventions: { {k: round(v, 4) for k, v in info['coverages'].items()} })", flush=True)
    if info["coverage"] < args.min_join_coverage:
        raise ValueError(f"{label}: join coverage {info['coverage']:.4f} < "
                         f"--min-join-coverage {args.min_join_coverage}")
    if not keep.all():
        print(f"  WARNING: dropping {int((~keep).sum())} events without cache inputs", flush=True)
        loaded = {m: {side: {k: v[keep] for k, v in dd.items()}
                      for side, dd in mm.items()} for m, mm in loaded.items()}
        ev = ev[keep]

    out = {"engine": engine, "pred_root": root, "n_events": int(len(ev)),
           "join": info["join"], "join_coverage": info["coverage"]}
    mlp_kw = dict(n_boot=args.n_boot, epochs=args.epochs, device=args.device)

    for mod in args.modalities:
        if mod not in loaded:
            print(f"  {mod}: not in root — skipping", flush=True)
            out[mod] = {"_status": "missing"}
            continue
        d = loaded[mod]
        d["target"] = ensure_cossin(d["target"])
        d["reco"] = ensure_cossin(d["reco"])
        pe.alignment_gate(cache, d["target"], mod, rows, keep, f"{label}/{mod}")
        r = {}

        # (1a) marginal, event-aggregate MLP
        et = _build_event_features(d["target"], mod)
        er = _build_event_features(d["reco"], mod)
        r["marginal_event_agg"] = pe.c2st_bootstrap(et, er, **mlp_kw)

        # (1b) joint, event-aggregate MLP (truthpart inputs ⊕ outputs)
        r["joint_event_agg"] = c2st_joint_bootstrap(
            cache, d, mod, ev, input_modalities=INPUT_MODALITIES,
            join=info["join"], min_coverage=args.min_join_coverage, **mlp_kw)

        ajt = pe._build_joint_event_features(
            cache, d["target"], mod, ev, input_modalities=INPUT_MODALITIES,
            join=info["join"])
        ajr = pe._build_joint_event_features(
            cache, d["reco"], mod, ev, input_modalities=INPUT_MODALITIES,
            join=info["join"])
        mlp_width = int(r["joint_event_agg"]["feature_dim"])
        if ajt.shape[1] != ajr.shape[1] or ajt.shape[1] != mlp_width:
            raise RuntimeError(f"{label}/{mod}: joint seed widths "
                               f"{ajt.shape[1]}/{ajr.shape[1]} != MLP {mlp_width}")
        r["joint_seed_width"] = int(ajt.shape[1])

        # (2a) marginal, set-transformer (output set only; load_pred_root
        # aliases 'pt' for topo, so event_sets_pflow works for both modalities)
        n_max = int(max(ak.num(d["target"]["eta"], axis=1).to_numpy().max(),
                        ak.num(d["reco"]["eta"], axis=1).to_numpy().max()))
        Xt, Mt = event_sets_pflow(d["target"], n_max)
        Xr, Mr = event_sets_pflow(d["reco"], n_max)
        m, s = _boot_set(Xt, Mt, Xr, Mr, args.n_boot, args.set_epochs, args.device,
                         At=et, Ar=er)
        r["marginal_set"] = {"auc_mean": m, "auc_std": s}

        # (2b) joint, set-transformer (union set: truthpart ∪ outputs)
        Xjt, Mjt = build_joint_event_sets(cache, d["target"], mod, ev,
                                          input_modalities=INPUT_MODALITIES,
                                          join=info["join"], n_max_out=n_max)
        Xjr, Mjr = build_joint_event_sets(cache, d["reco"], mod, ev,
                                          input_modalities=INPUT_MODALITIES,
                                          join=info["join"], n_max_out=n_max)
        m, s = _boot_set(Xjt, Mjt, Xjr, Mjr, args.n_boot, args.set_epochs,
                         args.device, paired=True, At=ajt, Ar=ajr)
        r["joint_set"] = {"auc_mean": m, "auc_std": s}

        _idx = np.arange(len(Xjt))
        _ha, _hb = _idx[_idx % 2 == 0], _idx[_idx % 2 == 1]
        _n = min(len(_ha), len(_hb)); _ha, _hb = _ha[:_n], _hb[:_n]
        m, s = _boot_set(Xjt[_ha], Mjt[_ha], Xjt[_hb], Mjt[_hb], args.n_boot,
                         args.set_epochs, args.device,
                         paired=False, At=ajt[_ha], Ar=ajt[_hb])
        null_ok = bool(np.isfinite(m) and abs(m - 0.5) <= args.null_tol)
        r["null_joint_set"] = {
            "auc_mean": m, "auc_std": s, "passed": null_ok,
            "tolerance": float(args.null_tol)}
        print(f"  {mod}: set null joint AUC {m:.6f} "
              f"(tolerance {args.null_tol:.3f}) "
              f"{'OK' if null_ok else 'FAIL'}", flush=True)

        print(f"  {mod}: event-agg  marginal {r['marginal_event_agg']['c2st_auc_mean']:.3f}"
              f"  joint {r['joint_event_agg']['c2st_auc_mean']:.3f}\n"
              f"  {mod}: set-trans  marginal {r['marginal_set']['auc_mean']:.3f}"
              f"  joint {r['joint_set']['auc_mean']:.3f}", flush=True)
        out[mod] = r
    return out

def main():
    ap = argparse.ArgumentParser(description="Marginal and joint C2ST, detector-simulation direction")
    ap.add_argument("--cache", required=True,
                    help=".npz from hep4m/judge/build_input_cache.py (must include truthpart)")
    ap.add_argument("--row", action="append", default=[], required=True,
                    metavar="label:engine:root",
                    help="model row (engine in {hep4m, nano_hep}); repeatable")
    ap.add_argument("--modalities", nargs="+", default=["topo", "track"],
                    help="generated (output) modalities to test")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100, help="MLP C2ST epochs")
    ap.add_argument("--set-epochs", type=int, default=60, help="set-transformer epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-join-coverage", type=float, default=0.995)
    ap.add_argument("--null-tol", type=float, default=0.02,
                    help="abort if |set-judge joint null AUC - 0.5| exceeds this")
    ap.add_argument("--out-dir", default="c2st_sim", help="output directory")
    args = ap.parse_args()

    cache = load_c2st_input_cache(args.cache)
    print(f"input cache: {args.cache} ({len(cache['event_numbers'])} events, "
          f"modalities {cache['modalities']})", flush=True)
    missing = [m for m in INPUT_MODALITIES if m not in cache["modalities"]]
    if missing:
        raise SystemExit(f"cache lacks joint-input modalities {missing}; rebuild with "
                         f"python -m hep4m.judge.build_input_cache --modalities ... {' '.join(missing)}")

    rows = []
    for spec in args.row:
        lab, eng, root = spec.split(":", 2)
        rows.append((lab, eng, root))

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for lab, eng, root in rows:
        name = lab.replace("\n", " ")
        if not Path(root).exists():
            print(f"SKIP {name}: missing root {root}", flush=True)
            continue
        print(f"== {name} ({eng}) ==", flush=True)
        results[lab] = run_model(lab, eng, root, cache, args)

    out_json = Path(args.out_dir) / "c2st_sim.json"
    clean = {k.replace("\n", " "): v for k, v in results.items()}
    agg_dim = {}
    nulls = {}
    for lab, model_result in clean.items():
        agg_dim[lab] = {mod: {"marginal": 8,
                              "joint": values.get("joint_seed_width")}
                        for mod, values in model_result.items()
                        if isinstance(values, dict) and "joint_set" in values}
        nulls[lab] = {mod: values["null_joint_set"]
                      for mod, values in model_result.items()
                      if isinstance(values, dict) and "null_joint_set" in values}
    meta = run_metadata(
        args, results, cache_path=args.cache,
        prediction_roots={k.replace("\n", " "): v["pred_root"] for k, v in results.items()},
        seeding_function={"marginal": "evaluate._build_event_features",
                          "joint": "evaluate._build_joint_event_features"},
        agg_dim=agg_dim, null_result=nulls)
    json.dump({"_meta": meta, **clean},
              open(out_json, "w"), indent=1, default=str)
    print(f"-> {out_json}")

    failed_nulls = [f"{lab}/{mod}" for lab, model_result in clean.items()
                    for mod, values in model_result.items()
                    if isinstance(values, dict) and "null_joint_set" in values
                    and not values["null_joint_set"]["passed"]]
    if failed_nulls:
        raise SystemExit(f"set-judge null failed for {failed_nulls}; results aborted")

    # one paired-bar figure per generated modality
    for mod in args.modalities:
        per_mod = {lab: r[mod] for lab, r in results.items()
                   if isinstance(r.get(mod), dict) and "marginal_event_agg" in r[mod]}
        if per_mod:
            plot_results(per_mod, Path(args.out_dir) / f"c2st_sim_{mod}.png")

if __name__ == "__main__":
    main()
