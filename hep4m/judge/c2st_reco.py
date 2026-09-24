"""Marginal and joint (conditional) C2ST for a reconstruction direction.

Marginal: truth outputs vs model outputs, p(y_truth) vs p(y_model); blind to
event-wise errors that keep the distribution. Joint: each output set is paired with
the same event's inputs (topo and track, decoded from the tokenised store and
matched by event number), so class A = inputs + truth outputs and class B =
inputs + model outputs; any AUC above 0.5 comes from p(y | x).

Both variants run at two levels:
  (1) event-aggregate MLP  - hand-crafted per-event summary features,
  (2) set transformer      - attends over the per-event object set.
A split-half null (truth events vs truth events) runs with every model and must
give AUC near 0.5; results are withheld if it does not (see --null-tol). --quick is
for smoke tests on a few events: 1 bootstrap, 5 epochs, and the null result is
recorded but not enforced.

Inputs cache: ``python -m hep4m.judge.build_input_cache``. Models are given as
``--row "label:engine:/path/to/prediction.root"`` (engine: hep4m, nano_hep or
hgpflow). Writes c2st_reco.json and a bar chart to --out-dir.

Example (CPU, small):
    python -m hep4m.judge.c2st_reco --cache inputs.npz --row "nanoHEP:nano_hep:argmax.root" \
        --max-events 200 --n-boot 1 --device cpu
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
import awkward as ak
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from hep4m.performance import evaluate as pe  # noqa: E402
from hep4m.performance.evaluate import (  # noqa: E402
    _build_event_features,
    build_joint_event_sets,
    c2st_joint_bootstrap,
    load_c2st_input_cache,
    match_cache_rows,
)
from hep4m.judge.c2st_set import c2st_set, ensure_cossin, event_sets_pflow  # noqa: E402


def _boot_set(Xt, Mt, Xr, Mr, n_boot, epochs, device, paired=False, At=None, Ar=None):
    """Seed-bootstrap the set-transformer C2ST -> (auc_mean, auc_std).

    Fold ONCE on the seed-mean, not per seed: folding max(a, 1-a) per seed
    rectifies noise and biases the mean upward by ~0.016 at a 0.02 seed
    spread near 0.5.
    """
    aucs = []
    for seed in range(n_boot):
        a = c2st_set(Xt, Mt, Xr, Mr, seed=seed, epochs=epochs, device=device,
                     paired=paired, At=At, Ar=Ar)
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    mean = float(np.mean(aucs))
    return max(mean, 1.0 - mean), float(np.std(aucs))

def _mlp_pair(res):
    return float(res["c2st_auc_mean"]), float(np.nan_to_num(
        0.5 * (res["c2st_auc_p95"] - res["c2st_auc_p5"]), nan=0.0))

def run_model(label, engine, root, cache, args):
    """All four AUCs for one model. Returns a dict (json-ready)."""
    run = pe.RunSpec(case="pflow", run_id=label, engine=engine, pred_root=root,
                     n_classes=5 if engine == "hgpflow" else 3)
    d = pe.load_pred_root(run, entry_stop=args.max_events)["truthpart"]
    d["target"] = ensure_cossin(d["target"])
    d["reco"] = ensure_cossin(d["reco"])
    ev = pe.load_pred_event_numbers(run, entry_stop=args.max_events)

    # ---- alignment guardrails: join + coverage report, drop unmatched ----
    rows, keep, info = match_cache_rows(cache, ev)
    print(f"  join={info['join']} coverage={info['coverage']:.4f} "
          f"({info['n_matched']}/{info['n_events']} events; "
          f"all conventions: { {k: round(v, 4) for k, v in info['coverages'].items()} })", flush=True)
    if info["coverage"] < args.min_join_coverage:
        raise ValueError(f"{label}: join coverage {info['coverage']:.4f} < "
                         f"--min-join-coverage {args.min_join_coverage}")
    pe.alignment_gate(cache, d["target"], "truthpart", rows, keep, label)
    if not keep.all():
        print(f"  WARNING: dropping {int((~keep).sum())} events without cache inputs", flush=True)
        d = {side: {k: v[keep] for k, v in dd.items()} for side, dd in d.items()}
        ev = ev[keep]

    out = {"engine": engine, "pred_root": root, "n_events": int(len(ev)),
           "join": info["join"], "join_coverage": info["coverage"]}
    mlp_kw = dict(n_boot=args.n_boot, epochs=args.epochs, device=args.device)

    # (1a) marginal, event-aggregate MLP
    et = _build_event_features(d["target"], "truthpart")
    er = _build_event_features(d["reco"], "truthpart")
    out["marginal_event_agg"] = pe.c2st_bootstrap(et, er, **mlp_kw)

    # (1b) joint, event-aggregate MLP (same bootstrap protocol)
    out["joint_event_agg"] = c2st_joint_bootstrap(
        cache, d, "truthpart", ev, join=info["join"],
        min_coverage=args.min_join_coverage, **mlp_kw)

    # The joint SetC2ST aggregate vector is the joint MLP feature vector: cached
    # input summaries concatenated with the corresponding output summary.
    ajt = pe._build_joint_event_features(
        cache, d["target"], "truthpart", ev, join=info["join"])
    ajr = pe._build_joint_event_features(
        cache, d["reco"], "truthpart", ev, join=info["join"])
    joint_mlp_width = int(out["joint_event_agg"]["feature_dim"])
    joint_seed_width = int(ajt.shape[1])
    if ajr.shape[1] != joint_seed_width or joint_seed_width != joint_mlp_width:
        raise RuntimeError(
            f"{label}: joint aggregate width mismatch: seed truth/reco "
            f"{ajt.shape[1]}/{ajr.shape[1]}, MLP {joint_mlp_width}")
    print(f"  joint seed width={joint_seed_width}; joint MLP feature width="
          f"{joint_mlp_width} (match)", flush=True)
    out["joint_seed_width"] = joint_seed_width
    out["joint_mlp_feature_width"] = joint_mlp_width

    # (2a) marginal, set-transformer (output set only)
    n_max = int(max(ak.num(d["target"]["eta"], axis=1).to_numpy().max(),
                    ak.num(d["reco"]["eta"], axis=1).to_numpy().max()))
    Xt, Mt = event_sets_pflow(d["target"], n_max)
    Xr, Mr = event_sets_pflow(d["reco"], n_max)
    m, s = _boot_set(Xt, Mt, Xr, Mr, args.n_boot, args.set_epochs, args.device,
                     At=et, Ar=er)
    out["marginal_set"] = {"auc_mean": m, "auc_std": s}

    # (2b) joint, set-transformer (union set: inputs ∪ outputs, shared n_max_out)
    Xjt, Mjt = build_joint_event_sets(cache, d["target"], "truthpart", ev,
                                      join=info["join"], n_max_out=n_max)
    Xjr, Mjr = build_joint_event_sets(cache, d["reco"], "truthpart", ev,
                                      join=info["join"], n_max_out=n_max)
    # paired split: same-event twins share input objects (see c2st_set docstring)
    m, s = _boot_set(Xjt, Mjt, Xjr, Mjr, args.n_boot, args.set_epochs, args.device,
                     paired=True, At=ajt, Ar=ajr)
    out["joint_set"] = {"auc_mean": m, "auc_std": s}

    ev_idx = np.arange(len(Xjt))
    ha, hb = ev_idx[ev_idx % 2 == 0], ev_idx[ev_idx % 2 == 1]
    n_half = min(len(ha), len(hb)); ha, hb = ha[:n_half], hb[:n_half]
    m, s = _boot_set(Xjt[ha], Mjt[ha], Xjt[hb], Mjt[hb],
                     args.n_boot, args.set_epochs, args.device,
                     paired=False, At=ajt[ha], Ar=ajt[hb])
    null_ok = bool(np.isfinite(m) and abs(m - 0.5) <= args.null_tol)
    out["null_joint_set"] = {"auc_mean": m, "auc_std": s, "passed": null_ok,
                             "tolerance": float(args.null_tol)}
    print(f"  set null joint AUC {m:.6f} (tolerance {args.null_tol:.3f}) "
          f"{'OK' if null_ok else ('FAIL (not enforced: --quick)' if args.quick else 'FAIL')}",
          flush=True)
    return out

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"

def run_metadata(args, results, *, cache_path, prediction_roots,
                      seeding_function, agg_dim, null_result, **extra):
    """Provenance header written with every C2ST result file."""
    meta = {
        "protocol": "hep4m.judge C2ST",
        "git_commit": _git_commit(),
        "cache_path": str(Path(cache_path).resolve()),
        "renorm_phi": bool(os.environ.get("HEP4M_C2ST_RENORM_PHI", "0") == "1"),
        "event_count": {k.replace("\n", " "): int(v["n_events"])
                        for k, v in results.items()},
        "agg_dim": agg_dim,
        "seeding_function": seeding_function,
        "prediction_root_paths": prediction_roots,
        "null_control": null_result,
    }
    meta.update(extra)
    return meta

def plot_results(results, out_png):
    """Paired bars (marginal solid | joint hatched) per model, one panel per level."""
    labels = list(results)
    panels = [("event-aggregate MLP", "marginal_event_agg", "joint_event_agg"),
              ("set-transformer", "marginal_set", "joint_set")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    x = np.arange(len(labels))
    w = 0.38
    for ax, (title, k_marg, k_joint) in zip(axes, panels):
        for i, lab in enumerate(labels):
            color = f"C{i}"
            r = results[lab]
            marg = (_mlp_pair(r[k_marg]) if "c2st_auc_mean" in r[k_marg]
                    else (r[k_marg]["auc_mean"], r[k_marg]["auc_std"]))
            joint = (_mlp_pair(r[k_joint]) if "c2st_auc_mean" in r[k_joint]
                     else (r[k_joint]["auc_mean"], r[k_joint]["auc_std"]))
            ax.bar(x[i] - w / 2, marg[0], w, yerr=marg[1], capsize=4,
                   color=color, alpha=0.9, label="marginal" if i == 0 else None)
            ax.bar(x[i] + w / 2, joint[0], w, yerr=joint[1], capsize=4,
                   color=color, alpha=0.55, hatch="//",
                   label="joint (⊕ inputs)" if i == 0 else None)
        ax.axhline(0.5, color="0.5", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("C2ST AUC  (0.5 = indistinguishable)")
    axes[0].set_ylim(bottom=0.45)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"-> {out_png}")

def main():
    ap = argparse.ArgumentParser(description="Marginal and joint C2ST, reconstruction direction")
    ap.add_argument("--cache", required=True,
                    help=".npz from hep4m/judge/build_input_cache.py")
    ap.add_argument("--row", action="append", default=[], required=True,
                    metavar="label:engine:root",
                    help="model row (engine in {hep4m, nano_hep, hgpflow}); repeatable")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100, help="MLP C2ST epochs")
    ap.add_argument("--set-epochs", type=int, default=60, help="set-transformer epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-join-coverage", type=float, default=0.995)
    ap.add_argument("--null-tol", type=float, default=0.02,
                    help="abort if |set-judge joint null AUC - 0.5| exceeds this")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test on a few events: --n-boot 1, 5 epochs for both "
                         "judges, null result recorded but not enforced")
    ap.add_argument("--out-dir", default="c2st_reco", help="output directory")
    args = ap.parse_args()
    if args.quick:
        args.n_boot, args.epochs, args.set_epochs = 1, 5, 5

    cache = load_c2st_input_cache(args.cache)
    print(f"input cache: {args.cache} ({len(cache['event_numbers'])} events, "
          f"modalities {cache['modalities']})", flush=True)

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
        r = results[lab]
        print(f"  event-agg  marginal {r['marginal_event_agg']['c2st_auc_mean']:.3f}"
              f"  joint {r['joint_event_agg']['c2st_auc_mean']:.3f}\n"
              f"  set-trans  marginal {r['marginal_set']['auc_mean']:.3f}"
              f"  joint {r['joint_set']['auc_mean']:.3f}", flush=True)

    out_json = Path(args.out_dir) / "c2st_reco.json"
    clean = {k.replace("\n", " "): v for k, v in results.items()}
    meta = run_metadata(
        args, results, cache_path=args.cache,
        prediction_roots={k.replace("\n", " "): v["pred_root"] for k, v in results.items()},
        seeding_function={"marginal": "evaluate._build_event_features",
                          "joint": "evaluate._build_joint_event_features"},
        agg_dim={k.replace("\n", " "): {"marginal": 8,
                  "joint": v["joint_seed_width"]} for k, v in results.items()},
        null_result={k.replace("\n", " "): v["null_joint_set"]
                     for k, v in results.items()},
        quick=bool(args.quick))
    json.dump({"_meta": meta, **clean},
              open(out_json, "w"), indent=1, default=str)
    print(f"-> {out_json}")
    failed_nulls = [k.replace("\n", " ") for k, v in results.items()
                    if not v["null_joint_set"]["passed"]]
    if failed_nulls and not args.quick:
        raise SystemExit(f"set-judge null failed for {failed_nulls}; results aborted")
    if results:
        plot_results(results, Path(args.out_dir) / "c2st_reco.png")

if __name__ == "__main__":
    main()
