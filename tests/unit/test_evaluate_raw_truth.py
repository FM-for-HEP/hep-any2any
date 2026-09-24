"""evaluate metrics --raw-truth: raw truth particles as the target, matched events only
(synthetic ROOT files, no data needed)."""
from __future__ import annotations

import json

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

from hep4m.performance import evaluate as ev  # noqa: E402

N = 60


def _jagged(rng, counts, lo, hi):
    return ak.unflatten(rng.uniform(lo, hi, int(counts.sum())), counts)


@pytest.fixture()
def files(tmp_path):
    rng = np.random.default_rng(7)
    n_raw = rng.integers(1, 9, N)
    n_store = n_raw.copy()
    lost = rng.choice(N, 10, replace=False)
    n_store[lost] -= 1                       # tokenisation dropped a particle in 10 events
    raw = {v: _jagged(rng, n_raw, *r) for v, r in (("pt", (1, 50)), ("eta", (-2, 2)), ("phi", (-3, 3)))}
    raw_ev = np.sort(rng.choice(np.arange(2 * N), N, replace=False)).astype(np.int64)
    cls5 = ak.unflatten(rng.integers(0, 5, int(n_raw.sum())), n_raw)
    hgp = tmp_path / "hgpflow.root"
    with uproot.recreate(hgp) as f:
        f["event_tree"] = {"truth_pt": raw["pt"], "truth_eta": raw["eta"], "truth_phi": raw["phi"],
                           "truth_class": cls5, "event_number": raw_ev}
    # the raw COCOA file: same particles as float32, plus one pdgid -999 particle in event 3
    pdg = ak.unflatten(rng.choice([22, 211, -211, 130], int(n_raw.sum())), n_raw)
    trk = ak.unflatten(rng.integers(-1, 3, int(n_raw.sum())), n_raw)

    def add999(a, val):
        lst = ak.to_list(a)
        lst[3] = lst[3] + [val]
        return ak.Array(lst)

    cocoa = tmp_path / "raw.root"
    with uproot.recreate(cocoa) as f:
        f["EventTree"] = {"particle_pt": ak.values_astype(add999(raw["pt"], 5.0), np.float32),
                          "particle_eta": ak.values_astype(add999(raw["eta"], 0.1), np.float32),
                          "particle_phi": ak.values_astype(add999(raw["phi"], 0.2), np.float32),
                          "particle_pdgid": add999(pdg, -999), "particle_track_idx": add999(trk, -1),
                          "eventNumber": raw_ev}
    # prediction file (HEP4M layout): reco = raw particles smeared; tokenised truth = raw
    reco_pt = raw["pt"] * ak.unflatten(rng.normal(1, 0.1, int(n_raw.sum())), n_raw)
    pred = tmp_path / "pred.root"
    with uproot.recreate(pred) as f:
        f["event_tree"] = {
            **{f"truthpart_truth_{k}": v for k, v in (("pt", raw["pt"]), ("eta", raw["eta"]),
                                                      ("cosphi", np.cos(raw["phi"])),
                                                      ("sinphi", np.sin(raw["phi"])))},
            **{f"truthpart_reco_{k}": v for k, v in (("pt", reco_pt), ("eta", raw["eta"]),
                                                     ("cosphi", np.cos(raw["phi"])),
                                                     ("sinphi", np.sin(raw["phi"])))},
            "event_number": raw_ev}
    store = tmp_path / "store"
    store.mkdir()
    off = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(n_store, out=off[1:])
    off.tofile(store / "truthpart_offsets.npy")
    keep = np.ones(N, bool)
    keep[lost] = False
    return dict(hgp=hgp, cocoa=cocoa, pred=pred, store=store, keep=keep, raw=raw, reco_pt=reco_pt)


def _expected_iqr(f, keep):
    raw = f["raw"]
    t = np.hypot(np.asarray(ak.sum(raw["pt"] * np.cos(raw["phi"]), axis=1)),
                 np.asarray(ak.sum(raw["pt"] * np.sin(raw["phi"]), axis=1)))
    r = np.hypot(np.asarray(ak.sum(f["reco_pt"] * np.cos(raw["phi"]), axis=1)),
                 np.asarray(ak.sum(f["reco_pt"] * np.sin(raw["phi"]), axis=1)))
    x = (r / t)[keep]
    return np.median(x), np.percentile(x, 75) - np.percentile(x, 25)


@pytest.mark.parametrize("truth_file", ["hgp", "cocoa"])
def test_cli_raw_truth_matched(files, truth_file, tmp_path, monkeypatch):
    out = tmp_path / f"m_{truth_file}.json"
    monkeypatch.setattr("sys.argv", ["evaluate", "metrics", "--root-path", str(files["pred"]), "--engine", "hep4m",
                                     "--case", "pflow", "--run-id", "t", "--metrics-out", str(out),
                                     "--raw-truth", str(files[truth_file]), "--store", str(files["store"])])
    ev.main()
    m = json.loads(out.read_text())
    assert m["_selection"]["n_matched"] == int(files["keep"].sum()) == N - 10
    med, iqr = _expected_iqr(files, files["keep"])
    assert m["truthpart"]["median_jet_pt_response"] == pytest.approx(med, rel=1e-6)
    assert m["truthpart"]["iqr_jet_pt_response"] == pytest.approx(iqr, rel=1e-6)


def test_raw_truth_files_agree(files):
    a, ev_a, n_a = ev.load_raw_truth(files["hgp"])
    b, ev_b, n_b = ev.load_raw_truth(files["cocoa"])
    assert (n_a, n_b) == (5, 3)
    assert np.array_equal(ev_a, ev_b)
    for k in ("pt", "eta", "phi"):   # the -999 particle is dropped, float32 values equal
        assert ak.all(ak.values_astype(a[k], np.float32) == b[k])


def test_raw_truth_rejects_shuffled_events(files):
    loaded = ev.load_pred_root(ev.RunSpec("pflow", "t", "hep4m", str(files["pred"])))
    with pytest.raises(ValueError, match="first events"):
        ev.raw_truth_selection(loaded, files["hgp"], files["store"], np.arange(N)[::-1] + 1000)
