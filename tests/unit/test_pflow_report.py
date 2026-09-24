"""Tests for hep4m.performance.pflow_report and related modules."""

import numpy as np
import matplotlib
matplotlib.use("Agg")


def test_jet_helper_four_vectors():
    """JetHelper computes jets from pt, eta, phi, E arrays."""
    from hep4m.performance.jet_helper import JetHelper

    jh = JetHelper()
    # 2 events, 3 particles each
    pts = np.array([[30.0, 20.0, 10.0], [50.0, 25.0, 15.0]])
    etas = np.array([[0.5, -0.3, 1.0], [0.0, 0.8, -0.5]])
    phis = np.array([[0.1, 1.2, -0.5], [0.0, -1.0, 2.0]])
    es = np.array([[35.0, 22.0, 15.0], [55.0, 30.0, 18.0]])
    jets = jh.compute_jets_PtEtaPhiE(pts, etas, phis, es)
    assert jets is not None


def test_cardinality_plot():
    """helper_fns.cardinality_plot runs on synthetic indicator scores."""
    import awkward as ak
    from hep4m.performance.helper_fns import cardinality_plot

    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10, size=100)
    ind = ak.unflatten(rng.uniform(size=int(counts.sum())), counts)
    truth_card = rng.integers(1, 10, size=100)
    fig = cardinality_plot(ind, truth_card, ind_th=0.5)
    assert fig is not None


def test_marginal_distributions():
    """helper_fns.marginal_distributions runs without error on synthetic data."""
    from hep4m.performance.helper_fns import marginal_distributions

    n_samples = 200
    data_flat_dict = {
        "truth": {
            "pt": np.random.exponential(10, n_samples),
            "eta": np.random.normal(0, 2, n_samples),
        },
        "reco": {
            "pt": np.random.exponential(10, n_samples),
            "eta": np.random.normal(0, 2, n_samples),
        },
    }
    cosmetic_dict = {"COLORS": {"truth": "black", "reco": "red"}}
    fig = marginal_distributions(
        data_flat_dict, names=["truth", "reco"], vars=["pt", "eta"],
        log_var_pos=[0], cosmetic_dict=cosmetic_dict,
    )
    assert fig is not None
