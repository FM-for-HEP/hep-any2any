"""(d) The set-transformer C2ST judge (hep4m.judge.c2st_set / c2st_reco) on tiny
synthetic particle sets.

Checks the three properties the judge relies on:
  * identical distributions -> AUC near 0.5 (null),
  * a shifted distribution -> AUC well above 0.5,
  * the event-paired (joint) split and the seed bootstrap run and fold to >= 0.5.
Sizes and epochs are small so the test runs in about a minute on CPU; the
thresholds are loose accordingly.
"""
from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

pytestmark = pytest.mark.workflow


def _events(rng, n, pt_scale=1.0, eta_shift=0.0):
    counts = rng.integers(2, 12, size=n)
    tot = counts.sum()
    pt = rng.exponential(10.0, tot) * pt_scale
    eta = rng.normal(eta_shift, 1.0, tot)
    phi = rng.uniform(-np.pi, np.pi, tot)
    return {"pt": ak.unflatten(pt, counts), "eta": ak.unflatten(eta, counts),
            "phi": ak.unflatten(phi, counts)}


def _sets(d, n_max=16):
    from hep4m.judge.c2st_set import ensure_cossin, event_sets_pflow

    return event_sets_pflow(ensure_cossin(dict(d)), n_max)


def test_event_sets_shapes():
    rng = np.random.default_rng(0)
    X, M = _sets(_events(rng, 50))
    assert X.shape == (50, 16, 4) and M.shape == (50, 16)
    assert np.all(X[~M] == 0)


def test_c2st_null_and_shift():
    from hep4m.judge.c2st_set import c2st_set

    rng = np.random.default_rng(1)
    n = 600
    Xa, Ma = _sets(_events(rng, n))
    Xb, Mb = _sets(_events(rng, n))
    Xs, Ms = _sets(_events(rng, n, pt_scale=2.0, eta_shift=0.7))
    null = c2st_set(Xa, Ma, Xb, Mb, seed=0, epochs=6, device="cpu")
    shift = c2st_set(Xa, Ma, Xs, Ms, seed=0, epochs=6, device="cpu")
    print(f"[c2st] null AUC={null:.3f} shifted AUC={shift:.3f}")
    assert abs(null - 0.5) < 0.1, null
    assert shift > 0.8, shift


def test_c2st_paired_bootstrap():
    from hep4m.judge.c2st_reco import _boot_set

    rng = np.random.default_rng(2)
    n = 300
    Xa, Ma = _sets(_events(rng, n))
    Xb, Mb = _sets(_events(rng, n, pt_scale=1.5))
    mean, std = _boot_set(Xa, Ma, Xb, Mb, n_boot=2, epochs=4, device="cpu", paired=True)
    print(f"[c2st] paired bootstrap AUC={mean:.3f} +- {std:.3f}")
    assert 0.5 <= mean <= 1.0 and np.isfinite(std)
