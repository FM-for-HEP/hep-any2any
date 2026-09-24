"""Set-transformer classifier two-sample test (C2ST) for sets of reconstructed objects.

A small set transformer is trained to tell events of sample A (e.g. true particles)
from events of sample B (e.g. a model's particles). Each event is a padded set of
objects with features (log pT, eta, cos phi, sin phi) plus an optional per-event
aggregate vector. The test AUC is the score: 0.5 means the classifier cannot tell
the samples apart. ``paired=True`` splits train/test by event so that the two
samples of the same event never straddle the split (needed when both samples
share inputs, as in the joint test of ``hep4m.judge.c2st_reco``).

Set ``HEP4M_C2ST_RENORM_PHI=1`` to renormalise (cos phi, sin phi) onto the unit
circle before the test; generated angles that are off the unit circle are
otherwise an easy tell for the classifier.
"""

import os

import numpy as np
import awkward as ak
import torch
import torch.nn as nn
from scipy.stats import rankdata

DEV = "cuda" if torch.cuda.is_available() else "cpu"

RENORM_PHI = os.environ.get("HEP4M_C2ST_RENORM_PHI", "0") == "1"

def renorm_cossin(c, s):
    """Unit-circle renormalisation; works for both numpy and (jagged) awkward."""
    r = (c ** 2 + s ** 2) ** 0.5
    try:
        r = ak.where(r > 0, r, 1.0)          # jagged awkward path
    except Exception:
        r = np.where(np.asarray(r) > 0, r, 1.0)
    return c / r, s / r

def ensure_cossin(d):
    if "cosphi" not in d and "phi" in d:
        d["cosphi"] = np.cos(d["phi"]); d["sinphi"] = np.sin(d["phi"])
    if RENORM_PHI and "cosphi" in d and "sinphi" in d:
        d["cosphi"], d["sinphi"] = renorm_cossin(d["cosphi"], d["sinphi"])
    return d

def event_sets_pflow(d, n_max):
    """Per-event padded truthpart set (N_evt, n_max, 4) + mask; F = log-pT, eta, cos, sin."""
    lk = np.log1p(ak.where(d["pt"] > 0, d["pt"], 0.0))
    c, sn = d["cosphi"], d["sinphi"]
    if RENORM_PHI:
        r = np.sqrt(c ** 2 + sn ** 2)
        r = ak.where(r > 0, r, 1.0)
        c, sn = c / r, sn / r
    feat = ak.concatenate([lk[..., None], d["eta"][..., None],
                           c[..., None], sn[..., None]], axis=-1)
    padded = ak.pad_none(feat, n_max, axis=1, clip=True)
    X = np.ma.filled(ak.to_numpy(padded, allow_missing=True), 0.0).astype(np.float32)
    counts = ak.to_numpy(ak.num(feat, axis=1))
    mask = (np.arange(n_max)[None, :] < counts[:, None])
    return X, mask

class SetC2ST(nn.Module):
    def __init__(self, f_dim, d=48, heads=4, layers=2, agg_dim=0):
        super().__init__()
        self.embed = nn.Linear(f_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        # Aggregate-seeded CLS: inject hand-crafted per-event summary features
        # (multiplicity, summed/mean kinematics - the event-MLP features) into
        # the CLS initialisation, so the set model strictly contains the
        # aggregate classifier's information. Counting padded elements through
        # attention is what a small transformer does worst; this hands it N.
        self.agg_embed = nn.Linear(agg_dim, d) if agg_dim > 0 else None
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.0,
                                         batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask, agg=None):
        h = self.embed(x)
        cls = self.cls.expand(h.shape[0], -1, -1)
        if self.agg_embed is not None and agg is not None:
            cls = cls + self.agg_embed(agg).unsqueeze(1)
        h = torch.cat([cls, h], dim=1)
        kpm = torch.cat([torch.zeros(h.shape[0], 1, dtype=torch.bool, device=x.device), ~mask], dim=1)
        h = self.tr(h, src_key_padding_mask=kpm)
        return self.head(h[:, 0]).squeeze(-1)

def _auc(prob, y):
    # Average tied ranks, matching evaluate._compute_c2st and the
    # Mann--Whitney definition of ROC AUC (important for exact null twins).
    ranks = rankdata(prob, method="average")
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    U = ranks[y == 1].sum() - npos * (npos + 1) / 2
    return float(U / (npos * nneg))

def c2st_set(Xt, Mt, Xr, Mr, seed=0, epochs=60, device=None, paired=False, At=None, Ar=None):
    """paired=True: event-paired 70/30 split — row i of Xt and Xr are the SAME
    event (joint C2ST), so both twins must land on the same side or the model
    memorizes a train twin's shared input objects and labels its held-out twin
    (the null then reads AUC >> 0.5). Default False: random split."""
    device = device or DEV
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = np.concatenate([Xt, Xr]); M = np.concatenate([Mt, Mr])
    y = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xr))]).astype(np.float32)
    flat = X[M]; mu = flat.mean(0); sd = flat.std(0); sd[sd == 0] = 1.0
    X = ((X - mu) / sd) * M[..., None]
    A = None
    if At is not None and Ar is not None:
        A = np.concatenate([At, Ar]).astype(np.float32)
        amu = A.mean(0); asd = A.std(0); asd[asd == 0] = 1.0
        A = (A - amu) / asd
    n = len(y)
    if paired:
        assert len(Xt) == len(Xr), "paired c2st_set needs aligned (same-event) rows"
        ev = rng.permutation(len(Xt)); sp_ev = int(0.7 * len(Xt))
        tr = np.concatenate([ev[:sp_ev], ev[:sp_ev] + len(Xt)])
        te = np.concatenate([ev[sp_ev:], ev[sp_ev:] + len(Xt)])
    else:
        idx = rng.permutation(n); sp = int(0.7 * n); tr, te = idx[:sp], idx[sp:]
    Xtr, Mtr, ytr = (torch.from_numpy(a).to(device) for a in (X[tr], M[tr], y[tr]))
    Xte, Mte = torch.from_numpy(X[te]).to(device), torch.from_numpy(M[te]).to(device)
    Atr = Ate = None
    if A is not None:
        Atr = torch.from_numpy(A[tr]).to(device); Ate = torch.from_numpy(A[te]).to(device)
    model = SetC2ST(X.shape[-1], agg_dim=(A.shape[-1] if A is not None else 0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    L = X.shape[1]
    bs = max(8, min(256, 65536 // max(L, 1)))
    MAX_STEPS = 35000
    step_count = 0
    for ep in range(epochs):
        if step_count >= MAX_STEPS:
            break
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), bs):
            if step_count >= MAX_STEPS:
                break
            s = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[s], Mtr[s], None if Atr is None else Atr[s]), ytr[s]); loss.backward(); opt.step()
            step_count += 1
    model.eval()
    # score in chunks: one full-test-set forward overflows int32 indexing in
    # the nested-tensor attention kernel at joint-set sizes (illegal memory access)
    with torch.no_grad():
        probs = []
        ec = max(64, 4 * bs)
        for i in range(0, len(Xte), ec):
            probs.append(torch.sigmoid(model(
                Xte[i:i + ec], Mte[i:i + ec],
                None if Ate is None else Ate[i:i + ec])).cpu())
        prob = torch.cat(probs).numpy()
    return _auc(prob, y[te])
