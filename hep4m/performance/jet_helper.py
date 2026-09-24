import awkward as ak
import numpy as np


class JetHelper:
    def __init__(self, mode='vec_sum'):
        assert mode in ['vec_sum'], \
            f"Unsupported mode {mode}. Only 'vec_sum' is currently supported."
        self.mode = mode
    

    def compute_jets_PtEtaPhiE(self, pts, etas, phis, es):

        pxs = pts * np.cos(phis)
        pys = pts * np.sin(phis)
        pzs = pts * np.sinh(etas)

        jet_es = ak.sum(es, axis=-1)

        jet_pxs = ak.sum(pxs, axis=-1)
        jet_pys = ak.sum(pys, axis=-1)
        jet_pzs = ak.sum(pzs, axis=-1)

        jet_pts = np.sqrt(jet_pxs**2 + jet_pys**2)
        jet_pts = ak.where(jet_pts == 0, 1e-10, jet_pts)
        jet_etas = np.arcsinh(jet_pzs / jet_pts)
        jet_phis = np.arctan2(jet_pys, jet_pxs)

        n_const = ak.count(pts, axis=-1)

        return jet_pts, jet_etas, jet_phis, jet_es, n_const


    def compute_jets_PtEtaPhiM(self, pts, etas, phis, ms):

        pxs = pts * np.cos(phis)
        pys = pts * np.sin(phis)
        pzs = pts * np.sinh(etas)
        es = np.sqrt(ms**2 + pxs**2 + pys**2 + pzs**2)
        
        jet_es = ak.sum(es, axis=-1)

        jet_pxs = ak.sum(pxs, axis=-1)
        jet_pys = ak.sum(pys, axis=-1)
        jet_pzs = ak.sum(pzs, axis=-1)

        jet_pts = np.sqrt(jet_pxs**2 + jet_pys**2)
        jet_pts = ak.where(jet_pts == 0, 1e-10, jet_pts)
        jet_etas = np.arcsinh(jet_pzs / jet_pts)
        jet_phis = np.arctan2(jet_pys, jet_pxs)

        n_const = ak.count(pts, axis=-1)

        return jet_pts, jet_etas, jet_phis, jet_es, n_const


    def compute_jets_EEtaPhiM(self, es, etas, phis, ms):

        ps = np.sqrt(es**2 - ms**2)
        pts = ps / np.cosh(etas)

        return self.compute_jets_PtEtaPhiM(pts, etas, phis, ms)
    

