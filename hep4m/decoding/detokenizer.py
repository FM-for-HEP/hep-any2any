"""Shared per-modality token detokenizer.

Loads the frozen VQ-VAE tokenisers and decodes (content, pos) token tensors back
into physics features, keeping both the continuous and the categorical (e.g.
particle ``class``) decoder heads. Used by the HEP4M and nanoHEP inference helpers.

Schema convention (matches HEP4M):

    config_v["features"][f"{modality}_feat0"] = [
        N,                         # max_token_cardinality
        [cont_feature_names...],   # continuous, in output_net_cont order
        [[cat_name, n_classes], ...],  # categorical
        [gpos_feature_names...],   # eta, cosphi, sinphi typically
    ]
"""
from __future__ import annotations

from hep4m.paths import safe_load_expanded as _hp_safe_load
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from hep4m.utility.var_transformation import VarTransformation


LOG = logging.getLogger(__name__)


def _load_vqvae(config_path_v: str, config_path_m: str, checkpoint_path: str, device: str):
    """Load and freeze a HEP4M VQ-VAE tokenizer."""
    from hep4m.models.vqvae import VQVAE
    with open(config_path_v) as f:
        cfg_v = _hp_safe_load(f)
    with open(config_path_m) as f:
        cfg_m = _hp_safe_load(f)
    cfg_m["_config_v"] = cfg_v
    vq = VQVAE(cfg_m)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()
    }
    vq.load_state_dict(state, strict=False)
    vq.to(device).eval()
    for p in vq.parameters():
        p.requires_grad = False
    return vq, cfg_v


class Detokenizer:
    """Per-modality VQ-VAE + pos-tokenizer detokenizer.

    Parameters
    ----------
    modality_dict_path : str
        Path to the YAML file mapping modality -> {config_path_v,
        config_path_m, checkpoint_path}. Same schema as HEP4M's
        ``modality_dict.yml``.
    modalities : iterable of str
        Which modalities to load. Pass only the ones you'll decode.
    device : str
        Where to keep the frozen VQ-VAEs (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, modality_dict_path: str | Path, modalities: Iterable[str], device: str = "cuda"):
        self.device = device
        with open(modality_dict_path) as f:
            self.modality_dict = _hp_safe_load(f)

        modalities = list(modalities)
        missing = [m for m in modalities if m not in self.modality_dict]
        if missing:
            raise KeyError(f"modalities {missing} not in modality_dict at {modality_dict_path}")

        self.vqs: Dict[str, torch.nn.Module] = {}
        self.config_v: Dict[str, dict] = {}
        self.features_cont: Dict[str, List[str]] = {}
        self.features_cat: Dict[str, List[str]] = {}
        self.features_gpos: Dict[str, List[str]] = {}
        self.transforms: Dict[str, Dict[str, VarTransformation]] = {}
        self.data_type: Dict[str, str] = {}

        for m in modalities:
            entry = self.modality_dict[m]
            vq, cv = _load_vqvae(
                entry["config_path_v"], entry["config_path_m"],
                entry["checkpoint_path"], device,
            )
            self.vqs[m] = vq
            self.config_v[m] = cv
            self.data_type[m] = cv.get("data_type", "set")

            feats = cv["features"][f"{m}_feat0"]
            self.features_cont[m] = list(feats[1])
            self.features_cat[m] = [pair[0] for pair in feats[2]]
            self.features_gpos[m] = list(feats[3]) if len(feats) > 3 else []

            tdict = cv.get("transformation_dict", {})
            self.transforms[m] = {k: VarTransformation(v) for k, v in tdict.items()}

        # Lazy-load pos tokenizer; not all modalities use it.
        self._pos_tok = None

    @property
    def pos_tok(self):
        if self._pos_tok is None:
            from hep4m.models.pos_tokenizer import PosTokenizer
            self._pos_tok = PosTokenizer().to(self.device).eval()
            for p in self._pos_tok.parameters():
                p.requires_grad = False
        return self._pos_tok

    def feature_names(self, modality: str) -> List[str]:
        """All feature names (continuous + categorical + gpos) for one modality, in output order."""
        return (
            list(self.features_cont[modality])
            + list(self.features_cat[modality])
            + list(self.features_gpos[modality])
        )

    @torch.no_grad()
    def decode_tokens(
        self,
        modality: str,
        content_tokens: torch.Tensor,
        pos_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode a batch of tokens to physics features.

        Parameters
        ----------
        modality : str
        content_tokens : (B, N, num_q) or (B, N) long
            Per-element VQ codebook indices.
        pos_tokens : (B, N, num_q_pos) long, optional
            Per-element position codebook indices.
        mask : (B, N) bool, optional
            True for real elements, False for padding. If None, treats
            all positions as real.

        Returns
        -------
        dict feat_name -> tensor
            Continuous features: (B, N) after inverse VarTransformation.
            Categorical features: (B, N) class IDs from argmax.
            gpos features: (B, N) values from the pos tokenizer.
        """
        if modality not in self.vqs:
            raise KeyError(
                f"modality {modality!r} not loaded; pass it in modalities= at __init__"
            )
        vq = self.vqs[modality]
        device = self.device

        ct = content_tokens.to(device)
        if mask is None:
            mask = torch.ones(ct.shape[:2], dtype=torch.bool, device=device)
        m_dev = mask.to(device).bool()

        z_q = vq.indices_to_zq(ct, m_dev)
        x_hat_cont, x_hat_cat = vq.decode(z_q, x_mask=m_dev)

        out: Dict[str, torch.Tensor] = {}

        # Continuous features (in output_net_cont order)
        for feat_i, feat_name in enumerate(self.features_cont[modality]):
            vals = x_hat_cont[..., feat_i]
            tform = self.transforms[modality].get(feat_name)
            if tform is not None:
                vals = tform.inverse(vals)
            out[feat_name] = vals.detach().cpu()

        # Categorical features (argmax over last dim per dict entry)
        for feat_name in self.features_cat[modality]:
            if feat_name not in x_hat_cat:
                # Sometimes the model didn't have this categorical head; skip.
                continue
            cls = torch.argmax(x_hat_cat[feat_name], dim=-1)
            out[feat_name] = cls.detach().cpu()

        # gpos features (only if pos_tokens supplied AND modality has gpos)
        if pos_tokens is not None and self.features_gpos[modality]:
            pt_dev = pos_tokens.to(device)
            x_gpos_hat = self.pos_tok.decode(pt_dev)  # (B, N, len(gpos))
            if (self.data_type.get(modality) == "global"
                    and x_gpos_hat.shape[1] != x_hat_cont.shape[1]):
                # Global modalities such as truthjet can encode one physical
                # object with several latent content rows. Position codes are
                # stored/broadcast on the latent axis; keep one position row
                # per decoded physical object.
                LOG.debug(
                    "global decoded-axis alignment modality=%s position_len=%d "
                    "physical_continuous_len=%d; keeping the leading position rows",
                    modality, x_gpos_hat.shape[1], x_hat_cont.shape[1])
                x_gpos_hat = x_gpos_hat[:, :x_hat_cont.shape[1], ...]
            for feat_i, feat_name in enumerate(self.features_gpos[modality]):
                if feat_i >= x_gpos_hat.shape[-1]:
                    break
                out[feat_name] = x_gpos_hat[..., feat_i].detach().cpu()

        return out


    def _cell_helper(self, modality: str):
        if not hasattr(self, "_cellimg_helpers"):
            self._cellimg_helpers = {}
        if modality not in self._cellimg_helpers:
            from hep4m.datasets.cell_image_helper import CellImageHelper
            h = CellImageHelper(pow2_scale=self.config_v[modality]["pow2_scale"])
            h.to_device(self.device)
            self._cellimg_helpers[modality] = h
        return self._cellimg_helpers[modality]

    def _cell_lbirbi_from_gpos(self, helper, gpos: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Invert ``CellImageHelper.set_to_image``'s window geometry from decoded
        per-grid-token gpos (eta, cosphi, sinphi) patch midpoints.

        Only the layer-0 slice is needed: its patch grid spans the (constant-size)
        window. eta snaps to an integer layer-5 bin (clamped exactly like
        ``get_left_right_edge_idx``); phi unwraps around the circular mean, and any
        2π branch is equivalent after ``image_to_set`` re-wraps (bin mids repeat
        with period 2π every ``gran`` indices), so we shift into valid range.
        """
        import math
        n0 = helper.num_patches_per_layer_1d[0]
        g0 = gpos[: n0 * n0]
        keep5 = helper.bins_to_keep_dict[5]
        gran5 = helper.grans[5]

        # eta: meshgrid 'ij' flattening -> row-major, rows = eta
        eta_mids = g0[:, 0].view(n0, n0)[:, 0]
        bin5_w = 6.0 / gran5
        eta_w = keep5 * bin5_w
        left_eta = float(eta_mids.min()) - (eta_w / n0) / 2
        left_eta_bin5 = int(round((left_eta + 3.0) / bin5_w))
        left_eta_bin5 = max(0, min(left_eta_bin5, gran5 - keep5))
        right_eta_bin5 = left_eta_bin5 + keep5

        # phi: unwrap around circular mean, then snap; shift branch into range
        cosb = g0[:, 1].view(n0, n0)[0]
        sinb = g0[:, 2].view(n0, n0)[0]
        phi = torch.atan2(sinb, cosb)
        m = float(torch.atan2(sinb.mean(), cosb.mean()))
        phi_u = m + torch.remainder(phi - m + math.pi, 2 * math.pi) - math.pi
        phi5_w = (4 * math.pi) / (2 * gran5)
        phi_w = keep5 * phi5_w
        left_phi = float(phi_u.min()) - (phi_w / n0) / 2
        left_phi_bin5 = int(round((left_phi + 2 * math.pi) / phi5_w))
        while left_phi_bin5 < 0:
            left_phi_bin5 += gran5
        while left_phi_bin5 + keep5 > 2 * gran5:
            left_phi_bin5 -= gran5
        right_phi_bin5 = left_phi_bin5 + keep5

        lbirbi = {}
        for li in range(helper.n_layers):
            eta_lbi = int(helper.grans[li] / helper.grans[5] * left_eta_bin5)
            eta_rbi = int(helper.grans[li] / helper.grans[5] * right_eta_bin5) + 1
            phi_lbi = int(helper.grans[li] / helper.grans[-1] * left_phi_bin5)
            phi_rbi = int(helper.grans[li] / helper.grans[-1] * right_phi_bin5) + 1
            lbirbi[li] = torch.tensor([[eta_lbi, eta_rbi], [phi_lbi, phi_rbi]])
        return lbirbi

    @torch.no_grad()
    def decode_cell_image_event(
        self,
        modality: str,
        content_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        ind_logit_th: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Decode ONE event of a ``cell_image`` modality to a cell set.

        Parameters
        ----------
        content_tokens : (N_grid, num_q) long — all grid tokens of the event
            (fixed N_grid, e.g. 156 for pow2_scale [3,3,3,3,3,2]).
        pos_tokens : (N_grid, 3) long — pos-tokenizer codes of the grid-token
            (eta, cosphi, sinphi) patch midpoints; used to invert the window.
        ind_logit_th : float — indicator threshold for ``image_to_set``.

        Returns
        -------
        dict short-feat -> np.ndarray over cells:
        continuous features (inverse-transformed), then ``eta``, ``phi``,
        ``layer``, ``ind_logit`` — matching the HEP4M cell-image decode path.
        """
        vq = self.vqs[modality]
        helper = self._cell_helper(modality)

        ct = content_tokens.unsqueeze(0).to(self.device).long()
        mask = torch.ones(1, ct.shape[1], dtype=torch.bool, device=self.device)
        z_q = vq.indices_to_zq(ct, mask)
        img_dict = vq.decode(z_q)  # {feat0_i: (1, H, W, C_cont+1)}; ind channel last

        gpos = self.pos_tok.decode(pos_tokens.unsqueeze(0).to(self.device).long())[0]
        lbirbi = self._cell_lbirbi_from_gpos(helper, gpos)
        lbirbi = {k: v.to(self.device) for k, v in lbirbi.items()}

        feat_dict = {k: v[0][..., :-1] for k, v in img_dict.items()}
        ind_logit_dict = {k.replace("feat0", "ind"): v[0][..., -1]
                          for k, v in img_dict.items()}
        cells = helper.image_to_set(feat_dict, ind_logit_dict, lbirbi,
                                    ind_logit_th=ind_logit_th)
        # columns: [cont feats..., eta, phi, layer, ind_logit]
        n_cont = len(self.features_cont[modality])
        out: Dict[str, np.ndarray] = {}
        for fi, fn in enumerate(self.features_cont[modality]):
            vals = cells[:, fi]
            tform = self.transforms[modality].get(fn)
            if tform is not None:
                vals = tform.inverse(vals)
            out[fn.removeprefix(f"{modality}_")] = vals.detach().cpu().numpy()
        for off, name in enumerate(("eta", "phi", "layer", "ind_logit")):
            out[name] = cells[:, n_cont + off].detach().cpu().numpy()
        return out

    def to_jagged(
        self,
        decoded: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        modality: Optional[str] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Convert a ``(B, N)`` decoded-feature dict to per-event jagged arrays.

        Useful for handing off to ``hep4m.utility.tree_writer.TreeWriter``.

        Parameters
        ----------
        decoded : dict feat_name -> (B, N) tensor
        mask : (B, N) bool
            True for real elements; False positions are dropped per event.
        """
        latent_mask = mask.bool().cpu()
        B = latent_mask.shape[0]
        out: Dict[str, List[np.ndarray]] = {}
        for feat_name, vals in decoded.items():
            v_np = vals.detach().cpu().numpy()
            if vals.shape[1] != latent_mask.shape[1]:
                LOG.debug(
                    "decoded-axis mask mismatch modality=%s component=%s "
                    "q_mask_len=%d decoded_len=%d; using all-ones decoded mask",
                    modality or "unknown", feat_name, latent_mask.shape[1], vals.shape[1])
                component_mask = torch.ones(vals.shape[:2], dtype=torch.bool)
            else:
                component_mask = latent_mask
            m_np = component_mask.numpy()
            out[feat_name] = [v_np[b][m_np[b]] for b in range(B)]
        return out
