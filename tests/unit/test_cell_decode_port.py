"""nano cell decoding: geometry-inversion regression.

``Detokenizer._cell_lbirbi_from_gpos`` inverts the window geometry that
``CellImageHelper.set_to_image`` encoded into per-grid-token gpos midpoints,
through the LOSSY PosTokenizer roundtrip. At phi ~ ±pi the reconstructed window
may sit on a different (equivalent) 2pi branch, so the contract is NOT
index-equality but *cell-set equality* through ``image_to_set`` (which re-wraps
phi). Covers eta-edge clamping and the phi wrap.
"""
import numpy as np
import pytest
import torch
import awkward as ak

from hep4m.datasets.cell_image_helper import CellImageHelper
from hep4m.models.pos_tokenizer import PosTokenizer
from hep4m.decoding.detokenizer import Detokenizer


@pytest.fixture(scope="module")
def helper():
    return CellImageHelper(pow2_scale=[3, 3, 3, 3, 3, 2])


@pytest.fixture(scope="module")
def ptok():
    return PosTokenizer().eval()


def _true_window(helper, ref_eta, ref_phi):
    empty = ak.Array(np.zeros(0))
    empty_i = ak.Array(np.zeros(0, dtype=int))
    _, _, lbirbi, mids = helper.set_to_image(
        ref=(ref_eta, ref_phi), coordinates=(empty, empty, empty_i),
        var_dict={"cell_e": empty})
    return lbirbi, mids


@pytest.mark.parametrize("ref_eta", [-2.9, -1.1, 0.0, 2.3, 2.9])
@pytest.mark.parametrize("ref_phi", [-3.14, -1.6, 0.0, 2.0, 3.14])
def test_lbirbi_inversion_gives_identical_cell_sets(helper, ptok, ref_eta, ref_phi):
    lb_true, mids = _true_window(helper, ref_eta, ref_phi)
    with torch.no_grad():
        gpos = ptok.decode(ptok(mids.unsqueeze(0).float()))[0]
    lb_hat = Detokenizer._cell_lbirbi_from_gpos(None, helper, gpos)

    rng = np.random.default_rng(42)
    feat, ind = {}, {}
    for li, n in enumerate([64, 64, 32, 16, 16, 8]):
        feat[f"feat0_{li}"] = torch.from_numpy(rng.normal(size=(n, n, 1))).float()
        ind[f"ind_{li}"] = torch.from_numpy(rng.normal(size=(n, n))).float()
    a = helper.image_to_set(feat, ind, lb_true, ind_logit_th=0.0)
    b = helper.image_to_set(feat, ind, lb_hat, ind_logit_th=0.0)
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-6), (
        f"ref=({ref_eta},{ref_phi}): cell sets differ, "
        f"maxdiff={float((a - b).abs().max()):.2e}"
    )
