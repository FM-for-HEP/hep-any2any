"""
Test that the three primary dataset classes (track, topo, truthpart) can be
instantiated from the fixture ROOT file and return correctly-shaped tensors.

Key note: the fixture uses tree name 'EventTree' (raw COCOA convention).
All three modality classes default to 'event_tree' (tokenizer output
convention), so we must pass tree_name="EventTree" explicitly.
"""

import torch

from hep4m.datasets.dataset_modalities import get_dataset_class

REDUCE_DS = 50   # only load 50 events — fast enough for CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_getitem(item: dict, feat_key: str):
    """Shared shape/dtype assertions for a single __getitem__ return value.

    In double_tok, __getitem__ returns:
      - '{feat_key}'           : tensor (N_objects, N_cont_features)
      - '{feat_key}_mask'      : bool tensor (N_objects,)  -- per-feature mask
      - '{feat_key}_categorical': dict
      - 'event_number'         : scalar
    """
    mask_key = f"{feat_key}_mask"
    assert feat_key   in item, f"Missing feature key '{feat_key}'"
    assert mask_key   in item, f"Missing mask key '{mask_key}' (got keys: {list(item.keys())})"
    assert "event_number" in item, "Missing 'event_number'"

    x    = item[feat_key]
    mask = item[mask_key]

    assert isinstance(x,    torch.Tensor), f"x is {type(x)}, expected Tensor"
    assert isinstance(mask, torch.Tensor), f"mask is {type(mask)}, expected Tensor"

    assert x.dtype    == torch.float32, f"x.dtype={x.dtype}, expected float32"
    assert mask.dtype == torch.bool,    f"mask.dtype={mask.dtype}, expected bool"

    assert x.ndim    == 2, f"x.ndim={x.ndim}, expected 2"
    assert mask.ndim == 1, f"mask.ndim={mask.ndim}, expected 1"
    assert x.shape[0] == mask.shape[0], "x and mask first-dim mismatch"

    # No NaNs in the real (non-padded) part of the feature tensor
    real_x = x[mask]
    assert not torch.isnan(real_x).any(), "NaN found in non-padded feature tensor"


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

# Note: in double_tok, dataset_modalities subclass __init__ methods do not
# expose tree_name — only dataset_base does, and its default is 'EventTree',
# which matches the raw COCOA fixture.  No override needed here.

class TestCOCOADatasetTrack:

    def test_instantiation(self, fixture_path, track_config_v):
        COCOADatasetTrack = get_dataset_class('track')
        ds = COCOADatasetTrack(
            filename=fixture_path,
            config_v=track_config_v,
            reduce_ds=REDUCE_DS,
        )
        assert len(ds) == REDUCE_DS

    def test_getitem_keys_and_shapes(self, fixture_path, track_config_v):
        COCOADatasetTrack = get_dataset_class('track')
        ds = COCOADatasetTrack(
            filename=fixture_path,
            config_v=track_config_v,
            reduce_ds=REDUCE_DS,
        )
        item = ds[0]
        _check_getitem(item, feat_key="track_feat0")

    def test_getitem_feature_dim(self, fixture_path, track_config_v):
        """Continuous features: pt, d0, z0 → shape[1]==3.
        Positional vars (eta, cosphi, sinphi) go into track_feat0_gpos, not the main tensor.
        """
        COCOADatasetTrack = get_dataset_class('track')
        ds = COCOADatasetTrack(
            filename=fixture_path,
            config_v=track_config_v,
            reduce_ds=REDUCE_DS,
        )
        item = ds[0]
        x    = item["track_feat0"]
        gpos = item.get("track_feat0_gpos")
        # main tensor: only the 3 continuous vars (pt, d0, z0)
        assert x.shape[1] == 3, f"Expected 3 cont features, got {x.shape[1]}"
        # gpos tensor: 3 positional vars (eta, cosphi, sinphi)
        if gpos is not None:
            assert gpos.shape[1] == 3, f"Expected 3 gpos features, got {gpos.shape[1]}"

    def test_raw_branches_stored(self, fixture_path, track_config_v):
        """branches_store_raw populates self.data_dict['*_raw'] for use in metrics/plotting.
        They are NOT included in __getitem__ output — check data_dict directly.
        """
        COCOADatasetTrack = get_dataset_class('track')
        ds = COCOADatasetTrack(
            filename=fixture_path,
            config_v=track_config_v,
            reduce_ds=REDUCE_DS,
        )
        assert "track_pt_raw"  in ds.data_dict, "track_pt_raw not in data_dict"
        assert "track_eta_raw" in ds.data_dict, "track_eta_raw not in data_dict"
        assert sum([len(x) for x in ds.data_dict["track_pt_raw"]]) == len(ds), "raw length mismatch"


# ---------------------------------------------------------------------------
# Topo
# ---------------------------------------------------------------------------

class TestCOCOADatasetTopo:

    def test_instantiation(self, fixture_path, topo_config_v):
        COCOADatasetTopo = get_dataset_class('topo')
        ds = COCOADatasetTopo(
            filename=fixture_path,
            config_v=topo_config_v,
            reduce_ds=REDUCE_DS,
        )
        assert len(ds) == REDUCE_DS

    def test_getitem_keys_and_shapes(self, fixture_path, topo_config_v):
        COCOADatasetTopo = get_dataset_class('topo')
        ds = COCOADatasetTopo(
            filename=fixture_path,
            config_v=topo_config_v,
            reduce_ds=REDUCE_DS,
        )
        item = ds[0]
        _check_getitem(item, feat_key="topo_feat0")

    def test_em_frac_derived(self, fixture_path, topo_config_v):
        """additional_processing should derive topo_em_frac from ecal/hcal branches."""
        COCOADatasetTopo = get_dataset_class('topo')
        ds = COCOADatasetTopo(
            filename=fixture_path,
            config_v=topo_config_v,
            reduce_ds=REDUCE_DS,
        )
        # topo_ecal_e / topo_hcal_e must have been consumed; em_frac must exist
        assert "topo_em_frac"     in ds.data_dict, "topo_em_frac not derived"
        assert "topo_ecal_e"  not in ds.data_dict, "topo_ecal_e should be deleted after processing"
        assert "topo_hcal_e"  not in ds.data_dict, "topo_hcal_e should be deleted after processing"


# ---------------------------------------------------------------------------
# TruthPart
# ---------------------------------------------------------------------------

class TestCOCOADatasetTruthPart:

    def test_instantiation(self, fixture_path, truthpart_config_v):
        COCOADatasetTruthPart = get_dataset_class('truthpart')
        ds = COCOADatasetTruthPart(
            filename=fixture_path,
            config_v=truthpart_config_v,
            reduce_ds=REDUCE_DS,
        )
        assert len(ds) == REDUCE_DS

    def test_getitem_keys_and_shapes(self, fixture_path, truthpart_config_v):
        COCOADatasetTruthPart = get_dataset_class('truthpart')
        ds = COCOADatasetTruthPart(
            filename=fixture_path,
            config_v=truthpart_config_v,
            reduce_ds=REDUCE_DS,
        )
        item = ds[0]
        _check_getitem(item, feat_key="truthpart_feat0")

    def test_particle_class_derived(self, fixture_path, truthpart_config_v):
        """additional_processing should derive truthpart_class (0/1/2) from pdgid."""
        COCOADatasetTruthPart = get_dataset_class('truthpart')
        import awkward as ak
        ds = COCOADatasetTruthPart(
            filename=fixture_path,
            config_v=truthpart_config_v,
            reduce_ds=REDUCE_DS,
        )
        assert "truthpart_class" in ds.data_dict, "truthpart_class not derived"
        cated = ak.concatenate(ds.data_dict["truthpart_class"])
        flat = ak.to_numpy(ak.flatten(cated))
        assert set(flat).issubset({0, 1, 2}), f"Unexpected class values: {set(flat)}"
