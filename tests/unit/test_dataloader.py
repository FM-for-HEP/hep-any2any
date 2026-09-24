"""
Test that dataset classes can be wrapped in a DataLoader and produce
correctly-shaped batches.  num_workers=0 avoids forking issues in CI.
"""

import torch
from torch.utils.data import DataLoader

from hep4m.datasets.dataset_modalities import get_dataset_class

REDUCE_DS  = 50
BATCH_SIZE = 8


def _get_one_batch(dataset_cls, fixture_path, config_v, feat_key: str) -> dict:
    # double_tok subclasses don't expose tree_name; base default is 'EventTree'
    ds = dataset_cls(
        filename=fixture_path,
        config_v=config_v,
        reduce_ds=REDUCE_DS,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    return next(iter(loader)), feat_key


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

class TestTrackDataLoader:

    def test_batch_shapes(self, fixture_path, track_config_v):
        COCOADatasetTrack = get_dataset_class('track')
        batch, feat_key = _get_one_batch(
            COCOADatasetTrack, fixture_path, track_config_v, "track_feat0"
        )
        x    = batch[feat_key]
        mask = batch[f"{feat_key}_mask"]

        assert x.ndim    == 3, f"x.ndim={x.ndim}, expected 3 (batch, objects, features)"
        assert mask.ndim == 2, f"mask.ndim={mask.ndim}, expected 2 (batch, objects)"
        assert x.shape[0]    == BATCH_SIZE
        assert mask.shape[0] == BATCH_SIZE
        assert x.shape[1]    == mask.shape[1], "object-dim mismatch between x and mask"

    def test_batch_dtypes(self, fixture_path, track_config_v):
        COCOADatasetTrack = get_dataset_class('track')
        batch, feat_key = _get_one_batch(
            COCOADatasetTrack, fixture_path, track_config_v, "track_feat0"
        )
        assert batch[feat_key].dtype               == torch.float32
        assert batch[f"{feat_key}_mask"].dtype     == torch.bool


# ---------------------------------------------------------------------------
# Topo
# ---------------------------------------------------------------------------

class TestTopoDataLoader:

    def test_batch_shapes(self, fixture_path, topo_config_v):
        COCOADatasetTopo = get_dataset_class('topo')
        batch, feat_key = _get_one_batch(
            COCOADatasetTopo, fixture_path, topo_config_v, "topo_feat0"
        )
        x    = batch[feat_key]
        mask = batch[f"{feat_key}_mask"]

        assert x.ndim    == 3
        assert mask.ndim == 2
        assert x.shape[0]    == BATCH_SIZE
        assert mask.shape[0] == BATCH_SIZE

    def test_no_nans_in_batch(self, fixture_path, topo_config_v):
        COCOADatasetTopo = get_dataset_class('topo')
        batch, feat_key = _get_one_batch(
            COCOADatasetTopo, fixture_path, topo_config_v, "topo_feat0"
        )
        mask = batch[f"{feat_key}_mask"]
        real = batch[feat_key][mask]  # only non-padded entries
        assert not torch.isnan(real).any(), "NaN found in topo batch"


# ---------------------------------------------------------------------------
# TruthPart
# ---------------------------------------------------------------------------

class TestTruthPartDataLoader:

    def test_batch_shapes(self, fixture_path, truthpart_config_v):
        COCOADatasetTruthPart = get_dataset_class('truthpart')
        batch, feat_key = _get_one_batch(
            COCOADatasetTruthPart, fixture_path, truthpart_config_v, "truthpart_feat0"
        )
        x    = batch[feat_key]
        mask = batch[f"{feat_key}_mask"]

        assert x.ndim    == 3
        assert mask.ndim == 2
        assert x.shape[0]    == BATCH_SIZE
        assert mask.shape[0] == BATCH_SIZE
