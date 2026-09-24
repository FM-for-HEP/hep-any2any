"""
Shared pytest fixtures for the HEP4M test suite.

Test fixtures (not in the repository; tests that need them are skipped):
  HEP4M_FIXTURE_ROOT       1000-event slice of a raw COCOA single-jet ROOT file
                           (default tests/unit/fixtures/singlejet_fixture_1000events.root)
  HEP4M_TOKENIZED_FIXTURE  100-event tokenised store, made with
                           tests/unit/fixtures/generate_tokenized_fixture.py
                           (default tests/unit/fixtures/tokenized_100events)
"""

import os
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Fixture ROOT file
# ---------------------------------------------------------------------------
_REPO_FIXTURE = Path(__file__).parent / "fixtures" / "singlejet_fixture_1000events.root"
_DEFAULT_FIXTURE = str(_REPO_FIXTURE)
FIXTURE_ROOT = os.environ.get("HEP4M_FIXTURE_ROOT", _DEFAULT_FIXTURE)


@pytest.fixture(scope="session")
def fixture_path() -> str:
    p = Path(FIXTURE_ROOT)
    if not p.exists():
        pytest.skip(f"Fixture file not found: {p}")
    return str(p)


# ---------------------------------------------------------------------------
# Tokenized fixture (100 events x 4 modalities), made by
# tests/unit/fixtures/generate_tokenized_fixture.py.
# ---------------------------------------------------------------------------
_TOKENIZED_FIXTURE = Path(os.environ.get(
    "HEP4M_TOKENIZED_FIXTURE", Path(__file__).parent / "fixtures" / "tokenized_100events"))


@pytest.fixture(scope="session")
def tokenized_fixture_root() -> str:
    """Root directory of the tokenized fixture. Tests find per-split
    subdirs (``val/``) underneath."""
    p = _TOKENIZED_FIXTURE
    if not p.exists():
        pytest.skip(f"Tokenized fixture not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def tokenized_fixture_val(tokenized_fixture_root) -> str:
    """Path to the val/ subdir of the tokenized fixture."""
    p = Path(tokenized_fixture_root) / "val"
    if not p.exists():
        pytest.skip(f"Tokenized val fixture not found: {p}")
    return str(p)


# ---------------------------------------------------------------------------
# Variable configs  (copied verbatim from configs/tokenizers/variables/*.yml so tests
# have no filesystem dependency beyond the fixture ROOT file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def track_config_v() -> dict:
    return {
        "detector": "COCOA",
        "modality": "track",
        "features": {
            "track_feat0": [
                20,
                ["track_pt", "track_d0", "track_z0"],
                [],
                ["track_eta", "track_cosphi", "track_sinphi"],
            ]
        },
        "sort_by_var": "track_pt",
        "data_loading": {
            "branches_to_read": [
                "track_pt", "track_eta", "track_phi", "track_d0", "track_z0",
                "particle_eta", "particle_phi",
                "eventNumber",
            ],
            "branches_store_raw": ["track_pt", "track_eta"],
            "branches_rename": {"eventNumber": "event_number"},
        },
        "data_processing": {
            "vars_sin_cos": ["track_phi"],
            "vars_sin_cos_og_delete": [],
            "vars_relatives": {},
            "cuts": {
                "ref_eta_var":  "particle_eta",
                "ref_eta_var2": "track_eta",
                "eta_var":      "track_eta",
                "ref_phi_var":  "particle_phi",
                "ref_phi_var2": "track_phi",
                "phi_var":      "track_phi",
                "deta_min": -0.75, "deta_max": 0.75,
                "dphi_min": -0.75, "dphi_max": 0.75,
                "vars_to_delete": ["particle_eta", "particle_phi"],
            },
            "vars_to_transform": ["track_pt", "track_d0", "track_z0"],
        },
        "getitem_return": [],
        "transformation_dict": {
            "track_pt": {
                "transformation": "pow(x,m)", "m": 0.5,
                "scale_mode": "min_max",
                "mean": None, "std": None,
                "min": 0.962, "max": 15.619, "range": [-1, 1],
            },
            "track_d0": {
                "transformation": None,
                "scale_mode": "min_max",
                "mean": 0, "std": 0.0317,
                "min": -0.2, "max": 0.2, "range": [-1, 1],
            },
            "track_z0": {
                "transformation": None,
                "scale_mode": "min_max",
                "mean": 0, "std": 0.0863,
                "min": -0.4, "max": 0.4, "range": [-1, 1],
            },
        },
    }


@pytest.fixture(scope="session")
def topo_config_v() -> dict:
    return {
        "detector": "COCOA",
        "modality": "topo",
        "features": {
            "topo_feat0": [
                70,
                ["topo_e", "topo_rho", "topo_em_frac"],
                [["topo_em_category", 3]],
                ["topo_eta", "topo_cosphi", "topo_sinphi"],
            ]
        },
        "sort_by_var": "topo_e",
        "data_loading": {
            "branches_to_read": [
                "topo_eta", "topo_phi", "topo_e", "topo_rho",
                "particle_eta", "particle_phi",
                "topo_ecal_e", "topo_hcal_e",
                "eventNumber",
            ],
            "branches_store_raw": ["topo_e", "topo_eta"],
            "branches_rename": {"eventNumber": "event_number"},
        },
        "data_processing": {
            "vars_sin_cos": ["topo_phi"],
            "vars_sin_cos_og_delete": [],
            "vars_relatives": {},
            "cuts": {
                "ref_eta_var":  "particle_eta",
                "ref_eta_var2": "topo_eta",
                "eta_var":      "topo_eta",
                "ref_phi_var":  "particle_phi",
                "ref_phi_var2": "topo_phi",
                "phi_var":      "topo_phi",
                "deta_min": -0.75, "deta_max": 0.75,
                "dphi_min": -0.75, "dphi_max": 0.75,
                "vars_to_delete": ["particle_eta", "particle_phi"],
            },
            "vars_to_transform": ["topo_e", "topo_rho", "topo_em_frac"],
        },
        "getitem_return": [],
        "transformation_dict": {
            "topo_e": {
                "transformation": "pow(x,m)", "m": 0.5,
                "scale_mode": "min_max",
                "mean": None, "std": None,
                "min": 1, "max": 15, "range": [-1, 1],
            },
            "topo_rho": {
                "transformation": None,
                "scale_mode": "min_max",
                "mean": 1905.9, "std": 484.053,
                "min": 1547.056, "max": 3824.707, "range": [-1, 1],
            },
            "topo_em_frac": {
                "transformation": "inv_sigmoid",
                "scale_mode": "min_max",
                "mean": None, "std": None,
                "min": -13.8, "max": 13.8, "range": [-1, 1],
            },
        },
    }


@pytest.fixture(scope="session")
def truthpart_config_v() -> dict:
    return {
        "detector": "COCOA",
        "modality": "truthpart",
        "features": {
            "truthpart_feat0": [
                32,
                ["truthpart_pt", "truthpart_e"],
                [["truthpart_class", 3]],
                ["truthpart_eta", "truthpart_cosphi", "truthpart_sinphi"],
            ]
        },
        "sort_by_var": "truthpart_pt",
        "data_loading": {
            "branches_to_read": [
                "particle_pt", "particle_eta", "particle_phi",
                "particle_pdgid", "particle_e", "particle_track_idx",
                "eventNumber",
            ],
            "branches_store_raw": ["truthpart_pt", "truthpart_eta", "truthpart_e"],
            "branches_rename": {
                "particle_pt":  "truthpart_pt",
                "particle_eta": "truthpart_eta",
                "particle_phi": "truthpart_phi",
                "particle_e":   "truthpart_e",
                "eventNumber":  "event_number",
            },
        },
        "data_processing": {
            "vars_sin_cos": ["truthpart_phi"],
            "vars_sin_cos_og_delete": [],
            "vars_relatives": {},
            "cuts": {
                "ref_eta_var":  "truthpart_eta",
                "ref_eta_var2": "truthpart_eta",
                "eta_var":      "truthpart_eta",
                "ref_phi_var":  "truthpart_phi",
                "ref_phi_var2": "truthpart_phi",
                "phi_var":      "truthpart_phi",
                "deta_min": -0.75, "deta_max": 0.75,
                "dphi_min": -0.75, "dphi_max": 0.75,
                "vars_to_delete": [],
            },
            "vars_to_transform": ["truthpart_pt", "truthpart_e"],
        },
        "getitem_return": [],
        "transformation_dict": {
            "truthpart_pt": {
                "transformation": "pow(x,m)", "m": 0.5,
                "scale_mode": "min_max",
                "mean": None, "std": None,
                "min": 0.962, "max": 15.619, "range": [-1, 1],
            },
            "truthpart_e": {
                "transformation": "pow(x,m)", "m": 0.5,
                "scale_mode": "min_max",
                "mean": 0.289, "std": 0.375,
                "min": 0.962, "max": 100.0, "range": [-1, 1],
            },
        },
    }


# ---------------------------------------------------------------------------
# Tiny VQVAE model config  (CPU-friendly: flash_attn off, tiny dims)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_track_model_config() -> dict:
    emb = 32
    return {
        "n_feat_cont": 3,
        "common_emb_dim": emb,
        "codebook_dim": 4,
        "categorical_vars": [],
        "categorical": {},
        "encoder": {
            "init_net": {
                "input_size": None,
                "output_size": emb,
                "hidden_layers": [32],
                "activation": "LeakyReLU",
                "final_activation": None,
                "norm_layer": None,
                "norm_final_layer": False,
                "dropout": 0.0,
                "context_size": 0,
            },
            "transformer": {
                "embed_dim": emb,
                "num_layers": 2,
                "mha_config": {"enable_flash_attn": False, "num_heads": 4},
                "dense_config": {"embed_dim": emb, "hidden_dim": emb},
                "out_dim": emb,
                "layer_scale": True,
                "apply_norm_after_last_proj": True,
            },
        },
        "quantizer": {
            "type": "ResidualVQ",
            "dim": emb,
            "num_quantizers": 2,
            "codebook_dim": 4,
            "codebook_size": 64,
            "kmeans_init": False,
        },
        "decoder": {
            "transformer": {
                "embed_dim": emb,
                "num_layers": 2,
                "mha_config": {"enable_flash_attn": False, "num_heads": 4},
                "dense_config": {"embed_dim": emb, "hidden_dim": emb},
                "out_dim": emb,
                "layer_scale": True,
                "apply_norm_after_last_proj": True,
            },
            "output_net_cont": {
                "input_size": emb,
                "output_size": 3,   # n_feat_cont
                "hidden_layers": [32],
                "activation": "LeakyReLU",
                "final_activation": None,
                "norm_layer": "LayerNorm",
                "norm_final_layer": False,
                "dropout": 0.0,
                "context_size": 0,
            },
            "output_net_categorical": {
                "input_size": emb,
                "output_size": None,
                "hidden_layers": [32],
                "activation": "LeakyReLU",
                "final_activation": None,
                "norm_layer": "LayerNorm",
                "norm_final_layer": False,
                "dropout": 0.0,
                "context_size": 0,
            },
        },
    }
