"""
Validate that every YAML config in the configs/ tree can be parsed and
contains the minimum required top-level keys for its role.

These tests catch typos, missing keys, and YAML syntax errors early —
before any GPU time is spent.
"""

from hep4m.paths import safe_load_expanded as _hp_safe_load
from pathlib import Path
import pytest

CONFIGS_DIR = Path(__file__).parents[2] / "configs"


def _load(path: Path) -> dict:
    with open(path) as f:
        return _hp_safe_load(f) or {}


# ---------------------------------------------------------------------------
# Variable configs  (configs/tokenizers/variables/*.yml)
# ---------------------------------------------------------------------------
VARIABLE_REQUIRED_KEYS = {"modality", "features", "data_loading", "data_processing",
                          "transformation_dict"}

@pytest.mark.parametrize("cfg_path", sorted((CONFIGS_DIR / "tokenizers" / "variables").glob("*.yml")))
def test_variable_config_keys(cfg_path):
    cfg = _load(cfg_path)
    missing = VARIABLE_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{cfg_path.name} missing keys: {missing}"


@pytest.mark.parametrize("cfg_path", sorted((CONFIGS_DIR / "tokenizers" / "variables").glob("*.yml")))
def test_variable_config_data_loading_keys(cfg_path):
    cfg = _load(cfg_path)
    dl = cfg.get("data_loading", {})
    for key in ("branches_to_read", "branches_store_raw", "branches_rename"):
        assert key in dl, f"{cfg_path.name}: data_loading missing '{key}'"


# ---------------------------------------------------------------------------
# Tokenizer model configs  (configs/tokenizers/*/model.yml)
# ---------------------------------------------------------------------------
MODEL_REQUIRED_KEYS = {"n_feat_cont", "common_emb_dim", "encoder", "quantizer", "decoder"}

@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("tokenizers/*/model.yml")))
def test_tokenizer_model_config_keys(cfg_path):
    cfg = _load(cfg_path)
    missing = MODEL_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{cfg_path.parent.name}/model.yml missing keys: {missing}"


@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("tokenizers/*/model.yml")))
def test_tokenizer_model_quantizer_keys(cfg_path):
    cfg = _load(cfg_path)
    q = cfg.get("quantizer", {})
    # All quantizer types must have at least dim and codebook_size.
    # Note: vq_vae_truthjet uses a MultiheadVQ variant that has no 'type' key.
    for key in ("dim", "codebook_size"):
        assert key in q, f"{cfg_path.parent.name}/model.yml quantizer missing '{key}'"


# ---------------------------------------------------------------------------
# Tokenizer train configs  (configs/tokenizers/*/train.yml)
# ---------------------------------------------------------------------------
TRAIN_REQUIRED_KEYS = {"learning_rate", "batchsize_train", "batchsize_val",
                       "num_epochs", "path_train", "path_val"}

@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("tokenizers/*/train.yml")))
def test_tokenizer_train_config_keys(cfg_path):
    cfg = _load(cfg_path)
    missing = TRAIN_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{cfg_path.parent.name}/train.yml missing keys: {missing}"


# ---------------------------------------------------------------------------
# HEP4M model and train configs  (configs/train/hep4m_*/config_{m,t}*.yml)
# ---------------------------------------------------------------------------
HEP4M_MODEL_REQUIRED_KEYS = {"embedding_dim"}
HEP4M_TRAIN_REQUIRED_KEYS = {"learning_rate", "num_epochs", "batchsize_train"}


@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("train/hep4m_*/config_m*.yml")))
def test_hep4m_model_config_parses(cfg_path):
    cfg = _load(cfg_path)
    assert cfg, f"{cfg_path} is empty or unparseable"
    missing = HEP4M_MODEL_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{cfg_path.relative_to(CONFIGS_DIR)} missing keys: {missing}"


@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("train/hep4m_*/config_t*.yml")))
def test_hep4m_train_config_parses(cfg_path):
    cfg = _load(cfg_path)
    assert cfg, f"{cfg_path} is empty or unparseable"
    missing = HEP4M_TRAIN_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{cfg_path.relative_to(CONFIGS_DIR)} missing keys: {missing}"


# ---------------------------------------------------------------------------
# nanoHEP train configs  (configs/train/nanohep_*.yml)
# ---------------------------------------------------------------------------
NANO_TRAIN_REQUIRED_KEYS = {"gpt_config", "vocab_args", "tokenized_root", "all_modalities",
                            "sampling_dict", "block_size", "batch_size"}


@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("train/nanohep_*.yml")))
def test_nanohep_train_config_parses(cfg_path):
    cfg = _load(cfg_path)
    missing = NANO_TRAIN_REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{cfg_path.relative_to(CONFIGS_DIR)} missing keys: {missing}"
    assert set(cfg["vocab_args"]["modalities"]) >= set(cfg["all_modalities"])


# ---------------------------------------------------------------------------
# Inference configs  (configs/infer/*.yml)
# ---------------------------------------------------------------------------
HEP4M_ITEM_KEYS = {"input_modalities", "output_modalities", "use_truth_cardinality", "card_topk_dict",
                   "card_temperature_dict", "top_k_token_dict", "top_k_gpos_token_dict",
                   "temperature_token_dict", "temperature_gpos_token_dict", "filepath_dict",
                   "dir_flag", "reduce_ds"}
NANO_ITEM_KEYS = {"sampling_dict", "argmax", "max_new_tokens", "dir_flag", "suffix"}


@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("infer/*.yml")))
def test_inference_config_parses(cfg_path):
    cfg = _load(cfg_path)
    assert {"init", "items"} <= cfg.keys(), cfg_path.name
    init = cfg["init"]
    assert "output_dir" in init and "model" in init, cfg_path.name
    if init.get("model_type", "hep4m") == "nano_hep":
        assert {"checkpoint_path", "modality_dict_path"} <= init["model"].keys()
        assert "nano_hep" in init
        required = NANO_ITEM_KEYS
    else:
        assert {"config_path_m", "checkpoint_path", "modality_dict_path"} <= init["model"].keys()
        required = HEP4M_ITEM_KEYS
    for it in cfg["items"]:
        missing = required - it.keys()
        assert not missing, f"{cfg_path.name} item {it.get('info')!r} missing keys: {missing}"


# ---------------------------------------------------------------------------
# Tokenise configs  (configs/tokenize/*.yml)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_path", sorted(CONFIGS_DIR.glob("tokenize/*.yml")))
def test_tokenize_config_parses(cfg_path):
    cfg = _load(cfg_path)
    assert {"config_path_v", "config_path_m", "checkpoint_path"} <= cfg["init"]["model"].keys()
    for it in cfg["items"]:
        assert {"input_path", "dir_flag", "reduce_ds"} <= it.keys()


# ---------------------------------------------------------------------------
# Modality dicts  (configs/modality_dicts/*.yml, configs/train/*/modality_dict.yml)
# ---------------------------------------------------------------------------
MODALITY_ENTRY_KEYS = {"config_path_v", "config_path_m", "checkpoint_path", "codebook_loss_wts"}
_MODALITY_DICTS = sorted(CONFIGS_DIR.glob("modality_dicts/*.yml")) + sorted(CONFIGS_DIR.glob("train/*/modality_dict.yml"))


@pytest.mark.parametrize("cfg_path", _MODALITY_DICTS)
def test_modality_dict_config_parses(cfg_path):
    cfg = _load(cfg_path)
    assert cfg, f"{cfg_path} is empty or unparseable"
    for modality, entry in cfg.items():
        if not isinstance(entry, dict):
            continue
        missing = MODALITY_ENTRY_KEYS - entry.keys()
        assert not missing, f"{cfg_path.relative_to(CONFIGS_DIR)} modality '{modality}' missing keys: {missing}"


def test_config_references_exist():
    """Every ${HEP4M_REPO}/configs/... path named in a config exists in the repository."""
    import re
    bad = []
    for p in CONFIGS_DIR.rglob("*.yml"):
        for ref in re.findall(r"\$\{HEP4M_REPO\}/(configs/[\w./-]+\.yml)", p.read_text()):
            if not (CONFIGS_DIR.parent / ref).exists():
                bad.append(f"{p.relative_to(CONFIGS_DIR)} -> {ref}")
    assert not bad, bad
