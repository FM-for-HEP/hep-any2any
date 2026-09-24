"""(a) Tokeniser encode/decode round trips, one test per modality.

test_config_quantiser_roundtrip: builds each of the seven tokenisers from the
    configs in configs/tokenizers (random weights, no data needed) and
    checks encode -> codes -> indices_to_zq reproduces the quantised latents,
    and that decode returns the expected shapes.
test_released_tokeniser_roundtrip: loads the released tokeniser checkpoints and
    runs stored tokens -> decode -> features -> encode -> tokens on 16 events of
    the test store. The decoder is lossy, so the round trip is not an identity;
    the test records the per-level code agreement and requires level-0
    agreement >= 0.5 (a random or mismatched tokeniser gives ~1/codebook_size).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from conftest import ALL7, TEST_STORE, TEST_STORE_SPLIT, load_cfg

pytestmark = pytest.mark.workflow


def _build_from_repo_config(mod: str):
    from hep4m.models.vqvae import VQVAE

    cfg_m = load_cfg(f"tokenizers/{mod}/model.yml")
    cfg_v = load_cfg(f"tokenizers/variables/{mod}.yml")

    def no_flash(o):
        if isinstance(o, dict):
            for k in list(o):
                if k in ("enable_flash_attn", "enable_flex_attn"):
                    o[k] = False
                else:
                    no_flash(o[k])
        elif isinstance(o, list):
            for x in o:
                no_flash(x)

    no_flash(cfg_m)
    cfg_m["_config_v"] = cfg_v
    torch.manual_seed(0)
    return VQVAE(cfg_m), cfg_m, cfg_v


@pytest.mark.parametrize("mod", ALL7)
def test_config_quantiser_roundtrip(mod):
    vq, cfg_m, cfg_v = _build_from_repo_config(mod)
    vq.eval()
    B = 3
    with torch.no_grad():
        if vq.vae_type == "transformer":
            N = 5
            x_cont = torch.randn(B, N, cfg_m["n_feat_cont"])
            x_cat = {k: torch.randint(0, n, (B, N)) for k, (n, _) in cfg_m["categorical"].items()}
            mask = torch.ones(B, N, dtype=torch.bool)
            mask[0, 3:] = False
            z_q, idx, _ = vq.encode(x_cont, x_cat, mask)
            z_back = vq.indices_to_zq(idx, mask)
            assert torch.allclose(z_back[mask], z_q[mask], atol=1e-4)
            x_hat, x_hat_cat = vq.decode(z_q, x_mask=mask)
            assert x_hat.shape == (B, N, cfg_m["n_feat_cont"])
            for k, (n, _) in cfg_m["categorical"].items():
                assert x_hat_cat[k].shape == (B, N, n)
        elif vq.vae_type == "mlp_bottleneck":
            n_in = cfg_m["encoder"]["init_net_bottleneck"]["input_size"]
            x_cont = torch.randn(B, 1, n_in)  # one global object per event
            z_q, idx, _ = vq.encode(x_cont, {}, None)
            z_back = vq.indices_to_zq(idx.squeeze(-1) if idx.shape[-1] == 1 else idx, None)
            assert torch.allclose(z_back.reshape(z_q.shape), z_q, atol=1e-4)
            x_hat, _ = vq.decode(z_q)
            assert torch.isfinite(x_hat).all()
        else:  # cellimg_transformer
            ntok = cfg_m["decoder"]["num_tokens_per_layer"]
            x = {}
            for i, (n, p) in enumerate(zip(ntok, cfg_m["pow2_scale"])):
                side = int(math.isqrt(n)) * 2 ** p
                x[f"feat0_{i}"] = torch.randn(B, side, side, cfg_m["encoder"]["conv2d"]["in_channels"])
            z_q, idx, _ = vq.encode(x, None, None)
            mask = torch.ones(idx.shape[:2], dtype=torch.bool)
            z_back = vq.indices_to_zq(idx, mask)
            assert torch.allclose(z_back, z_q, atol=1e-4)
            x_hat = vq.decode(z_q)
            for k, v in x.items():
                assert x_hat[k].shape[:3] == v.shape[:3]


N_EVENTS = 16


@pytest.mark.parametrize("mod", ALL7)
def test_released_tokeniser_roundtrip(mod, modality_dict_7mod, store_split_dir, tmp_path_factory):
    from hep4m.datasets.tokenized_memmap import ModalityMemmap
    from hep4m.decoding.detokenizer import _load_vqvae

    e = modality_dict_7mod[mod]
    vq, _ = _load_vqvae(e["config_path_v"], e["config_path_m"], e["checkpoint_path"], "cpu")
    mm = ModalityMemmap.load(TEST_STORE, TEST_STORE_SPLIT, mod)
    codes = [torch.from_numpy(mm.event_codes(i, mm.n_codebooks)).long() for i in range(N_EVENTS)]

    with torch.no_grad():
        if vq.vae_type == "transformer":
            L = max(1, max(c.shape[0] for c in codes))
            idx = torch.zeros(N_EVENTS, L, mm.n_codebooks, dtype=torch.long)
            mask = torch.zeros(N_EVENTS, L, dtype=torch.bool)
            for i, c in enumerate(codes):
                idx[i, :len(c)] = c
                mask[i, :len(c)] = True
            z_q = vq.indices_to_zq(idx, mask)
            x_hat, x_hat_cat = vq.decode(z_q, x_mask=mask)
            cat = {k: v.argmax(-1) for k, v in x_hat_cat.items()}
            _, idx2, _ = vq.encode(x_hat, cat, mask)
            agree = (idx2[mask] == idx[mask]).float().mean(0)
        elif vq.vae_type == "mlp_bottleneck":
            idx = torch.stack(codes)
            z_q = vq.indices_to_zq(idx.squeeze(-1), None)
            x_hat, _ = vq.decode(z_q)
            _, idx2, _ = vq.encode(x_hat, {}, None)
            agree = (idx2.reshape(idx.shape) == idx).float().mean((0, 1))
        else:
            idx = torch.stack(codes)
            mask = torch.ones(idx.shape[:2], dtype=torch.bool)
            x_hat = vq.decode(vq.indices_to_zq(idx, mask))
            # decoder emits (energy, indicator logit) per pixel; the encoder takes energy
            x_in = {k: v[..., :1] * (v[..., 1:2] > 0).float() for k, v in x_hat.items()}
            _, idx2, _ = vq.encode(x_in, None, None)
            agree = (idx2 == idx).float().mean((0, 1))

    agree = [round(float(a), 4) for a in np.atleast_1d(agree.numpy())]
    out = tmp_path_factory.getbasetemp() / "tokeniser_roundtrip.jsonl"
    with open(out, "a") as fh:
        fh.write(json.dumps({"modality": mod, "per_level_code_agreement": agree,
                             "events": N_EVENTS}) + "\n")
    print(f"[roundtrip] {mod}: per-level code agreement {agree}")
    assert agree[0] >= 0.5, f"{mod}: level-0 agreement {agree[0]} (mismatched tokeniser/store?)"
