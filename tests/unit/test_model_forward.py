"""
Test a CPU forward+backward pass of the VQVAE model with a tiny config.

Uses synthetic tensors — no fixture file required — so this test is fast
and needs no data.

'kmeans_init: False' is critical: it avoids the first-batch codebook
initialisation that would require a full dataset pass.
"""

import torch


BATCH   = 4
N_OBJ   = 20   # number of objects per event
# The VQVAE init_net receives only the N continuous features (n_feat_cont).
# Positional features (sin/cos/eta) are handled separately as gpos and are
# not passed through the encoder init_net.
N_CONT  = 3    # matches tiny_track_model_config n_feat_cont


def _make_batch(batch=BATCH, n_obj=N_OBJ, n_cont=N_CONT):
    """Synthetic (x_cont, x_categorical, mask) triple on CPU.
    VQVAE.forward(x_cont, x_categorical, x_mask) -> (embedding_loss, x_hat, indices)
    """
    x_cont = torch.randn(batch, n_obj, n_cont)
    x_cat  = {}   # no categorical vars in the tiny track config
    mask   = torch.ones(batch, n_obj, dtype=torch.bool)
    return x_cont, x_cat, mask


# ---------------------------------------------------------------------------
# VQVAE construction
# ---------------------------------------------------------------------------

class TestVQVAEConstruction:

    def test_instantiates_on_cpu(self, tiny_track_model_config):
        from hep4m.models.vqvae import VQVAE
        model = VQVAE(tiny_track_model_config)
        assert model is not None

    def test_parameters_exist(self, tiny_track_model_config):
        from hep4m.models.vqvae import VQVAE
        model = VQVAE(tiny_track_model_config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestVQVAEForwardPass:

    def test_output_shape_matches_input(self, tiny_track_model_config):
        from hep4m.models.vqvae import VQVAE
        model = VQVAE(tiny_track_model_config)
        model.eval()
        x_cont, x_cat, mask = _make_batch()
        with torch.no_grad():
            embedding_loss, (x_hat_cont, x_hat_cat), indices = model(x_cont, x_cat, mask)
        # continuous reconstruction: same shape as input
        assert x_hat_cont.shape == x_cont.shape, (
            f"Reconstruction shape {x_hat_cont.shape} != input shape {x_cont.shape}"
        )

    def test_output_is_finite(self, tiny_track_model_config):
        from hep4m.models.vqvae import VQVAE
        model = VQVAE(tiny_track_model_config)
        model.eval()
        x_cont, x_cat, mask = _make_batch()
        with torch.no_grad():
            embedding_loss, (x_hat_cont, x_hat_cat), indices = model(x_cont, x_cat, mask)
        assert torch.isfinite(x_hat_cont).all(), "Non-finite values in reconstruction"


# ---------------------------------------------------------------------------
# Backward pass  (gradients flow)
# ---------------------------------------------------------------------------

class TestVQVAEBackwardPass:

    def test_loss_is_finite_scalar(self, tiny_track_model_config):
        from hep4m.models.vqvae import VQVAE
        model = VQVAE(tiny_track_model_config)
        model.train()
        x_cont, x_cat, mask = _make_batch()
        embedding_loss, (x_hat_cont, x_hat_cat), indices = model(x_cont, x_cat, mask)
        # embedding_loss is (B,); reduce before combining with mse_loss
        loss = torch.nn.functional.mse_loss(x_hat_cont, x_cont) + embedding_loss.mean()
        assert loss.ndim == 0,             "Loss is not a scalar"
        assert torch.isfinite(loss).all(), "Loss is not finite"

    def test_gradients_flow(self, tiny_track_model_config):
        from hep4m.models.vqvae import VQVAE
        model = VQVAE(tiny_track_model_config)
        model.train()
        x_cont, x_cat, mask = _make_batch()
        embedding_loss, (x_hat_cont, x_hat_cat), indices = model(x_cont, x_cat, mask)
        loss = torch.nn.functional.mse_loss(x_hat_cont, x_cont) + embedding_loss.mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients were computed"
        assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradient found"
