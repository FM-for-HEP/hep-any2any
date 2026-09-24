"""Map flat metric names from pflow_report / generative_report to wandb
hierarchical keys using forward-slash subsections.

Wandb groups panels by the first '/' segment (and optionally by all
segments, configurable in the workspace settings), so a name like
"val/pflow/jet/pt_response/mean" creates a nested Panel Section structure.

Examples:
    to_wandb_subkey("mean_jet_pt_response", "pflow")
        -> "val/pflow/jet/pt_response/mean"
    to_wandb_subkey("ks_particle_pt", "pflow")
        -> "val/pflow/ks/particle_pt"
    to_wandb_subkey("quick_marginal_particle_auc", "gen")
        -> "val/gen/c2st/quick_marginal_particle_auc"
    to_wandb_subkey("frac_truth_class_0", "pflow")
        -> "val/pflow/class/frac_truth_class_0"
"""

_STATS = ("mean", "median", "std", "iqr")
# Metric families that act as their own value (no aggregation).
_GROUP_NO_STAT = ("ks", "wasserstein", "c2st", "chi2", "tv")


def to_wandb_subkey(raw: str, namespace: str = "pflow") -> str:
    """Return the wandb dashboard key for a flat metric name.

    Args:
        raw: e.g. "mean_jet_pt_response" or "ks_particle_pt"
        namespace: "pflow" or "gen" — controls the val/<namespace>/... prefix
    """
    # Strip leading stat token if present
    stat = None
    for s in _STATS:
        if raw.startswith(s + "_"):
            stat = s
            raw = raw[len(s) + 1:]
            break

    # Categorize the remainder
    if raw.startswith("jet_"):
        group, leaf = "jet", raw[4:]
    elif raw.startswith("quick_marginal_"):
        group, leaf = "c2st", raw
    elif raw.startswith("particle_"):
        group, leaf = "particle", raw[len("particle_"):]
    elif raw.startswith(_GROUP_NO_STAT):
        first, _, rest = raw.partition("_")
        # chi2_/tv_ are occupancy variants — keep them together
        if first in ("chi2", "tv"):
            group, leaf = "occupancy", raw
        else:
            group, leaf = first, rest
    elif raw.startswith("frac_"):
        group, leaf = "class", raw
    elif raw.startswith("event_"):
        group, leaf = "event", raw[len("event_"):]
    elif raw == "n_events" or raw.startswith("n_pred") or raw.startswith("n_true") \
            or raw.startswith("cardinality") or raw.startswith("reco_cardinality"):
        group, leaf = "summary", raw
    elif raw.startswith("mean_reco_cardinality"):
        # quirk: 'mean' wasn't stripped (it was — but the resulting body still
        # starts with 'reco_cardinality'; routed above)
        group, leaf = "summary", raw
    else:
        group, leaf = "misc", raw

    key = f"val/{namespace}/{group}/{leaf}"
    if stat:
        key += f"/{stat}"
    return key


# Image upload keys — map from PNG filename to the wandb hierarchical key.
PFLOW_IMAGES = {
    "jet_response.png":       "val/pflow/images/jet_response",
    "marginals.png":          "val/pflow/images/marginals",
    "cardinality.png":        "val/pflow/images/cardinality",
    "particle_residuals.png": "val/pflow/images/particle_residuals",
    "jet_residuals.png":      "val/pflow/images/jet_residuals",
    "event_displays.png":     "val/pflow/images/event_displays",
}

# generative-report produces occupancy_{modality}.png; the caller fills in modality.
def gen_image_key(modality: str) -> dict:
    return {f"occupancy_{modality}.png": "val/gen/images/occupancy"}


# Cardinality scalar shorts that are self.log'd directly from nano_hep_lightning.
CARDINALITY_REMAP = {
    "cardinality_acc":  "val/cardinality/acc",
    "cardinality_mae":  "val/cardinality/mae",
    "cardinality_bias": "val/cardinality/bias",
    "cardinality_std":  "val/cardinality/std",
    "n_pred_mean":      "val/cardinality/n_pred_mean",
    "n_pred_std":       "val/cardinality/n_pred_std",
    "n_true_mean":      "val/cardinality/n_true_mean",
    "n_true_std":       "val/cardinality/n_true_std",
}
