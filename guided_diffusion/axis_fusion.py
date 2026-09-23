"""Per-step fusion rules for the three TriCoRe view predictions."""

import torch as th


FUSION_MODES = (
    "mean",
    "median",
    "agreement",
    "xstart_agreement",
    "xstart_mean",
    "xstart_median",
    "xstart_loo",
    "xstart_fixed",
    "adaptive_xstart",
    "final_mean",
    "final_loo",
)

# Every mode in this tuple fuses clean-state x_start predictions and then
# takes one shared DDPM posterior step, so comparisons across these modes
# differ only in the fusion rule (same posterior path and shared noise).
XSTART_FUSION_MODES = (
    "xstart_agreement",
    "xstart_mean",
    "xstart_median",
    "xstart_loo",
    "xstart_fixed",
    "adaptive_xstart",
)

FINAL_FUSION_MODES = ("final_mean", "final_loo")


def is_xstart_fusion_mode(mode):
    return mode in XSTART_FUSION_MODES


def is_final_fusion_mode(mode):
    return mode in FINAL_FUSION_MODES


def fuse_axis_predictions(
    predictions,
    mode="mean",
    temperature=0.05,
    timestep=None,
    num_timesteps=None,
):
    """Fuse spatially aligned x/y/z predictions and return their voxel weights.

    ``agreement`` treats a view as less reliable where it differs from the
    three-view consensus. ``xstart_agreement`` uses leave-one-out consensus;
    it is intended for fusion of denoised x_start predictions before sampling
    the next diffusion state. ``xstart_fixed`` is the controlled non-LOO arm:
    the same fixed-temperature softmax, but against the self-inclusive
    three-view consensus. ``adaptive_xstart`` uses the leave-one-out consensus
    and changes the temperature over the trajectory: conservative averaging at
    high-noise steps and sharper selection at low-noise steps.
    """
    if len(predictions) == 0:
        raise ValueError("TriCoRe fusion requires at least one prediction")
    if mode not in FUSION_MODES:
        raise ValueError(f"Unknown fusion mode {mode!r}; choose from {FUSION_MODES}")
    if temperature <= 0:
        raise ValueError("fusion temperature must be positive")

    stacked = th.stack(predictions, dim=0)
    if len(predictions) == 1:
        return stacked[0], th.ones_like(stacked)
    effective_mode = {
        "xstart_mean": "mean",
        "xstart_median": "median",
        "xstart_loo": "xstart_agreement",
        # Fixed-weight reliability arm: same softmax weighting and the same
        # fixed temperature as xstart_agreement, but the consensus includes
        # the scored view itself (three-view mean). Isolates the leave-one-out
        # change under an otherwise identical clean-state fusion protocol.
        "xstart_fixed": "agreement",
        "final_mean": "mean",
        "final_loo": "xstart_agreement",
    }.get(mode, mode)

    if effective_mode == "mean":
        weights = th.full_like(stacked, 1.0 / len(predictions))
        return stacked.mean(dim=0), weights
    elif effective_mode == "median":
        median = stacked.median(dim=0).values
        # Report deterministic one-hot weights for the axis closest to the
        # voxelwise median; ties resolve to the first axis.
        weights = th.zeros_like(stacked)
        closest = (stacked - median.unsqueeze(0)).abs().argmin(dim=0, keepdim=True)
        weights.scatter_(0, closest, 1.0)
        return median, weights
    elif effective_mode == "agreement":
        consensus = stacked.mean(dim=0, keepdim=True)
        disagreement = (stacked - consensus).abs()
        weights = th.softmax(-disagreement / temperature, dim=0)
    else:
        # Compare each axis with the mean of the other two axes. This avoids
        # allowing an outlier to pull the consensus toward itself.
        consensus = (stacked.sum(dim=0, keepdim=True) - stacked) / 2.0
        disagreement = (stacked - consensus).abs()
        effective_temperature = temperature
        if effective_mode == "adaptive_xstart":
            if timestep is None or num_timesteps is None or num_timesteps <= 1:
                raise ValueError(
                    "adaptive_xstart requires timestep and num_timesteps"
                )
            # t is traversed from high noise to zero. The explicit endpoint
            # factors make this schedule reproducible for any starting t.
            progress = float(timestep) / float(num_timesteps - 1)
            progress = min(max(progress, 0.0), 1.0)
            effective_temperature = temperature * (0.5 + 1.5 * progress)
        weights = th.softmax(-disagreement / effective_temperature, dim=0)

    fused = (stacked * weights).sum(dim=0)
    return fused, weights
