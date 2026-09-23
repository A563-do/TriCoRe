import pytest
import torch

from guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from .distill_transition import ddim_transition_from_xstart, stage_transition_indices


def test_stage_indices_are_two_to_one():
    starts, middles, ends = stage_transition_indices(1000, 100, 50)
    assert len(starts) == 49
    assert all(start > middle > end for start, middle, end in zip(starts, middles, ends))


def test_ddim_transition_is_finite():
    diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule("linear", 1000),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    x = torch.randn(2, 1, 8, 8)
    x_start = torch.randn_like(x)
    result = ddim_transition_from_xstart(
        diffusion, x, x_start, torch.tensor([800, 600]), torch.tensor([780, 580])
    )
    assert result.shape == x.shape
    assert torch.isfinite(result).all()


def test_stage_requires_two_to_one():
    with pytest.raises(ValueError):
        stage_transition_indices(1000, 100, 40)
