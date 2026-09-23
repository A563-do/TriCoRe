"""Deterministic DDIM transition targets for progressive distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from guided_diffusion.respace import space_timesteps


def retained_timesteps(diffusion_steps: int, schedule_steps: int) -> list[int]:
    values = sorted(space_timesteps(diffusion_steps, f"ddim{schedule_steps}"))
    if len(values) != schedule_steps:
        raise ValueError(f"expected {schedule_steps} retained steps, got {len(values)}")
    return values


def stage_transition_indices(diffusion_steps, teacher_steps, student_steps):
    """Return matching two-teacher-step intervals in descending order."""
    if teacher_steps != 2 * student_steps:
        raise ValueError("teacher_steps must equal 2 * student_steps")
    teacher = retained_timesteps(diffusion_steps, teacher_steps)
    student = retained_timesteps(diffusion_steps, student_steps)
    teacher_index = {value: index for index, value in enumerate(teacher)}
    if not set(student).issubset(teacher_index):
        missing = sorted(set(student) - set(teacher))
        raise ValueError(f"student timesteps are not retained by teacher: {missing[:8]}")
    starts, middles, ends = [], [], []
    for end_index in range(len(student) - 1):
        start, end = student[end_index + 1], student[end_index]
        start_index, end_teacher_index = teacher_index[start], teacher_index[end]
        if start_index - end_teacher_index != 2:
            raise ValueError(f"interval {start}->{end} is not two teacher intervals")
        starts.append(start)
        middles.append(teacher[start_index - 1])
        ends.append(end)
    return starts, middles, ends


def _extract(values, timestep, shape):
    result = torch.as_tensor(values, device=timestep.device, dtype=torch.float32)[timestep]
    return result.view(timestep.shape[0], *([1] * (len(shape) - 1)))


def ddim_transition_from_xstart(diffusion, x_t, x_start, t_start, t_end):
    """Apply deterministic DDIM transition from t_start to t_end."""
    alpha_start = _extract(diffusion.alphas_cumprod, t_start, x_t.shape)
    alpha_end = _extract(diffusion.alphas_cumprod, t_end, x_t.shape)
    eps = (x_t * torch.rsqrt(alpha_start) - x_start) / torch.sqrt(
        torch.clamp(1.0 / alpha_start - 1.0, min=1.0e-12)
    )
    return torch.sqrt(alpha_end) * x_start + torch.sqrt(1.0 - alpha_end) * eps


@torch.no_grad()
def teacher_two_step_target(diffusion, teacher, x_t, cond, t_start, t_middle, t_end):
    """Generate x_t_end after two deterministic teacher DDIM transitions."""
    first = diffusion.p_mean_variance(teacher, x_t, cond, t_start, clip_denoised=True)
    x_middle = ddim_transition_from_xstart(
        diffusion, x_t, first["pred_xstart"], t_start, t_middle
    )
    second = diffusion.p_mean_variance(
        teacher, x_middle, cond, t_middle, clip_denoised=True
    )
    return ddim_transition_from_xstart(
        diffusion, x_middle, second["pred_xstart"], t_middle, t_end
    )


def student_transition_loss(diffusion, student, x_t, cond, t_start, t_end, target):
    output = diffusion.p_mean_variance(student, x_t, cond, t_start, clip_denoised=True)
    prediction = ddim_transition_from_xstart(
        diffusion, x_t, output["pred_xstart"], t_start, t_end
    )
    gradient = F.l1_loss(prediction.diff(dim=2), target.diff(dim=2))
    gradient = gradient + F.l1_loss(prediction.diff(dim=3), target.diff(dim=3))
    return F.l1_loss(prediction, target) + 0.1 * gradient, prediction
