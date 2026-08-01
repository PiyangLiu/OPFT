from __future__ import annotations

import torch


@torch.no_grad()
def _velocity(model, field, relperm, t, observation, cfg_scale, t_eps):
    batch = field.shape[0]
    t_batch = torch.full((batch,), float(t), device=field.device)
    field_cond, theta_cond = model(field, relperm, t_batch, observation)
    denominator = max(1.0 - float(t), t_eps)
    if cfg_scale == 1.0:
        return (
            (field_cond - field) / denominator,
            (theta_cond - relperm) / denominator,
        )
    unconditional = torch.full_like(observation, -2.0)
    field_uncond, theta_uncond = model(field, relperm, t_batch, unconditional)
    field_velocity = (
        (field_uncond - field) + cfg_scale * (field_cond - field_uncond)
    ) / denominator
    theta_velocity = (
        (theta_uncond - relperm) + cfg_scale * (theta_cond - theta_uncond)
    ) / denominator
    return field_velocity, theta_velocity


@torch.no_grad()
def generate_joint_samples(model, observation, num_steps=30, cfg_scale=1.0, t_eps=5e-2):
    model.eval()
    batch = observation.shape[0]
    device = observation.device
    field = torch.randn(batch, 10, 128, 128, device=device)
    relperm = torch.randn(batch, 6, device=device)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    for step in range(num_steps):
        t = timesteps[step]
        t_next = timesteps[step + 1]
        dt = t_next - t
        field_velocity, theta_velocity = _velocity(
            model, field, relperm, t, observation, cfg_scale, t_eps
        )
        field = field + dt * field_velocity
        relperm = relperm + dt * theta_velocity
    return field.clamp(-1.0, 1.0), relperm.clamp(-1.0, 1.0)
