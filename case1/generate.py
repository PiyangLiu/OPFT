import torch
import torch.nn as nn


class ImprovedSampler:
    def __init__(
        self,
        model,
        cfg_scale=2.0,
        cfg_interval=(0.0, 1.0),
        t_eps=5e-2,
        special_value=-2.0,
    ):
        self.model = model
        self.cfg_scale = cfg_scale
        self.cfg_interval = cfg_interval
        self.t_eps = t_eps
        self.special_value = special_value

    @torch.no_grad()
    def forward_sample(self, z, t, labels):

        if t.dim() == 0:
            t_batch = torch.full((z.shape[0],), t.item(), device=z.device)
        else:
            t_batch = t

        x_cond = self.model(z, t_batch, labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        labels_uncond = torch.full_like(labels, self.special_value)
        x_uncond = self.model(z, t_batch, labels_uncond)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def euler_step(self, z, t, t_next, labels):

        v_pred = self.forward_sample(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def heun_step(self, z, t, t_next, labels):

        v_pred_t = self.forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self.forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def rk4_step(self, z, t, t_next, labels):

        dt = t_next - t

        k1 = self.forward_sample(z, t, labels)

        z2 = z + 0.5 * dt * k1
        t2 = t + 0.5 * dt
        k2 = self.forward_sample(z2, t2, labels)

        z3 = z + 0.5 * dt * k2
        k3 = self.forward_sample(z3, t2, labels)

        z4 = z + dt * k3
        t4 = t + dt
        k4 = self.forward_sample(z4, t4, labels)

        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def improved_generate_samples(model, labels, args, config, method="euler"):

    model.eval()
    device = config.device
    bsz = labels.shape[0]

    sampler = ImprovedSampler(
        model=model,
        cfg_scale=getattr(args, "cfg_scale", 2.0),
        cfg_interval=(
            getattr(args, "cfg_interval_min", 0.0),
            getattr(args, "cfg_interval_max", 1.0),
        ),
        t_eps=getattr(config, "t_eps", 5e-2),
        special_value=-2.0,
    )

    z = args.noise_scale * torch.randn(
        bsz, config.channels, config.image_size[0], config.image_size[1], device=device
    )

    timesteps = torch.linspace(0.0, 1.0, args.num_steps + 1, device=device)

    if method == "heun":
        stepper = sampler.heun_step
    elif method == "rk4":
        stepper = sampler.rk4_step
    else:
        stepper = sampler.euler_step

    for i in range(args.num_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        z = stepper(z, t, t_next, labels)

    return z


def generate_samples(model, labels, args, config, method="euler"):
    return improved_generate_samples(model, labels, args, config, method)


def rk4_integrate(model, z0, labels, timesteps, device, guidance_scale=2.0, t_eps=5e-2):

    sampler = ImprovedSampler(
        model, guidance_scale, (0.0, 1.0), t_eps, special_value=-2.0
    )
    z = z0.to(device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        z = sampler.rk4_step(z, t, t_next, labels)

    return z


def euler_integrate(
    model, z0, labels, timesteps, device, guidance_scale=2.0, t_eps=5e-2
):

    sampler = ImprovedSampler(
        model, guidance_scale, (0.0, 1.0), t_eps, special_value=-2.0
    )
    z = z0.to(device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        z = sampler.euler_step(z, t, t_next, labels)

    return z
