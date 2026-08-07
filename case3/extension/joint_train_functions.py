from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

from utils import EMA


class JointFlowMatcher:
    def __init__(
        self,
        p_mean=-0.8,
        p_std=0.8,
        time_sampling="uniform",
        noise_scale=1.0,
        t_eps=5e-2,
        label_drop_prob=0.0,
        lambda_perm=1.0,
        lambda_poro=1.0,
        lambda_relperm=2.0,
        well_loss_weight=3.0,
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.time_sampling = time_sampling
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.label_drop_prob = label_drop_prob
        self.lambda_perm = lambda_perm
        self.lambda_poro = lambda_poro
        self.lambda_relperm = lambda_relperm
        self.well_loss_weight = well_loss_weight
        self.special_value = -2.0
        self.well_coords = torch.tensor(
            [
                [31, 31],
                [31, 95],
                [95, 31],
                [95, 95],
                [13, 19],
                [35, 110],
                [88, 80],
                [100, 40],
                [25, 50],
                [64, 68]
            ],
            dtype=torch.long,
        )

    def sample_t(self, batch_size, device):
        if self.time_sampling == "uniform":
            return torch.rand(batch_size, device=device)
        if self.time_sampling == "logistic":
            return torch.sigmoid(
                torch.randn(batch_size, device=device) * self.p_std + self.p_mean
            )
        raise ValueError(f"unknown time sampling: {self.time_sampling}")

    def drop_observations(self, observations, is_training):
        if not is_training or self.label_drop_prob <= 0:
            return observations
        mask = (
            torch.rand(observations.shape[0], device=observations.device)
            < self.label_drop_prob
        )
        unconditional = torch.full_like(observations, self.special_value)
        return torch.where(mask[:, None], unconditional, observations)

    def compute_well_constraint_loss(self, true_perm, predicted_perm):

        coordinates = self.well_coords.to(true_perm.device)
        rows = coordinates[:, 0]
        columns = coordinates[:, 1]
        true_values = true_perm[:, :, rows, columns]
        predicted_values = predicted_perm[:, :, rows, columns]
        return (true_values - predicted_values).square().mean()

    def compute_loss(self, model, field, relperm, observations, is_training=True):
        batch_size = field.shape[0]
        t = self.sample_t(batch_size, field.device)
        t_field = t[:, None, None, None]
        t_vector = t[:, None]
        field_noise = torch.randn_like(field) * self.noise_scale
        relperm_noise = torch.randn_like(relperm) * self.noise_scale
        z_field = t_field * field + (1.0 - t_field) * field_noise
        z_relperm = t_vector * relperm + (1.0 - t_vector) * relperm_noise
        target_field_velocity = (field - z_field) / (1.0 - t_field).clamp_min(
            self.t_eps
        )
        target_relperm_velocity = (relperm - z_relperm) / (1.0 - t_vector).clamp_min(
            self.t_eps
        )

        condition = self.drop_observations(observations, is_training)
        field_prediction, relperm_prediction = model(z_field, z_relperm, t, condition)
        field_velocity = (field_prediction - z_field) / (1.0 - t_field).clamp_min(
            self.t_eps
        )
        relperm_velocity = (relperm_prediction - z_relperm) / (
            1.0 - t_vector
        ).clamp_min(self.t_eps)

        perm_loss = (
            (target_field_velocity[:, :5] - field_velocity[:, :5]).square().mean()
        )
        poro_loss = (
            (target_field_velocity[:, 5:] - field_velocity[:, 5:]).square().mean()
        )
        relperm_loss = (target_relperm_velocity - relperm_velocity).square().mean()
        flow_total = (
            self.lambda_perm * perm_loss
            + self.lambda_poro * poro_loss
            + self.lambda_relperm * relperm_loss
        )
        if is_training and self.well_loss_weight > 0.0:
            well_loss = self.compute_well_constraint_loss(
                field[:, :5], field_prediction[:, :5]
            )
        else:
            well_loss = field.new_zeros(())
        total = flow_total + self.well_loss_weight * well_loss
        return total, perm_loss, poro_loss, relperm_loss, well_loss


def _epoch(model, loader, matcher, optimizer=None, ema=None, device="cuda"):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(5, dtype=torch.float64)
    progress = tqdm(loader, leave=False)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for field, relperm, observations in progress:
            field = field.to(device, non_blocking=True)
            relperm = relperm.to(device, non_blocking=True)
            observations = observations.to(device, non_blocking=True)
            losses = matcher.compute_loss(model, field, relperm, observations, training)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            totals += torch.tensor([loss.detach().item() for loss in losses])
            progress.set_postfix(loss=f"{losses[0].item():.4f}")
    return (totals / max(len(loader), 1)).tolist()


def train_joint_model(model, train_loader, val_loader, args):
    matcher = JointFlowMatcher(
        p_mean=args.p_mean,
        p_std=args.p_std,
        time_sampling=args.time_sampling,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        label_drop_prob=args.label_drop_prob,
        lambda_perm=args.lambda_perm,
        lambda_poro=args.lambda_poro,
        lambda_relperm=args.lambda_relperm,
        well_loss_weight=args.well_loss_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ema = EMA(model, decay=args.ema_decay)
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        "Joint training: objective=velocity_matching "
        f"time_sampling={args.time_sampling} "
        f"label_drop_prob={args.label_drop_prob} "
        f"ema_decay={args.ema_decay} "
        f"well_loss_weight={args.well_loss_weight} "
        f"well_count={len(matcher.well_coords)}"
    )
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_values = _epoch(
            model,
            train_loader,
            matcher,
            optimizer=optimizer,
            ema=ema,
            device=args.device,
        )
        val_values = _epoch(
            model,
            val_loader,
            matcher,
            optimizer=None,
            ema=None,
            device=args.device,
        )
        history.append(train_values + val_values)
        print(
            f"Epoch {epoch + 1:03d} train={train_values[0]:.6f} "
            f"val={val_values[0]:.6f} K={val_values[1]:.6f} "
            f"PHI={val_values[2]:.6f} relperm={val_values[3]:.6f} "
            f"train_well={train_values[4]:.6f}"
        )
        if val_values[0] < best:
            best = val_values[0]
            serializable_args = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_params": ema.ema_params,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best,
                    "args": serializable_args,
                },
                os.path.join(args.output_dir, "model_best.pth"),
            )
    torch.save({"history": history}, os.path.join(args.output_dir, "loss_history.pth"))
    return history, best
