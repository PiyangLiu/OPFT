from __future__ import annotations

import argparse
import csv
import os
import pathlib
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joint_generate import generate_joint_samples
from joint_jit_model import JointJiT
from main_joint import corey_tables


ROOT = Path(__file__).resolve().parent
GRID_SHAPE = (5, 128, 128)
INJ_X = [31, 31, 95, 95]
INJ_Y = [31, 95, 31, 95]
MON_X = [13, 35, 88, 100, 25, 64]
MON_Y = [19, 110, 80, 40, 50, 68]


def dataset_root():
    for candidate in (ROOT, ROOT / "dataset", ROOT.parent / "dataset"):
        if (candidate / "data" / "train_data_joint.h5").exists():
            return candidate
    return ROOT.parent / "dataset"


def arguments():
    data_root = dataset_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=data_root / "data" / "train_data_joint.h5"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "model_ckpt_joint" / "model_best.pth"
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=ROOT / "model_ckpt_joint" / "normalization.npz",
    )
    parser.add_argument("--indices", default="0,100,500,999")
    parser.add_argument("--samples-per-condition", type=int, default=10)
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--denoise-time", type=float, default=0.3)
    parser.add_argument(
        "--weights",
        choices=("raw", "ema"),
        default="raw",
        help="checkpoint weights to evaluate; old joint checkpoints must use raw",
    )
    parser.add_argument("--backbone", default="jit_small")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "training_reconstruction_check"
    )
    return parser.parse_args()


def scale(values, minimum, maximum):
    return np.clip(
        2.0 * (values - minimum) / (maximum - minimum + 1e-8) - 1.0,
        -1.0,
        1.0,
    )


def inverse_scale(values, minimum, maximum):
    return 0.5 * (values + 1.0) * (maximum - minimum) + minimum


def load_model(args, device):
    model = JointJiT(
        backbone=args.backbone,
        patch_size=(args.patch_size, args.patch_size),
    ).to(device)
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        try:
            checkpoint = torch.load(
                args.checkpoint, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(args.checkpoint, map_location=device)
    finally:
        pathlib.PosixPath = original_posix_path
    state_dict = checkpoint["model_state_dict"].copy()
    for key in tuple(state_dict):
        if key.endswith(("rope.cos_cached", "rope.sin_cached")):
            state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    if args.weights == "ema" and checkpoint.get("ema_params") is None:
        raise KeyError("checkpoint does not contain ema_params")
    if args.weights == "ema":
        for parameter, ema_parameter in zip(
            model.parameters(), checkpoint["ema_params"]
        ):
            parameter.data.copy_(ema_parameter.to(device))
    print(f"Loaded {args.weights} weights from {args.checkpoint}")
    model.eval()
    return model


def correlation_rows(prediction, truth):
    truth_centered = truth - truth.mean()
    truth_norm = np.sqrt(np.sum(truth_centered**2))
    output = []
    for row in prediction:
        centered = row - row.mean()
        denominator = np.sqrt(np.sum(centered**2)) * truth_norm
        output.append(
            float(np.sum(centered * truth_centered) / denominator)
            if denominator
            else 0.0
        )
    return np.asarray(output)


def add_wells(axis):
    for i, (x, y) in enumerate(zip(INJ_X, INJ_Y)):
        axis.scatter(x, y, marker=".", c="k", s=10)
        axis.text(x - 4, y - 1, f"I{i + 1}", color="k", fontsize=8)
    for i, (x, y) in enumerate(zip(MON_X, MON_Y)):
        axis.scatter(x, y, marker=".", c="k", s=10)
        axis.text(x - 4, y - 1, f"M{i + 1}", color="k", fontsize=8)


def plot_fields(truth, posterior, output_dir, prefix, limits):
    truth = truth.reshape(GRID_SHAPE)
    posterior = posterior.reshape((-1,) + GRID_SHAPE)
    mean = posterior.mean(axis=0)
    std = posterior.std(axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(5):
        fig, axes = plt.subplots(1, 3, figsize=(8, 4), dpi=300)
        fig.suptitle(f"Layer {layer + 1}", y=1.02)
        panels = (
            (truth[layer], "True", limits),
            (mean[layer], "Posterior Mean", limits),
            (std[layer], "Posterior Std", (None, None)),
        )
        for axis, (field, title, color_limits) in zip(axes, panels):
            image = axis.imshow(
                field,
                cmap="jet",
                vmin=color_limits[0],
                vmax=color_limits[1],
            )
            axis.set_title(title)
            plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            add_wells(axis)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_layer_{layer + 1}.png")
        plt.close(fig)


def generate_condition(model, observation, args, device):
    fields, parameters = [], []
    remaining = args.samples_per_condition
    while remaining > 0:
        batch = min(args.generation_batch, remaining)
        condition = torch.from_numpy(np.repeat(observation, batch, axis=0)).to(device)
        field, relperm = generate_joint_samples(
            model,
            condition,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )
        fields.append(field.cpu().numpy())
        parameters.append(relperm.cpu().numpy())
        remaining -= batch
    return np.concatenate(fields), np.concatenate(parameters)


@torch.no_grad()
def teacher_forced_denoise(model, field, relperm, observation, time, device):
    field = torch.from_numpy(field).to(device)
    relperm = torch.from_numpy(relperm).to(device)
    observation = torch.from_numpy(observation).to(device)
    t = torch.full((field.shape[0],), time, device=device)
    t_field = t[:, None, None, None]
    t_vector = t[:, None]
    z_field = t_field * field + (1.0 - t_field) * torch.randn_like(field)
    z_relperm = t_vector * relperm + (1.0 - t_vector) * torch.randn_like(relperm)
    prediction_field, prediction_relperm = model(z_field, z_relperm, t, observation)
    return (
        prediction_field.clamp(-1.0, 1.0).cpu().numpy(),
        prediction_relperm.clamp(-1.0, 1.0).cpu().numpy(),
    )


@torch.no_grad()
def patch_roundtrip_error(model, field, device):
    field = torch.from_numpy(field).to(device)
    fusion = model.spatial_model.fusion_layer
    ph, pw = fusion.patch_size
    sh, sw = fusion.stride
    patches = torch.stack(
        [
            field[:, :, i * sh : i * sh + ph, j * sw : j * sw + pw]
            for i in range(fusion.grid_h)
            for j in range(fusion.grid_w)
        ],
        dim=1,
    )
    weights = (
        torch.sigmoid(fusion.weight_param)
        if fusion.use_learnable_weights
        else fusion.weight_param
    )
    reconstruction = fusion._reconstruct_with_weights(patches, weights)
    return float((reconstruction - field).abs().max().item())


def main():
    args = arguments()
    indices = [int(value) for value in args.indices.split(",")]
    if args.samples_per_condition < 1:
        raise ValueError("--samples-per-condition must be positive")
    if not 0.0 < args.denoise_time < 1.0:
        raise ValueError("--denoise-time must lie strictly between zero and one")
    for path in (args.dataset, args.checkpoint, args.normalization):
        if not path.exists():
            raise FileNotFoundError(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    normalization = dict(np.load(args.normalization))

    with h5py.File(args.dataset, "r") as h5:
        sample_count = h5["x_perm"].shape[0]
        if min(indices) < 0 or max(indices) >= sample_count:
            raise IndexError(f"indices must lie in [0, {sample_count - 1}]")
        if "completed" in h5 and not np.all(h5["completed"][indices]):
            raise ValueError(
                "one or more requested training simulations are incomplete"
            )
        true_perm = h5["x_perm"][indices].astype(np.float32)
        true_poro = h5["x_poro"][indices].astype(np.float32)
        true_relperm = h5["x_relperm"][indices].astype(np.float32)
        observations = h5["y"][indices].astype(np.float32)

    model = load_model(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = args.output_dir / "training_reconstruction.h5"
    metrics_csv = args.output_dir / "metrics.csv"
    records = []

    with h5py.File(output_h5, "w") as output:
        output.attrs["checkpoint"] = str(args.checkpoint)
        output.attrs["checkpoint_weights"] = args.weights
        output.attrs["samples_per_condition"] = args.samples_per_condition
        output.attrs["denoise_time"] = args.denoise_time
        for position, dataset_index in enumerate(indices):
            obs_norm = scale(
                observations[position : position + 1],
                normalization["obs_min"],
                normalization["obs_max"],
            ).astype(np.float32)
            true_field_norm = np.concatenate(
                (
                    scale(
                        np.log1p(true_perm[position]).reshape((1,) + GRID_SHAPE),
                        normalization["logk_min"],
                        normalization["logk_max"],
                    ),
                    scale(
                        true_poro[position].reshape((1,) + GRID_SHAPE),
                        normalization["phi_min"],
                        normalization["phi_max"],
                    ),
                ),
                axis=1,
            ).astype(np.float32)
            true_relperm_norm = scale(
                true_relperm[position : position + 1],
                normalization["relperm_min"],
                normalization["relperm_max"],
            ).astype(np.float32)

            if position == 0:
                roundtrip_error = patch_roundtrip_error(model, true_field_norm, device)
                output.attrs["patch_roundtrip_max_abs_error"] = roundtrip_error
                print(f"Patch round-trip max abs error: {roundtrip_error:.3e}")

            denoised_field_norm, denoised_relperm_norm = teacher_forced_denoise(
                model,
                true_field_norm,
                true_relperm_norm,
                obs_norm,
                args.denoise_time,
                device,
            )
            denoised_perm = np.expm1(
                inverse_scale(
                    denoised_field_norm[:, :5],
                    normalization["logk_min"],
                    normalization["logk_max"],
                )
            ).reshape(1, -1)
            denoised_poro = inverse_scale(
                denoised_field_norm[:, 5:],
                normalization["phi_min"],
                normalization["phi_max"],
            ).reshape(1, -1)
            denoised_relperm = inverse_scale(
                denoised_relperm_norm,
                normalization["relperm_min"],
                normalization["relperm_max"],
            )
            field_norm, relperm_norm = generate_condition(model, obs_norm, args, device)
            post_perm = np.expm1(
                inverse_scale(
                    field_norm[:, :5],
                    normalization["logk_min"],
                    normalization["logk_max"],
                )
            ).reshape(args.samples_per_condition, -1)
            post_poro = inverse_scale(
                field_norm[:, 5:],
                normalization["phi_min"],
                normalization["phi_max"],
            ).reshape(args.samples_per_condition, -1)
            post_relperm = inverse_scale(
                relperm_norm,
                normalization["relperm_min"],
                normalization["relperm_max"],
            )

            true_logk = np.log1p(true_perm[position])
            post_logk = np.log1p(post_perm)
            logk_rmse = np.sqrt(np.mean((post_logk - true_logk[None, :]) ** 2, axis=1))
            poro_rmse = np.sqrt(
                np.mean((post_poro - true_poro[position][None, :]) ** 2, axis=1)
            )
            relperm_rmse = np.sqrt(
                np.mean((post_relperm - true_relperm[position][None, :]) ** 2, axis=1)
            )
            logk_corr = correlation_rows(post_logk, true_logk)
            poro_corr = correlation_rows(post_poro, true_poro[position])

            mean_logk_rmse = float(
                np.sqrt(np.mean((post_logk.mean(axis=0) - true_logk) ** 2))
            )
            mean_poro_rmse = float(
                np.sqrt(np.mean((post_poro.mean(axis=0) - true_poro[position]) ** 2))
            )
            mean_relperm_rmse = float(
                np.sqrt(
                    np.mean((post_relperm.mean(axis=0) - true_relperm[position]) ** 2)
                )
            )
            mean_logk_corr = float(
                correlation_rows(post_logk.mean(axis=0, keepdims=True), true_logk)[0]
            )
            mean_poro_corr = float(
                correlation_rows(
                    post_poro.mean(axis=0, keepdims=True), true_poro[position]
                )[0]
            )

            group = output.create_group(f"sample_{dataset_index}")
            group.create_dataset("true_perm", data=true_perm[position])
            group.create_dataset("true_poro", data=true_poro[position])
            group.create_dataset("true_relperm", data=true_relperm[position])
            group.create_dataset("observation", data=observations[position])
            group.create_dataset(
                "post_perm", data=post_perm.astype(np.float32), compression="gzip"
            )
            group.create_dataset(
                "post_poro", data=post_poro.astype(np.float32), compression="gzip"
            )
            group.create_dataset("post_relperm", data=post_relperm.astype(np.float32))
            group.create_dataset("denoised_perm", data=denoised_perm.astype(np.float32))
            group.create_dataset("denoised_poro", data=denoised_poro.astype(np.float32))
            group.create_dataset(
                "denoised_relperm", data=denoised_relperm.astype(np.float32)
            )
            group.create_dataset("logk_rmse", data=logk_rmse)
            group.create_dataset("poro_rmse", data=poro_rmse)
            group.create_dataset("relperm_rmse", data=relperm_rmse)
            group.create_dataset("logk_corr", data=logk_corr)
            group.create_dataset("poro_corr", data=poro_corr)
            wsf, gsf = zip(*(corey_tables(row) for row in post_relperm))
            group.create_dataset("post_wsf", data=np.asarray(wsf, dtype=np.float32))
            group.create_dataset("post_gsf", data=np.asarray(gsf, dtype=np.float32))

            record = {
                "dataset_index": dataset_index,
                "mean_logk_rmse": mean_logk_rmse,
                "mean_logk_corr": mean_logk_corr,
                "mean_poro_rmse": mean_poro_rmse,
                "mean_poro_corr": mean_poro_corr,
                "mean_relperm_rmse": mean_relperm_rmse,
                "best_logk_rmse": float(logk_rmse.min()),
                "best_poro_rmse": float(poro_rmse.min()),
                "best_relperm_rmse": float(relperm_rmse.min()),
                "denoise_logk_rmse": float(
                    np.sqrt(np.mean((np.log1p(denoised_perm[0]) - true_logk) ** 2))
                ),
                "denoise_logk_corr": float(
                    correlation_rows(np.log1p(denoised_perm), true_logk)[0]
                ),
                "denoise_poro_rmse": float(
                    np.sqrt(np.mean((denoised_poro[0] - true_poro[position]) ** 2))
                ),
                "denoise_poro_corr": float(
                    correlation_rows(denoised_poro, true_poro[position])[0]
                ),
                "denoise_relperm_rmse": float(
                    np.sqrt(
                        np.mean((denoised_relperm[0] - true_relperm[position]) ** 2)
                    )
                ),
            }
            records.append(record)
            print(record)

            sample_dir = args.output_dir / f"sample_{dataset_index}"
            plot_fields(
                true_logk,
                post_logk,
                sample_dir / "permeability",
                "permeability",
                (4.9, 7.1),
            )
            plot_fields(
                true_poro[position],
                post_poro,
                sample_dir / "porosity",
                "porosity",
                (0.10, 0.30),
            )

    with open(metrics_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved reconstruction data to {output_h5}")
    print(f"Saved metrics to {metrics_csv}")


if __name__ == "__main__":
    main()
