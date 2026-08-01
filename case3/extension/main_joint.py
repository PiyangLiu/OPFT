from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from joint_generate import generate_joint_samples
from joint_jit_model import JointJiT
from joint_train_functions import train_joint_model


ROOT = Path(__file__).resolve().parent
CASE_ROOT = ROOT.parent
DEFAULT_DATASET = CASE_ROOT / "dataset" / "data" / "train_data_joint.h5"
DEFAULT_REFERENCE = CASE_ROOT / "dataset" / "data" / "ini_data_joint.h5"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "generate"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", default=str(ROOT / "model_ckpt_joint"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--post-output",
        type=Path,
        default=ROOT / "post_data_joint.h5",
        help="posterior HDF5 path used in generate mode",
    )
    parser.add_argument("--backbone", default="jit_small")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--t-eps", type=float, default=5e-2)
    parser.add_argument("--p-mean", type=float, default=-0.8)
    parser.add_argument("--p-std", type=float, default=0.8)
    parser.add_argument(
        "--time-sampling", choices=("uniform", "logistic"), default="uniform"
    )
    parser.add_argument("--label-drop-prob", type=float, default=0.0)
    parser.add_argument("--lambda-perm", type=float, default=1.0)
    parser.add_argument("--lambda-poro", type=float, default=1.0)
    parser.add_argument("--lambda-relperm", type=float, default=2.0)
    parser.add_argument("--well-loss-weight", type=float, default=3.0)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--weights",
        choices=("raw", "ema"),
        default="raw",
        help="checkpoint weights used for generation; old joint checkpoints must use raw",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint is None:
        args.checkpoint = Path(args.output_dir) / "model_best.pth"
    return args


def _scale(values, minimum, maximum):
    return np.clip(
        2.0 * (values - minimum) / (maximum - minimum + 1e-8) - 1.0, -1.0, 1.0
    )


def _inverse_scale(values, minimum, maximum):
    return 0.5 * (values + 1.0) * (maximum - minimum) + minimum


def load_training_arrays(path, train_count):
    with h5py.File(path, "r") as h5:
        completed = h5["completed"][:]
        valid = np.flatnonzero(completed)
        if valid.size < 2:
            raise ValueError("joint dataset has fewer than two completed simulations")
        permeability = h5["x_perm"][valid].astype(np.float32)
        porosity = h5["x_poro"][valid].astype(np.float32)
        relperm = h5["x_relperm"][valid].astype(np.float32)
        observations = h5["y"][valid].astype(np.float32)
        relperm_bounds = np.asarray(h5.attrs["relperm_bounds"], dtype=np.float32)
        petrophysics = json.loads(h5.attrs["petrophysics_config"])

    split = min(train_count, valid.size - 1)
    log_permeability = np.log1p(permeability)
    logk_min = np.quantile(log_permeability[:split], 0.001).astype(np.float32)
    logk_max = np.quantile(log_permeability[:split], 0.999).astype(np.float32)
    phi_min = np.float32(petrophysics["phi_min"])
    phi_max = np.float32(petrophysics["phi_max"])
    obs_min = observations[:split].min(axis=0)
    obs_max = observations[:split].max(axis=0)
    normalization = {
        "logk_min": logk_min,
        "logk_max": logk_max,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "relperm_min": relperm_bounds[:, 0],
        "relperm_max": relperm_bounds[:, 1],
        "obs_min": obs_min,
        "obs_max": obs_max,
    }

    perm_norm = _scale(log_permeability, logk_min, logk_max).reshape(-1, 5, 128, 128)
    poro_norm = _scale(porosity, phi_min, phi_max).reshape(-1, 5, 128, 128)
    field = np.concatenate((perm_norm, poro_norm), axis=1).astype(np.float32)
    theta = _scale(relperm, relperm_bounds[:, 0], relperm_bounds[:, 1]).astype(
        np.float32
    )
    obs = _scale(observations, obs_min, obs_max).astype(np.float32)
    return field, theta, obs, split, normalization


def make_model(args):
    return JointJiT(
        backbone=args.backbone,
        patch_size=(args.patch_size, args.patch_size),
    ).to(args.device)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    field, theta, obs, split, normalization = load_training_arrays(
        args.dataset, args.train_count
    )
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(Path(args.output_dir) / "normalization.npz", **normalization)
    train_dataset = TensorDataset(
        torch.from_numpy(field[:split]),
        torch.from_numpy(theta[:split]),
        torch.from_numpy(obs[:split]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(field[split:]),
        torch.from_numpy(theta[split:]),
        torch.from_numpy(obs[split:]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    model = make_model(args)
    _, best = train_joint_model(model, train_loader, val_loader, args)
    print(f"Best joint validation loss: {best:.6f}")


def corey_tables(theta, points=20):
    swc, sgc, krw_end, krg_end, nw, ng = theta
    sw = np.linspace(swc, 1.0, points)
    krw = krw_end * np.clip((sw - swc) / (1.0 - swc), 0.0, 1.0) ** nw
    sg_max = 1.0 - swc
    sg = np.linspace(0.0, sg_max, points)
    krg = krg_end * np.clip((sg - sgc) / max(sg_max - sgc, 1e-8), 0.0, 1.0) ** ng
    reference_pc = np.array(
        [
            0.0689,
            0.0746898,
            0.0813404,
            0.0890425,
            0.0980449,
            0.108677,
            0.121386,
            0.136786,
            0.155751,
            0.179555,
            0.210206,
            0.250606,
            0.305877,
            0.385006,
            0.505416,
            0.705165,
            1.08334,
            1.98422,
            5.58326,
            5.58326,
        ]
    )
    pc = np.interp(
        sg / max(sg_max, 1e-8), np.linspace(0.0, 1.0, reference_pc.size), reference_pc
    )
    return np.column_stack((sw, krw)), np.column_stack((sg, krg, pc))


def load_weights(model, checkpoint_path, device, weights="raw"):
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:

            checkpoint = torch.load(checkpoint_path, map_location=device)
    finally:
        pathlib.PosixPath = original_posix_path
    state_dict = checkpoint["model_state_dict"].copy()
    for key in tuple(state_dict):
        if key.endswith(("rope.cos_cached", "rope.sin_cached")):
            state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    ema = checkpoint.get("ema_params")
    if weights == "ema" and ema is None:
        raise KeyError("checkpoint does not contain ema_params")
    if weights == "ema":
        for parameter, ema_parameter in zip(model.parameters(), ema):
            parameter.data.copy_(ema_parameter.to(device))
    print(f"Loaded {weights} weights from {checkpoint_path}")


def generate(args):
    torch.manual_seed(args.seed)
    normalization_path = Path(args.output_dir) / "normalization.npz"
    if not normalization_path.exists():
        raise FileNotFoundError(normalization_path)
    norm = dict(np.load(normalization_path))
    with h5py.File(args.reference, "r") as h5:
        observation = h5["dobstrue"][:].reshape(1, -1).astype(np.float32)
    observation = _scale(observation, norm["obs_min"], norm["obs_max"])
    model = make_model(args)
    load_weights(model, args.checkpoint, args.device, args.weights)

    fields, thetas = [], []
    for start in range(0, args.num_samples, args.generation_batch):
        count = min(args.generation_batch, args.num_samples - start)
        condition = torch.from_numpy(np.repeat(observation, count, axis=0)).to(
            args.device
        )
        field, theta = generate_joint_samples(
            model, condition, args.num_steps, args.cfg_scale, args.t_eps
        )
        fields.append(field.cpu().numpy())
        thetas.append(theta.cpu().numpy())
    field = np.concatenate(fields)
    theta_norm = np.concatenate(thetas)
    permeability = np.expm1(
        _inverse_scale(field[:, :5], norm["logk_min"], norm["logk_max"])
    ).reshape(args.num_samples, -1)
    porosity = _inverse_scale(field[:, 5:], norm["phi_min"], norm["phi_max"]).reshape(
        args.num_samples, -1
    )
    theta = _inverse_scale(theta_norm, norm["relperm_min"], norm["relperm_max"])
    wsf, gsf = zip(*(corey_tables(row) for row in theta))

    output_path = args.post_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["checkpoint_weights"] = args.weights
        h5.create_dataset(
            "post_perm", data=permeability.astype(np.float32), compression="gzip"
        )
        h5.create_dataset(
            "post_poro", data=porosity.astype(np.float32), compression="gzip"
        )
        h5.create_dataset("post_relperm", data=theta.astype(np.float32))
        h5.create_dataset("post_wsf", data=np.asarray(wsf, dtype=np.float32))
        h5.create_dataset("post_gsf", data=np.asarray(gsf, dtype=np.float32))
    print(f"Saved {args.num_samples} joint posterior samples to {output_path}")


def main():
    args = arguments()
    if args.mode == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
