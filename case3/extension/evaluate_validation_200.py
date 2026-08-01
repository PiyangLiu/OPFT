from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_generate import generate_joint_samples
from joint_jit_model import JointJiT
from main_joint import load_weights


ROOT = Path(__file__).resolve().parent
CASE_ROOT = ROOT.parent.parent
GRID_SHAPE = (5, 128, 128)
FULL_GRID_SHAPE = (7, 128, 128)
FIELD_SIZE = int(np.prod(GRID_SHAPE))
RELPERM_NAMES = ("swc", "sgc", "krw_end", "krg_end", "nw", "ng")


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def arguments():
    checkpoint = first_existing(
        ROOT / "model_ckpt_joint" / "model_best.pth",
        ROOT.parent / "model_ckpt_joint" / "model_best.pth",
    )
    normalization = first_existing(
        checkpoint.parent / "normalization.npz",
        ROOT / "model_ckpt_joint" / "normalization.npz",
        ROOT.parent / "model_ckpt_joint" / "normalization.npz",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=CASE_ROOT / "dataset" / "data" / "train_data_joint.h5",
    )
    parser.add_argument(
        "--active-grid",
        type=Path,
        default=CASE_ROOT / "dataset" / "eclmodel" / "ACTIVE.INC",
    )
    parser.add_argument("--checkpoint", type=Path, default=checkpoint)
    parser.add_argument("--normalization", type=Path, default=normalization)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation_200_evaluation.h5",
    )
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--t-eps", type=float, default=5e-2)
    parser.add_argument("--backbone", default="jit_small")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def scale(values, minimum, maximum):
    return np.clip(
        2.0 * (values - minimum) / (maximum - minimum + 1e-8) - 1.0,
        -1.0,
        1.0,
    )


def inverse_scale(values, minimum, maximum):
    return 0.5 * (values + 1.0) * (maximum - minimum) + minimum


def load_active_mask(path):
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        lines.append(line.split("--", 1)[0])
    match = re.search(r"\bACTNUM\b(.*?)/", "\n".join(lines), re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError(f"ACTNUM block not found in {path}")
    values = []
    for token in match.group(1).split():
        if "*" in token:
            count, value = token.split("*", 1)
            values.extend([int(value)] * int(count))
        else:
            values.append(int(token))
    expected = int(np.prod(FULL_GRID_SHAPE))
    if len(values) != expected:
        raise ValueError(f"ACTNUM has {len(values)} values; expected {expected}")
    return np.asarray(values, dtype=bool).reshape(FULL_GRID_SHAPE)[2:].reshape(-1)


def validation_indices(dataset, train_count, validation_count):
    with h5py.File(dataset, "r") as h5:
        completed = h5["completed"][:]
        valid = np.flatnonzero(completed)
    split = min(train_count, valid.size - 1)
    available = valid.size - split
    if available < validation_count:
        raise ValueError(
            f"only {available} validation samples are available after "
            f"train-count={split}; requested {validation_count}"
        )
    return valid[split : split + validation_count], split, valid.size


def make_output(path, count, dataset_indices, args, active_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(path, "w")
    h5.attrs["dataset"] = str(args.dataset.resolve())
    h5.attrs["checkpoint"] = str(args.checkpoint.resolve())
    h5.attrs["checkpoint_weights"] = args.weights
    h5.attrs["seed"] = args.seed
    h5.attrs["num_steps"] = args.num_steps
    h5.attrs["cfg_scale"] = args.cfg_scale
    h5.attrs["t_eps"] = args.t_eps
    h5.attrs["train_count"] = args.train_count
    h5.attrs["validation_count"] = count
    h5.attrs["realizations_per_condition"] = 1
    h5.attrs["field_rmse_primary_mask"] = "active ACTNUM cells"
    h5.attrs["active_storage_cell_count"] = active_count
    h5.create_dataset("dataset_index", data=dataset_indices)
    field_chunks = (1, FIELD_SIZE)
    h5.create_dataset(
        "pred_perm",
        (count, FIELD_SIZE),
        dtype="f4",
        chunks=field_chunks,
        compression="gzip",
        compression_opts=4,
    )
    h5.create_dataset(
        "pred_poro",
        (count, FIELD_SIZE),
        dtype="f4",
        chunks=field_chunks,
        compression="gzip",
        compression_opts=4,
    )
    h5.create_dataset("pred_relperm", (count, 6), dtype="f4")
    h5.create_dataset("true_relperm", (count, 6), dtype="f4")
    for name in (
        "perm_rmse_md_active",
        "perm_rmse_md_full",
        "log1p_perm_rmse_active",
        "log1p_perm_rmse_full",
        "poro_rmse_active",
        "poro_rmse_full",
        "relperm_rmse",
        "relperm_normalized_rmse",
    ):
        h5.create_dataset(name, (count,), dtype="f8")
    h5.create_dataset("relperm_error", (count, 6), dtype="f8")
    return h5


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "global_rmse": float(np.sqrt(np.mean(values**2))),
        "mean_sample_rmse": float(np.mean(values)),
        "std_sample_rmse": float(np.std(values)),
        "median_sample_rmse": float(np.median(values)),
        "min_sample_rmse": float(np.min(values)),
        "max_sample_rmse": float(np.max(values)),
    }


def write_reports(output_path, output, parameter_names):
    metric_names = (
        "perm_rmse_md_active",
        "perm_rmse_md_full",
        "log1p_perm_rmse_active",
        "log1p_perm_rmse_full",
        "poro_rmse_active",
        "poro_rmse_full",
        "relperm_rmse",
        "relperm_normalized_rmse",
    )
    metrics = {name: output[name][:] for name in metric_names}
    relperm_error = output["relperm_error"][:]
    summary = {name: describe(values) for name, values in metrics.items()}
    summary["relperm_parameter_rmse"] = {
        name: float(np.sqrt(np.mean(relperm_error[:, i] ** 2)))
        for i, name in enumerate(parameter_names)
    }
    summary["sample_count"] = int(output["dataset_index"].shape[0])
    summary["checkpoint_weights"] = str(output.attrs["checkpoint_weights"])

    csv_path = output_path.with_name(output_path.stem + "_metrics.csv")
    fieldnames = ["validation_position", "dataset_index", *metric_names]
    fieldnames.extend(f"{name}_error" for name in parameter_names)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        indices = output["dataset_index"][:]
        for i, dataset_index in enumerate(indices):
            row = {"validation_position": i, "dataset_index": int(dataset_index)}
            row.update({name: float(metrics[name][i]) for name in metric_names})
            row.update(
                {
                    f"{name}_error": float(relperm_error[i, j])
                    for j, name in enumerate(parameter_names)
                }
            )
            writer.writerow(row)

    json_path = output_path.with_name(output_path.stem + "_summary.json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path, summary


def main():
    args = arguments()
    for path in (args.dataset, args.active_grid, args.checkpoint, args.normalization):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.generation_batch < 1 or args.validation_count < 1:
        raise ValueError("generation-batch and validation-count must be positive")

    dataset_indices, split, valid_count = validation_indices(
        args.dataset, args.train_count, args.validation_count
    )
    active = load_active_mask(args.active_grid)
    if active.size != FIELD_SIZE or not np.any(active):
        raise ValueError("invalid storage-layer ACTNUM mask")
    print(
        f"Completed={valid_count}, training={split}, validation={len(dataset_indices)}, "
        f"dataset indices={dataset_indices[0]}..{dataset_indices[-1]}"
    )
    print(f"Active storage cells={int(active.sum())}/{active.size}")
    print(f"Device={args.device}, checkpoint weights={args.weights}")
    if args.dry_run:
        print("Dry run completed; model generation was not started")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    normalization = dict(np.load(args.normalization))
    model = JointJiT(
        backbone=args.backbone,
        patch_size=(args.patch_size, args.patch_size),
    ).to(args.device)
    load_weights(model, args.checkpoint, args.device, args.weights)
    model.eval()

    with h5py.File(args.dataset, "r") as source, make_output(
        args.output,
        len(dataset_indices),
        dataset_indices,
        args,
        int(active.sum()),
    ) as output:
        raw_names = source.attrs.get("relperm_parameter_names", RELPERM_NAMES)
        parameter_names = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_names
        )
        output.attrs["relperm_parameter_names"] = np.asarray(
            parameter_names, dtype="S16"
        )

        for start in range(0, len(dataset_indices), args.generation_batch):
            stop = min(start + args.generation_batch, len(dataset_indices))
            indices = dataset_indices[start:stop]
            true_perm = source["x_perm"][indices].astype(np.float32)
            true_poro = source["x_poro"][indices].astype(np.float32)
            true_relperm = source["x_relperm"][indices].astype(np.float32)
            observations = source["y"][indices].astype(np.float32)
            obs_norm = scale(
                observations, normalization["obs_min"], normalization["obs_max"]
            ).astype(np.float32)

            condition = torch.from_numpy(obs_norm).to(args.device)
            field_norm, relperm_norm = generate_joint_samples(
                model,
                condition,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                t_eps=args.t_eps,
            )
            field_norm = field_norm.cpu().numpy()
            relperm_norm = relperm_norm.cpu().numpy()
            pred_perm = np.expm1(
                inverse_scale(
                    field_norm[:, :5],
                    normalization["logk_min"],
                    normalization["logk_max"],
                )
            ).reshape(stop - start, FIELD_SIZE)
            pred_poro = inverse_scale(
                field_norm[:, 5:],
                normalization["phi_min"],
                normalization["phi_max"],
            ).reshape(stop - start, FIELD_SIZE)
            pred_relperm = inverse_scale(
                relperm_norm,
                normalization["relperm_min"],
                normalization["relperm_max"],
            )

            perm_error = pred_perm - true_perm
            logk_error = np.log1p(pred_perm) - np.log1p(true_perm)
            poro_error = pred_poro - true_poro
            relperm_error = pred_relperm - true_relperm
            relperm_norm_error = scale(
                pred_relperm,
                normalization["relperm_min"],
                normalization["relperm_max"],
            ) - scale(
                true_relperm,
                normalization["relperm_min"],
                normalization["relperm_max"],
            )

            target = slice(start, stop)
            output["pred_perm"][target] = pred_perm.astype(np.float32)
            output["pred_poro"][target] = pred_poro.astype(np.float32)
            output["pred_relperm"][target] = pred_relperm.astype(np.float32)
            output["true_relperm"][target] = true_relperm
            output["perm_rmse_md_active"][target] = np.sqrt(
                np.mean(perm_error[:, active] ** 2, axis=1)
            )
            output["perm_rmse_md_full"][target] = np.sqrt(
                np.mean(perm_error**2, axis=1)
            )
            output["log1p_perm_rmse_active"][target] = np.sqrt(
                np.mean(logk_error[:, active] ** 2, axis=1)
            )
            output["log1p_perm_rmse_full"][target] = np.sqrt(
                np.mean(logk_error**2, axis=1)
            )
            output["poro_rmse_active"][target] = np.sqrt(
                np.mean(poro_error[:, active] ** 2, axis=1)
            )
            output["poro_rmse_full"][target] = np.sqrt(np.mean(poro_error**2, axis=1))
            output["relperm_rmse"][target] = np.sqrt(np.mean(relperm_error**2, axis=1))
            output["relperm_normalized_rmse"][target] = np.sqrt(
                np.mean(relperm_norm_error**2, axis=1)
            )
            output["relperm_error"][target] = relperm_error
            output.flush()
            print(
                f"Generated validation samples {start + 1}-{stop}/{len(dataset_indices)}"
            )

        csv_path, json_path, summary = write_reports(
            args.output, output, parameter_names
        )

    print("Overall validation metrics:")
    for name in (
        "perm_rmse_md_active",
        "log1p_perm_rmse_active",
        "poro_rmse_active",
        "relperm_rmse",
        "relperm_normalized_rmse",
    ):
        values = summary[name]
        print(
            f"  {name}: global={values['global_rmse']:.10g}, "
            f"mean={values['mean_sample_rmse']:.10g}, "
            f"std={values['std_sample_rmse']:.10g}"
        )
    print("Relative-permeability parameter RMSE:")
    for name, value in summary["relperm_parameter_rmse"].items():
        print(f"  {name}: {value:.10g}")
    print(f"Saved predictions and metrics to {args.output}")
    print(f"Saved per-sample metrics to {csv_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
