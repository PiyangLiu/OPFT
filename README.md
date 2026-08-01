# OPFT



Official implementation of the Overlapping Patch Fusion Transformer (OPFT) described in *Monitoring-Informed Generative Inverse Modeling for Geological CO2 Storage Using an Overlapping Patch Fusion Transformer*.

OPFT is a monitoring-conditioned generative inversion framework for geological CO2 storage. It extends direct clean-field prediction to reconstruct high-dimensional geological properties from transient pressure and CO2-saturation observations. The implementation combines overlapping patch extraction, weighted overlapping-patch fusion, convolutional residual refinement, a well-location-weighted loss, and deterministic ODE sampling.

## Release scope

This project provides:

- source code for Cases 1, 2, and 3;
- the Case 1 training dataset as a GitHub Release asset;
- the small Case 1 reference dataset in the repository;
- the joint-inversion extension used for Case 3.

This repository does not contain the manuscript, Case 2 or Case 3 datasets, simulator input/output files, pretrained checkpoints, or generated results.

## Repository layout

```text
.
|-- case1/
|   |-- OPFT.py
|   |-- generate.py
|   |-- main1.py
|   |-- train_functions.py
|   |-- utils.py
|   |-- download_data.py
|   |-- train_data.sha256
|   `-- data/ini_data.h5
|-- case2/
|   `-- source code only
|-- case3/
|   |-- source code only
|   `-- extension/        joint permeability, porosity, and Corey inversion
|-- DATASET.md
|-- DATA_LICENSE
|-- LICENSE
`-- requirements.txt
```

All model implementation files formerly named `jit_model.py` are published as `OPFT.py`.

## Method summary

The network receives a noised geological field, a continuous time coordinate, and a flattened dynamic-observation vector. It directly predicts the clean geological field. The corresponding velocity is derived analytically and integrated with a deterministic ODE solver.

The main components are:

1. an MLP observation encoder for transient pressure and saturation responses;
2. adaptive normalization modulation of the Transformer backbone;
3. overlapping spatial patches to retain information shared by adjacent tokens;
4. weighted patch aggregation and convolutional residual refinement to suppress patch-boundary artifacts;
5. a soft well-location reconstruction penalty;
6. Euler, Heun, or RK4 generation routines.

In the manuscript comparison, OPFT and DDIM use an equal budget of 30 network evaluations. For that setting, use 30 Euler steps.

## Environment

- Python 3.9 or later
- PyTorch 2.1 or later
- A CUDA-capable GPU is required by the current training loops

Install PyTorch for the CUDA version available on your system, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Case 1 dataset

Case 1 is a 60 x 60 two-dimensional heterogeneous CO2-storage benchmark. Each paired sample contains one permeability field and a 500-dimensional monitoring vector formed from 10 response series at 50 reporting times.

The 39.4 MB training file is distributed as the `train_data.h5` asset on the [latest GitHub Release](https://github.com/PiyangLiu/OPFT/releases/latest). It is intentionally excluded from Git history. Download and verify it from the repository root with:

```bash
python case1/download_data.py
```

The script places the verified file at `case1/train_data.h5`. Its SHA-256 checksum is recorded in `case1/train_data.sha256`.

`case1/train_data.h5` contains:

- `x`: shape `(1200, 3600)`, permeability fields flattened in row-major order;
- `y`: shape `(1200, 500)`, flattened dynamic-observation vectors.

`case1/data/ini_data.h5` contains:

- `true_para`: shape `(3600, 1)`, the reference permeability field;
- `dobstrue`: shape `(500, 1)`, the corresponding reference observation vector.

The code uses samples 0-999 for training and samples 1000-1199 for evaluation. Permeability is transformed with `ln(k + 1)` and both fields and observations are min-max normalized to `[-1, 1]`.

See [DATASET.md](DATASET.md) for the complete data and licensing notes.

## Running Case 1

Run commands from the `case1` directory because the scripts use local imports and local output paths.

```bash
python case1/download_data.py
cd case1
python main1.py --num_steps 30 --integrate_method euler
```

The current Case 1 entry point has a `mode` variable near the end of `main1.py`:

- `mode = 0` trains the model;
- `mode = 1` loads `model_ckpt/model_best.pth` and generates conditional samples.

Pretrained checkpoints are not included. Training writes checkpoints to `case1/model_ckpt/`. Generation writes conditional realizations to `post_data.h5` and may also write diagnostic figures and HDF5 comparison files.

## Cases 2 and 3

The Case 2 and Case 3 folders contain implementation code only. Their `main1.py` scripts expect `train_data.h5` in the case directory and `data/ini_data.h5` beneath it. These datasets are intentionally not part of this release.

The `case3/extension` entry point exposes explicit train and generate modes for joint reconstruction:

```bash
cd case3/extension
python main_joint.py --mode train --dataset PATH_TO_TRAIN_DATA --reference PATH_TO_REFERENCE_DATA
python main_joint.py --mode generate --dataset PATH_TO_TRAIN_DATA --reference PATH_TO_REFERENCE_DATA --checkpoint PATH_TO_CHECKPOINT
```

## Reproducibility notes

- The released Case 1 data comprise 1,000 training pairs and 200 held-out pairs.
- The manuscript's matched OPFT/DDIM comparison uses 30 network evaluations per target.
- Model checkpoints and proprietary forward-simulator resources are excluded.
- The scripts preserve the research-code training and output conventions used for the study; inspect resource requirements before launching a full run.

## Citation

If this code or dataset supports your work, cite the associated manuscript:

```text
P. Liu, J. Wang, J. Zhang, L. Zhang, Z. Tao, K. Zhang, Z. Zhang, and J. Yao,
"Monitoring-Informed Generative Inverse Modeling for Geological CO2 Storage
Using an Overlapping Patch Fusion Transformer."
```

Bibliographic metadata can be updated after the article receives its final journal citation.

## Licenses

Source code is licensed under the [Apache License 2.0](LICENSE). The Case 1 HDF5 files are licensed separately under [CC BY 4.0](DATA_LICENSE).
