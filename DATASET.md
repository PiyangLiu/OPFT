# Case 1 Dataset

## Scope

Only the Case 1 dataset is publicly distributed. It supports the 60 x 60 two-dimensional heterogeneous CO2-storage benchmark described in the associated OPFT manuscript. The training file is a GitHub Release asset, while the small reference file remains in the repository. Case 2 and Case 3 data are not included.

## Download and integrity

Download the release asset from the repository root:

```bash
python case1/download_data.py
```

The downloader retrieves `train_data.h5` from the [latest GitHub Release](https://github.com/PiyangLiu/OPFT/releases/latest), saves it under `case1/`, and verifies its SHA-256 digest before use:

```text
fff52d66f0adef89a4f27fb1b9e1e34ec1c9b3c18ec06900b5e306df6851c3b9  train_data.h5
```

## Files and schema

### `case1/train_data.h5` (GitHub Release asset)

| Key | Shape | Type | Description |
| --- | --- | --- | --- |
| `x` | `(1200, 3600)` | `float64` | Flattened permeability fields |
| `y` | `(1200, 500)` | `float64` | Flattened monitoring-response vectors |

### `case1/data/ini_data.h5` (tracked in the repository)

| Key | Shape | Type | Description |
| --- | --- | --- | --- |
| `true_para` | `(3600, 1)` | `float64` | Reference permeability field |
| `dobstrue` | `(500, 1)` | `float64` | Reference monitoring-response vector |

All arrays contain finite values. The model code reshapes each permeability vector to `(1, 60, 60)`. Each observation vector contains 10 response series sampled at 50 fixed reporting times and stored in feature-major flattened order.

## Split and preprocessing

- Training indices: `[0, 1000)`
- Evaluation indices: `[1000, 1200)`
- Permeability transform: `ln(k + 1)`
- Field normalization: per-cell min-max normalization to `[-1, 1]`
- Observation normalization: per-feature min-max normalization to `[-1, 1]`

Normalization statistics are computed from the 1,200 paired samples by the released entry point. Users conducting new benchmark comparisons should report any change to this preprocessing convention.

## Provenance

The manuscript describes Case 1 as a synthetic dataset generated from 60 x 60 heterogeneous permeability realizations and corresponding multiphase-flow responses. The full forward-simulation setup is described in the associated article. Proprietary simulator files are not redistributed here.

## License and attribution

The two HDF5 files listed above are licensed under Creative Commons Attribution 4.0 International. See [DATA_LICENSE](DATA_LICENSE). A suitable attribution is:

```text
OPFT Authors (2026), Case 1 dataset for "Monitoring-Informed Generative
Inverse Modeling for Geological CO2 Storage Using an Overlapping Patch
Fusion Transformer," CC BY 4.0.
```
