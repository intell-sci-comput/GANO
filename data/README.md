# Data Preparation

GANO has three data pipelines. The Helmholtz benchmark is generated locally;
the airfoil and vehicle benchmarks use public datasets that must be downloaded
under their own licenses. All commands in this document assume the current
working directory is the repository root.

Large generated files are ignored by Git. Before preprocessing, allow roughly
60 GB for the Airfoil 9k source file and additional space for processed tensors.
For DrivAerNet++, download only the geometry and pressure modalities needed by
GANO rather than the complete multimodal collection.

## Directory Layout

```text
data/
├── README.md
├── hh/
│   ├── genshape.py
│   ├── gensdf.py
│   ├── genpde.py
│   ├── normalize_pde.py
│   ├── scattering_shapes_256.npz
│   ├── scattering_sdf_dataset_mixed.npz
│   ├── scattering_dataset_scat_fields_k7.npz
│   ├── scattering_dataset_normalized.npz
│   └── normalization_stats.pt
├── airfoil/
│   ├── raw/
│   │   └── airfoil_9k_data.h5
│   ├── preprocess_sdf_airfoil.py
│   ├── preprocess_physics_airfoil.py
│   ├── airfoil_sdf_train.pt
│   └── airfoil_physics_train.pt
└── car/
    ├── raw/
    │   ├── 3DMeshesSTL/              # May contain nested directories
    │   │   └── **/<vehicle-id>.stl
    │   └── PressureVTK/              # May contain nested directories
    │       └── **/<vehicle-id>.vtk
    ├── sdf/
    │   └── **/<vehicle-id>.npz
    ├── pressure/
    │   ├── **/<vehicle-id>.npz
    │   └── dataset_stats.json
    └── split/
        ├── train.txt
        └── test.txt
```

The archive-specific intermediate directory names are not significant. The
preprocessing scripts search recursively and preserve relative subdirectories.
For vehicle data, the STL and VTK basenames are significant: matching files
must share the same `<vehicle-id>`.

## 1. 2D Helmholtz

No external download is needed. The scripts generate random smooth obstacles,
sample their signed distance functions, solve the Helmholtz scattering problem,
and normalize the resulting complex fields.

### Generate the dataset

```bash
python data/hh/genshape.py
python data/hh/gensdf.py
python data/hh/genpde.py
python data/hh/normalize_pde.py
```

The stages must be run in this order:

1. `genshape.py` generates 1,000 Fourier-parameterized obstacles. Each obstacle
   is rasterized on a $256\times256$ grid over $[-1,1]^2$.
2. `gensdf.py` resamples 10,000 query points per obstacle: 40% use near-boundary
   noise $\sigma=0.005$, 40% use $\sigma=0.025$, and 20% are uniform in the
   domain. SDF values are negative inside the obstacle and positive outside.
3. `genpde.py` solves the scattering problem using a finite-difference solver
   with a PML. The defaults are wavenumber $k=7$, 10 incident angles uniformly
   spaced over $[0,2\pi)$, and a $256\times256$ field grid.
4. `normalize_pde.py` splits each complex field into real and imaginary channels
   and standardizes each channel using statistics over all shapes, angles, and
   grid points.

### Generated formats

| File | Keys and shapes | Description |
|---|---|---|
| `scattering_shapes_256.npz` | `masks [1000,256,256]`, `points [1000,10000,2]`, `sdfs [1000,10000]` | Initial shapes and SDF samples |
| `scattering_sdf_dataset_mixed.npz` | same array shapes as above | Multi-scale SDF samples used by StableSDF |
| `scattering_dataset_scat_fields_k7.npz` | `fields [1000,10,256,256]` (`complex64`) plus masks, points, and SDFs | Full scattered fields before normalization |
| `scattering_dataset_normalized.npz` | `fields_norm [1000,10,256,256,2]`, `mean [2]`, `std [2]` | Real/imaginary targets used by GI-Transolver |
| `normalization_stats.pt` | `mean [2]`, `std [2]` | Statistics loaded by the inversion script |

StableSDF uses all 1,000 generated shapes. GI-Transolver treats each
shape/incident-angle pair as one example and creates a seeded 90/10 random
training/validation split. Sparse inverse-scattering sensors are not stored in
these files; `scripts/hh/optimize_hh.py` samples 100 sensor locations at runtime
from an annulus with radii 0.50--0.51.

### Changing the generation size

The main settings can be overridden without editing the source:

```bash
GANO_HH_SHAPE_NUM_SAMPLES=100 \
GANO_HH_SHAPE_N_JOBS=8 \
python data/hh/genshape.py

GANO_HH_SDF_N_JOBS=8 python data/hh/gensdf.py
GANO_HH_PDE_N_JOBS=8 python data/hh/genpde.py
```

Keep the shape, SDF, and PDE resolutions consistent when overriding their
defaults.

## 2. 2D Airfoil

### Source and license

GANO uses the
[Airfoil Computational Fluid Dynamics - 9k shapes, 2 AoAs](https://data.openei.org/submissions/5889)
dataset from NREL's Open Energy Data Initiative (DOI:
[10.25984/2222587](https://doi.org/10.25984/2222587)). It contains 8,996 shapes
simulated at $4^\circ$ and $12^\circ$ angles of attack. The source dataset is
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite
the dataset and follow its attribution requirements when using it.

GANO uses all 8,996 shapes and only the `alpha+04` flow group. The source
simulations use Mach 0.1 and Reynolds number $9\times10^6$.

### Download

The HDF5 file is approximately 52.7 GB. It can be downloaded through the
[OEDI data page](https://data.openei.org/submissions/5889) or anonymously from
the public S3 bucket using the AWS CLI:

```bash
mkdir -p data/airfoil/raw
aws s3 cp \
    s3://nrel-pds-windai/aerodynamic_shapes/2D/9k_airfoils/v1.0.0/airfoil_9k_data.h5 \
    data/airfoil/raw/airfoil_9k_data.h5 \
    --no-sign-request
```

If the file is stored elsewhere, set `GANO_AIRFOIL_H5_PATH` to its absolute
path when running either preprocessing script.

The scripts expect the following HDF5 entries:

```text
shape/landmarks                         # [8996, boundary_points, 2]
alpha+04/flow_field/<sample-id>/x
alpha+04/flow_field/<sample-id>/y
alpha+04/flow_field/<sample-id>/rho
alpha+04/flow_field/<sample-id>/rho_u
alpha+04/flow_field/<sample-id>/rho_v
alpha+04/flow_field/<sample-id>/e
```

### Prepare SDF samples

```bash
python data/airfoil/preprocess_sdf_airfoil.py
```

For every airfoil, the script writes 4,096 query points in the box
$[-0.2,1.2]\times[-0.25,0.25]$. Twenty percent are sampled uniformly; the
remaining points are sampled near the boundary using Gaussian noise with
$\sigma=0.005$ or $0.05$. The output is:

```text
data/airfoil/airfoil_sdf_train.pt
├── coords   float32 [8996, 4096, 2]
├── sdfs     float32 [8996, 4096, 1]
└── indices  int64   [8996]
```

`indices` records the source HDF5 row for every successfully processed shape
and must be preserved so that geometry codes remain aligned with flow fields.

### Prepare flow fields

The flow-field dataset includes learned geometry codes, so it must be created
after StableSDF training:

```bash
python scripts/airfoil/train_stablesdf_airfoil.py
python data/airfoil/preprocess_physics_airfoil.py
```

By default, the second command loads
`checkpoints/airfoil_stablesdf/latents_latest.pth`. A different checkpoint can
be selected with `GANO_AIRFOIL_LATENTS_PATH`.

For each shape, the preprocessing script:

- keeps CFD nodes satisfying $-1\le x\le2$ and $-1\le y\le1$;
- converts conservative variables to $u=(\rho u)/\rho$, $v=(\rho v)/\rho$, and
  $p=(\gamma-1)(e-\frac12\rho(u^2+v^2))$, with $\gamma=1.4$;
- computes one global mean and standard deviation for the $(u,v,p)$ channels;
- stores normalized targets while retaining the original coordinates.

The output format is:

```text
data/airfoil/airfoil_physics_train.pt
├── latents  float32 [8996, 64]
├── coords   list of 8996 tensors [num_nodes_i, 2]
├── targets  list of 8996 tensors [num_nodes_i, 3]  # normalized u, v, p
├── indices  int64 [8996]
└── stats
    ├── mean  float32 [3]
    └── std   float32 [3]
```

`scripts/airfoil/train_gi_transolver.py` creates a deterministic 90/10 random
split over these 8,996 airfoils using seed 42. It samples 4,096 CFD nodes per
airfoil during training.

## 3. 3D Vehicle

### Source and license

GANO uses geometry and surface-pressure data from
[DrivAerNet++](https://github.com/Mohamedelrefaie/DrivAerNet). Download the
**3D Meshes (STL)** and **Pressure (VTK)** subsets from the
[Harvard Dataverse collection](https://dataverse.harvard.edu/dataverse/DrivAerNet).
The complete DrivAerNet++ collection is about 39 TB and is not required for
GANO. Harvard Dataverse may direct large downloads through Globus; follow the
current instructions linked by the official repository.

DrivAerNet++ is distributed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). In particular,
commercial use is not permitted by that dataset license without separate
permission. The dataset license is independent of GANO's code license.

### Arrange the raw files

Extract or link the two downloaded modalities as follows:

```text
data/car/raw/3DMeshesSTL/**/<vehicle-id>.stl
data/car/raw/PressureVTK/**/<vehicle-id>.vtk
```

Nested directories are allowed. A geometry and its pressure field must have
the same filename stem, for example:

```text
data/car/raw/3DMeshesSTL/.../E_S_WW_WM_395.stl
data/car/raw/PressureVTK/.../E_S_WW_WM_395.vtk
```

External storage can be used without copying the dataset into the repository:

```bash
GANO_CAR_MESH_ROOT=/path/to/3DMeshesSTL \
python data/car/preprocess_sdf_car.py

GANO_CAR_PRESSURE_RAW_ROOT=/path/to/PressureVTK \
python data/car/preprocess_pressure_car.py
```

### Prepare vehicle SDF samples

```bash
python data/car/preprocess_sdf_car.py
```

Each STL mesh is cleaned, centered at its axis-aligned bounding-box center, and
scaled so that the bounding-box diagonal is 1.9. The script then generates
100,000 SDF queries per vehicle:

- 45% near-surface samples with Gaussian noise $\sigma=0.005$;
- 45% near-surface samples with Gaussian noise $\sigma=0.025$;
- 10% uniform samples in $[-1,1]^3$.

Near-surface base points are drawn equally from mesh vertices and area-weighted
surface samples. Every output file contains:

```text
data/car/sdf/**/<vehicle-id>.npz
├── coords  float32 [100000, 3]
└── sdf     float32 [100000, 1]
```

The default uses 32 CPU workers. For a smaller machine, for example:

```bash
GANO_CAR_SDF_NUM_WORKERS=8 python data/car/preprocess_sdf_car.py
```

### Prepare surface-pressure samples

```bash
python data/car/preprocess_pressure_car.py
```

The script reads the VTK point array named `p`. If `p` is absent, it falls back
to the first scalar point-data array, so inspect unusual VTK files before
training. Coordinates receive the same bounding-box centering and diagonal-1.9
scaling used for SDF preprocessing. Pressure is standardized using the fixed
statistics from the released experiments:

```text
mean = -93.427311
std  = 120.596359
```

These values are fixed in the script rather than recomputed. If a different
vehicle subset or physical convention is used, recompute the statistics and
update both preprocessing and optimization configurations consistently.

Each pressure output contains:

```text
data/car/pressure/**/<vehicle-id>.npz
├── coords  float32 [num_surface_nodes, 3]
├── data    float32 [num_surface_nodes, 1]  # normalized pressure
└── meta    centroid, scale factor, normalization statistics, source path
```

The preprocessing summary is written to
`data/car/pressure/dataset_stats.json`.

### Vehicle IDs and splits

The GANO split files contain the 8,129 designs successfully processed in the
released experiments:

| Split | Number of vehicles |
|---|---:|
| `data/car/split/train.txt` | 7,316 |
| `data/car/split/test.txt` | 813 |

There is no overlap between the two lists. These are GANO's experiment splits,
not the three-way split published by DrivAerNet++. StableSDF is trained on all
available vehicle SDF files and writes their shuffled order to
`checkpoints/car_training_h800_all/file_list.json`. GI-Transolver uses the file
basenames to align that latent-code order with pressure files and split IDs.
The current training script uses `train.txt` for optimization and `test.txt` as
its validation set.

The split covers 5,340 fastback (`F_*`), 1,386 estateback (`E_*`), and 1,403
notchback (`N_*`) designs.

The same vehicle ID must therefore appear consistently in four places:

```text
STL basename -> SDF NPZ basename -> pressure NPZ basename -> split entry
```

### Optional component meshes for part-wise optimization

`scripts/car/optimize_vehicle.py` additionally expects component-level STL
meshes for the selected vehicle:

```text
data/car/parts/<vehicle-id>/*.stl
```

This directory is not produced by the two preprocessing scripts. Component
filenames beginning with `Mirrors`, `Underbody`, or `wheels` are treated as
fixed rather than optimizable; this prefix matching is case-sensitive. Mirror
meshes supply null-space constraint points, with wheel meshes used as a
fallback. Set `GANO_CAR_OPT_PARTS_ROOT` if these files are stored elsewhere.

## Smoke-test mode

The preprocessing scripts support reduced workloads for pipeline checks:

```bash
GANO_SMOKE_TEST=1 python data/hh/genshape.py
GANO_SMOKE_TEST=1 python data/airfoil/preprocess_sdf_airfoil.py
GANO_SMOKE_TEST=1 python data/car/preprocess_sdf_car.py
```

Smoke-test outputs use the normal filenames and may overwrite full processed
datasets. Run them in a disposable copy or back up existing outputs first.
