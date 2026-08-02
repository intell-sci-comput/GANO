#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

usage() {
    echo "Usage: bash scripts/reproduce/vehicle.sh [prepare|train-sdf|train-physics|optimize|all]"
}

prepare() {
    python data/car/preprocess_sdf_car.py
    python data/car/preprocess_pressure_car.py
}

train_sdf() {
    : "${GANO_CAR_STABLESDF_RESUME:=0}"
    : "${GANO_CAR_STABLESDF_START_EPOCH:=0}"
    export GANO_CAR_STABLESDF_RESUME GANO_CAR_STABLESDF_START_EPOCH
    python scripts/car/train_stablesdf_car.py
}

train_physics() {
    python scripts/car/train_gi_transolver_car.py
}

optimize() {
    python scripts/car/optimize_vehicle.py
}

stage="${1:-all}"
case "${stage}" in
    prepare)
        prepare
        ;;
    train-sdf)
        train_sdf
        ;;
    train-physics)
        train_physics
        ;;
    optimize)
        optimize
        ;;
    all)
        prepare
        train_sdf
        train_physics
        optimize
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
