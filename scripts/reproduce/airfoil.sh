#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

usage() {
    echo "Usage: bash scripts/reproduce/airfoil.sh [prepare-sdf|train-sdf|prepare-physics|train-physics|optimize|all]"
}

prepare_sdf() {
    python data/airfoil/preprocess_sdf_airfoil.py
}

train_sdf() {
    python scripts/airfoil/train_stablesdf_airfoil.py
}

prepare_physics() {
    python data/airfoil/preprocess_physics_airfoil.py
}

train_physics() {
    python scripts/airfoil/train_gi_transolver.py
}

optimize() {
    python scripts/airfoil/optimize_airfoil.py
}

stage="${1:-all}"
case "${stage}" in
    prepare-sdf)
        prepare_sdf
        ;;
    train-sdf)
        train_sdf
        ;;
    prepare-physics)
        prepare_physics
        ;;
    train-physics)
        train_physics
        ;;
    optimize)
        optimize
        ;;
    all)
        prepare_sdf
        train_sdf
        prepare_physics
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
