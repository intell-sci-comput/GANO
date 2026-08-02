#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

usage() {
    echo "Usage: bash scripts/reproduce/helmholtz.sh [prepare|train-sdf|train-physics|invert|all]"
}

prepare() {
    python data/hh/genshape.py
    python data/hh/gensdf.py
    python data/hh/genpde.py
    python data/hh/normalize_pde.py
}

train_sdf() {
    python scripts/hh/train_stablesdf.py
}

train_physics() {
    python scripts/hh/train_gi_transolver.py
}

invert() {
    python scripts/hh/optimize_hh.py
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
    invert)
        invert
        ;;
    all)
        prepare
        train_sdf
        train_physics
        invert
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
