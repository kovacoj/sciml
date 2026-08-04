#!/usr/bin/env bash

# Environment script for OpenFOAM + LibTorch project
# Must be sourced before any build or execution commands

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source OpenFOAM v14 (suppress ZSH warnings)
set +u 2>/dev/null
source /opt/openfoam14/etc/bashrc
set -u 2>/dev/null || true

# Set LibTorch environment
export LIBTORCH_ROOT="$PROJECT_ROOT/.deps/libtorch-2.0.1-cpu-cxx11"
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Thread control (default: 8 cores; override by setting before sourcing)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

# Project root reference
export LHO_PROJECT_ROOT="$PROJECT_ROOT"

echo "=== OpenFOAM + LibTorch Environment ==="
echo "OpenFOAM: $(foamVersion)"
echo "WM_OPTIONS: $WM_OPTIONS"
echo "LIBTORCH_ROOT: $LIBTORCH_ROOT"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "========================================"
