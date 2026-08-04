#!/usr/bin/env bash

# Environment script for the shallow HFDIB physics demo.
# Must be sourced before any build or execution commands.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source OpenFOAM v14
set +u 2>/dev/null
source /opt/openfoam14/etc/bashrc
set -u 2>/dev/null || true

# Reuse the LibTorch 2.0.1 (CPU, CXX11 ABI) installation of the LHO project
export HFDIB_DEMO_ROOT="$PROJECT_ROOT"
export LHO_PROJECT_ROOT="$PROJECT_ROOT/../openfoam_libtorch_lho"
export LIBTORCH_ROOT="$LHO_PROJECT_ROOT/.deps/libtorch-2.0.1-cpu-cxx11"
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Thread control (default: 8 cores; override by setting before sourcing)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

echo "=== OpenFOAM + LibTorch HFDIB demo environment ==="
echo "OpenFOAM: $(foamVersion)"
echo "WM_OPTIONS: $WM_OPTIONS"
echo "LIBTORCH_ROOT: $LIBTORCH_ROOT"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "==================================================="
