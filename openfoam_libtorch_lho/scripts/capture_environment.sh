#!/usr/bin/env bash
set -euo pipefail

# Capture complete environment information for reproducibility

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

OUTPUT_DIR="$ROOT/outputs"
mkdir -p "$OUTPUT_DIR"

ENV_FILE="$OUTPUT_DIR/environment.txt"

echo "=== Capturing environment to $ENV_FILE ==="

{
    echo "=========================================="
    echo "Environment Capture"
    echo "Date: $(date --iso-8601=seconds)"
    echo "=========================================="
    echo ""
    echo "=== OpenFOAM ==="
    foamVersion
    echo "WM_PROJECT_VERSION: $WM_PROJECT_VERSION"
    echo "WM_OPTIONS: $WM_OPTIONS"
    echo "WM_CXX: $WM_CXX"
    echo "WM_CXXFLAGS: $WM_CXXFLAGS"
    echo ""
    echo "=== Compiler ==="
    g++ --version
    echo ""
    echo "=== GLIBC Version ==="
    ldd --version | head -1
    echo ""
    echo "=== LibTorch Archive Checksum ==="
    if [[ -f "$ROOT/.deps/libtorch-archive.sha256" ]]; then
        cat "$ROOT/.deps/libtorch-archive.sha256"
    else
        echo "Checksum file not found"
    fi
    echo ""
    echo "=== Git Commit ==="
    cd "$ROOT"
    git rev-parse HEAD 2>/dev/null || echo "Not a git repository or no commit yet"
    echo ""
    echo "=== Operating System ==="
    cat /etc/os-release
    uname -a
    echo ""
    echo "=== LD_LIBRARY_PATH ==="
    echo "$LD_LIBRARY_PATH"
    echo ""
    echo "=== LDD on torchFoamSmoke (if built) ==="
    if [[ -f "$FOAM_USER_APPBIN/torchFoamSmoke" ]]; then
        ldd "$FOAM_USER_APPBIN/torchFoamSmoke" || true
    else
        echo "torchFoamSmoke not yet built"
    fi
    echo ""
    echo "=== LDD on fvNeuralLHO (if built) ==="
    if [[ -f "$FOAM_USER_APPBIN/fvNeuralLHO" ]]; then
        ldd "$FOAM_USER_APPBIN/fvNeuralLHO" || true
    else
        echo "fvNeuralLHO not yet built"
    fi
    echo ""
    echo "=========================================="
} > "$ENV_FILE"

echo "Environment captured to $ENV_FILE"
