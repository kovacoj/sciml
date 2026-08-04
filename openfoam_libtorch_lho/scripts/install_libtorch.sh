#!/usr/bin/env bash
set -euo pipefail

# Install pinned LibTorch 2.0.1+cpu with CXX11 ABI
# This is the shared CPU-only distribution with dependencies included

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$ROOT/.deps"
ARCHIVE="$DEPS/libtorch-2.0.1-cpu-cxx11.zip"
TARGET="$DEPS/libtorch-2.0.1-cpu-cxx11"

mkdir -p "$DEPS"

echo "=== Checking LibTorch installation ==="

if [[ -d "$TARGET" ]]; then
    echo "LibTorch already installed at $TARGET"
else
    echo "Downloading LibTorch 2.0.1+cpu (CXX11 ABI, shared)..."

    if [[ ! -f "$ARCHIVE" ]]; then
        wget -O "$ARCHIVE" \
          "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip"
    else
        echo "Archive already exists, skipping download"
    fi

    echo "Extracting LibTorch..."
    rm -rf "$DEPS/libtorch"
    unzip -q "$ARCHIVE" -d "$DEPS"
    mv "$DEPS/libtorch" "$TARGET"
fi

echo "=== Verifying LibTorch installation ==="

test -f "$TARGET/include/torch/torch.h" || { echo "ERROR: torch/torch.h not found"; exit 1; }
test -f "$TARGET/lib/libtorch.so" || { echo "ERROR: libtorch.so not found"; exit 1; }
test -f "$TARGET/lib/libtorch_cpu.so" || { echo "ERROR: libtorch_cpu.so not found"; exit 1; }
test -f "$TARGET/lib/libc10.so" || { echo "ERROR: libc10.so not found"; exit 1; }

echo "LibTorch headers and libraries verified."

# Compute and store checksum
sha256sum "$ARCHIVE" > "$DEPS/libtorch-archive.sha256"
echo "Checksum saved to $DEPS/libtorch-archive.sha256"

echo "=== LibTorch installation complete ==="
echo "Target directory: $TARGET"
echo ""
echo "To use LibTorch, source env.sh or set:"
echo "  export LIBTORCH_ROOT=$TARGET"
echo "  export LD_LIBRARY_PATH=\$LIBTORCH_ROOT/lib:\${LD_LIBRARY_PATH:-}"

# Output the target path for scripting
echo "$TARGET"
