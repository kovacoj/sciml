#!/usr/bin/env bash
set -euo pipefail

# Run test suite
# Usage: scripts/run_tests.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

echo "=== Running test suite ==="

echo ""
echo "1. Environment check..."
test -d "$LIBTORCH_ROOT" || { echo "ERROR: LIBTORCH_ROOT not set or missing"; exit 1; }
test -f "$LIBTORCH_ROOT/include/torch/torch.h" || { echo "ERROR: LibTorch headers missing"; exit 1; }
foamVersion

echo ""
echo "2. Building applications..."
./Allwmake

echo ""
echo "3. Generating test case (N=32)..."
python3 "$ROOT/scripts/generate_case.py" --cells 32 --half-width 8 --output "$ROOT/cases/lho_test"

echo ""
echo "4. Creating mesh..."
blockMesh -case "$ROOT/cases/lho_test"
checkMesh -case "$ROOT/cases/lho_test" | tail -10

echo ""
echo "5. Gate 0: Smoke test..."
torchFoamSmoke -case "$ROOT/cases/lho_test"

echo ""
echo "6. Direct eigensolver (small mesh)..."
fvNeuralLHO -case "$ROOT/cases/lho_test" -mode direct

echo ""
echo "7. Python matrix tests..."
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
pytest -q tests/test_matrix.py -v

echo ""
echo "8. Python spectrum tests..."
pytest -q tests/test_spectrum.py -v

echo ""
echo "9. Output validation..."
pytest -q tests/test_outputs.py -v

echo ""
echo "=== All tests complete ==="
