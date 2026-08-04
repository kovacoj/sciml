#!/usr/bin/env bash
set -euo pipefail

# Run fvNeuralLHO on a single case with all modes
# Usage: scripts/run_single_case.sh [caseDir] [--quick]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

CASE_DIR="${1:-$ROOT/cases/lho_256}"
QUICK=false

if [[ "${2:-}" == "--quick" ]]; then
    QUICK=true
fi

echo "=== Running single case ==="
echo "Case: $CASE_DIR"
echo "Quick mode: $QUICK"

blockMesh -case "$CASE_DIR"
checkMesh -case "$CASE_DIR" | tail -20

echo ""
echo "=== Gate 0: Smoke test ==="
torchFoamSmoke -case "$CASE_DIR"

echo ""
echo "=== Direct FV eigensolver ==="
fvNeuralLHO -case "$CASE_DIR" -mode direct

if [[ "$QUICK" == "false" ]]; then
    echo ""
    echo "=== Coefficient optimization ==="
    fvNeuralLHO -case "$CASE_DIR" -mode coefficients

    echo ""
    echo "=== Neural training ==="
    fvNeuralLHO -case "$CASE_DIR" -mode neural
else
    echo ""
    echo "Skipping coefficient and neural modes in quick mode"
fi

echo "=== Case complete ==="
