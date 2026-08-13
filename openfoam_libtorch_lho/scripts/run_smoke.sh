#!/usr/bin/env bash
set -euo pipefail

# Run smoke test for a single case
# Usage: scripts/run_smoke.sh [caseDir]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

CASE_DIR="${1:-$ROOT/cases/lho_template}"

echo "=== Running smoke test ==="
echo "Case: $CASE_DIR"

blockMesh -case "$CASE_DIR"
checkMesh -case "$CASE_DIR" | head -50

torchFoamSmoke -case "$CASE_DIR"

echo "=== Smoke test complete ==="
