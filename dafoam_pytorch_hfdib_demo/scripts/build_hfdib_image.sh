#!/usr/bin/env bash
# Build the derived Docker image with the HFDIB patch applied and both
# normal + ADR libraries compiled. Fails unless all build checks pass.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker build \
    -f "$ROOT/environment/Dockerfile.hfdib" \
    -t sciml-dafoam-torch-hfdib:latest \
    "$ROOT/.."

echo ""
echo "Image built: sciml-dafoam-torch-hfdib:latest"
echo "Run Gate G with: ./scripts/run_hfdib_gate_g.sh"
