#!/usr/bin/env bash
# Records environment provenance for the pinned image. Run on the HOST.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG=sciml-dafoam-torch:v5.0.0

docker image inspect "$IMG" > "$ROOT/environment/docker_image_inspect.json"
echo "environment/docker_image_inspect.json written"

docker run --rm "$IMG" bash -lc '
  source /home/dafoamuser/dafoam/loadDAFoam.sh
  python -m pip freeze
' > "$ROOT/environment/pip_freeze.txt"
echo "environment/pip_freeze.txt written"

docker run --rm "$IMG" bash -lc '
  source /home/dafoamuser/dafoam/loadDAFoam.sh
  env | sort
' > "$ROOT/environment/environment.txt"
echo "environment/environment.txt written"
