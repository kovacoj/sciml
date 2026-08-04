#!/usr/bin/env bash
# Build (once) and enter the pinned DAFoam+PyTorch container.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG=sciml-dafoam-torch:v5.0.0

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  docker build -t "$IMG" -f "$ROOT/environment/Dockerfile" "$ROOT"
fi

docker run \
    --name sciml-dafoam-dev \
    -it \
    --rm \
    --ipc=host \
    --mount type=bind,src=/home/cady/chapel/sciml,target=/home/dafoamuser/sciml \
    -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
    "$IMG" \
    "${@:-bash}"
