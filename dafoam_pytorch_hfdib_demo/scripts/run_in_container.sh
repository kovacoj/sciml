#!/usr/bin/env bash
# Run a command inside the pinned DAFoam+PyTorch container with the repo
# bind-mounted. Example:
#   ./scripts/run_in_container.sh mpirun -np 1 python python/probe_dafoam.py
set -euo pipefail

IMG=sciml-dafoam-torch:v5.0.0
HOST_REPO=/home/cady/chapel/sciml

docker run \
    --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/dafoamuser \
    --mount type=bind,src="$HOST_REPO",target=/home/dafoamuser/sciml \
    -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
    "$IMG" bash -lc "source \$HOME/activate_dafoam_torch.sh && $*"
