#!/usr/bin/env bash
# Run the HFDIB PINN inside the HFDIB-patched container image.
# Usage:
#   ./scripts/run_hfdib_pinn.sh --architecture mlp --k 8 --gradient-check-only
#   ./scripts/run_hfdib_pinn.sh --architecture cnn --k 8 --steps 50 --lr 1e-3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMG=sciml-dafoam-torch-hfdib:latest

docker run \
    --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/dafoamuser \
    --mount type=bind,src="$PROJECT/..",target=/home/dafoamuser/sciml \
    -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
    "$IMG" bash -lc '
        source $HOME/activate_dafoam_torch.sh
        export MPLCONFIGDIR=/tmp/mplcfg
        export PYTHONPATH=/home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo/python:${PYTHONPATH:-}
        cd /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
        # regenerate meshes
        for case in cases/channel_isothermal cases/single_obstacle; do
            cd $case
            find . -maxdepth 1 -name "[1-9]*" -type d -exec rm -rf {} +
            rm -rf processor* postProcessing constant/polyMesh log.*
            blockMesh > log.blockMesh 2>&1
            cd - > /dev/null
        done
        mpirun -np 1 python -m pinn.train_hfdib_pinn "$@"
    ' _ "$@"
