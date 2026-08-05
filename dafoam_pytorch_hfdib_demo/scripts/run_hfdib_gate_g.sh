#!/usr/bin/env bash
# Run Gate G inside the HFDIB-patched container image.
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
        cd /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
        # regenerate meshes first
        for case in cases/channel_isothermal cases/single_obstacle; do
            cd $case && rm -rf [0-9]* 0.* processor* postProcessing constant/polyMesh
            blockMesh > log.blockMesh 2>&1
            cd - > /dev/null
        done
        # run Gate G
        mpirun -np 1 python -m hfdib.gate_g
    '
