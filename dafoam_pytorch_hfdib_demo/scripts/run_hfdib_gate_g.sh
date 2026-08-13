#!/usr/bin/env bash
# Run Gate G inside the HFDIB-patched container image.
# Usage: ./scripts/run_hfdib_gate_g.sh [--smoke|--full]
set -euo pipefail

MODE="${1:---smoke}"
case "$MODE" in
    --smoke|--full) ;;
    *)
        echo "Usage: $0 [--smoke|--full]" >&2
        exit 2
        ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMG=sciml-dafoam-torch-hfdib:latest

docker run \
    --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/dafoamuser \
    -e GATE_G_MODE="$MODE" \
    --mount type=bind,src="$PROJECT/..",target=/home/dafoamuser/sciml \
    -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
    "$IMG" bash -lc '
        source $HOME/activate_dafoam_torch.sh
        export MPLCONFIGDIR=/tmp/mplcfg
        export PYTHONPATH=/home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo/python:${PYTHONPATH:-}
        cd /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
        # regenerate meshes (NEVER touch 0/ fields)
        for case in cases/channel_isothermal cases/single_obstacle; do
            cd $case
            find . -maxdepth 1 -name "[1-9]*" -type d -exec rm -rf {} +
            rm -rf processor* postProcessing constant/polyMesh log.*
            blockMesh > log.blockMesh 2>&1
            cd - > /dev/null
        done
        mpirun -np 1 python -m hfdib.gate_g "$GATE_G_MODE"
    '
