#!/usr/bin/env bash
# Run shared multi-topology HFDIB CNN training.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMG="${SCIML_HFDIB_IMAGE:-sciml-dafoam-torch-hfdib:latest}"

docker run \
    --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/dafoamuser \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    --mount type=bind,src="$PROJECT/..",target=/home/dafoamuser/sciml \
    -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
    "$IMG" bash -lc '
        source $HOME/activate_dafoam_torch.sh
        export MPLCONFIGDIR=/tmp/mplcfg
        export PYTHONPATH=$PWD/python:${PYTHONPATH:-}
        cd /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
        python -m multitopology.train_shared_network "$@"
    ' _ "$@"
