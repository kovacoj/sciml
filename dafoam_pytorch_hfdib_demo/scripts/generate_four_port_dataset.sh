#!/usr/bin/env bash
# Generate the 64x64 four-port case and topology dataset.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMG="${SCIML_HFDIB_IMAGE:-sciml-dafoam-torch-hfdib:latest}"

# Step 1: Generate case template + topology masks (host-side, no container needed)
cd "$PROJECT"
python -m unet.generate_case

# Step 2: Generate dataset (needs DAFoam for converged solves)
docker run \
    --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/dafoamuser \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    --mount type=bind,src="$PROJECT/..",target=/home/dafoamuser/sciml \
    -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
    "$IMG" bash -lc '
        source $HOME/activate_dafoam_torch.sh
        export MPLCONFIGDIR=/tmp/mplcfg
        export PYTHONPATH=$PWD/python:${PYTHONPATH:-}
        cd /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
        # Regenerate meshes
        cd cases/four_port_64x64 && rm -rf [1-9]* processor* postProcessing constant/polyMesh log.*
        blockMesh > log.blockMesh 2>&1
        cd - > /dev/null
        # Generate dataset
        mpirun -np 1 python -m unet.generate_dataset \
          --dataset topologies/four_port_64 \
          --output datasets/four_port_64 \
          --case-template cases/four_port_64x64
    '
