#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="$1"
CPUS="$2"
shift 2

ROOT="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(CDPATH= cd "$ROOT/.." && pwd)"
OUT="$ROOT/outputs/$RUN_NAME"
mkdir -p "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
CONTAINER="dafoam_${RUN_NAME}"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

nohup docker run --rm --name "$CONTAINER" --ipc=host --cpuset-cpus="$CPUS" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/dafoamuser \
    -e PYTHONPATH=/home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo/python \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 -e MPLBACKEND=Agg \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -v "$REPO:/home/dafoamuser/sciml" \
    -v "${TPFM_REFERENCE_ROOT:-$HOME/personal/tpfm_unet_reference}:/reference/tpfm:ro" \
    -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
    sciml-dafoam-torch-hfdib:latest bash -lc "source \$HOME/activate_dafoam_torch.sh; exec $*" \
    > "$OUT/stdout.log" 2>&1 < /dev/null &

echo $! > "$OUT/launcher.pid"
echo "Started $RUN_NAME"
echo "Container: $CONTAINER"
echo "CPUs: $CPUS"
echo "Log: $OUT/stdout.log"
