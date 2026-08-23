#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_NAME CONFIG [CPUS]" >&2
  exit 2
fi

RUN_NAME=$1
CONFIG_INPUT=$2
CPUS=${3:-0-1}
REPO_ROOT=/home/cady/personal/sciml
TPFM_ROOT=/home/cady/personal/tpfm_unet_reference
PROJECT_ROOT=$REPO_ROOT/firedrake_hfdib_weak_flow
WORKDIR=/workspace/sciml/firedrake_hfdib_weak_flow
OUT=$PROJECT_ROOT/outputs/$RUN_NAME
CONTAINER=fdhfdib_$RUN_NAME

if [[ "$CONFIG_INPUT" = /* ]]; then
  CONFIG=$(realpath -m "$CONFIG_INPUT")
else
  CONFIG=$(realpath -m "$PROJECT_ROOT/$CONFIG_INPUT")
fi
if [[ "$CONFIG" != "$PROJECT_ROOT/"* ]]; then
  echo "config must be inside $PROJECT_ROOT" >&2
  exit 2
fi
CONTAINER_CONFIG=$WORKDIR/${CONFIG#"$PROJECT_ROOT/"}
CONTAINER_OUT=$WORKDIR/outputs/$RUN_NAME
mkdir -p "$OUT"

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  if [[ $(docker inspect --format '{{.State.Running}}' "$CONTAINER") == "true" ]]; then
    echo "container $CONTAINER is already running" >&2
    exit 1
  fi
  docker rm "$CONTAINER" >/dev/null
fi

RESUME_ARGS=()
if [[ -f "$OUT/checkpoint_latest.pt" ]]; then
  RESUME_ARGS=(--resume "$CONTAINER_OUT/checkpoint_latest.pt")
fi

nohup docker run --rm --name "$CONTAINER" --cpuset-cpus="$CPUS" \
  --ipc=host --memory=4g --pids-limit=512 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e MPLBACKEND=Agg -e PYTHONUNBUFFERED=1 \
  -v "$REPO_ROOT:/workspace/sciml" \
  -v "$TPFM_ROOT:/workspace/sciml/tpfm_unet_reference:ro" -w "$WORKDIR" \
  firedrakeproject/firedrake:latest \
  bash -lc 'exec python3 -m src.train --config "$1" --output-dir "$2" "${@:3}"' \
  bash "$CONTAINER_CONFIG" "$CONTAINER_OUT" "${RESUME_ARGS[@]}" \
  >"$OUT/stdout.log" 2>&1 &
echo $! >"$OUT/launcher.pid"
echo "launched $CONTAINER pid=$! log=$OUT/stdout.log"
