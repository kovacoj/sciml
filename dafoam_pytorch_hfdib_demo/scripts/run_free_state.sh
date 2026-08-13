#!/usr/bin/env bash
# Run Gate E1 (warm-started free-state training) inside the container.
# Passes all extra args through to train_free_state.py.
# Example:
#   ./scripts/run_free_state.sh --k 8 --steps 800 --lr 1e-3 \
#       --no-mask-phi --gn-iters 12 --cg-iters 300 --pc-probes 8
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2086
"$HERE/run_in_container.sh" \
  "export MPLCONFIGDIR=/tmp/mplcfg && mpirun -np 1 python python/train_free_state.py $*"
