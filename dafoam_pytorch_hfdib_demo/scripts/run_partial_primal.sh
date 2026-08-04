#!/usr/bin/env bash
# Run the partial-primal k sweep inside the pinned container.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/run_in_container.sh" \
  "export MPLCONFIGDIR=/tmp/mplcfg && mpirun -np 1 python python/run_partial_primal.py"
