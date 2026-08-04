#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/run_in_container.sh" \
  "export MPLCONFIGDIR=/tmp/mplcfg && mpirun -np 1 python tests/test_residual_jtv.py"
