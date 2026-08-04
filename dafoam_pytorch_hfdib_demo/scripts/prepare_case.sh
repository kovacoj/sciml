#!/usr/bin/env bash
# The channel_baseline case ships the supported ConvergentChannel premade
# mesh (constant/polyMesh from DAFoam reg_test_files) -- nothing to prepare.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$HERE/../cases/channel_baseline/constant/polyMesh/owner" ]; then
  echo "polyMesh missing from cases/channel_baseline/constant/" >&2
  exit 1
fi
echo "channel_baseline ready (ConvergentChannel mesh present)."
