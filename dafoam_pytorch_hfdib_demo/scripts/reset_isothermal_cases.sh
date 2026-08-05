#!/usr/bin/env bash
# Regenerate clean meshes for the isothermal channel and single-obstacle cases.
# Removes ONLY generated time dirs and polyMesh, never the tracked 0/ fields.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

"$HERE/run_in_container.sh" "
for case in cases/channel_isothermal cases/single_obstacle; do
  echo '=== resetting '\$case' ==='
  cd \$case || exit 1
  # remove ONLY numeric time dirs (NOT 0/), processor dirs, postProcessing, polyMesh
  find . -maxdepth 1 -name '[1-9]*' -type d -exec rm -rf {} +
  rm -rf processor* postProcessing constant/polyMesh
  # regenerate mesh
  blockMesh > log.blockMesh 2>&1
  checkMesh > log.checkMesh 2>&1
  grep -E 'cells:|Mesh (OK|FAILED)' log.checkMesh | head -3
  cd - > /dev/null
done
echo '=== mesh regeneration complete ==='
"
