#!/usr/bin/env bash
# Regenerate clean meshes for the isothermal channel and single-obstacle cases.
# Removes all generated time dirs, processor dirs, postProcessing, and stale
# polyMesh, then runs blockMesh + checkMesh inside the container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

"$HERE/run_in_container.sh" "
for case in cases/channel_isothermal cases/single_obstacle; do
  echo '=== resetting '\$case' ==='
  cd \$case || exit 1
  # remove generated artifacts
  rm -rf [0-9]* 0.* processor* postProcessing constant/polyMesh
  # regenerate mesh
  blockMesh > log.blockMesh 2>&1
  checkMesh > log.checkMesh 2>&1
  grep -E 'cells:|Mesh (OK|FAILED)' log.checkMesh | head -3
  cd - > /dev/null
done
echo '=== mesh regeneration complete ==='
"
