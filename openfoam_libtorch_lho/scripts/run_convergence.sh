#!/usr/bin/env bash
set -euo pipefail

# Direct-FV convergence study across mesh resolutions, run in parallel.
# Each resolution is an independent case: embarrassingly parallel.
# Usage: scripts/run_convergence.sh [cells ...]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

NJOB="$(nproc 2>/dev/null || echo 4)"

if [[ $# -gt 0 ]]; then
    CELLS=("$@")
else
    CELLS=(64 128 256 512)
fi

echo "=== Convergence study ==="
echo "Cell counts: ${CELLS[@]}"
echo "Parallel jobs: $NJOB"

# Pass 1 (sequential): generate cases and meshes (fast, IO-bound)
for N in "${CELLS[@]}"; do
    CASE_DIR="$ROOT/cases/lho_$N"
    if [[ ! -d "$CASE_DIR" ]]; then
        python3 "$ROOT/scripts/generate_case.py" --cells "$N" --half-width 8 --output "$CASE_DIR"
    fi
    blockMesh -case "$CASE_DIR" > /dev/null
done

# Pass 2 (parallel): one fvNeuralLHO direct run per mesh resolution
printf "%s\n" "${CELLS[@]}" | xargs -P "$NJOB" -I{} bash -c '
    CASE="'"$ROOT"'/cases/lho_{}"
    fvNeuralLHO -case "$CASE" -mode direct > "$CASE/direct.log" 2>&1
    echo "N={} direct done"
'

echo ""
echo "=== Convergence complete. Direct-FV errors: ==="
printf "%-8s %-16s %-16s %-16s\n" "N" "dE0" "dE1" "dE2"
for N in "${CELLS[@]}"; do
    read -r _ _ _ e0 _ < <(grep "^State 0" "$ROOT/cases/lho_$N/direct.log" | sed -E 's/.*error = //')
    read -r e1 < <(grep "^State 1" "$ROOT/cases/lho_$N/direct.log" | sed -E 's/.*error = //')
    read -r e2 < <(grep "^State 2" "$ROOT/cases/lho_$N/direct.log" | sed -E 's/.*error = //')
    printf "%-8s %-16s %-16s %-16s\n" "$N" "$e0" "$e1" "$e2"
done
