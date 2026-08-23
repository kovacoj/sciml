#!/usr/bin/env bash
set -euo pipefail

RUN="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/$RUN"
CONTAINER="dafoam_${RUN}"

echo "=== CONTAINER ==="
docker ps --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'
echo
echo "=== LAST LOG ==="
tail -n 40 "$OUT/stdout.log" 2>/dev/null || true
echo
echo "=== MEMORY ==="
free -h
echo
echo "=== LOAD ==="
uptime
