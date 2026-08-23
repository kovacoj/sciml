#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN_NAME=$1
OUT=/home/cady/personal/sciml/firedrake_hfdib_weak_flow/outputs/$RUN_NAME
CONTAINER=fdhfdib_$RUN_NAME
echo "== container =="
docker ps -a --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'
echo "== heartbeat =="
if [[ -f "$OUT/heartbeat.json" ]]; then
  cat "$OUT/heartbeat.json"
else
  echo "no heartbeat"
fi
echo "== last 40 stdout lines =="
if [[ -f "$OUT/stdout.log" ]]; then
  tail -n 40 "$OUT/stdout.log"
else
  echo "no stdout.log"
fi
echo "== free =="
free -h
