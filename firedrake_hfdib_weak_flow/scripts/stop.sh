#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

CONTAINER=fdhfdib_$1
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker stop --time 60 "$CONTAINER"
else
  echo "container $CONTAINER does not exist"
fi
