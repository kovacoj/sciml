#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"

SOURCE_NOTEBOOK="$REPOSITORY_ROOT/notebooks/laplacian_eigenmodes.py"
OUTPUT_ROOT="$REPOSITORY_ROOT/public/notebooks"
OUTPUT_DIRECTORY="$OUTPUT_ROOT/laplacian-eigenmodes"

if ! command -v marimo >/dev/null 2>&1; then
  echo "Error: marimo is not available on PATH." >&2
  echo "Activate the Python environment and install requirements-notebooks.txt." >&2
  exit 1
fi

if [[ ! -f "$SOURCE_NOTEBOOK" ]]; then
  echo "Error: notebook not found: $SOURCE_NOTEBOOK" >&2
  exit 1
fi

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

EXTRA_ARGUMENTS=()

if marimo export html-wasm --help 2>&1 | grep -q -- "--execute"; then
  EXTRA_ARGUMENTS+=(--execute)
fi

marimo export html-wasm \
  "$SOURCE_NOTEBOOK" \
  -o "$OUTPUT_DIRECTORY" \
  --mode run \
  "${EXTRA_ARGUMENTS[@]}"

if [[ ! -f "$OUTPUT_DIRECTORY/index.html" ]]; then
  echo "Error: marimo export did not create index.html." >&2
  exit 1
fi

echo "Exported marimo notebook to:"
echo "  $OUTPUT_DIRECTORY"
