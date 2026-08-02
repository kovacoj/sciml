#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"

OUTPUT_ROOT="$REPOSITORY_ROOT/public/notebooks"

if ! command -v marimo >/dev/null 2>&1; then
  echo "Error: marimo is not available on PATH." >&2
  echo "Activate the Python environment and install requirements-notebooks.txt." >&2
  exit 1
fi

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

EXTRA_ARGUMENTS=()

if marimo export html-wasm --help 2>&1 | grep -q -- "--execute"; then
  EXTRA_ARGUMENTS+=(--execute)
fi

export_notebook() {
  local source_notebook="$REPOSITORY_ROOT/notebooks/$1"
  local output_directory="$OUTPUT_ROOT/$2"

  if [[ ! -f "$source_notebook" ]]; then
    echo "Error: notebook not found: $source_notebook" >&2
    exit 1
  fi

  marimo export html-wasm \
    "$source_notebook" \
    -o "$output_directory" \
    --mode run \
    "${EXTRA_ARGUMENTS[@]}"

  if [[ ! -f "$output_directory/index.html" ]]; then
    echo "Error: marimo export did not create index.html." >&2
    exit 1
  fi

  echo "Exported marimo notebook to:"
  echo "  $output_directory"
}

export_notebook "laplacian_eigenmodes.py" "laplacian-eigenmodes"
export_notebook "thermodynamic_linear_algebra.py" "thermodynamic-linear-algebra"
