# Interactive notebooks

Marimo notebooks are committed as Python source. Browser deployments use marimo's HTML/WASM export and are written to `public/notebooks/`; generated exports are not committed.

Public exports use `--mode run` to provide read-only applications executed in the reader's browser. Notebook dependencies must be compatible with Pyodide. Expensive scientific computation belongs outside the browser; future notebooks may load compact, public result artifacts produced elsewhere.

## Local use

```bash
source .venv/bin/activate
marimo edit notebooks/laplacian_eigenmodes.py
./scripts/export-notebooks.sh
```
