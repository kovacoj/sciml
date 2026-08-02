# Research Notebook Website

This orphan `gh-pages` branch contains only the source for the research notebook website. Scientific code will live on `main` later. The two branches intentionally have unrelated histories.

The site uses Fumadocs and Next.js.

## Local development

Node.js 22 or newer is required.

```bash
npm install
npm run dev
```

## Static build

```bash
npm run build
```

Generated output is written to `out/`.

GitHub Actions builds and deploys the site when changes are pushed to `gh-pages`.

## Documentation chat

The browser stores complete chat history locally and sends at most eight recent
messages through an n8n webhook. The Raspberry Pi deployment keeps n8n on
`127.0.0.1:8000`; only the production webhook path is intended for HTTPS
ingress. An internal Python gateway validates the 24,000-character request,
loads the published documentation, calls Siemens with a server-side key, and
caches request IDs for reload recovery. Secrets belong in mode-600 environment
files outside this repository.

## Interactive notebooks

Notebook source lives under `notebooks/`. The `scripts/export-notebooks.sh` script exports read-only HTML/WASM applications into the Git-ignored `public/notebooks/` directory. Fumadocs embeds these applications with the `MarimoNotebook` MDX component, and Next.js copies them into the combined `out/` artifact deployed to GitHub Pages.

Browser notebooks should remain lightweight and use only Pyodide-compatible dependencies. Edit and build locally with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
marimo edit notebooks/laplacian_eigenmodes.py
./scripts/export-notebooks.sh
npm run build
```

The Laplacian experiment page also uses `@marimo-team/mdx-marimo`. Its `python marimo` code fences are compiled into connected marimo islands within the Fumadocs page, while the full notebook export remains available for isolation and comparison.
