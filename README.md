# Research Notebook Website

This orphan `pages` branch contains only the source for the research notebook website. Scientific code will live on `main` later. The two branches intentionally have unrelated histories.

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

GitHub Actions builds and deploys the site when changes are pushed to `pages`.

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

The experiments section also includes a separate native MDX demonstration built with `@marimo-team/mdx-marimo`. Its `python marimo` code fences are compiled into connected marimo islands within the Fumadocs page, while the full notebook export remains available for isolation and comparison.
