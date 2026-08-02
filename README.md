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
