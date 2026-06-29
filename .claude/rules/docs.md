---
description: >-
  Map of the basin web docs site (deployed to GitHub Pages at https://basin.rs):
  the page routes, the machine-readable endpoints (llms.txt, sitemap.xml,
  robots.txt), and the rule that the llms.txt signpost must list the real site
  routes. Solver-catalogue sync lives in web-docs-sync.md.
paths:
  - "web/src/routes/**"
---

# The web docs site

`web/` is a SvelteKit app (static adapter) deployed to GitHub Pages at the apex
custom domain `https://basin.rs/`, served at root. It is a thin overview that
signposts to the authoritative API reference on docs.rs, not a copy of it.

## Page routes

- `/` landing page
- `/docs`, `/docs/getting-started`, `/docs/solvers`
- `/benchmarks`, `/benchmarks/backends`, `/benchmarks/competitors`, `/benchmarks/solvers`
- `/visualizer` (WASM, in-browser solver visualization)

## Machine-readable endpoints

These are prerendered `+server.ts` routes (not static files), because each needs
an absolute `https://basin.rs` URL that a static file can't carry. All three pin
the same `SITE_ORIGIN` constant and are reached by the `prerender.entries: ['*']`
crawl even though nothing links to them.

- `sitemap.xml`: globs `+page.{svelte,svx,md}` at build time, so it can't drift.
- `robots.txt`: allow-all plus the absolute `Sitemap:` URL.
- `llms.txt`: the [llmstxt.org](https://llmstxt.org) signpost (see below).

## Keep llms.txt in sync with the real routes

`llms.txt` is a hand-maintained signpost: an H1 name, a blockquote summary, then
`## Section` blocks of `- [title](url): note` links. Unlike `sitemap.xml` it does
**not** glob the routes, so it drifts when pages are added or removed. When you
add, remove, or rename a top-level section of the site, update `llms.txt` to
match:

- Core docs go under `## Docs`, the API reference and external links under
  `## Reference`, and secondary material an LLM can skip under tight context
  (benchmarks, the visualizer) under `## Optional`.
- Use absolute `${SITE_ORIGIN}/...` URLs with a trailing slash, matching
  `trailingSlash: 'always'`, so the file works when fetched on its own and avoids
  a redirect hop.
- It is a signpost, not a copy: link out rather than inlining doc content.

## Verifying

`cd web && pnpm build` (the lockfile is `pnpm-lock.yaml`) catches malformed
markdown and broken endpoints. The solver catalogue page has its own sync rule;
see `web-docs-sync.md`.
