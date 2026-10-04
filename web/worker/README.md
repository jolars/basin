# Markdown negotiation

The site is static on GitHub Pages. `pnpm build` writes an `index.md` beside
each page's `index.html`, using the rendered `<main>` content. The Cloudflare
Worker serves that file when a page request includes `Accept: text/markdown`.
The Worker includes the generated Markdown as a fallback while the file is
missing, so the live site can support negotiation before the next Pages release.

Run `pnpm build`, then deploy the Worker from `web/` with
`pnpm exec wrangler deploy --config worker/wrangler.jsonc`. The Cloudflare token
needs Workers Scripts edit and Workers Routes edit permissions for `basin.rs`.
The route patterns cover the page paths and leave the site's static assets
outside the Worker.

Check the response with
`curl -i -H 'Accept: text/markdown' https://basin.rs/docs/` and compare it to
`curl -i https://basin.rs/docs/`. The former should have `Content-Type:
text/markdown` and `Vary: Accept`; the latter should remain HTML.
