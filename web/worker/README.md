# Website hosting

Cloudflare Workers serves the prerendered SvelteKit site at <https://basin.rs/>.
`web/wrangler.jsonc` deploys `web/build/` as static assets alongside the Worker.
The `www.basin.rs` domain redirects to the apex domain, preserving paths and
query strings. Cloudflare canonicalizes page URLs with trailing slashes and
serves `404.html` with a 404 status for unknown paths.

`pnpm build` writes an `index.md` beside each page's `index.html`, using the
rendered `<main>` content. The Worker serves that file when a page request
includes `Accept: text/markdown`. Both representations include `Vary: Accept`.
The Worker runs before asset routing so matching HTML assets do not bypass
negotiation. All content comes from the `ASSETS` binding.

## Development and deployment

From `web/`, run:

```sh
pnpm build
pnpm test:markdown
pnpm deploy:check
pnpm preview:worker
```

`preview:worker` runs the built site in Cloudflare's local runtime. `pnpm dev`
still runs the Svelte development server. After building, `pnpm deploy` uploads
the Worker and assets together; it requires Wrangler authentication or a
`CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` in the environment. Production
builds must leave `BASIN_BASE_PATH` unset.

The Website workflow deploys only the current stable release, as described in
[MAINTENANCE.md](../../MAINTENANCE.md#website-deployment). Configure these GitHub
Actions settings before deploying:

- Secret `CLOUDFLARE_API_TOKEN`: Workers Scripts Edit for the hosting account,
  plus Workers Routes Edit and Zone Read for the `basin.rs` zone.
- Variable `CLOUDFLARE_ACCOUNT_ID`: the hosting account ID.

The token must support the custom domains declared in `wrangler.jsonc`. The
[Cloudflare custom-domain documentation](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/)
explains domain setup and existing DNS conflicts.

## Production

The site moved from GitHub Pages to Workers on October 5, 2026, using the
v1.15.1 site sources. Both custom domains point to `basin-web`. Keep the hosting
configuration and workflow synchronized between `main` and `v1`.

To republish a release that predates this workflow, dispatch Website on `main`
or `v1` with `release_tag` set to the current stable tag. The workflow builds
that tag's site, generates its Markdown pages with the selected branch's tools,
and deploys the assets and Worker together.

Check HTML, Markdown, redirects, and missing pages:

```sh
curl -i https://basin.rs/docs/
curl -i -H 'Accept: text/markdown' https://basin.rs/docs/
curl -I https://basin.rs/docs
curl -I https://www.basin.rs/docs/
curl -I https://basin.rs/does-not-exist/
```
