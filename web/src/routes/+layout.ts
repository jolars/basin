// Static site: prerender every route to real HTML. SSR stays ON here so
// landing/docs pages ship prerendered content (SEO + fast first paint).
// The wasm visualizer is the one browser-only route; it opts out of SSR
// in `visualizer/+page.ts`.
export const prerender = true;

// Canonicalize every URL with a trailing slash. This makes relative
// links inside docs Markdown (e.g. `../solvers/`) resolve consistently
// in dev, `preview`, and under the `/basin` base path on GitHub Pages —
// without it, `trailingSlash: 'never'` resolves `../foo` against the
// wrong directory.
export const trailingSlash = "always";
