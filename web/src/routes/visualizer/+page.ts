// The browser-only wasm lives in `#lib/Visualizer.svelte`, which the page
// loads via a dynamic import in `onMount`. The route itself imports no
// wasm, so it is SSR-safe: we leave SSR on (the default) and prerender it
// to a real shell with a proper <title>/og: head for crawlers. The
// interactive viz hydrates in on mount.
export const prerender = true;
