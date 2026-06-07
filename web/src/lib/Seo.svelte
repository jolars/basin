<script lang="ts">
import { page } from "$app/state";

// Per-page <title>, description, canonical URL, and the dynamic Open
// Graph / Twitter tags. The page-INVARIANT social tags (og:image,
// og:type, og:site_name, twitter:card, twitter:image) live in
// `app.html` once. Title/description are kept here — and ONLY here —
// per page: `app.html` deliberately omits them, because a static tag
// in the template plus a per-page one in `<svelte:head>` would emit
// two <title>/<meta name="description"> tags (SvelteKit does not
// dedupe meta across the template and svelte:head).
//
// Canonical origin matches the SITE_ORIGIN constant in
// sitemap.xml / robots.txt. `page.url.pathname` carries a trailing slash
// (trailingSlash: 'always') and, under the apex domain, no base prefix;
// only the origin can't be derived during prerender (placeholder host),
// hence the constant.
const SITE_ORIGIN = "https://basin.rs";

const DEFAULT_DESCRIPTION =
    "basin is a numerical optimization library for Rust: pluggable solvers, multiple linear-algebra backends, first-class constraints, and a wasm-first design.";

let {
    title,
    description = DEFAULT_DESCRIPTION,
}: { title: string; description?: string } = $props();

let canonical = $derived(`${SITE_ORIGIN}${page.url.pathname}`);
</script>

<svelte:head>
    <title>{title}</title>
    <meta name="description" content={description} />
    <link rel="canonical" href={canonical} />

    <meta property="og:title" content={title} />
    <meta property="og:description" content={description} />
    <meta property="og:url" content={canonical} />

    <meta name="twitter:title" content={title} />
    <meta name="twitter:description" content={description} />
</svelte:head>
