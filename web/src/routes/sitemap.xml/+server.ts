// Canonical production origin. The deployed site lives at the apex custom
// domain `https://basin.rs/`, served at root, so there is no base path to
// prefix here (a base-prefixed deploy would have a different origin too,
// making these canonical URLs wrong regardless). Only the origin can't be
// derived during prerender (which uses a placeholder host), hence the
// constant.
const SITE_ORIGIN = "https://basin.rs";

// Discover every page route at build time so the sitemap can't drift when
// routes are added or removed. Glob keys look like
// `/src/routes/docs/solvers/+page.svx`; map them to URL paths below.
const pageModules = import.meta.glob("/src/routes/**/+page.{svelte,svx,md}");

function routePaths(): string[] {
    return (
        Object.keys(pageModules)
            .map((file) =>
                file
                    .replace("/src/routes", "")
                    .replace(/\/\+page\.(svelte|svx|md)$/, ""),
            )
            // Drop dynamic routes: they can't be enumerated without data.
            .filter((path) => !path.includes("["))
            // The root `+page.svelte` maps to `''`; that's the landing page.
            .map((path) => (path === "" ? "/" : path))
            .sort()
    );
}

// `export const prerender = true` makes the static adapter emit
// `build/sitemap.xml`; the `prerender.entries: ['*']` crawl in
// svelte.config.js reaches it even though nothing links to it.
export const prerender = true;

export function GET() {
    const urls = routePaths().map((path) => {
        // origin + path, with a trailing slash to match
        // `trailingSlash: 'always'` (root `+layout.ts`) and avoid a
        // redirect hop. Paths are clean (no `&`/query), so no XML escaping
        // is needed in <loc>.
        const loc = `${SITE_ORIGIN}${path}`.replace(/\/?$/, "/");
        return `  <url>\n    <loc>${loc}</loc>\n  </url>`;
    });

    const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.join("\n")}
</urlset>
`;

    return new Response(body, {
        headers: {
            "Content-Type": "application/xml",
        },
    });
}
