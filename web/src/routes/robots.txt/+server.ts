// Same canonical origin as the sitemap: the apex custom domain
// `https://basin.rs/`, served at root (so there is no base path to
// prefix). A static `static/robots.txt` couldn't carry an absolute
// `Sitemap:` URL, so this is an endpoint like sitemap.xml rather than a
// static file.
const SITE_ORIGIN = "https://basin.rs";

// Prerendered into `build/robots.txt` by the static adapter; reached by
// the `prerender.entries: ['*']` crawl even though nothing links to it.
export const prerender = true;

export function GET() {
    const sitemap = `${SITE_ORIGIN}/sitemap.xml`;

    const body = `User-agent: *
Allow: /

Sitemap: ${sitemap}
`;

    return new Response(body, {
        headers: {
            "Content-Type": "text/plain",
        },
    });
}
