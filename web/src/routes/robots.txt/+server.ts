import { base } from '$app/paths';

// Same canonical origin as the sitemap: the apex custom domain
// `https://basin.rs/`, served at root (so `base` is empty). A static
// `static/robots.txt` couldn't carry an absolute `Sitemap:` URL, so this
// is an endpoint like sitemap.xml rather than a static file.
const SITE_ORIGIN = 'https://basin.rs';

// Prerendered into `build/robots.txt` by the static adapter; reached by
// the `prerender.entries: ['*']` crawl even though nothing links to it.
export const prerender = true;

export function GET() {
    const sitemap = `${SITE_ORIGIN}${base}/sitemap.xml`;

    const body = `User-agent: *
Allow: /

Sitemap: ${sitemap}
`;

    return new Response(body, {
        headers: {
            'Content-Type': 'text/plain',
        },
    });
}
