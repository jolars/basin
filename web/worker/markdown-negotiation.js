function acceptsMarkdown(value) {
    return (
        value?.split(",").some((part) => {
            const [type, ...parameters] = part.trim().split(";");
            if (type.toLowerCase() !== "text/markdown") return false;
            const quality = parameters.find((parameter) =>
                /^\s*q\s*=/i.test(parameter),
            );
            return !quality || Number(quality.split("=")[1]) > 0;
        }) ?? false
    );
}

function varyOnAccept(headers) {
    const vary = headers.get("Vary");
    if (!vary) headers.set("Vary", "Accept");
    else if (
        !vary.split(",").some((part) => part.trim().toLowerCase() === "accept")
    ) {
        headers.set("Vary", `${vary}, Accept`);
    }
}

export default {
    async fetch(request, env) {
        const url = new URL(request.url);
        if (url.hostname === "www.basin.rs") {
            url.hostname = "basin.rs";
            url.protocol = "https:";
            return Response.redirect(url, 308);
        }

        // Let asset routing canonicalize non-trailing-slash URLs before
        // negotiating, so relative links resolve the same way in both formats.
        const isPage = url.pathname.endsWith("/");
        if (
            isPage &&
            ["GET", "HEAD"].includes(request.method) &&
            acceptsMarkdown(request.headers.get("Accept"))
        ) {
            url.pathname += "index.md";
            const markdown = await env.ASSETS.fetch(new Request(url, request));
            if (markdown.ok || markdown.status === 304) {
                const headers = new Headers(markdown.headers);
                headers.set("Content-Type", "text/markdown; charset=utf-8");
                varyOnAccept(headers);
                if (request.method === "HEAD" || markdown.status === 304) {
                    return new Response(null, {
                        status: markdown.status,
                        headers,
                    });
                }
                const body = await markdown.text();
                headers.set(
                    "x-markdown-tokens",
                    String(Math.ceil(body.length / 4)),
                );
                return new Response(body, { status: markdown.status, headers });
            }
        }

        const response = await env.ASSETS.fetch(request);
        if (!isPage) return response;
        const headers = new Headers(response.headers);
        varyOnAccept(headers);
        return new Response(response.body, {
            status: response.status,
            headers,
        });
    },
};
