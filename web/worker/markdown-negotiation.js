import fallbackPages from "../.svelte-kit/markdown-pages.generated.js";

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
    async fetch(request) {
        const url = new URL(request.url);
        const isPage =
            url.pathname.endsWith("/") || url.pathname.endsWith("/index.html");
        if (
            !isPage ||
            !["GET", "HEAD"].includes(request.method) ||
            !acceptsMarkdown(request.headers.get("Accept"))
        ) {
            const response = await fetch(request);
            if (!isPage) return response;
            const headers = new Headers(response.headers);
            varyOnAccept(headers);
            return new Response(response.body, {
                status: response.status,
                headers,
            });
        }

        url.pathname = url.pathname.replace(/(?:index\.html)?$/, "index.md");
        const markdown = await fetch(
            new Request(url, { method: request.method }),
        );
        if (!markdown.ok) {
            const response = await fetch(
                request.method === "HEAD"
                    ? new Request(request.url, { method: "GET" })
                    : request,
            );
            const headers = new Headers(response.headers);
            varyOnAccept(headers);
            const route = new URL(request.url).pathname.replace(
                /index\.html$/,
                "",
            );
            const body = fallbackPages[route];
            if (response.ok && body) {
                headers.set("Content-Type", "text/markdown; charset=utf-8");
                headers.delete("Content-Length");
                headers.delete("Content-Encoding");
                headers.delete("ETag");
                headers.delete("Last-Modified");
                headers.set(
                    "x-markdown-tokens",
                    String(Math.ceil(body.length / 4)),
                );
                return new Response(request.method === "HEAD" ? null : body, {
                    status: response.status,
                    headers,
                });
            }
            return new Response(
                request.method === "HEAD" ? null : response.body,
                {
                    status: response.status,
                    headers,
                },
            );
        }

        const headers = new Headers(markdown.headers);
        headers.set("Content-Type", "text/markdown; charset=utf-8");
        headers.delete("Content-Length");
        headers.delete("Content-Encoding");
        headers.delete("ETag");
        headers.delete("Last-Modified");
        varyOnAccept(headers);
        if (request.method === "HEAD") {
            return new Response(null, { status: markdown.status, headers });
        }
        const body = await markdown.text();
        headers.set("x-markdown-tokens", String(Math.ceil(body.length / 4)));
        return new Response(body, { status: markdown.status, headers });
    },
};
