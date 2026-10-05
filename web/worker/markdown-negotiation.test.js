import assert from "node:assert/strict";
import { test } from "node:test";
import worker from "./markdown-negotiation.js";

function assets(fetch) {
    return { ASSETS: { fetch } };
}

test("serves Markdown from the asset binding and preserves HTML by default", async () => {
    const fetched = [];
    const env = assets(async (request) => {
        fetched.push(request.url);
        return new Response(
            request.url.endsWith("index.md") ? "# Basin\n" : "<h1>Basin</h1>",
            {
                headers: {
                    "Content-Type": "text/html",
                    Vary: "Accept-Encoding",
                    ETag: '"asset-hash"',
                },
            },
        );
    });

    const url = "https://basin.rs/docs/";
    const markdown = await worker.fetch(
        new Request(url, {
            headers: { Accept: "text/html, text/markdown;q=0.8" },
        }),
        env,
    );
    assert.equal(
        markdown.headers.get("Content-Type"),
        "text/markdown; charset=utf-8",
    );
    assert.equal(markdown.headers.get("Vary"), "Accept-Encoding, Accept");
    assert.equal(markdown.headers.get("ETag"), '"asset-hash"');
    assert.equal(await markdown.text(), "# Basin\n");
    assert.deepEqual(fetched, ["https://basin.rs/docs/index.md"]);

    const html = await worker.fetch(new Request(url), env);
    assert.equal(html.headers.get("Content-Type"), "text/html");
    assert.equal(html.headers.get("Vary"), "Accept-Encoding, Accept");
    assert.equal(await html.text(), "<h1>Basin</h1>");
});

test("keeps HTML when Markdown is refused and preserves missing-page status", async () => {
    const env = assets(async (request) => {
        if (request.url.includes("unknown")) {
            return new Response("Not found", { status: 404 });
        }
        assert.equal(request.url, "https://basin.rs/docs/");
        return new Response("<h1>Basin</h1>");
    });
    const refused = await worker.fetch(
        new Request("https://basin.rs/docs/", {
            headers: { Accept: "text/markdown;q=0" },
        }),
        env,
    );
    assert.equal(await refused.text(), "<h1>Basin</h1>");
    const missing = await worker.fetch(
        new Request("https://basin.rs/docs/unknown/", {
            headers: { Accept: "text/markdown" },
        }),
        env,
    );
    assert.equal(missing.status, 404);
    assert.equal(await missing.text(), "Not found");
});

test("preserves HEAD and conditional requests for Markdown", async () => {
    for (const status of [200, 304]) {
        const response = await worker.fetch(
            new Request("https://basin.rs/docs/?example=1", {
                method: "HEAD",
                headers: {
                    Accept: "text/markdown",
                    "If-None-Match": '"markdown"',
                },
            }),
            assets(async (request) => {
                assert.equal(
                    request.url,
                    "https://basin.rs/docs/index.md?example=1",
                );
                assert.equal(request.method, "HEAD");
                assert.equal(
                    request.headers.get("If-None-Match"),
                    '"markdown"',
                );
                return new Response(null, {
                    status,
                    headers: { ETag: '"markdown"' },
                });
            }),
        );
        assert.equal(response.status, status);
        assert.equal(response.headers.get("Vary"), "Accept");
        assert.equal(response.headers.get("ETag"), '"markdown"');
        assert.equal(await response.text(), "");
    }
});

test("lets asset routing handle redirects, static files, and unsupported methods", async () => {
    for (const [path, method, status] of [
        ["/docs", "GET", 307],
        ["/docs/index.html", "GET", 307],
        ["/_app/immutable/basin.wasm", "GET", 200],
        ["/docs/", "POST", 405],
    ]) {
        const request = new Request(`https://basin.rs${path}`, {
            method,
            headers: { Accept: "text/markdown" },
        });
        const response = await worker.fetch(
            request,
            assets(async (assetRequest) => {
                assert.equal(assetRequest, request);
                return new Response(null, {
                    status,
                    headers: { Location: "/docs/" },
                });
            }),
        );
        assert.equal(response.status, status);
        assert.equal(response.headers.get("Location"), "/docs/");
    }
});

test("redirects www to the canonical domain, preserving the path and query", async () => {
    const response = await worker.fetch(
        new Request("https://www.basin.rs/docs/?example=1"),
        assets(() => assert.fail("Redirects must not fetch an asset.")),
    );
    assert.equal(response.status, 308);
    assert.equal(
        response.headers.get("Location"),
        "https://basin.rs/docs/?example=1",
    );
});
