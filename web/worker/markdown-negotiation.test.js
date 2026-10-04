import assert from "node:assert/strict";
import { after, test } from "node:test";
import worker from "./markdown-negotiation.js";

const originalFetch = globalThis.fetch;
after(() => {
    globalThis.fetch = originalFetch;
});

test("serves Markdown for an accepted page and preserves HTML by default", async () => {
    const fetched = [];
    globalThis.fetch = async (request) => {
        fetched.push(request.url);
        return new Response(
            request.url.endsWith("index.md") ? "# Basin\n" : "<h1>Basin</h1>",
            {
                headers: {
                    "Content-Type": "text/html",
                    Vary: "Accept-Encoding",
                },
            },
        );
    };

    const url = "https://basin.rs/docs/";
    const markdown = await worker.fetch(
        new Request(url, {
            headers: { Accept: "text/html, text/markdown;q=0.8" },
        }),
    );
    assert.equal(
        markdown.headers.get("Content-Type"),
        "text/markdown; charset=utf-8",
    );
    assert.equal(markdown.headers.get("Vary"), "Accept-Encoding, Accept");
    assert.equal(await markdown.text(), "# Basin\n");
    assert.deepEqual(fetched, ["https://basin.rs/docs/index.md"]);

    const html = await worker.fetch(new Request(url));
    assert.equal(html.headers.get("Content-Type"), "text/html");
    assert.equal(html.headers.get("Vary"), "Accept-Encoding, Accept");
    assert.equal(await html.text(), "<h1>Basin</h1>");
});

test("keeps HTML for a refused Markdown type and missing Markdown page", async () => {
    globalThis.fetch = async (request) =>
        request.url.endsWith("index.md")
            ? new Response("Not found", { status: 404 })
            : new Response("<h1>Basin</h1>");

    const url = "https://basin.rs/docs/";
    const refused = await worker.fetch(
        new Request(url, { headers: { Accept: "text/markdown;q=0" } }),
    );
    assert.equal(await refused.text(), "<h1>Basin</h1>");
    const missing = await worker.fetch(
        new Request("https://basin.rs/docs/unknown/", {
            headers: { Accept: "text/markdown" },
        }),
    );
    assert.equal(missing.status, 200);
    assert.equal(await missing.text(), "<h1>Basin</h1>");
});

test("serves bundled Markdown while waiting for a Markdown asset", async () => {
    globalThis.fetch = async (request) =>
        request.url.endsWith("index.md")
            ? new Response("Not found", { status: 404 })
            : new Response(
                  "<html><nav>Site links</nav><main><h1>Basin</h1><p>Optimization.</p></main></html>",
                  { headers: { "Content-Type": "text/html" } },
              );

    const response = await worker.fetch(
        new Request("https://basin.rs/", {
            headers: { Accept: "text/markdown" },
        }),
    );
    assert.equal(
        response.headers.get("Content-Type"),
        "text/markdown; charset=utf-8",
    );
    assert.match(await response.text(), /^# Numerical Optimization in Rust\n/m);
});
