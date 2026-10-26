import assert from "node:assert/strict";
import { globSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { extractSearchDocument } from "../src/lib/docs/search-document.ts";

test("indexes guide prose, links, headings, and code without HTML metadata", () => {
    const source = `<script lang="ts">import Example from "./Example.svelte";</script>
# Getting started

Use **Basin** and [the guide](/guide/).
<h2 id="example">Example</h2><p>First</p><p>Second</p>
\`\`\`rust
let result = solver.run();
\`\`\`
<style>p { color: red; }</style>
<!-- compile: example -->`;

    assert.deepEqual(extractSearchDocument(source), {
        title: "Getting started",
        text: "Getting started Use Basin and the guide. Example First Second rust let result = solver.run();",
    });
});

test("excludes mixed-case scripts and malformed closing tags", () => {
    const source = `<ScRiPt data-label=">">
# Hidden heading
hidden();
</sCrIpT ignored="true">
# Visible heading
<STYLE>hidden styles</STYLE>
Visible text`;

    assert.deepEqual(extractSearchDocument(source), {
        title: "Visible heading",
        text: "Visible heading Visible text",
    });
});

test("does not reassemble tags by deleting nested script sequences", () => {
    const { text } = extractSearchDocument(
        "<scrip<script>removed</script>t>alert(123)</script>",
    );

    assert.doesNotMatch(text, /<script/i);
    assert.match(text, /alert\(123\)/);
});

test("extracts encoded markup as literal text", () => {
    assert.deepEqual(
        extractSearchDocument(
            "&lt;script&gt;literal&lt;/script&gt; &amp; text",
        ),
        {
            title: "Documentation",
            text: "<script>literal</script> & text",
        },
    );
});

test("keeps every guide title and excludes component imports", () => {
    const directory = new URL("../src/routes/guide/", import.meta.url);
    const paths = globSync("**/+page.svx", { cwd: directory });
    assert.ok(paths.length > 0);

    for (const path of paths) {
        const source = readFileSync(new URL(path, directory), "utf8");
        const document = extractSearchDocument(source);
        assert.equal(document.title, source.match(/^#\s+(.+)$/m)?.[1], path);
        assert.doesNotMatch(document.text, /import\s+\w+\s+from\s+["']/, path);
    }
});
