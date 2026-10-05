import assert from "node:assert/strict";
import { test } from "node:test";
import { htmlToMarkdown } from "./html-to-markdown.js";

test("keeps main content and code lines without site chrome", () => {
    const html = `<header>Site navigation</header>
        <main><aside>Documentation links</aside><h1>Example</h1>
        <pre><code><span class="code-line">let x = 1;</span><span class="code-line">let y = 2;</span></code></pre></main>
        <footer>Copyright</footer>`;

    const markdown = htmlToMarkdown(html);
    assert.match(markdown, /^# Example/m);
    assert.match(markdown, /let x = 1;\nlet y = 2;/);
    assert.doesNotMatch(
        markdown,
        /Site navigation|Documentation links|Copyright/,
    );
});
