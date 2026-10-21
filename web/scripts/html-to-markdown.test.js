import assert from "node:assert/strict";
import { test } from "node:test";
import katex from "katex";
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

test("exports inline KaTeX as its original TeX once within prose", () => {
    const source = String.raw`\nabla f(x_k)^\top d_k < 0 \quad \text{cost \& gradient}`;
    const math = katex.renderToString(source);
    const html = `<main><p>Choose a <span>direction</span> with ${math} before the next step.</p></main>`;

    assert.equal(
        htmlToMarkdown(html),
        `Choose a direction with $${source}$ before the next step.\n`,
    );
});

test("exports display KaTeX as its original TeX block once between paragraphs", () => {
    const source = String.raw`\begin{aligned}
v_{k+1} &= \beta v_k - \alpha g_k \\
x_{k+1} &= x_k + v_{k+1}
\end{aligned}`;
    const math = katex.renderToString(source, { displayMode: true });
    const html = `<main><h1>Update</h1><p>Before the step.</p>${math}<p>After the step.</p></main>`;

    assert.equal(
        htmlToMarkdown(html),
        `# Update\n\nBefore the step.\n\n$$\n${source}\n$$\n\nAfter the step.\n`,
    );
});
