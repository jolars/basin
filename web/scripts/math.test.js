import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { test } from "node:test";
import { compile as compileMarkdown } from "mdsvex";
import remarkMath from "remark-math";
import { compile as compileSvelte } from "svelte/compiler";
import { render } from "svelte/server";
import { renderMath } from "./math.js";

async function renderDocument(source) {
    const result = await compileMarkdown(source, {
        remarkPlugins: [remarkMath, renderMath],
        highlight: false,
    });
    assert.ok(result);
    const compiled = compileSvelte(result.code, {
        filename: "math.test.svelte",
        generate: "server",
    });
    // A data URL has no package scope, so resolve Svelte's runtime import first.
    const moduleCode = compiled.js.code.replaceAll(
        /(['"])svelte\/internal\/server\1/g,
        JSON.stringify(import.meta.resolve("svelte/internal/server")),
    );
    const moduleUrl = `data:text/javascript;base64,${Buffer.from(moduleCode).toString("base64")}`;
    const { default: Component } = await import(moduleUrl);
    return render(Component).body;
}

test("renders inline TeX within prose as Svelte-compatible KaTeX", async () => {
    const source = String.raw`\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k) + \beta \mathbf{v}_k`;
    const html = await renderDocument(`Update $${source}$ before continuing.`);

    assert.match(html, /<p>Update /);
    assert.match(html, /class="katex"/);
    assert.match(
        html,
        /<math xmlns="http:\/\/www.w3.org\/1998\/Math\/MathML">/,
    );
    assert.match(html, /<mi mathvariant="bold">x<\/mi>/);
    assert.match(html, /<mi>α<\/mi>/);
    assert.match(html, /<mi>β<\/mi>/);
    assert.equal(
        html.match(
            /<annotation encoding="application\/x-tex">([^<]*)<\/annotation>/,
        )?.[1],
        source,
    );
    assert.match(html, /before continuing\.<\/p>/);
    assert.doesNotMatch(html, /katex-display/);
});

test("renders a multiline display equation between paragraphs", async () => {
    const html = await renderDocument(String.raw`Before the step.

$$
\begin{aligned}
v_{k+1} &= \beta v_k - \alpha \nabla f(x_k) \\
x_{k+1} &= x_k + v_{k+1}
\end{aligned}
$$

After the step.`);

    assert.match(html, /<p>Before the step\.<\/p>/);
    assert.match(html, /class="katex-display"/);
    assert.match(html, /<mtable\b/);
    assert.match(html, /<mi>β<\/mi>/);
    assert.match(html, /<p>After the step\.<\/p>/);
    assert.doesNotMatch(html, /\$\$/);
});

test("keeps dollar-delimited text inside a fenced Rust example", async () => {
    const html = await renderDocument(
        '```rust\nlet message = "$x_{k+1}$ and $$f(x)$$";\n```',
    );

    assert.match(html, /<pre><code class="language-rust">/);
    assert.match(html, /\$x_\{k\+1\}\$ and \$\$f\(x\)\$\$/);
    assert.doesNotMatch(html, /katex|<math\b|\{@html/);
});

test("rejects malformed TeX during Markdown compilation", async () => {
    await assert.rejects(
        renderDocument(String.raw`Broken $\frac{1}{2$.`),
        /KaTeX parse error:.*expected '\}'/,
    );
});
