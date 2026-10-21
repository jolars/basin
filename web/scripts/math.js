import katex from "katex";

/**
 * @typedef {{ type: string, value?: string, children?: MarkdownNode[], data?: object }} MarkdownNode
 */

/** Render math at build time so the browser only needs KaTeX's CSS and fonts. */
export function renderMath() {
    /** @param {MarkdownNode} tree */
    return (tree) => {
        /** @param {MarkdownNode} node */
        function visit(node) {
            if (node.type === "math" || node.type === "inlineMath") {
                const html = katex.renderToString(node.value ?? "", {
                    displayMode: node.type === "math",
                    throwOnError: true,
                    output: "htmlAndMathml",
                });
                node.type = "html";
                node.value = `{@html ${JSON.stringify(html)}}`;
                delete node.data;
                delete node.children;
            }
            for (const child of node.children ?? []) visit(child);
        }
        visit(tree);
    };
}
