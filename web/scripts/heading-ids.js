/**
 * @typedef {{ type: string, value?: string, children?: MarkdownNode[], data?: { hProperties?: Record<string, string> } }} MarkdownNode
 */

/** Give documentation headings stable anchors, including repeated titles. */
export function headingIds() {
    /** @param {MarkdownNode} tree */
    return (tree) => {
        const used = new Set();
        /** @param {MarkdownNode} node
         * @returns {string} */
        function text(node) {
            return node.value ?? (node.children ?? []).map(text).join("");
        }
        /** @param {MarkdownNode} node */
        function visit(node) {
            if (node.type === "heading") {
                const base =
                    text(node)
                        .toLowerCase()
                        .replace(/[^\p{L}\p{N}\s-]/gu, "")
                        .trim()
                        .replace(/\s+/g, "-") || "section";
                let id = base;
                let suffix = 1;
                while (used.has(id)) id = `${base}-${suffix++}`;
                used.add(id);
                node.data ??= {};
                node.data.hProperties ??= {};
                node.data.hProperties.id = id;
            }
            for (const child of node.children ?? []) visit(child);
        }
        visit(tree);
    };
}
