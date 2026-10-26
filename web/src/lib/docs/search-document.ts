import { type DefaultTreeAdapterTypes, parseFragment } from "parse5";

function textContent(node: DefaultTreeAdapterTypes.Node): string {
    if (
        "tagName" in node &&
        (node.tagName === "script" || node.tagName === "style")
    ) {
        return "";
    }
    if ("value" in node) return node.value;
    if ("childNodes" in node) {
        return node.childNodes.map(textContent).join(" ");
    }
    return "";
}

export function extractSearchDocument(source: string) {
    // Extract text from the parsed tree so filtering cannot create new markup.
    const markdown = textContent(parseFragment(source));
    return {
        title: markdown.match(/^#\s+(.+)$/m)?.[1] ?? "Documentation",
        text: markdown
            .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
            .replace(/[`#*_|]/g, "")
            .replace(/\s+/g, " ")
            .trim(),
    };
}
