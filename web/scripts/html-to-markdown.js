import { NodeHtmlMarkdown } from "node-html-markdown";

export function htmlToMarkdown(html) {
    const start = html.search(/<main(?:\s|>)/i);
    const end = html.indexOf("</main>", start);
    if (start < 0 || end < 0) return null;

    const main = html
        .slice(start, end + "</main>".length)
        .replace(/(<span class="code-line\b)/g, "\n$1");
    const markdown = NodeHtmlMarkdown.translate(main, {
        codeBlockStyle: "fenced",
        bulletMarker: "-",
        ignore: ["aside", "button", "img", "input", "script", "style", "svg"],
    }).trim();
    return markdown ? `${markdown}\n` : null;
}
