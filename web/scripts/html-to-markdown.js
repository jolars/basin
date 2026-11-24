import { NodeHtmlMarkdown } from "node-html-markdown";
import { parse, serializeOuter } from "parse5";

export function htmlToMarkdown(html) {
    function find(node, predicate) {
        if (predicate(node)) return node;
        for (const child of node.childNodes ?? []) {
            const match = find(child, predicate);
            if (match) return match;
        }
        return null;
    }
    const document = parse(html);
    const content =
        find(document, (node) =>
            node.attrs?.some(
                ({ name, value }) => name === "id" && value === "VPContent",
            ),
        ) ?? find(document, (node) => node.tagName === "main");
    if (!content) return null;

    function removeChrome(node) {
        if (!node.childNodes) return;
        node.childNodes = node.childNodes.filter(
            (child) =>
                !child.attrs?.some(
                    ({ name, value }) =>
                        name === "class" &&
                        /(?:^|\s)(?:header-anchor|VPDocFooter|lang)(?:\s|$)/.test(
                            value,
                        ),
                ),
        );
        for (const child of node.childNodes) removeChrome(child);
    }
    removeChrome(content);
    const heroName = find(
        content,
        (node) =>
            node.tagName === "span" &&
            node.attrs?.some(
                ({ name, value }) =>
                    name === "class" && value.split(" ").includes("clip"),
            ),
    );
    if (heroName?.parentNode?.tagName === "h1") {
        heroName.childNodes.push({
            nodeName: "#text",
            value: " ",
            parentNode: heroName,
        });
    }
    const main = serializeOuter(content).replace(
        /(<span class="code-line\b)/g,
        "\n$1",
    );
    const markdown = NodeHtmlMarkdown.translate(
        main,
        {
            codeBlockStyle: "fenced",
            bulletMarker: "-",
            ignore: [
                "aside",
                "button",
                "img",
                "input",
                "script",
                "style",
                "svg",
            ],
        },
        {
            span: ({ node }) => {
                const display = node.classList.contains("katex-display");
                if (!display && !node.classList.contains("katex")) return {};

                const annotation = node.querySelector(
                    'annotation[encoding="application/x-tex"]',
                );
                if (!annotation) return {};

                // KaTeX includes the equation twice for visual and MathML rendering.
                return {
                    content: display
                        ? `$$\n${annotation.textContent}\n$$`
                        : `$${annotation.textContent}$`,
                    recurse: false,
                    surroundingNewlines: display ? 2 : 0,
                };
            },
        },
    ).trim();
    return markdown ? `${markdown}\n` : null;
}
