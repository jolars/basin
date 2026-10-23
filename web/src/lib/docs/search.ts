import type { Path } from "$app/types";

const pages = import.meta.glob<string>("/src/routes/guide/**/+page.svx", {
    query: "?raw",
    import: "default",
    eager: true,
});

export const documents = Object.entries(pages).map(([path, source]) => {
    const markdown = source.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "");
    return {
        title: markdown.match(/^#\s+(.+)$/m)?.[1] ?? "Documentation",
        text: markdown
            .replace(/<[^>]+>/g, " ")
            .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
            .replace(/[`#*_|]/g, "")
            .replace(/\s+/g, " ")
            .trim(),
        href: path.replace("/src/routes/", "").replace("+page.svx", "") as Path,
    };
});
