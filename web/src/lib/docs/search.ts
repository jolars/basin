import type { Path } from "$app/types";
import { extractSearchDocument } from "./search-document";

const pages = import.meta.glob<string>("/src/routes/guide/**/+page.svx", {
    query: "?raw",
    import: "default",
    eager: true,
});

export const documents = Object.entries(pages).map(([path, source]) => ({
    ...extractSearchDocument(source),
    href: path.replace("/src/routes/", "").replace("+page.svx", "") as Path,
}));
