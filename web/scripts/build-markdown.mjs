import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { htmlToMarkdown } from "./html-to-markdown.js";

const build = new URL("../build/", import.meta.url);

async function pages(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    const found = [];
    for (const entry of entries) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) found.push(...(await pages(path)));
        else if (entry.name === "index.html") found.push(path);
    }
    return found;
}

for (const path of await pages(build.pathname)) {
    const html = await readFile(path, "utf8");
    const markdown = htmlToMarkdown(html);
    if (!markdown) throw new Error(`Empty Markdown content in ${path}`);

    await writeFile(path.replace(/index\.html$/, "index.md"), markdown);
}
