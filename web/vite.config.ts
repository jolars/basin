import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { escapeSvelte, type MdsvexOptions, mdsvex } from "mdsvex";
import { createHighlighter } from "shiki";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import { headingIds } from "./scripts/heading-ids.js";

// Production serves basin.rs at the root. Alternate deployments can set
// BASIN_BASE_PATH when they need a URL prefix.
const base = (process.env.BASIN_BASE_PATH ?? "") as "" | `/${string}`;

// Build-time syntax highlighter. Shiki runs only during preprocess/prerender
// (Node), so none of it ships to the client. Dual-theme output emits
// `--shiki-light`/`--shiki-dark` CSS variables (defaultColor: false); app.css
// maps them to the active theme via the `.dark` class toggle. The landing
// page's Playground uses the same GitHub palettes for its Prism tokens.
const LANGS = ["rust", "bash", "sh", "toml", "json", "js", "ts"];
const highlighter = await createHighlighter({
    themes: ["github-light-default", "github-dark-dimmed"],
    langs: LANGS,
});

const mdsvexConfig: MdsvexOptions = {
    remarkPlugins: [headingIds],
    extensions: [".svx", ".md"],
    // Every `.svx`/`.md` page is wrapped in this layout, which applies
    // the `prose` typography styling once instead of per-page.
    layout: {
        _: new URL("./src/lib/docs/mdsvex-layout.svelte", import.meta.url)
            .pathname,
    },
    // Highlight fenced code at build time with shiki (above). Unknown or
    // untagged languages fall back to plain `text`.
    highlight: {
        highlighter(code, lang) {
            const language = lang && LANGS.includes(lang) ? lang : "text";
            const html = highlighter.codeToHtml(code, {
                lang: language,
                themes: {
                    light: "github-light-default",
                    dark: "github-dark-dimmed",
                },
                defaultColor: false,
            });
            // Escape so Svelte doesn't parse `{`, backticks, etc. in the code.
            return `{@html \`${escapeSvelte(html)}\`}`;
        },
    },
};

// Read the basin crate version from Cargo.toml at config time so the
// site footer can show what version the docs/playground are pinned to.
const cargoToml = readFileSync(
    fileURLToPath(new URL("../crates/basin/Cargo.toml", import.meta.url)),
    "utf8",
);
const versionMatch = cargoToml.match(
    /\[package\][\s\S]*?\nversion\s*=\s*"([^"]+)"/,
);
if (!versionMatch) {
    throw new Error(
        "could not parse basin version from crates/basin/Cargo.toml",
    );
}
const basinVersion = versionMatch[1];

export default defineConfig({
    define: {
        __BASIN_VERSION__: JSON.stringify(basinVersion),
    },
    // `~icons/<set>/<name>` imports are resolved at build time by
    // unplugin-icons and compiled to Svelte components, so only the icons
    // actually imported are bundled (no runtime icon library). Icon data
    // comes from `@iconify-json/lucide`.
    plugins: [
        tailwindcss(),
        wasm(),
        sveltekit({
            // Top-level extensions so SvelteKit's router treats `.svx`/`.md` as
            // route files; mdsvex's own `extensions` (above) controls which files
            // it transforms.
            extensions: [".svelte", ".svx", ".md"],
            preprocess: [vitePreprocess(), mdsvex(mdsvexConfig)],
            // Every linked route is prerendered to its own `index.html`, so
            // docs/landing ship real HTML (SEO + fast load) — this is NOT SPA
            // mode (no `index.html` catch-all). The `404.html` fallback is the
            // one client-rendered page: Workers serves it with a 404 status
            // for unmatched paths.
            adapter: adapter({ fallback: "404.html" }),
            paths: { base },
            prerender: { entries: ["*"] },
        }),
        Icons({ compiler: "svelte" }),
    ],
    // The wasm-pack output uses ESM `import.meta.url` to find the .wasm
    // sibling. Marking the package as not-pre-bundled lets vite serve it
    // as-is in dev and preserves that resolution.
    optimizeDeps: { exclude: ["#lib/basin-wasm/basin_wasm.js"] },
});
