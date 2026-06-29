import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import { escapeSvelte, mdsvex } from "mdsvex";
import { createHighlighter } from "shiki";

// GitHub Pages serves repo sites under `<user>.github.io/<repo>/`. Set
// the prefix via `BASIN_BASE_PATH` when deploying (`/basin` for Pages,
// empty for `npm run dev`/`preview` and any custom-domain deploy).
const base = process.env.BASIN_BASE_PATH ?? "";

// Build-time syntax highlighter. Shiki runs only during preprocess/prerender
// (Node), so none of it ships to the client. Dual-theme output emits
// `--shiki-light`/`--shiki-dark` CSS variables (defaultColor: false); app.css
// maps them to the active theme via the `.dark` class toggle. The Everforest
// themes are the green-forward, soft-contrast "natural" palette that matches
// the logo's mossy basin floor; the landing page's Playground hand-colors Rust
// with the same Everforest material palette.
const LANGS = ["rust", "bash", "sh", "toml", "json", "js", "ts"];
const highlighter = await createHighlighter({
    themes: ["everforest-light", "everforest-dark"],
    langs: LANGS,
});

/** @type {import('mdsvex').MdsvexOptions} */
const mdsvexConfig = {
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
                    light: "everforest-light",
                    dark: "everforest-dark",
                },
                defaultColor: false,
            });
            // Escape so Svelte doesn't parse `{`, backticks, etc. in the code.
            return `{@html \`${escapeSvelte(html)}\`}`;
        },
    },
};

/** @type {import('@sveltejs/kit').Config} */
const config = {
    // Top-level extensions so SvelteKit's router treats `.svx`/`.md` as
    // route files; mdsvex's own `extensions` (above) controls which files
    // it transforms.
    extensions: [".svelte", ".svx", ".md"],
    preprocess: [vitePreprocess(), mdsvex(mdsvexConfig)],
    kit: {
        // Every linked route is prerendered to its own `index.html`, so
        // docs/landing ship real HTML (SEO + fast load) — this is NOT SPA
        // mode (no `index.html` catch-all). The `404.html` fallback is the
        // one client-rendered page: GitHub Pages serves it for unmatched
        // paths, giving a styled not-found instead of Pages' default.
        adapter: adapter({ fallback: "404.html" }),
        paths: { base },
        prerender: { entries: ["*"] },
    },
};

export default config;
