import { readFileSync } from "node:fs";
import { katex } from "@mdit/plugin-katex";
import Icons from "unplugin-icons/vite";
import { type DefaultTheme, defineConfig } from "vitepress";
import {
    BENCHMARK_LINKS,
    DOCS_LINKS,
    type DocsNavItem,
    SOLVER_LINKS,
} from "../src/lib/nav";

const origin = "https://basin.rs";
const base = (process.env.BASIN_BASE_PATH ?? "").replace(/^\/+|\/+$/g, "");
const version = readFileSync(
    new URL("../../crates/basin/Cargo.toml", import.meta.url),
    "utf8",
).match(/^version\s*=\s*"([^"]+)"/m)?.[1];
const description =
    "Solve optimization problems in Rust with a wide range of algorithms, support for constraints, and your choice of linear-algebra library. Runs natively and in the browser.";

function sidebar(items: DocsNavItem[]): DefaultTheme.SidebarItem[] {
    return items.map((item) =>
        "children" in item
            ? {
                  text: item.label,
                  collapsed: true,
                  items: sidebar(item.children),
              }
            : { text: item.label, link: "/" + item.href },
    );
}

export default defineConfig({
    srcDir: "src",
    srcExclude: ["lib/**", "public/**"],
    outDir: "build",
    base: base ? `/${base}/` : "/",
    cleanUrls: true,
    title: "Basin",
    description,
    lang: "en-US",
    sitemap: { hostname: origin },
    markdown: {
        anchor: {
            slugify: (text) =>
                text
                    .toLowerCase()
                    .replace(/[^\p{L}\p{N}\s-]/gu, "")
                    .trim()
                    .replace(/\s+/g, "-") || "section",
        },
        config(md) {
            md.use(katex);
            const renderLink = md.renderer.rules.link_open;
            md.renderer.rules.link_open = (
                tokens,
                index,
                options,
                env,
                renderer,
            ) => {
                const token = tokens[index];
                const href = token.attrGet("href");
                // The included files resolve relative links from the repository root.
                if (
                    (env.relativePath === "guide/changelog/index.md" ||
                        env.relativePath === "guide/contributing/index.md") &&
                    href &&
                    !/^(?:[a-z][a-z\d+.-]*:|[/#])/i.test(href)
                ) {
                    token.attrSet(
                        "href",
                        new URL(
                            href,
                            "https://github.com/jolars/basin/blob/main/",
                        ).href,
                    );
                }
                return renderLink
                    ? renderLink(tokens, index, options, env, renderer)
                    : renderer.renderToken(tokens, index, options);
            };
        },
    },
    head: [
        ["link", { rel: "icon", href: "/favicon.svg", type: "image/svg+xml" }],
        ["link", { rel: "apple-touch-icon", href: "/apple-touch-icon.png" }],
        ["link", { rel: "manifest", href: "/site.webmanifest" }],
        ["meta", { property: "og:image", content: `${origin}/og.png` }],
        ["meta", { property: "og:site_name", content: "Basin" }],
        ["meta", { name: "twitter:card", content: "summary_large_image" }],
        ["meta", { name: "twitter:image", content: `${origin}/og.png` }],
    ],
    transformHead({ pageData, title, description }) {
        if (pageData.relativePath === "404.md") return [];
        const url =
            origin + "/" + pageData.relativePath.replace(/index\.md$/, "");
        return [
            ["link", { rel: "canonical", href: url }],
            ["meta", { property: "og:url", content: url }],
            ["meta", { property: "og:title", content: title }],
            ["meta", { property: "og:description", content: description }],
            ["meta", { name: "twitter:title", content: title }],
            ["meta", { name: "twitter:description", content: description }],
        ];
    },
    themeConfig: {
        logo: { light: "/logo.svg", dark: "/logo-dark.svg" },
        nav: [
            {
                text: "Guide",
                link: "/guide/getting-started/",
                activeMatch: "^/guide/(?!solvers(?:/|$))",
            },
            {
                text: "Solvers",
                link: "/guide/solvers/",
                activeMatch: "^/guide/solvers/",
            },
            { text: "Visualizer", link: "/visualizer/" },
            { text: "API", link: "https://docs.rs/basin" },
        ],
        sidebar: {
            "/guide/": [
                { text: "Introduction", items: sidebar(DOCS_LINKS) },
                { text: "Benchmarks", items: sidebar(BENCHMARK_LINKS) },
                {
                    text: "Development",
                    items: [
                        {
                            text: "Migration guides",
                            collapsed: true,
                            items: [
                                {
                                    text: "Migrating to Basin 2.0",
                                    link: "/guide/migrating-to-v2/",
                                },
                                {
                                    text: "Migrating to Basin 3.0",
                                    link: "/guide/migrating-to-v3/",
                                },
                            ],
                        },
                        { text: "Changelog", link: "/guide/changelog/" },
                        {
                            text: "Contributing to Basin",
                            link: "/guide/contributing/",
                        },
                    ],
                },
            ],
            "/guide/solvers/": [
                { text: "Solvers", items: sidebar(SOLVER_LINKS) },
            ],
        },
        outline: [2, 3],
        search: { provider: "local" },
        socialLinks: [
            { icon: "github", link: "https://github.com/jolars/basin" },
        ],
        footer: { message: `Basin v${version}` },
    },
    vite: { plugins: [Icons({ compiler: "vue3" })] },
});
