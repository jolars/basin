// Same canonical origin as sitemap.xml / robots.txt: the apex custom
// domain `https://basin.rs/`, served at root (so there is no base path to
// prefix). Internal doc links are absolute so the file is useful when
// fetched on its own.
const SITE_ORIGIN = "https://basin.rs";

// llms.txt format: https://llmstxt.org—an H1 name, a blockquote summary,
// then sections of `- [title](url): note` links. This is a *signpost*,
// not a copy of the docs: the web pages are a thin overview and the
// authoritative API reference lives on docs.rs, so we link out to it
// rather than inlining content.
export const prerender = true;

export function GET() {
    const docs = `${SITE_ORIGIN}/docs`;

    const body = `# basin

> A numerical optimization library for Rust, inspired by argmin: a generic
> executor loop drives a solver over a state, calling into the problem traits
> you implement (CostFunction, Gradient, and friends). Works on plain
> Vec<f64> out of the box, with opt-in nalgebra / ndarray / faer backends.

## Docs

- [Overview](${docs}/): how the problem, solver, state, and executor pieces fit together
- [Getting started](${docs}/getting-started/): install, backend features, and a first solve
- [Solvers](${docs}/solvers/): catalog of available solvers and what each one needs

## Reference

- [API documentation (docs.rs)](https://docs.rs/basin): full, authoritative API reference
- [crates.io](https://crates.io/crates/basin): published releases
- [Source (GitHub)](https://github.com/jolars/basin): repository, issues, and changelog
`;

    return new Response(body, {
        headers: {
            "Content-Type": "text/plain; charset=utf-8",
        },
    });
}
