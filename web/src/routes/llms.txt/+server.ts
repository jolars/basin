// Same canonical origin as sitemap.xml / robots.txt: the apex custom
// domain `https://basin.rs/`, served at root (so there is no base path to
// prefix). Internal doc links are absolute so the file is useful when
// fetched on its own.
const SITE_ORIGIN = "https://basin.rs";

// llms.txt format: https://llmstxt.org, an H1 name, a blockquote summary,
// then sections of `- [title](url): note` links. This is a *signpost*,
// not a copy of the docs: the web pages are a thin overview and the
// authoritative API reference lives on docs.rs, so we link out to it
// rather than inlining content.
export const prerender = true;

export function GET() {
    const docs = `${SITE_ORIGIN}/docs`;

    const body = `# basin

> Solve optimization problems in Rust with a wide range of algorithms,
> support for constraints, and your choice of linear-algebra library.
> Runs natively and in the browser. Use plain Vec<f64> or opt into
> nalgebra, ndarray, or faer.

## Docs

- [Overview](${docs}/): how the problem, solver, state, and executor pieces fit together
- [Getting started](${docs}/getting-started/): install, backend features, and a first solve
- [Migrating from Argmin](${docs}/migrating-from-argmin/): map Argmin 0.11 problem traits, solvers, errors, bounds, observers, and backends to Basin
- [Migrating to Basin 2.0](${docs}/migrating-to-v2/): upgrade from Basin 1.x, including constraints, states, termination reports, solver settings, and checkpoints
- [Migrating to Basin 3.0](${docs}/migrating-to-v3/): development API with optional constraint declarations and standalone objective and residual traits
- [Choosing a solver](${docs}/solvers/): selection guide and full catalog grouped by algorithm family, including constraint support and scalar root finding
- [Gradient descent](${docs}/solvers/first-order/gradient-descent/): fixed steps, line searches, heavy-ball momentum, and a Rust example
- [Nonlinear conjugate gradient](${docs}/solvers/first-order/nonlinear-cg/): Hager–Zhang and Polak–Ribière+ updates, restarts, and a Rust example
- [Stochastic gradient descent](${docs}/solvers/first-order/sgd/): mini-batch gradients, seeded sampling, momentum, cost refreshes, and a Rust example
- [Projected gradient descent](${docs}/solvers/first-order/projected-gradient-descent/): problem-side box bounds, projection, stationarity tests, and a Rust example
- [L-BFGS](${docs}/solvers/quasi-newton/l-bfgs/): when to use limited-memory BFGS, a Rust example, convergence settings, backend requirements, and original references

## Reference

- [API documentation (docs.rs)](https://docs.rs/basin): full, authoritative API reference
- [crates.io](https://crates.io/crates/basin): published releases
- [Source (GitHub)](https://github.com/jolars/basin): repository, issues, and changelog

## Optional

- [Benchmarks](${SITE_ORIGIN}/benchmarks/): performance against competing libraries, across backends and solvers
- [Visualizer](${SITE_ORIGIN}/visualizer/): interactive in-browser solver visualization (WASM)
`;

    return new Response(body, {
        headers: {
            "Content-Type": "text/plain; charset=utf-8",
        },
    });
}
