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
    const guide = `${SITE_ORIGIN}/guide`;

    const body = `# basin

> Solve optimization problems in Rust with a wide range of algorithms,
> support for constraints, and your choice of linear-algebra library.
> Runs natively and in the browser. Use plain Vec<f64> or opt into
> nalgebra, ndarray, or faer.

## Guide

- [Overview](${guide}/): how the problem, solver, state, and executor pieces fit together
- [Getting started](${guide}/getting-started/): install, backend features, and a first solve
- [Migrating from Argmin](${guide}/migrating-from-argmin/): map Argmin 0.11 problem traits, solvers, errors, bounds, observers, and backends to Basin
- [Migrating to Basin 2.0](${guide}/migrating-to-v2/): upgrade from Basin 1.x, including constraints, states, termination reports, solver settings, and checkpoints
- [Migrating to Basin 3.0](${guide}/migrating-to-v3/): development API with optional constraint declarations and standalone objective and residual traits
- [Choosing a solver](${guide}/solvers/): selection guide and full catalog grouped by algorithm family, including constraint support and scalar root finding
- [Gradient descent](${guide}/solvers/first-order/gradient-descent/): fixed steps, line searches, heavy-ball momentum, and a Rust example
- [Nonlinear conjugate gradient](${guide}/solvers/first-order/nonlinear-cg/): Hager–Zhang and Polak–Ribière+ updates, restarts, and a Rust example
- [Stochastic gradient descent](${guide}/solvers/first-order/sgd/): mini-batch gradients, seeded sampling, momentum, cost refreshes, and a Rust example
- [Projected gradient descent](${guide}/solvers/first-order/projected-gradient-descent/): problem-side box bounds, projection, stationarity tests, and a Rust example
- [BFGS](${guide}/solvers/quasi-newton/bfgs/): dense inverse-Hessian models, initialization, stopping tests, and a Rust example
- [L-BFGS](${guide}/solvers/quasi-newton/l-bfgs/): when to use limited-memory BFGS, a Rust example, convergence settings, backend requirements, and original references
- [L-BFGS-B](${guide}/solvers/quasi-newton/l-bfgs-b/): problem-side box bounds, limited-memory curvature, stopping tests, and a Rust example
- [Nelder–Mead](${guide}/solvers/derivative-free/nelder-mead/): simplex search, adaptive coefficients, box projection, stopping tests, and a Rust example
- [MADS](${guide}/solvers/derivative-free/mads/): OrthoMADS polling, box and nonlinear constraints, poll-size stopping, and a Rust example
- [NEWUOA](${guide}/solvers/derivative-free/newuoa/): unconstrained quadratic interpolation, trust-region radii, and a Rust example
- [BOBYQA](${guide}/solvers/derivative-free/bobyqa/): quadratic interpolation with box bounds, trust-region radii, and a Rust example
- [LINCOA](${guide}/solvers/derivative-free/lincoa/): quadratic models with linear constraints, feasible starts, and a Rust example
- [COBYLA](${guide}/solvers/derivative-free/cobyla/): linear interpolation models, general constraints, feasibility caveats, and a Rust example
- [Solis–Wets](${guide}/solvers/derivative-free/solis-wets/): seeded random local search, adaptive step sizes, stopping tests, and a Rust example

## Reference

- [API documentation (docs.rs)](https://docs.rs/basin): full, authoritative API reference
- [crates.io](https://crates.io/crates/basin): published releases
- [Source (GitHub)](https://github.com/jolars/basin): repository, issues, and changelog

## Optional

- [Benchmarks](${SITE_ORIGIN}/benchmarks/): how to read the solver, library, and backend comparisons
- [Comparing solvers](${SITE_ORIGIN}/benchmarks/solvers/): convergence from shared starting points under a time budget
- [Comparing libraries](${SITE_ORIGIN}/benchmarks/competitors/): solver-by-solver comparisons with setup, interpretation, and methodology
- [Choosing a backend](${SITE_ORIGIN}/benchmarks/backends/): measured scaling for vector methods, dense matrices, and sparse least squares
- [Visualizer](${SITE_ORIGIN}/visualizer/): interactive in-browser solver visualization (WASM)
`;

    return new Response(body, {
        headers: {
            "Content-Type": "text/plain; charset=utf-8",
        },
    });
}
