// Shared navigation model. `href` is an app path without a leading slash.
// Calling `resolve()` prefixes the configured base, so links also work in
// deployments under a URL prefix. `section` is the first path segment, used
// for active-state matching across a whole section (e.g. any `/docs/*` page
// lights up the "Docs" link).

import type { Path } from "$app/types";

export type NavLink = {
    label: string;
    href: Path;
    /** First path segment for active-state matching. Omit for external links. */
    section?: string;
    external?: boolean;
};

export type DocsNavItem =
    | (NavLink & { fragment?: string })
    | { label: string; children: DocsNavItem[] };

// `href`s carry a trailing slash to match `trailingSlash: 'always'`
// (set in the root `+layout.ts`), so links hit the canonical URL with no
// redirect hop.

/** Top-level site navigation, shown in the header. */
export const NAV_LINKS: NavLink[] = [
    { label: "Docs", href: "docs/getting-started/", section: "docs" },
    { label: "Visualizer", href: "visualizer/", section: "visualizer" },
    { label: "Benchmarks", href: "benchmarks/", section: "benchmarks" },
];

/** Sidebar links for the benchmark guides. */
export const BENCHMARK_LINKS: NavLink[] = [
    { label: "Overview", href: "benchmarks/" },
    { label: "Comparing solvers", href: "benchmarks/solvers/" },
    { label: "Comparing libraries", href: "benchmarks/competitors/" },
    { label: "Choosing a backend", href: "benchmarks/backends/" },
];

/** Sidebar links for the docs section. */
export const DOCS_LINKS: NavLink[] = [
    { label: "Overview", href: "docs/", section: "docs" },
    {
        label: "Getting started",
        href: "docs/getting-started/",
        section: "docs",
    },
    {
        label: "Migrating from Argmin",
        href: "docs/migrating-from-argmin/",
        section: "docs",
    },
    {
        label: "Migrating to Basin 2.0",
        href: "docs/migrating-to-v2/",
        section: "docs",
    },
    {
        label: "Migrating to Basin 3.0",
        href: "docs/migrating-to-v3/",
        section: "docs",
    },
];

/** Sidebar links grouped by solver family. */
export const SOLVER_LINKS: DocsNavItem[] = [
    { label: "Choosing a solver", href: "docs/solvers/", fragment: "" },
    {
        label: "First-order",
        children: [
            {
                label: "Gradient descent",
                href: "docs/solvers/first-order/gradient-descent/",
            },
            {
                label: "Nonlinear CG",
                href: "docs/solvers/first-order/nonlinear-cg/",
            },
            {
                label: "Stochastic gradient descent",
                href: "docs/solvers/first-order/sgd/",
            },
            {
                label: "Projected gradient descent",
                href: "docs/solvers/first-order/projected-gradient-descent/",
            },
        ],
    },
    {
        label: "Quasi-Newton",
        children: [
            solver("BFGS", "bfgs"),
            {
                label: "L-BFGS",
                href: "docs/solvers/quasi-newton/l-bfgs/",
            },
            solver("L-BFGS-B", "lbfgsb"),
        ],
    },
    {
        label: "Derivative-free",
        children: [
            solver("Nelder–Mead", "nelder-mead"),
            solver("MADS", "mads"),
            solver("NEWUOA", "newuoa"),
            solver("BOBYQA", "bobyqa"),
            solver("LINCOA", "lincoa"),
            solver("COBYLA", "cobyla"),
            solver("Solis–Wets", "solis-wets"),
        ],
    },
    {
        label: "Trust-region",
        children: [solver("Trust-region Newton", "trust-region-newton")],
    },
    {
        label: "Nonlinear least squares",
        children: [
            solver("Gauss–Newton", "gauss-newton"),
            solver("Levenberg–Marquardt", "levenberg-marquardt"),
            solver("Levenberg–Marquardt QR", "levenberg-marquardt-qr"),
            solver(
                "Bounded Levenberg–Marquardt",
                "bounded-levenberg-marquardt",
            ),
            solver("Trust-region reflective", "trust-region-reflective"),
        ],
    },
    {
        label: "Global and stochastic",
        children: [
            solver("DIRECT", "direct"),
            solver("Globalized bounded Nelder–Mead", "gbnm"),
            solver("Simulated annealing", "simulated-annealing"),
            solver("CMA-ES", "cma-es"),
            solver("Bounded CMA-ES", "bounded-cma-es"),
            solver("Differential evolution", "de"),
            solver("Particle swarm", "global-best-pso"),
            solver("Random search", "random-search"),
            solver("Steady-state genetic algorithm", "ssga"),
            solver("Basin-hopping", "basin-hopping"),
        ],
    },
    {
        label: "Memetic",
        children: [
            solver("CMA-ES with local search", "cma-inject"),
            solver("Bounded CMA-ES with local search", "bounded-cma-inject"),
            solver("DE with local search", "de-inject"),
            solver("Local-search chains", "ma-ls-ch"),
            solver("CMA-ES chains", "ma-ls-ch-cma"),
            solver("Solis–Wets chains", "ma-ls-ch-sw"),
        ],
    },
    {
        label: "General constrained",
        children: [
            solver("SLSQP", "slsqp"),
            solver("Barrier method", "barrier-method"),
            solver("Augmented Lagrangian", "augmented-lagrangian-method"),
        ],
    },
    {
        label: "Scalar minimization",
        children: [
            solver("Brent", "brent"),
            solver("Brent with derivatives", "brent-derivative"),
            solver("Golden-section search", "golden-section"),
        ],
    },
    {
        label: "Root finding",
        children: [
            solver("Brent", "brent-root"),
            solver("Secant", "secant-root"),
            solver("Newton", "newton-root"),
            solver("Halley", "halley-root"),
            solver("TOMS 748", "toms748-root"),
        ],
    },
];

function solver(label: string, fragment: string): DocsNavItem {
    return { label, href: "docs/solvers/", fragment };
}

/**
 * The active section for a route. Returns the first path segment
 * (`'docs'`, `'visualizer'`, …) or `''` for the landing page.
 *
 * Derived from `page.route.id`, which is already base-independent
 * (`/docs/solvers`, never `/basin/...`), so there's no `base` prefix to
 * strip, unlike `page.url.pathname`.
 */
export function activeSection(routeId: string | null): string {
    return (routeId ?? "").split("/")[1] ?? "";
}
