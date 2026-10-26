// Shared navigation model. `href` is an app path without a leading slash.
// Calling `resolve()` prefixes the configured base, so links also work in
// deployments under a URL prefix. `section` is the first path segment, used
// for active-state matching across a whole section (e.g. any `/guide/*` page
// lights up the "Guide" link).

import { resolve } from "$app/paths";
import type { Path, ResolvedPathname } from "$app/types";

// A single-path signature avoids TypeScript's limit on Kit's generated union
// of argument tuples while preserving validation against the site's paths.
export const resolvePath = resolve as (path: Path) => ResolvedPathname;

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
    { label: "Guide", href: "guide/getting-started/", section: "guide" },
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

/** Sidebar links for the guide section. */
export const DOCS_LINKS: NavLink[] = [
    { label: "Overview", href: "guide/", section: "guide" },
    {
        label: "Getting started",
        href: "guide/getting-started/",
        section: "guide",
    },
    {
        label: "Migrating from Argmin",
        href: "guide/migrating-from-argmin/",
        section: "guide",
    },
    {
        label: "Migrating to Basin 2.0",
        href: "guide/migrating-to-v2/",
        section: "guide",
    },
    {
        label: "Migrating to Basin 3.0",
        href: "guide/migrating-to-v3/",
        section: "guide",
    },
];

/** Sidebar links grouped by solver family. */
export const SOLVER_LINKS: DocsNavItem[] = [
    { label: "Choosing a solver", href: "guide/solvers/", fragment: "" },
    {
        label: "First-order",
        children: [
            {
                label: "Gradient descent",
                href: "guide/solvers/first-order/gradient-descent/",
            },
            {
                label: "Nonlinear CG",
                href: "guide/solvers/first-order/nonlinear-cg/",
            },
            {
                label: "Stochastic gradient descent",
                href: "guide/solvers/first-order/sgd/",
            },
            {
                label: "Projected gradient descent",
                href: "guide/solvers/first-order/projected-gradient-descent/",
            },
        ],
    },
    {
        label: "Quasi-Newton",
        children: [
            {
                label: "BFGS",
                href: "guide/solvers/quasi-newton/bfgs/",
            },
            {
                label: "L-BFGS",
                href: "guide/solvers/quasi-newton/l-bfgs/",
            },
            {
                label: "L-BFGS-B",
                href: "guide/solvers/quasi-newton/l-bfgs-b/",
            },
        ],
    },
    {
        label: "Derivative-free",
        children: [
            {
                label: "Nelder–Mead",
                href: "guide/solvers/derivative-free/nelder-mead/",
            },
            {
                label: "MADS",
                href: "guide/solvers/derivative-free/mads/",
            },
            {
                label: "NEWUOA",
                href: "guide/solvers/derivative-free/newuoa/",
            },
            {
                label: "BOBYQA",
                href: "guide/solvers/derivative-free/bobyqa/",
            },
            {
                label: "LINCOA",
                href: "guide/solvers/derivative-free/lincoa/",
            },
            {
                label: "COBYLA",
                href: "guide/solvers/derivative-free/cobyla/",
            },
            {
                label: "Solis–Wets",
                href: "guide/solvers/derivative-free/solis-wets/",
            },
        ],
    },
    {
        label: "Trust-region",
        children: [
            {
                label: "Trust-region Newton",
                href: "guide/solvers/trust-region/trust-region-newton/",
            },
        ],
    },
    {
        label: "Nonlinear least squares",
        children: [
            {
                label: "Gauss–Newton",
                href: "guide/solvers/nonlinear-least-squares/gauss-newton/",
            },
            {
                label: "Levenberg–Marquardt",
                href: "guide/solvers/nonlinear-least-squares/levenberg-marquardt/",
            },
            {
                label: "Levenberg–Marquardt QR",
                href: "guide/solvers/nonlinear-least-squares/levenberg-marquardt-qr/",
            },
            {
                label: "Bounded Levenberg–Marquardt",
                href: "guide/solvers/nonlinear-least-squares/bounded-levenberg-marquardt/",
            },
            {
                label: "Trust-region reflective",
                href: "guide/solvers/nonlinear-least-squares/trust-region-reflective/",
            },
        ],
    },
    {
        label: "Global and stochastic",
        children: [
            {
                label: "DIRECT",
                href: "guide/solvers/global-and-stochastic/direct/",
            },
            {
                label: "Globalized bounded Nelder–Mead",
                href: "guide/solvers/global-and-stochastic/gbnm/",
            },
            {
                label: "Simulated annealing",
                href: "guide/solvers/global-and-stochastic/simulated-annealing/",
            },
            {
                label: "CMA-ES",
                href: "guide/solvers/global-and-stochastic/cma-es/",
            },
            {
                label: "Bounded CMA-ES",
                href: "guide/solvers/global-and-stochastic/bounded-cma-es/",
            },
            {
                label: "Differential evolution",
                href: "guide/solvers/global-and-stochastic/de/",
            },
            {
                label: "Particle swarm",
                href: "guide/solvers/global-and-stochastic/global-best-pso/",
            },
            {
                label: "Random search",
                href: "guide/solvers/global-and-stochastic/random-search/",
            },
            {
                label: "Steady-state genetic algorithm",
                href: "guide/solvers/global-and-stochastic/ssga/",
            },
            {
                label: "Basin-hopping",
                href: "guide/solvers/global-and-stochastic/basin-hopping/",
            },
        ],
    },
    {
        label: "Memetic",
        children: [
            {
                label: "CMA-ES with local search",
                href: "guide/solvers/memetic/cma-inject/",
            },
            {
                label: "Bounded CMA-ES with local search",
                href: "guide/solvers/memetic/bounded-cma-inject/",
            },
            {
                label: "DE with local search",
                href: "guide/solvers/memetic/de-inject/",
            },
            {
                label: "Local-search chains",
                href: "guide/solvers/memetic/ma-ls-ch/",
            },
            {
                label: "CMA-ES chains",
                href: "guide/solvers/memetic/ma-ls-ch-cma/",
            },
            {
                label: "Solis–Wets chains",
                href: "guide/solvers/memetic/ma-ls-ch-sw/",
            },
        ],
    },
    {
        label: "General constrained",
        children: [
            {
                label: "SLSQP",
                href: "guide/solvers/general-constrained/slsqp/",
            },
            {
                label: "Barrier method",
                href: "guide/solvers/general-constrained/barrier-method/",
            },
            {
                label: "Augmented Lagrangian",
                href: "guide/solvers/general-constrained/augmented-lagrangian-method/",
            },
        ],
    },
    {
        label: "Scalar minimization",
        children: [
            {
                label: "Brent",
                href: "guide/solvers/scalar-minimization/brent/",
            },
            {
                label: "Brent with derivatives",
                href: "guide/solvers/scalar-minimization/brent-derivative/",
            },
            {
                label: "Golden-section search",
                href: "guide/solvers/scalar-minimization/golden-section/",
            },
        ],
    },
    {
        label: "Root finding",
        children: [
            {
                label: "Bisection",
                href: "guide/solvers/root-finding/bisection/",
            },
            {
                label: "Brent",
                href: "guide/solvers/root-finding/brent/",
            },
            {
                label: "Secant",
                href: "guide/solvers/root-finding/secant/",
            },
            {
                label: "Newton",
                href: "guide/solvers/root-finding/newton/",
            },
            {
                label: "Halley",
                href: "guide/solvers/root-finding/halley/",
            },
            {
                label: "TOMS 748",
                href: "guide/solvers/root-finding/toms748/",
            },
        ],
    },
];

/**
 * The active section for a route. Returns the first path segment
 * (`'guide'`, `'visualizer'`, …) or `''` for the landing page.
 *
 * Derived from `page.route.id`, which is already base-independent
 * (`/guide/solvers`, never `/basin/...`), so there's no `base` prefix to
 * strip, unlike `page.url.pathname`.
 */
export function activeSection(routeId: string | null): string {
    return (routeId ?? "").split("/")[1] ?? "";
}
