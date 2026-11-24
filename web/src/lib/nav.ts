export type NavLink = {
    label: string;
    href: string;
};

export type DocsNavItem = NavLink | { label: string; children: DocsNavItem[] };

// Keep trailing slashes so links match the canonical URLs.

/** Sidebar links for the benchmark guides. */
export const BENCHMARK_LINKS: DocsNavItem[] = [
    {
        label: "Competitors",
        children: [
            {
                label: "Smooth unconstrained optimization",
                href: "guide/benchmarks/smooth-unconstrained/",
            },
            {
                label: "Constrained optimization",
                href: "guide/benchmarks/constrained/",
            },
            {
                label: "Derivative-free and global search",
                href: "guide/benchmarks/derivative-free-and-global/",
            },
            {
                label: "Nonlinear least squares",
                href: "guide/benchmarks/least-squares/",
            },
        ],
    },
    { label: "Backends", href: "guide/benchmarks/backends/" },
];

/** Sidebar links for the guide section. */
export const DOCS_LINKS: NavLink[] = [
    { label: "Overview", href: "guide/" },
    { label: "Getting started", href: "guide/getting-started/" },
];

/** Sidebar links grouped by solver family. */
export const SOLVER_LINKS: DocsNavItem[] = [
    { label: "Choosing a solver", href: "guide/solvers/" },
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
