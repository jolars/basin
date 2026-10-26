<script lang="ts">
    import { page } from "$app/state";
    import { DOCS_LINKS, SOLVER_LINKS } from "#lib/nav.ts";
    import Seo from "#lib/Seo.svelte";
    import DocsLayout from "#lib/docs/DocsLayout.svelte";

    let { children } = $props();

    // Per-doc-page <title>/description. The docs `.svx` pages set no
    // <svelte:head> of their own, so this layout is the single title
    // source for the whole section (one tag per page, no duplicates).
    // Keyed by `page.route.id`, which is base-independent (`/guide/solvers`,
    // never `/basin/...`), unlike `resolve()` from `$app/paths`, which
    // returns a page-relative string (`../..`) under the default
    // `paths.relative`.
    const DOCS_META: Record<string, { title: string; description: string }> = {
        "/guide": {
            title: "Documentation – Basin",
            description:
                "Learn how to define an optimization problem, choose a solver, and run it with Basin.",
        },
        "/guide/getting-started": {
            title: "Getting started – Basin",
            description:
                "Install Basin and run your first solve: implement CostFunction, add a Gradient when needed, then drive a solver with the Executor.",
        },
        "/guide/migrating-from-argmin": {
            title: "Migrating from Argmin – Basin",
            description:
                "Migrate an Argmin 0.11 application to Basin, including executors, solvers, typed errors, finite differences, bounds, observers, and backends.",
        },
        "/guide/migrating-to-v2": {
            title: "Migrating to Basin 2.0 – Basin",
            description:
                "Upgrade from Basin 1.x to 2.0: migrate problem constraints, progress states, termination reports, solver settings, and checkpoints.",
        },
        "/guide/solvers": {
            title: "Choosing a solver – Basin",
            description:
                "Choose a Basin solver based on your derivatives, constraints, and problem structure, then browse the full solver catalog.",
        },
        "/guide/solvers/first-order/gradient-descent": {
            title: "Gradient descent – Basin",
            description:
                "When to use gradient descent, how fixed steps, line searches, and heavy-ball momentum work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/first-order/nonlinear-cg": {
            title: "Nonlinear conjugate gradient – Basin",
            description:
                "When to use nonlinear conjugate gradient, how Hager–Zhang and Polak–Ribière+ updates work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/first-order/sgd": {
            title: "Stochastic gradient descent – Basin",
            description:
                "When to use SGD, how mini-batch sampling, momentum, and cost refreshes work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/first-order/projected-gradient-descent": {
            title: "Projected gradient descent – Basin",
            description:
                "When to use projected gradient descent, how box bounds and projected stationarity work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/quasi-newton/l-bfgs": {
            title: "L-BFGS – Basin",
            description:
                "When to use L-BFGS, how limited-memory quasi-Newton optimization works, and how to configure and run it in Basin, with original references.",
        },
        "/guide/solvers/derivative-free/nelder-mead": {
            title: "Nelder–Mead – Basin",
            description:
                "When to use Nelder–Mead, how simplex search and box projection work, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/derivative-free/mads": {
            title: "MADS – Basin",
            description:
                "When to use mesh adaptive direct search, how OrthoMADS polling and constraint barriers work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/derivative-free/newuoa": {
            title: "NEWUOA – Basin",
            description:
                "When to use NEWUOA, how quadratic interpolation and trust regions work, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/derivative-free/bobyqa": {
            title: "BOBYQA – Basin",
            description:
                "When to use BOBYQA, how quadratic models handle box bounds, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/derivative-free/lincoa": {
            title: "LINCOA – Basin",
            description:
                "When to use LINCOA, how quadratic models handle linear constraints, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/derivative-free/cobyla": {
            title: "COBYLA – Basin",
            description:
                "When to use COBYLA, how linear interpolation models handle constraints, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/derivative-free/solis-wets": {
            title: "Solis–Wets – Basin",
            description:
                "When to use Solis–Wets, how adaptive random local search works, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/direct": {
            title: "DIRECT – Basin",
            description:
                "When to use DIRECT, how deterministic box subdivision works, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/gbnm": {
            title: "Globalized bounded Nelder–Mead – Basin",
            description:
                "When to use globalized bounded Nelder–Mead, how simplex searches and probabilistic restarts work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/simulated-annealing": {
            title: "Simulated annealing – Basin",
            description:
                "When to use simulated annealing, how neighbors, Metropolis acceptance, cooling, and restarts work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/cma-es": {
            title: "CMA-ES – Basin",
            description:
                "When to use CMA-ES, how covariance adaptation and step-size control work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/bounded-cma-es": {
            title: "Bounded CMA-ES – Basin",
            description:
                "When to use bounded CMA-ES, how box repair and adaptive penalties work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/de": {
            title: "Differential evolution – Basin",
            description:
                "When to use differential evolution, how mutation and crossover work, and how to configure and run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/global-best-pso": {
            title: "Particle swarm – Basin",
            description:
                "When to use global-best particle swarm optimization, how velocity and boundary policies work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/random-search": {
            title: "Random search – Basin",
            description:
                "When to use random search, how uniform box sampling and elitist selection work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/ssga": {
            title: "Steady-state genetic algorithm – Basin",
            description:
                "When to use the steady-state genetic algorithm, how mating, crossover, mutation, and replacement work, and how to run it in Basin, with references.",
        },
        "/guide/solvers/global-and-stochastic/basin-hopping": {
            title: "Basin-hopping – Basin",
            description:
                "When to use basin-hopping, how local solves, Metropolis acceptance, and adaptive steps work, and how to run it in Basin, with references.",
        },
        "/guide/migrating-to-v3": {
            title: "Migrating to Basin 3.0 – Basin",
            description:
                "Upgrade from Basin 2.0 to 3.0: optional constraint declarations, standalone evaluation traits, and unconstrained solver behavior.",
        },
    };
    const FALLBACK_META = {
        title: "Documentation – Basin",
        description:
            "Documentation for Basin, a numerical optimization library for Rust.",
    };

    let meta = $derived(DOCS_META[page.route.id ?? ""] ?? FALLBACK_META);
</script>

<Seo title={meta.title} description={meta.description} />

<DocsLayout
    navigation={[
        { label: "Documentation", items: DOCS_LINKS },
        { label: "Solvers", items: SOLVER_LINKS },
    ]}
>
    {@render children()}
</DocsLayout>
