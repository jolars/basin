<script lang="ts">
    import { page } from "$app/state";
    import { DOCS_LINKS, SOLVER_LINKS } from "#lib/nav.ts";
    import Seo from "#lib/Seo.svelte";
    import DocsLayout from "#lib/docs/DocsLayout.svelte";

    let { children } = $props();

    // Per-doc-page <title>/description. The docs `.svx` pages set no
    // <svelte:head> of their own, so this layout is the single title
    // source for the whole section (one tag per page, no duplicates).
    // Keyed by `page.route.id`, which is base-independent (`/docs/solvers`,
    // never `/basin/...`), unlike `resolve()` from `$app/paths`, which
    // returns a page-relative string (`../..`) under the default
    // `paths.relative`.
    const DOCS_META: Record<string, { title: string; description: string }> = {
        "/docs": {
            title: "Documentation – Basin",
            description:
                "Learn how to define an optimization problem, choose a solver, and run it with Basin.",
        },
        "/docs/getting-started": {
            title: "Getting Started – Basin",
            description:
                "Install Basin and run your first solve: implement CostFunction, add a Gradient when needed, then drive a solver with the Executor.",
        },
        "/docs/migrating-from-argmin": {
            title: "Migrating from Argmin – Basin",
            description:
                "Migrate an Argmin 0.11 application to Basin, including executors, solvers, typed errors, finite differences, bounds, observers, and backends.",
        },
        "/docs/migrating-to-v2": {
            title: "Migrating to Basin 2.0 – Basin",
            description:
                "Upgrade from Basin 1.x to 2.0: migrate problem constraints, progress states, termination reports, solver settings, and checkpoints.",
        },
        "/docs/solvers": {
            title: "Choosing a Solver – Basin",
            description:
                "Choose a Basin solver based on your derivatives, constraints, and problem structure, then browse the full solver catalog.",
        },
        "/docs/solvers/first-order/gradient-descent": {
            title: "Gradient Descent – Basin",
            description:
                "When to use gradient descent, how fixed steps, line searches, and heavy-ball momentum work, and how to run it in Basin, with references.",
        },
        "/docs/solvers/first-order/nonlinear-cg": {
            title: "Nonlinear Conjugate Gradient – Basin",
            description:
                "When to use nonlinear conjugate gradient, how Hager–Zhang and Polak–Ribière+ updates work, and how to run it in Basin, with references.",
        },
        "/docs/solvers/first-order/sgd": {
            title: "Stochastic Gradient Descent – Basin",
            description:
                "When to use SGD, how mini-batch sampling, momentum, and cost refreshes work, and how to run it in Basin, with references.",
        },
        "/docs/solvers/first-order/projected-gradient-descent": {
            title: "Projected Gradient Descent – Basin",
            description:
                "When to use projected gradient descent, how box bounds and projected stationarity work, and how to run it in Basin, with references.",
        },
        "/docs/solvers/quasi-newton/l-bfgs": {
            title: "L-BFGS – Basin",
            description:
                "When to use L-BFGS, how limited-memory quasi-Newton optimization works, and how to configure and run it in Basin, with original references.",
        },
        "/docs/migrating-to-v3": {
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
