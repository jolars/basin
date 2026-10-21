<script lang="ts">
    import { page } from "$app/state";
    import {
        DOCS_LINKS,
        SOLVER_LINKS,
        type DocsNavItem as NavigationItem,
    } from "#lib/nav.ts";
    import Seo from "#lib/Seo.svelte";
    import DocsNavItem from "#lib/docs/DocsNavItem.svelte";
    import PageOutline from "#lib/docs/PageOutline.svelte";

    let { children } = $props();
    let content = $state<HTMLElement>();

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

{#snippet navigation(label: string, items: NavigationItem[])}
    <section>
        <h2
            class="text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500 mb-3"
        >
            {label}
        </h2>
        <nav aria-label={label} class="text-sm">
            <ul class="space-y-1">
                {#each items as item (item.label)}
                    <DocsNavItem {item} />
                {/each}
            </ul>
        </nav>
    </section>
{/snippet}

{#snippet sidebar()}
    <div class="space-y-8">
        {@render navigation("Documentation", DOCS_LINKS)}
        {@render navigation("Solvers", SOLVER_LINKS)}
    </div>
{/snippet}

<div class="nav-container docs-layout py-10 grid gap-8">
    <aside
        class="lg:max-w-64 lg:sticky lg:top-20 lg:self-start lg:max-h-[calc(100dvh-7rem)] lg:overflow-y-auto"
    >
        <div class="hidden lg:block">
            {@render sidebar()}
        </div>
        <details
            class="rounded-lg border border-stone-200 dark:border-stone-800 lg:hidden"
        >
            <summary class="cursor-pointer px-3 py-2 text-sm font-medium">
                Navigation
            </summary>
            <div class="px-2 pb-2">
                {@render sidebar()}
            </div>
        </details>
    </aside>

    <div bind:this={content} class="docs-content min-w-0">
        {@render children()}
    </div>
    <aside
        class="hidden xl:block sticky top-20 self-start max-h-[calc(100dvh-7rem)] overflow-y-auto"
    >
        <PageOutline {content} />
    </aside>
</div>

<style>
    @media (min-width: 64rem) and (max-width: 79.999rem) {
        .docs-layout {
            grid-template-columns: 14rem minmax(0, 1fr);
            padding-inline-end: 4rem;
        }
    }

    @media (min-width: 80rem) {
        .docs-layout {
            /* Equal side columns keep the article centered in the viewport. */
            grid-template-columns: minmax(0, 1fr) minmax(0, 44rem) minmax(
                    0,
                    1fr
                );
        }
    }

    .docs-content :global([id]) {
        scroll-margin-top: 6rem;
    }
</style>
