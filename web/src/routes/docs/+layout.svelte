<script lang="ts">
    import { resolve } from "$app/paths";
    import { page } from "$app/state";
    import { DOCS_LINKS } from "#lib/nav.ts";
    import Seo from "#lib/Seo.svelte";
    import PageOutline from "#lib/docs/PageOutline.svelte";

    let { children } = $props();
    let content = $state<HTMLElement>();

    // Exact-match active state, tolerant of a missing/extra trailing
    // slash so highlighting survives either URL form.
    const strip = (s: string) => s.replace(/\/+$/, "");
    let path = $derived(strip(page.url.pathname));

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
            title: "Solvers – Basin",
            description:
                "Basin's solver catalog (first-order, derivative-free, nonlinear least-squares, and evolutionary methods) and the backend each supports.",
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

<div class="nav-container docs-layout py-10 grid gap-8">
    <aside
        class="lg:max-w-48 lg:sticky lg:top-20 lg:self-start lg:max-h-[calc(100dvh-7rem)] lg:overflow-y-auto"
    >
        <p
            class="text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500 mb-3"
        >
            Documentation
        </p>
        <nav class="flex flex-col gap-1 text-sm">
            {#each DOCS_LINKS as link}
                <a
                    href={resolve(link.href)}
                    aria-current={path === strip(resolve(link.href))
                        ? "page"
                        : undefined}
                    class="px-3 py-1.5 rounded-md transition-colors {path ===
                    strip(resolve(link.href))
                        ? 'bg-stone-100 text-stone-900 font-medium dark:bg-stone-800 dark:text-stone-100'
                        : 'text-stone-600 hover:text-stone-900 hover:bg-stone-100 dark:text-stone-400 dark:hover:text-stone-100 dark:hover:bg-stone-800'}"
                >
                    {link.label}
                </a>
            {/each}
        </nav>
    </aside>

    <div bind:this={content} class="docs-content min-w-0">
        {@render children()}
    </div>
    <aside
        class="hidden lg:block sticky top-20 self-start max-h-[calc(100dvh-7rem)] overflow-y-auto"
    >
        <PageOutline {content} />
    </aside>
</div>

<style>
    @media (min-width: 64rem) {
        .docs-layout {
            /* Equal side columns keep the article centered in the viewport. */
            grid-template-columns: minmax(0, 1fr) minmax(0, 44rem) minmax(
                    0,
                    1fr
                );
        }
    }

    .docs-content :global(h2),
    .docs-content :global(h3) {
        scroll-margin-top: 6rem;
    }

    @media (min-width: 64rem) and (max-width: 79.999rem) {
        .docs-layout {
            grid-template-columns: 11rem minmax(0, 1fr) 11rem;
        }
    }
</style>
