<script lang="ts">
    import { resolve } from "$app/paths";
    import { page } from "$app/state";
    import { DOCS_LINKS } from "$lib/nav";
    import Seo from "$lib/Seo.svelte";

    let { children } = $props();

    // Exact-match active state, tolerant of a missing/extra trailing
    // slash so highlighting survives either URL form.
    const strip = (s: string) => s.replace(/\/+$/, "");
    let path = $derived(strip(page.url.pathname));

    // Per-doc-page <title>/description. The docs `.svx` pages set no
    // <svelte:head> of their own, so this layout is the single title
    // source for the whole section (one tag per page, no duplicates).
    // Keyed by `page.route.id`, which is base-independent (`/docs/solvers`,
    // never `/basin/...`) — unlike `resolve()` from `$app/paths`, which
    // returns a page-relative string (`../..`) under the default
    // `paths.relative`.
    const DOCS_META: Record<string, { title: string; description: string }> = {
        "/docs": {
            title: "Documentation — Basin",
            description:
                "Overview of Basin: a generic executor loop drives a solver over a state, calling the problem traits you implement.",
        },
        "/docs/getting-started": {
            title: "Getting started — Basin",
            description:
                "Install Basin and run your first solve: implement CostFunction, add a Gradient when needed, then drive a solver with the Executor.",
        },
        "/docs/solvers": {
            title: "Solvers — Basin",
            description:
                "Basin's solver catalogue — first-order, derivative-free, nonlinear least-squares, and evolutionary methods — and the backend each supports.",
        },
    };
    const FALLBACK_META = {
        title: "Documentation — Basin",
        description:
            "Documentation for Basin, a numerical optimization library for Rust.",
    };

    let meta = $derived(DOCS_META[page.route.id ?? ""] ?? FALLBACK_META);
</script>

<Seo title={meta.title} description={meta.description} />

<div
    class="max-w-screen-2xl mx-auto px-4 md:px-8 py-10 grid lg:grid-cols-[14rem_1fr] gap-10"
>
    <aside class="lg:sticky lg:top-20 lg:self-start">
        <p
            class="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3"
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
                        ? 'bg-slate-100 text-slate-900 font-medium dark:bg-slate-800 dark:text-slate-100'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-100 dark:hover:bg-slate-800'}"
                >
                    {link.label}
                </a>
            {/each}
        </nav>
    </aside>

    <div class="min-w-0">
        {@render children()}
    </div>
</div>
