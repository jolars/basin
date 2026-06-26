<script lang="ts">
    import { resolve } from "$app/paths";
    import type { Pathname } from "$app/types";
    import Seo from "$lib/Seo.svelte";

    // The three-axis overview. Each card links to its subpage when live.
    const axes: {
        title: string;
        status: string;
        href: Pathname;
        body: string;
    }[] = [
        {
            title: "Backends",
            status: "Live",
            href: "/benchmarks/backends/",
            body: "A curated set of solver + problem pairs across Vec, nalgebra, ndarray, and faer: isolating the cost of the linear-algebra layer, and showing where a backend can't run a solver at all.",
        },
        {
            title: "Solvers",
            status: "Live",
            href: "/benchmarks/solvers/",
            body: "Head-to-head runs of GD, Nelder–Mead, BFGS, L-BFGS, and CMA-ES on Rosenbrock from six starting points: suboptimality against wall-clock time under a fixed time budget per run.",
        },
        {
            title: "Competitors",
            status: "Live",
            href: "/benchmarks/competitors/",
            body: "basin against established crates such as argmin on matched problems: suboptimality against wall-clock time, since the implementations differ.",
        },
    ];
</script>

<Seo
    title="Basin—benchmarks"
    description="Benchmarks for the Basin optimization library, along three axes: linear-algebra backends, solver families, and competing crates such as argmin."
/>

<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <h1 class="text-3xl md:text-4xl font-semibold tracking-tight">
        Benchmarks
    </h1>
    <p class="mt-4 max-w-2xl text-slate-600 dark:text-slate-300">
        Basin's benchmark suite is built along three axes: <strong
            >backends</strong
        >,
        <strong>solvers</strong>, and
        <strong>competitors</strong>. Pick one to drill in.
    </p>

    <div class="mt-10 grid gap-6 sm:grid-cols-3">
        {#each axes as axis}
            {@const Tag = axis.href ? "a" : "div"}
            <svelte:element
                this={Tag}
                href={axis.href ? resolve(axis.href) : undefined}
                class="block rounded-xl border border-slate-200 dark:border-slate-800 p-5 {axis.href
                    ? 'transition-colors hover:border-slate-300 dark:hover:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-800/40'
                    : ''}"
            >
                <div class="flex items-center justify-between gap-2">
                    <h2 class="font-semibold">{axis.title}</h2>
                    <span
                        class="text-xs font-mono uppercase tracking-widest {axis.status ===
                        'Live'
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-slate-400 dark:text-slate-500'}"
                    >
                        {axis.status}
                    </span>
                </div>
                <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
                    {axis.body}
                </p>
                {#if axis.href}
                    <p
                        class="mt-3 text-sm font-medium text-indigo-600 dark:text-indigo-400"
                    >
                        View benchmarks →
                    </p>
                {/if}
            </svelte:element>
        {/each}
    </div>
</section>
