<script lang="ts">
    import { resolve } from "$app/paths";
    import type { Pathname } from "$app/types";
    import Seo from "$lib/Seo.svelte";

    // The three-axis overview. Each card links to its subpage.
    const axes: {
        title: string;
        href: Pathname;
        body: string;
    }[] = [
        {
            title: "Backends",
            href: "/benchmarks/backends/",
            body: "A curated set of solver and problem pairs across the Vec, nalgebra, ndarray, and faer backends.",
        },
        {
            title: "Solvers",
            href: "/benchmarks/solvers/",
            body: "Head-to-head runs of optimizers on different problems, showing suboptimality against wall-clock time under a fixed time budget per run.",
        },
        {
            title: "Competitors",
            href: "/benchmarks/competitors/",
            body: "Basin compared against established crates such as argmin on matched problems.",
        },
    ];
</script>

<Seo
    title="Benchmarks – Basin"
    description="Benchmarks for the Basin optimization library, along three axes: linear-algebra backends, solver families, and competing crates such as argmin."
/>

<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <h1 class="text-3xl md:text-4xl font-semibold tracking-tight">
        Benchmarks
    </h1>
    <p class="mt-4 max-w-2xl text-stone-600 dark:text-stone-300">
        Basin's benchmark suite is divided into three parts: <strong
            >backends</strong
        >, <strong>solvers</strong>, and <strong>competitors</strong>.
    </p>

    <div class="mt-10 grid gap-6 sm:grid-cols-3">
        {#each axes as axis}
            {@const Tag = axis.href ? "a" : "div"}
            <svelte:element
                this={Tag}
                href={axis.href ? resolve(axis.href) : undefined}
                class="block rounded-xl border border-stone-200 dark:border-stone-800 p-5 {axis.href
                    ? 'transition-colors hover:border-stone-300 dark:hover:border-stone-600 hover:bg-stone-50 dark:hover:bg-stone-800/40'
                    : ''}"
            >
                <h2 class="font-semibold">{axis.title}</h2>
                <p class="mt-2 text-sm text-stone-600 dark:text-stone-300">
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
