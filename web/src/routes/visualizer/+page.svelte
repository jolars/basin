<script lang="ts">
    import { onMount } from "svelte";
    import type { Component } from "svelte";
    import Seo from "$lib/Seo.svelte";

    // The visualizer is browser-only: it loads the wasm module and drives a
    // requestAnimationFrame loop, so it can't render on the server. The
    // heavy component lives in `$lib/Visualizer.svelte` and is pulled in via
    // a dynamic import on mount, so this route stays SSR-safe and
    // prerenders a real <title>/og: head for crawlers (see `+page.ts`),
    // while the wasm module is never imported on the server.
    let Visualizer = $state<Component | null>(null);

    onMount(async () => {
        Visualizer = (await import("$lib/Visualizer.svelte")).default;
    });
</script>

<Seo
    title="Solver Visualizer – Basin"
    description="Live wasm-driven 2D optimization trajectories from the Basin Rust library."
/>

{#if Visualizer}
    <Visualizer />
{:else}
    <section
        class="min-h-[calc(100vh-8rem)] max-w-screen-2xl w-full mx-auto px-4 md:px-8 py-6 flex flex-col gap-6"
    >
        <header>
            <h1 class="text-2xl md:text-3xl font-semibold tracking-tight">
                Solver visualizer
            </h1>
            <p class="text-stone-600 dark:text-stone-400 text-sm mt-1">
                Live wasm-driven 2D trajectories. Click on the contour to reset
                the start point.
            </p>
        </header>
        <p class="text-stone-500 dark:text-stone-400">Loading visualizer…</p>
    </section>
{/if}
