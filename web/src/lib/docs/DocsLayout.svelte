<script lang="ts">
    import "katex/dist/katex.min.css";
    import type { Snippet } from "svelte";
    import type { DocsNavItem as NavigationItem } from "#lib/nav.ts";
    import DocsNavItem from "./DocsNavItem.svelte";
    import PageOutline from "./PageOutline.svelte";

    let {
        children,
        navigation,
    }: {
        children: Snippet;
        navigation: { label: string; items: NavigationItem[] }[];
    } = $props();
    let content = $state<HTMLElement>();
</script>

{#snippet sidebar()}
    <div class="space-y-8">
        {#each navigation as group (group.label)}
            <section>
                <h2
                    class="text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500 mb-3"
                >
                    {group.label}
                </h2>
                <nav aria-label={group.label} class="text-sm">
                    <ul class="space-y-1">
                        {#each group.items as item (item.label)}
                            <DocsNavItem {item} />
                        {/each}
                    </ul>
                </nav>
            </section>
        {/each}
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

    .docs-content :global(.katex-display) {
        overflow-x: auto;
        overflow-y: hidden;
    }
</style>
