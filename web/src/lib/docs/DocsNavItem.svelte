<script lang="ts">
    import { page } from "$app/state";
    import { type DocsNavItem, resolvePath } from "#lib/nav.ts";
    import DocsNavEntry from "./DocsNavItem.svelte";
    import IconChevronRight from "~icons/lucide/chevron-right";

    let { item }: { item: DocsNavItem } = $props();

    const strip = (path: string) => path.replace(/\/+$/, "");

    function isActive(entry: DocsNavItem): boolean {
        if ("children" in entry) {
            return entry.children.some(isActive);
        }
        return (
            strip(page.route.id ?? "") === strip(`/${entry.href}`) &&
            (entry.fragment === undefined ||
                page.url.hash === (entry.fragment ? `#${entry.fragment}` : ""))
        );
    }

    let active = $derived(isActive(item));
</script>

<li>
    {#if "children" in item}
        <details open={active}>
            <summary
                class="flex cursor-pointer list-none items-center gap-1 rounded-md px-2 py-1.5 text-stone-700 transition-colors hover:bg-stone-100 dark:text-stone-300 dark:hover:bg-stone-800"
            >
                <IconChevronRight
                    class="size-3.5 shrink-0 transition-transform"
                />
                <span class:font-medium={active}>{item.label}</span>
            </summary>
            <ul
                class="ml-3 mt-1 space-y-1 border-l border-stone-200 pl-2 dark:border-stone-800"
            >
                {#each item.children as child (child.label)}
                    <DocsNavEntry item={child} />
                {/each}
            </ul>
        </details>
    {:else}
        <a
            href={`${resolvePath(item.href)}${item.fragment ? `#${item.fragment}` : ""}`}
            aria-current={active
                ? item.fragment
                    ? "location"
                    : "page"
                : undefined}
            class="block rounded-md px-3 py-1.5 transition-colors {active
                ? 'bg-stone-100 font-medium text-stone-900 dark:bg-stone-800 dark:text-stone-100'
                : 'text-stone-600 hover:bg-stone-100 hover:text-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-100'}"
        >
            {item.label}
        </a>
    {/if}
</li>

<style>
    summary::-webkit-details-marker {
        display: none;
    }

    details[open] > summary :global(svg) {
        transform: rotate(90deg);
    }
</style>
