<script lang="ts">
    import { tick } from "svelte";
    import { page } from "$app/state";

    let { content }: { content: HTMLElement | undefined } = $props();
    let headings = $state<HTMLElement[]>([]);
    let active = $state("");

    function updateActive() {
        let current = headings[0]?.id ?? "";
        for (const heading of headings) {
            if (heading.getBoundingClientRect().top > 96) break;
            current = heading.id;
        }
        active = current;
    }

    $effect(() => {
        page.url.pathname;
        const element = content;
        let cancelled = false;
        void tick().then(() => {
            if (cancelled) return;
            headings = Array.from(
                element?.querySelectorAll<HTMLElement>("h2[id], h3[id]") ?? [],
            );
            updateActive();
        });
        return () => {
            cancelled = true;
        };
    });
</script>

<svelte:window onscroll={updateActive} onresize={updateActive} />

{#if headings.length}
    <nav
        aria-label="On this page"
        class="border-l border-stone-200 dark:border-stone-800 pl-4 text-sm"
    >
        <p class="font-semibold text-stone-900 dark:text-stone-100 mb-3">
            On this page
        </p>
        <ul class="space-y-2">
            {#each headings as heading (heading.id)}
                <li class:pl-3={heading.tagName === "H3"}>
                    <a
                        href={`#${encodeURIComponent(heading.id)}`}
                        aria-current={active === heading.id
                            ? "location"
                            : undefined}
                        class="block transition-colors hover:text-stone-900 dark:hover:text-stone-100 {active ===
                        heading.id
                            ? 'text-stone-900 dark:text-stone-100 font-medium'
                            : 'text-stone-500 dark:text-stone-400'}"
                        >{heading.textContent}</a
                    >
                </li>
            {/each}
        </ul>
    </nav>
{/if}
