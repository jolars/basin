<script lang="ts">
    import IconCopy from "~icons/lucide/copy";
    import IconCheck from "~icons/lucide/check";

    // Tabbed citation block for the docs pages: the same reference in a
    // human-readable form and the two BibTeX dialects people actually paste
    // into a manuscript. The panels are plain <pre> text rather than shiki
    // output because shiki only runs over fenced code in `.svx` (see
    // svelte.config.js), and a copy button needs the raw string anyway.

    const APA = `Larsson, J. (2026). Basin: Efficient and Extensible Numerical \
Optimization in Rust (arXiv:2608.11279). arXiv. \
https://doi.org/10.48550/arXiv.2608.11279`;

    const BIBTEX = `@misc{larsson2026basin,
  title         = {Basin: Efficient and Extensible Numerical Optimization in {{Rust}}},
  shorttitle    = {Basin},
  author        = {Larsson, Johan},
  year          = {2026},
  month         = aug,
  number        = {arXiv:2608.11279},
  eprint        = {2608.11279},
  primaryclass  = {cs.LG},
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2608.11279},
  archiveprefix = {arXiv}
}`;

    const BIBLATEX = `@online{larsson2026basin,
  title       = {Basin: Efficient and Extensible Numerical Optimization in {{Rust}}},
  shorttitle  = {Basin},
  author      = {Larsson, Johan},
  date        = {2026-08-11},
  eprint      = {2608.11279},
  eprinttype  = {arXiv},
  eprintclass = {cs.LG},
  doi         = {10.48550/arXiv.2608.11279},
  pubstate    = {prepublished}
}`;

    type Format = { id: string; label: string; text: string };

    const FORMATS: Format[] = [
        { id: "apa", label: "APA", text: APA },
        { id: "bibtex", label: "BibTeX", text: BIBTEX },
        { id: "biblatex", label: "BibLaTeX", text: BIBLATEX },
    ];

    let active = $state(0);
    let copied = $state(false);
    let timer: ReturnType<typeof setTimeout> | undefined;
    let tabs: HTMLButtonElement[] = $state([]);

    async function copy() {
        try {
            await navigator.clipboard.writeText(FORMATS[active].text);
            copied = true;
            clearTimeout(timer);
            timer = setTimeout(() => (copied = false), 1500);
        } catch {
            // Clipboard access can be denied (insecure context, permissions);
            // the text stays selectable, so there is nothing to recover from.
        }
    }

    // Roving-tabindex keyboard support: only the active tab is in the tab
    // order, and arrow keys move between tabs (WAI-ARIA tabs pattern).
    function onkeydown(event: KeyboardEvent) {
        const delta =
            event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
        if (delta === 0) return;
        event.preventDefault();
        active = (active + delta + FORMATS.length) % FORMATS.length;
        copied = false;
        tabs[active]?.focus();
    }
</script>

<div
    class="not-prose my-6 rounded-lg border border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-900"
>
    <div
        class="flex items-center justify-between gap-2 border-b border-stone-200 px-2 dark:border-stone-800"
    >
        <div role="tablist" aria-label="Citation format" class="flex gap-1">
            {#each FORMATS as format, i (format.id)}
                <button
                    bind:this={tabs[i]}
                    type="button"
                    role="tab"
                    id="cite-tab-{format.id}"
                    aria-selected={active === i}
                    aria-controls="cite-panel-{format.id}"
                    tabindex={active === i ? 0 : -1}
                    {onkeydown}
                    onclick={() => {
                        active = i;
                        copied = false;
                    }}
                    class="-mb-px border-b-2 px-3 py-2 text-sm transition-colors {active ===
                    i
                        ? 'border-stone-800 font-medium text-stone-900 dark:border-stone-200 dark:text-stone-100'
                        : 'border-transparent text-stone-500 hover:text-stone-900 dark:text-stone-400 dark:hover:text-stone-100'}"
                >
                    {format.label}
                </button>
            {/each}
        </div>

        <button
            type="button"
            onclick={copy}
            aria-label="Copy citation"
            class="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-100"
        >
            {#if copied}
                <IconCheck width="14" height="14" aria-hidden="true" />
                Copied
            {:else}
                <IconCopy width="14" height="14" aria-hidden="true" />
                Copy
            {/if}
        </button>
    </div>

    {#each FORMATS as format, i (format.id)}
        <div
            role="tabpanel"
            id="cite-panel-{format.id}"
            aria-labelledby="cite-tab-{format.id}"
            hidden={active !== i}
            tabindex="0"
        >
            <pre
                class="overflow-x-auto px-4 py-3 text-sm text-stone-700 dark:text-stone-300"><code
                    class="font-mono {format.id === 'apa'
                        ? 'whitespace-pre-wrap'
                        : ''}">{format.text}</code
                ></pre>
        </div>
    {/each}
</div>
