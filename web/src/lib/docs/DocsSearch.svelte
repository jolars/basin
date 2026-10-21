<script lang="ts">
    import { resolve } from "$app/paths";
    import IconSearch from "~icons/lucide/search";
    import IconClose from "~icons/lucide/x";
    import type { documents as searchDocuments } from "./search";

    let dialog: HTMLDialogElement;
    let query = $state("");
    let documents = $state<typeof searchDocuments>([]);
    let loading = $state(false);
    let failed = $state(false);
    const results = $derived.by(() => {
        const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
        if (!terms.length) return [];
        return documents
            .filter((doc) =>
                terms.every((term) =>
                    `${doc.title} ${doc.text}`.toLowerCase().includes(term),
                ),
            )
            .map((doc) => {
                const start = Math.max(
                    0,
                    doc.text.toLowerCase().indexOf(terms[0]) - 60,
                );
                return {
                    ...doc,
                    score: terms.filter((term) =>
                        doc.title.toLowerCase().includes(term),
                    ).length,
                    excerpt: `${start ? "…" : ""}${doc.text.slice(start, start + 200)}${doc.text.length > start + 200 ? "…" : ""}`,
                };
            })
            .sort((a, b) => b.score - a.score);
    });

    async function open() {
        dialog.showModal();
        dialog.querySelector("input")?.focus();
        if (documents.length) return;
        loading = true;
        failed = false;
        try {
            documents = (await import("./search")).documents;
        } catch {
            failed = true;
        } finally {
            loading = false;
        }
    }
</script>

<button
    type="button"
    onclick={open}
    aria-label="Search documentation"
    aria-haspopup="dialog"
    class="inline-flex items-center gap-2 rounded-md p-2 sm:px-3 sm:w-28 lg:w-48 sm:border border-stone-200 dark:border-stone-700 text-sm text-stone-500 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
>
    <IconSearch width="18" height="18" aria-hidden="true" />
    <span class="hidden sm:inline">Search</span>
</button>

<dialog
    bind:this={dialog}
    aria-labelledby="docs-search-title"
    class="fixed inset-0 m-auto w-[calc(100%-2rem)] max-w-xl max-h-[80dvh] rounded-xl border border-stone-200 bg-white p-0 text-stone-900 shadow-xl backdrop:bg-black/40 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100"
>
    <div class="flex items-center justify-between px-4 pt-4">
        <h2 id="docs-search-title" class="font-semibold">
            Search documentation
        </h2>
        <button
            type="button"
            onclick={() => dialog.close()}
            aria-label="Close search"
            class="rounded-md p-2 hover:bg-stone-100 dark:hover:bg-stone-800"
            ><IconClose width="18" height="18" aria-hidden="true" /></button
        >
    </div>
    <div class="p-4">
        <label
            class="flex items-center gap-2 rounded-md border border-stone-300 px-3 focus-within:ring-2 focus-within:ring-stone-400 dark:border-stone-700"
        >
            <IconSearch width="18" height="18" aria-hidden="true" />
            <input
                bind:value={query}
                type="search"
                aria-label="Search documentation"
                placeholder="Search"
                class="min-w-0 w-full bg-transparent py-3 outline-none"
            />
        </label>
        <p
            role="status"
            class="mt-3 text-sm text-stone-500 dark:text-stone-400"
        >
            {#if loading}Loading search…{:else if failed}Search could not load.
                Close and reopen to try again.{:else if !query.trim()}Search the
                guides, solvers, and migration docs.{:else if !results.length}No
                results found. Try another search.{:else}{results.length}
                {results.length === 1 ? "result" : "results"}{/if}
        </p>
        <ul class="mt-3 space-y-1">
            {#each results as result (result.href)}
                <li>
                    <a
                        href={resolve(result.href)}
                        onclick={() => dialog.close()}
                        class="block rounded-md p-3 hover:bg-stone-100 focus-visible:bg-stone-100 dark:hover:bg-stone-800 dark:focus-visible:bg-stone-800"
                        ><span class="font-medium">{result.title}</span>
                        <p
                            class="mt-1 text-sm text-stone-500 dark:text-stone-400"
                        >
                            {result.excerpt}
                        </p></a
                    >
                </li>
            {/each}
        </ul>
    </div>
</dialog>
