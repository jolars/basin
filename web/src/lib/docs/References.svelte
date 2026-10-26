<script lang="ts">
    import {
        formatAuthors,
        formatEdition,
        getReference,
    } from "./references.ts";

    let { entries }: { entries: string[] } = $props();
</script>

<section aria-labelledby="references">
    <h2 id="references">References</h2>
    <ol>
        {#each entries as id (id)}
            {@const reference = getReference(id)}
            <li id={`ref-${id}`}>
                {formatAuthors(reference.author)}
                ({reference.issued["date-parts"][0][0]}).
                {#if reference.type === "webpage"}
                    <cite>{reference.title}</cite>. {reference.publisher}.
                {:else if reference.type === "book"}
                    <cite>{reference.title}</cite>
                    {#if reference.edition}
                        ({formatEdition(reference.edition)})
                    {/if}.
                    {reference.publisher}.
                {:else if reference.type === "chapter"}
                    {reference.title}. In
                    <cite>{reference["container-title"]}</cite>
                    ({#if reference.edition}{formatEdition(reference.edition)},
                    {/if}pp. {reference.page?.replaceAll("-", "–")}).
                    {reference.publisher}.
                {:else if reference.type === "paper-conference"}
                    {reference.title}. In
                    <cite>{reference["container-title"]}</cite>
                    ({#if reference.volume}Vol. {reference.volume},
                    {/if}pp. {reference.page?.replaceAll("-", "–")}).
                    {reference.publisher}.
                {:else if reference.type === "report"}
                    <cite>{reference.title}</cite>. Report {reference.number}.
                    {reference.publisher}.
                {:else}
                    {reference.title}.
                    <cite>{reference["container-title"]}</cite>,
                    {reference.volume}{#if reference.issue}({reference.issue}){/if},
                    {reference.page?.replaceAll("-", "–")}.
                {/if}
                {#if reference.DOI}
                    <a href={`https://doi.org/${reference.DOI}`}
                        >doi:{reference.DOI}</a
                    >.
                {:else if reference.URL}
                    <a href={reference.URL}>Full text</a>.
                {/if}
            </li>
        {/each}
    </ol>
</section>
