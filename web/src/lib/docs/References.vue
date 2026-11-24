<script setup lang="ts">
import { formatAuthors, formatEdition, getReference } from "./references.ts";

let { entries } = defineProps<{ entries: string[] }>();
</script>

<template>
    <section aria-labelledby="references">
        <h2 id="references">References</h2>
        <ol>
            <template
                v-for="{ id, reference } in entries.map((id) => ({
                    id,
                    reference: getReference(id),
                }))"
                :key="id"
            >
                <li :id="`ref-${id}`">
                    {{ formatAuthors(reference.author) }}
                    ({{ reference.issued["date-parts"][0][0] }}).
                    <template v-if="reference.type === 'webpage'">
                        <cite>{{ reference.title }}</cite>.
                        {{ reference.publisher }}. </template
                    ><template v-else-if="reference.type === 'book'">
                        <cite>{{ reference.title }}</cite>
                        <template v-if="reference.edition">
                            ({{ formatEdition(reference.edition) }}) </template
                        >. {{ reference.publisher }}. </template
                    ><template v-else-if="reference.type === 'chapter'">
                        {{ reference.title }}. In
                        <cite>{{ reference["container-title"] }}</cite>
                        (<template v-if="reference.edition"
                            >{{ formatEdition(reference.edition) }}, </template
                        >pp. {{ reference.page?.replaceAll("-", "–") }}).
                        {{ reference.publisher }}. </template
                    ><template
                        v-else-if="reference.type === 'paper-conference'"
                    >
                        {{ reference.title }}. In
                        <cite>{{ reference["container-title"] }}</cite>
                        (<template v-if="reference.volume"
                            >Vol. {{ reference.volume }}, </template
                        >pp. {{ reference.page?.replaceAll("-", "–") }}).
                        {{ reference.publisher }}. </template
                    ><template v-else-if="reference.type === 'report'">
                        <cite>{{ reference.title }}</cite>. Report
                        {{ reference.number }}.
                        {{ reference.publisher }}. </template
                    ><template v-else>
                        {{ reference.title }}.
                        <cite>{{ reference["container-title"] }}</cite>,
                        {{ reference.volume
                        }}<template v-if="reference.issue"
                            >({{ reference.issue }})</template
                        >, {{ reference.page?.replaceAll("-", "–") }}.
                    </template>
                    <template v-if="reference.DOI">
                        <a :href="`https://doi.org/${reference.DOI}`"
                            >doi:{{ reference.DOI }}</a
                        >. </template
                    ><template v-else-if="reference.URL">
                        <a :href="reference.URL">Full text</a>.
                    </template>
                </li>
            </template>
        </ol>
    </section>
</template>
