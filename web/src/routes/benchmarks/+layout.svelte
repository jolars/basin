<script lang="ts">
    import DocsLayout from "#lib/docs/DocsLayout.svelte";
    import { BENCHMARK_LINKS } from "#lib/nav.ts";

    let { children } = $props();
</script>

<DocsLayout navigation={[{ label: "Benchmarks", items: BENCHMARK_LINKS }]}>
    {@render children()}
</DocsLayout>
