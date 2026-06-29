<script lang="ts">
    // Wraps every `.svx`/`.md` page (set as the default mdsvex `layout`
    // in svelte.config.js). mdsvex passes the rendered Markdown as the
    // default slot, which Svelte 5 surfaces here as the `children`
    // snippet. Frontmatter fields arrive as extra props and are ignored.
    let { children } = $props();
</script>

<article class="prose prose-stone dark:prose-invert max-w-none">
    {@render children()}
</article>
