<script lang="ts">
    import { theme } from "./theme.svelte";
    import IconSun from "~icons/lucide/sun";
    import IconMoon from "~icons/lucide/moon";

    // Single-button toggle: the icon shows the *current* effective theme
    // and clicking flips to the explicit opposite. There's no "auto"
    // button: `auto` (follow the OS) is the default for first-time
    // visitors and stays in effect until the first click overrides it.
    //
    // Icon visibility is driven by the `dark:` variant (i.e. the `.dark`
    // class on <html>), not by JS. The no-FOUC script in `app.html` sets
    // that class before first paint, so the correct icon shows with no
    // hydration flash.
</script>

<button
    type="button"
    class="inline-flex items-center justify-center rounded-md p-1.5 text-stone-600 hover:text-stone-900 hover:bg-stone-100 dark:text-stone-400 dark:hover:text-stone-100 dark:hover:bg-stone-800 transition-colors"
    aria-label={theme.effective === "dark"
        ? "Switch to light theme"
        : "Switch to dark theme"}
    title={theme.effective === "dark"
        ? "Switch to light theme"
        : "Switch to dark theme"}
    onclick={() => theme.set(theme.effective === "dark" ? "light" : "dark")}
>
    <IconSun
        width="18"
        height="18"
        class="block dark:hidden"
        aria-hidden="true"
    />
    <IconMoon
        width="18"
        height="18"
        class="hidden dark:block"
        aria-hidden="true"
    />
</button>
