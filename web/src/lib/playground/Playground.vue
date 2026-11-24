<script lang="ts">
import { createHighlighterCoreSync } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import rust from "shiki/langs/rust.mjs";
import githubDark from "shiki/themes/github-dark.mjs";
import githubLight from "shiki/themes/github-light.mjs";

const highlighter = createHighlighterCoreSync({
    langs: [rust],
    themes: [githubLight, githubDark],
    engine: createJavaScriptRegexEngine(),
});
</script>

<script setup lang="ts">
import { ref, computed, watchEffect, onUnmounted, useId } from "vue";
import type { Component } from "vue";
import {
    ALPHA_STEPS,
    BETA_MAX,
    BETA_MIN,
    BETA_STEP,
    DEFAULT_CONFIG,
    MAXITER_STEPS,
    buildOutputLine,
    generateSnippet,
    nearestIndex,
    rustFloat,
    rustInt,
    type PlaygroundConfig,
    type RunOutput,
} from "./codegen";

// The playground is server-renderable: `generateSnippet` is pure, so the
// landing page prerenders the default snippet as real HTML. Reactivity
// (and the clipboard copy) take over after hydration.
let cfg = ref<PlaygroundConfig>({ ...DEFAULT_CONFIG });
let codeExpanded = ref(false);
let snippetElement = ref<HTMLDivElement | undefined>();
const snippetId = useId();

const code = computed(() => generateSnippet(cfg.value));
// One element per line so we can flash the ones that change. The snippet's
// line structure is fixed (only literals inside fixed lines change), so a
// positional diff lands exactly on the changed line(s).
const lines = computed(() => code.value.replace(/\n+$/, "").split("\n"));
const highlightedLines = computed(
    () =>
        highlighter.codeToTokens(lines.value.join("\n"), {
            lang: "rust",
            // Use the same palettes and CSS variables as VitePress code blocks.
            themes: { light: githubLight.name, dark: githubDark.name },
            defaultColor: false,
        }).tokens,
);

// Indices of lines that just changed, briefly highlighted then faded out.
let flashed = ref<Set<number>>(new Set());
let prevLines: string[] | null = null;
let fadeTimer: ReturnType<typeof setTimeout> | null = null;

watchEffect(() => {
    const cur = lines.value; // track: re-run whenever the generated code changes
    if (prevLines === null) {
        prevLines = cur; // first render: nothing to flash yet
        return;
    }
    const changed = new Set<number>();
    const n = Math.max(cur.length, prevLines.length);
    for (let i = 0; i < n; i++) {
        if (cur[i] !== prevLines[i]) changed.add(i);
    }
    prevLines = cur;
    if (changed.size === 0) return;
    flashed.value = changed;
    if (fadeTimer) clearTimeout(fadeTimer);
    fadeTimer = setTimeout(() => {
        flashed.value = new Set();
        fadeTimer = null;
    }, 650);
});

// Clear the pending fade if the component is torn down mid-flash.
let copyTimer: ReturnType<typeof setTimeout> | undefined;
onUnmounted(() => {
    clearTimeout(copyTimer);
    if (fadeTimer) clearTimeout(fadeTimer);
});

let copied = ref(false);
async function copyCode() {
    try {
        await navigator.clipboard.writeText(code.value);
        copied.value = true;
        clearTimeout(copyTimer);
        copyTimer = setTimeout(() => (copied.value = false), 1500);
    } catch {
        // Clipboard unavailable (e.g. insecure context): no-op.
    }
}

// --- Live contour (Phase 2) ------------------------------------------
// The contour component pulls in wasm + ContourPlot, so it is loaded
// lazily and client-only: never imported on the server or in the hero
// bundle. An IntersectionObserver boots it the first time the plot
// scrolls into view, protecting the landing page's load and LCP.
type ContourProps = {
    alpha: number;
    beta: number;
    maxIter: number;
    start: [number, number];
    onPick: (p: { x: number; y: number }) => void;
    onResult: (o: RunOutput) => void;
};
let contourEl = ref<HTMLElement>();
let ContourComp = ref<Component<ContourProps> | null>(null);

// Latest run result for the live output console. Null until the contour
// has booted and reported (so SSR or pre-scroll shows a placeholder).
let output = ref<RunOutput | null>(null);
function handleResult(o: RunOutput) {
    output.value = o;
}
const outputText = computed(() =>
    output.value
        ? buildOutputLine(output.value.paramDebug, output.value.costDisplay)
        : "Run output appears here once the solver runs.",
);

watchEffect((onCleanup) => {
    if (!contourEl.value || ContourComp.value) return;
    const io = new IntersectionObserver(
        (entries) => {
            if (entries.some((e) => e.isIntersecting)) {
                io.disconnect();
                import("./PlaygroundContour.vue").then((m) => {
                    ContourComp.value = m.default as Component<ContourProps>;
                });
            }
        },
        { rootMargin: "200px" },
    );
    io.observe(contourEl.value);
    onCleanup(() => io.disconnect());
});

// Clicking the contour moves the start point (rounded for a clean code
// literal), which re-runs the solve and rewrites `FirstOrderState::new(...)`.
function handlePick(p: { x: number; y: number }) {
    cfg.value.start = [
        Math.round(p.x * 100) / 100,
        Math.round(p.y * 100) / 100,
    ];
}
</script>

<template>
    <div class="grid gap-8">
        <div class="min-w-0">
            <!-- Live solver. Lazily booted on scroll (see the IntersectionObserver
             above) so the wasm never weighs on the hero's initial load. -->
            <div
                ref="contourEl"
                class="relative mt-6 w-full aspect-[3/2] lg:max-h-[26rem] xl:max-h-[22rem] rounded-xl border border-stone-200 dark:border-stone-800 bg-stone-100 dark:bg-stone-900 overflow-hidden"
            >
                <template v-if="ContourComp">
                    <component
                        :is="ContourComp"
                        :alpha="cfg.alpha"
                        :beta="cfg.beta"
                        :maxIter="cfg.maxIter"
                        :start="cfg.start"
                        :onPick="handlePick"
                        :onResult="handleResult"
                    ></component> </template
                ><template v-else>
                    <div
                        class="absolute inset-0 grid place-items-center px-6 text-center text-xs text-stone-500 dark:text-stone-400"
                    >
                        Live solver: animates as it scrolls into view.
                    </div>
                </template>
            </div>

            <div class="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                <label class="flex flex-col gap-1">
                    <span
                        class="text-stone-700 dark:text-stone-300 uppercase text-xs tracking-wide"
                        >Step size α:
                        <span
                            class="font-mono text-stone-900 dark:text-stone-100"
                            >{{ rustFloat(cfg.alpha) }}</span
                        ></span
                    >
                    <input
                        type="range"
                        min="0"
                        :max="ALPHA_STEPS.length - 1"
                        step="1"
                        :value="nearestIndex(ALPHA_STEPS, cfg.alpha)"
                        @input="
                            (e) =>
                                (cfg.alpha =
                                    ALPHA_STEPS[
                                        Number(
                                            (
                                                e.currentTarget as HTMLInputElement
                                            ).value,
                                        )
                                    ])
                        "
                    />
                </label>

                <label class="flex flex-col gap-1">
                    <span
                        class="text-stone-700 dark:text-stone-300 uppercase text-xs tracking-wide"
                        >Momentum β:
                        <span
                            class="font-mono text-stone-900 dark:text-stone-100"
                            >{{
                                cfg.beta > 0 ? rustFloat(cfg.beta) : "off"
                            }}</span
                        ></span
                    >
                    <input
                        type="range"
                        :min="BETA_MIN"
                        :max="BETA_MAX"
                        :step="BETA_STEP"
                        :value="cfg.beta"
                        @input="
                            (e) =>
                                (cfg.beta =
                                    Math.round(
                                        Number(
                                            (
                                                e.currentTarget as HTMLInputElement
                                            ).value,
                                        ) * 100,
                                    ) / 100)
                        "
                    />
                </label>

                <label class="flex flex-col gap-1">
                    <span
                        class="text-stone-700 dark:text-stone-300 uppercase text-xs tracking-wide"
                        >Max iterations:
                        <span
                            class="font-mono text-stone-900 dark:text-stone-100"
                            >{{ rustInt(cfg.maxIter) }}</span
                        ></span
                    >
                    <input
                        type="range"
                        min="0"
                        :max="MAXITER_STEPS.length - 1"
                        step="1"
                        :value="nearestIndex(MAXITER_STEPS, cfg.maxIter)"
                        @input="
                            (e) =>
                                (cfg.maxIter =
                                    MAXITER_STEPS[
                                        Number(
                                            (
                                                e.currentTarget as HTMLInputElement
                                            ).value,
                                        )
                                    ])
                        "
                    />
                </label>
            </div>
        </div>

        <div class="flex flex-col gap-4 min-w-0">
            <div
                class="rounded-xl border border-stone-200 dark:border-stone-800 bg-[var(--vp-code-block-bg)] overflow-hidden [overflow-anchor:none]"
            >
                <div
                    class="px-4 py-2 border-b border-stone-200 dark:border-stone-800 flex items-center justify-between gap-3"
                >
                    <span
                        class="text-xs font-mono text-stone-500 dark:text-stone-400"
                        >rosenbrock.rs</span
                    >
                    <button
                        type="button"
                        @click="copyCode"
                        class="text-xs font-mono px-2 py-1 rounded border border-stone-300 dark:border-stone-700 text-stone-600 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
                    >
                        {{ copied ? "Copied!" : "Copy" }}
                    </button>
                </div>
                <div
                    :id="snippetId"
                    :class="
                        codeExpanded
                            ? ''
                            : 'max-h-72 overflow-hidden sm:max-h-none'
                    "
                >
                    <!-- prettier-ignore -->
                    <pre class="p-4 overflow-x-auto text-sm leading-relaxed"><code class="vp-code font-mono"><span v-for="(tokens, i) in highlightedLines" :key="i" class="code-line" :class="{ flash: flashed.has(i) }"><span v-for="token in tokens" :key="token.offset" :style="token.htmlStyle">{{ token.content }}</span><template v-if="tokens.length === 0"> </template></span></code></pre>
                </div>
                <button
                    type="button"
                    :aria-expanded="codeExpanded"
                    :aria-controls="snippetId"
                    @click="() => (codeExpanded = !codeExpanded)"
                    class="sm:hidden w-full border-t border-stone-200 dark:border-stone-800 px-4 py-3 text-sm font-medium text-stone-700 dark:text-stone-200 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
                >
                    {{ codeExpanded ? "Show less code" : "Show full code" }}
                </button>
            </div>

            <!-- Live program output. The values are Rust-formatted in wasm
         (Run.paramDebug and costDisplay), so this is the snippet's real
         stdout, not a JS approximation. -->
            <div
                class="rounded-xl border border-stone-200 dark:border-stone-800 bg-stone-50 dark:bg-stone-900 overflow-hidden"
            >
                <div
                    class="px-4 py-2 border-b border-stone-200 dark:border-stone-800 flex items-center justify-between gap-3"
                >
                    <span
                        class="text-xs font-mono text-stone-500 dark:text-stone-400"
                        >Output</span
                    >
                    <template v-if="output">
                        <span
                            class="text-[10px] font-mono uppercase tracking-wide text-stone-400 dark:text-stone-500"
                            >{{ output.done ? "done" : "running…" }}</span
                        >
                    </template>
                </div>
                <pre
                    class="px-4 py-3 overflow-x-auto text-sm leading-relaxed"
                ><code :class="'font-mono ' + (output
                        ? 'text-stone-700 dark:text-stone-300'
                        : 'text-stone-400 dark:text-stone-500')">{{ outputText }}</code></pre>
            </div>
        </div>
    </div>
</template>
<style scoped>
.code-line {
    display: block;
    border-radius: 3px;
    /* Fade-out when the flash class is removed. */
    transition: background-color 0.55s ease-out;
}
.code-line.flash {
    background-color: rgba(250, 204, 21, 0.38); /* amber-400 */
    /* Appear instantly on change; only the removal animates. */
    transition: none;
}
</style>
