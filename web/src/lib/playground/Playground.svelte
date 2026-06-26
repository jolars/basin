<script lang="ts">
    import type { Component } from "svelte";
    import Prism from "prismjs";
    import "prismjs/components/prism-rust";
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

    // We highlight by hand (per line), so stop Prism from auto-scanning the
    // document for `language-*` blocks on load.
    Prism.manual = true;

    function escapeHtml(s: string): string {
        return s.replace(/[&<>]/g, (c) =>
            c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;",
        );
    }

    // Highlight a single line of Rust to token-span HTML. Lines are highlighted
    // independently (no token in the generated snippet spans multiple lines),
    // which keeps the per-line structure the flash relies on. Runs on the
    // server too (Prism is isomorphic), so the prerendered HTML is highlighted.
    function highlightRust(line: string): string {
        if (line === "") return " ";
        const grammar = Prism.languages.rust;
        return grammar
            ? Prism.highlight(line, grammar, "rust")
            : escapeHtml(line);
    }

    // The playground is server-renderable: `generateSnippet` is pure, so the
    // landing page prerenders the default snippet as real HTML. Reactivity
    // (and the clipboard copy) take over after hydration.
    let cfg = $state<PlaygroundConfig>({ ...DEFAULT_CONFIG });

    const code = $derived(generateSnippet(cfg));
    // One element per line so we can flash the ones that change. The snippet's
    // line structure is fixed (only literals inside fixed lines change), so a
    // positional diff lands exactly on the changed line(s).
    const lines = $derived(code.replace(/\n+$/, "").split("\n"));
    const highlightedLines = $derived(lines.map(highlightRust));

    // Indices of lines that just changed, briefly highlighted then faded out.
    let flashed = $state<Set<number>>(new Set());
    let prevLines: string[] | null = null;
    let fadeTimer: ReturnType<typeof setTimeout> | null = null;

    $effect(() => {
        const cur = lines; // track: re-run whenever the generated code changes
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
        flashed = changed;
        if (fadeTimer) clearTimeout(fadeTimer);
        fadeTimer = setTimeout(() => {
            flashed = new Set();
            fadeTimer = null;
        }, 650);
    });

    // Clear the pending fade if the component is torn down mid-flash.
    $effect(() => () => {
        if (fadeTimer) clearTimeout(fadeTimer);
    });

    let copied = $state(false);
    async function copyCode() {
        try {
            await navigator.clipboard.writeText(code);
            copied = true;
            setTimeout(() => (copied = false), 1500);
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
    let contourEl = $state<HTMLElement>();
    let ContourComp = $state<Component<ContourProps> | null>(null);

    // Latest run result for the live output console. Null until the contour
    // has booted and reported (so SSR or pre-scroll shows a placeholder).
    let output = $state<RunOutput | null>(null);
    function handleResult(o: RunOutput) {
        output = o;
    }
    const outputText = $derived(
        output
            ? buildOutputLine(output.paramDebug, output.costDisplay)
            : "Run output appears here once the solver runs.",
    );

    $effect(() => {
        if (!contourEl || ContourComp) return;
        const io = new IntersectionObserver(
            (entries) => {
                if (entries.some((e) => e.isIntersecting)) {
                    io.disconnect();
                    import("./PlaygroundContour.svelte").then((m) => {
                        ContourComp = m.default as Component<ContourProps>;
                    });
                }
            },
            { rootMargin: "200px" },
        );
        io.observe(contourEl);
        return () => io.disconnect();
    });

    // Clicking the contour moves the start point (rounded for a clean code
    // literal), which re-runs the solve and rewrites `BasicState::new(...)`.
    function handlePick(p: { x: number; y: number }) {
        cfg.start = [Math.round(p.x * 100) / 100, Math.round(p.y * 100) / 100];
    }
</script>

<div class="grid xl:grid-cols-2 gap-8 items-start">
    <div class="min-w-0">
        <h2 class="text-2xl font-semibold tracking-tight">A Small Example</h2>
        <p class="mt-3 text-slate-600 dark:text-slate-300">
            Implement <code class="font-mono text-sm">CostFunction</code> and
            <code class="font-mono text-sm">Gradient</code>, then hand your
            problem, a solver, and a starting point to the
            <code class="font-mono text-sm">Executor</code>. Here it's gradient
            descent on the Rosenbrock valley: drag the sliders, or click the
            plot to move the start, and watch the run and the code update
            together.
        </p>

        <!-- Live solver. Lazily booted on scroll (see the IntersectionObserver
             above) so the wasm never weighs on the hero's initial load. -->
        <div
            bind:this={contourEl}
            class="relative mt-6 w-full max-w-2xl aspect-[3/2] mx-auto xl:max-w-none xl:mx-0 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-900 overflow-hidden"
        >
            {#if ContourComp}
                <ContourComp
                    alpha={cfg.alpha}
                    beta={cfg.beta}
                    maxIter={cfg.maxIter}
                    start={cfg.start}
                    onPick={handlePick}
                    onResult={handleResult}
                />
            {:else}
                <div
                    class="absolute inset-0 grid place-items-center px-6 text-center text-xs text-slate-500 dark:text-slate-400"
                >
                    Live solver: animates as it scrolls into view.
                </div>
            {/if}
        </div>

        <div class="mt-6 flex flex-col gap-4 text-sm">
            <label class="flex flex-col gap-1">
                <span
                    class="text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide"
                    >Step size α:
                    <span class="font-mono text-slate-900 dark:text-slate-100"
                        >{rustFloat(cfg.alpha)}</span
                    ></span
                >
                <input
                    type="range"
                    min="0"
                    max={ALPHA_STEPS.length - 1}
                    step="1"
                    value={nearestIndex(ALPHA_STEPS, cfg.alpha)}
                    oninput={(e) =>
                        (cfg.alpha =
                            ALPHA_STEPS[
                                Number(
                                    (e.currentTarget as HTMLInputElement).value,
                                )
                            ])}
                />
            </label>

            <label class="flex flex-col gap-1">
                <span
                    class="text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide"
                    >Momentum β:
                    <span class="font-mono text-slate-900 dark:text-slate-100"
                        >{cfg.beta > 0 ? rustFloat(cfg.beta) : "off"}</span
                    ></span
                >
                <input
                    type="range"
                    min={BETA_MIN}
                    max={BETA_MAX}
                    step={BETA_STEP}
                    value={cfg.beta}
                    oninput={(e) =>
                        (cfg.beta =
                            Math.round(
                                Number(
                                    (e.currentTarget as HTMLInputElement).value,
                                ) * 100,
                            ) / 100)}
                />
            </label>

            <label class="flex flex-col gap-1">
                <span
                    class="text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide"
                    >Max iterations:
                    <span class="font-mono text-slate-900 dark:text-slate-100"
                        >{rustInt(cfg.maxIter)}</span
                    ></span
                >
                <input
                    type="range"
                    min="0"
                    max={MAXITER_STEPS.length - 1}
                    step="1"
                    value={nearestIndex(MAXITER_STEPS, cfg.maxIter)}
                    oninput={(e) =>
                        (cfg.maxIter =
                            MAXITER_STEPS[
                                Number(
                                    (e.currentTarget as HTMLInputElement).value,
                                )
                            ])}
                />
            </label>
        </div>
    </div>

    <div class="flex flex-col gap-4 min-w-0">
        <div
            class="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 overflow-hidden"
        >
            <div
                class="px-4 py-2 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between gap-3"
            >
                <span
                    class="text-xs font-mono text-slate-500 dark:text-slate-400"
                    >rosenbrock.rs</span
                >
                <button
                    type="button"
                    onclick={copyCode}
                    class="text-xs font-mono px-2 py-1 rounded border border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                    {copied ? "Copied!" : "Copy"}
                </button>
            </div>
            <!-- prettier-ignore -->
            <pre class="p-4 overflow-x-auto text-sm leading-relaxed"><code class="rust-hl font-mono">{#each highlightedLines as html, i (i)}<span class="code-line" class:flash={flashed.has(i)}>{@html html}</span>{/each}</code></pre>
        </div>

        <!-- Live program output. The values are Rust-formatted in wasm
         (Run.paramDebug and costDisplay), so this is the snippet's real
         stdout, not a JS approximation. -->
        <div
            class="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 overflow-hidden"
        >
            <div
                class="px-4 py-2 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between gap-3"
            >
                <span
                    class="text-xs font-mono text-slate-500 dark:text-slate-400"
                    >Output</span
                >
                {#if output}
                    <span
                        class="text-[10px] font-mono uppercase tracking-wide text-slate-400 dark:text-slate-500"
                        >{output.done ? "done" : "running…"}</span
                    >
                {/if}
            </div>
            <pre class="px-4 py-3 overflow-x-auto text-sm leading-relaxed"><code
                    class="font-mono {output
                        ? 'text-slate-700 dark:text-slate-300'
                        : 'text-slate-400 dark:text-slate-500'}"
                    >{outputText}</code
                ></pre>
        </div>
    </div>
</div>

<style>
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

    /* Prism token theme: earthy, topographic palette (Gruvbox-derived) to
   match basin's geographical theme: sienna-red keywords, gold types,
   olive functions, rust numbers, teal "water" strings, warm stone
   neutrals. Tokens are injected via {@html}, so they need `:global`;
   colors come from CSS variables on `.rust-hl` (a real template element,
   so it carries the scope hash and the variables inherit into the
   injected token spans). One rule set serves both themes: the variables
   flip under the site's class-based `.dark`. */
    .rust-hl {
        --tok-comment: #928374; /* stone-gray */
        --tok-keyword: #9d0006; /* sienna red */
        --tok-fn: #79740e; /* olive green */
        --tok-macro: #8f3f71; /* muted plum */
        --tok-string: #427b58; /* basin teal */
        --tok-number: #af3a03; /* rust orange */
        --tok-type: #b57614; /* ochre gold */
        --tok-punct: #7c6f64; /* warm stone */
    }
    :global(.dark) .rust-hl {
        --tok-comment: #928374;
        --tok-keyword: #fb4934;
        --tok-fn: #b8bb26;
        --tok-macro: #d3869b;
        --tok-string: #8ec07c;
        --tok-number: #fe8019;
        --tok-type: #fabd2f;
        --tok-punct: #a89984;
    }
    .rust-hl :global(.token.comment) {
        color: var(--tok-comment);
        font-style: italic;
    }
    .rust-hl :global(.token.keyword) {
        color: var(--tok-keyword);
    }
    .rust-hl :global(.token.function),
    .rust-hl :global(.token.function-definition) {
        color: var(--tok-fn);
    }
    .rust-hl :global(.token.macro) {
        color: var(--tok-macro);
    }
    .rust-hl :global(.token.string),
    .rust-hl :global(.token.char) {
        color: var(--tok-string);
    }
    .rust-hl :global(.token.number),
    .rust-hl :global(.token.boolean) {
        color: var(--tok-number);
    }
    .rust-hl :global(.token.class-name),
    .rust-hl :global(.token.namespace) {
        color: var(--tok-type);
    }
    .rust-hl :global(.token.punctuation),
    .rust-hl :global(.token.operator) {
        color: var(--tok-punct);
    }
</style>
