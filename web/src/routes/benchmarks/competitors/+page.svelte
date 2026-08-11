<script lang="ts">
    import { resolve } from "$app/paths";
    import ConvergenceChart from "$lib/ConvergenceChart.svelte";
    import Seo from "$lib/Seo.svelte";
    import {
        COMPETITOR_BENCHMARKS as data,
        COMPETITOR_CASES,
        LIBRARY_COLORS,
        LIBRARY_LABELS,
        PROBLEM_LABELS,
        SOLVER_LABELS,
        librariesFor,
        type Solver,
    } from "$lib/data/competitors";

    // One convergence trace per library present for a case: suboptimality vs
    // wall-clock time, the curve each library actually walked.
    function seriesFor(solver: Solver, problem: string) {
        return librariesFor(solver, problem).map((library) => ({
            label: LIBRARY_LABELS[library],
            color: LIBRARY_COLORS[library],
            points:
                data.results.find(
                    (r) =>
                        r.solver === solver &&
                        r.problem === problem &&
                        r.library === library,
                )?.points ?? [],
        }));
    }
</script>

<Seo
    title="Competitor Benchmarks – Basin"
    description="Basin versus established Rust optimization crates such as argmin and gomez on matched problems, as suboptimality-vs-time convergence traces."
/>

<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <p class="text-sm text-stone-500 dark:text-stone-400">
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("/benchmarks/")}>Benchmarks</a
        >
        <span class="text-stone-400 dark:text-stone-600">/</span> Competitors
    </p>
    <h1 class="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">
        Competitor Benchmarks
    </h1>
    <p class="mt-3 max-w-3xl text-stone-600 dark:text-stone-300">
        This benchmark compars Basin against <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://argmin-rs.org/"
            target="_blank"
            rel="noreferrer">argmin</a
        >,
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://docs.rs/gomez/"
            target="_blank"
            rel="noreferrer">gomez</a
        >, and
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://nlopt.readthedocs.io/"
            target="_blank"
            rel="noreferrer">nlopt</a
        >. Each plot shows <em>suboptimality</em>,
        <code class="font-mono">f(x) − f*</code>, against wall-clock time on
        log–log axes: how far down the objective each library gets, and how long
        it spends getting there. Lower and further left is better.
    </p>

    <div class="mt-6 grid gap-6 lg:grid-cols-2">
        {#each COMPETITOR_CASES as c}
            <div
                class="rounded-xl border border-stone-200 dark:border-stone-800 p-5"
            >
                <h3 class="text-sm font-semibold">
                    {SOLVER_LABELS[c.solver]}
                    <span class="text-stone-400 dark:text-stone-500">·</span>
                    {PROBLEM_LABELS[c.problem]}
                </h3>
                <p class="mt-1 text-xs text-stone-500 dark:text-stone-400">
                    {c.blurb}
                </p>
                <div class="mt-3">
                    <ConvergenceChart
                        directLabels
                        series={seriesFor(c.solver, c.problem)}
                        ariaLabel={`${SOLVER_LABELS[c.solver]} on ${PROBLEM_LABELS[c.problem]}: suboptimality vs wall-clock time, one line per library`}
                    />
                </div>
            </div>
        {/each}
    </div>

    <p class="mt-8 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
        Measured {data.generatedAt} on {data.env.cpu}
        ({data.env.os}/{data.env.arch}). All libraries run on the
        <code class="font-mono">Vec&lt;f64&gt;</code> backend (gomez through its
        own bundled
        <code class="font-mono">nalgebra::DVector&lt;f64&gt;</code>). The GD,
        NM, and L-BFGS cases run from the classic Rosenbrock start to a
        {data.iterations}-iteration cap. Each point is the median wall-clock
        time per iteration over repeated runs (nlopt, which exposes no
        per-iteration hook, is sampled per function evaluation as a best-so-far
        curve); the solvers are deterministic, so only the timing varies.
    </p>

    <p class="mt-6 text-sm text-stone-500 dark:text-stone-400">
        To watch basin's solvers converge interactively, try the <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("/visualizer/")}>visualizer</a
        >.
    </p>
</section>
