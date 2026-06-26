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
    title="Basin—competitor benchmarks"
    description="basin versus established Rust optimization crates such as argmin and gomez on matched problems, as suboptimality-vs-time convergence traces."
/>

<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <p class="text-sm text-slate-500 dark:text-slate-400">
        <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href={resolve("/benchmarks/")}>Benchmarks</a
        >
        <span class="text-slate-400 dark:text-slate-600">/</span> Competitors
    </p>
    <h1 class="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">
        Competitors: basin vs argmin, gomez, and nlopt, convergence over time
    </h1>
    <p class="mt-3 max-w-3xl text-slate-600 dark:text-slate-300">
        basin against <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href="https://argmin-rs.org/"
            target="_blank"
            rel="noreferrer">argmin</a
        >,
        <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href="https://docs.rs/gomez/"
            target="_blank"
            rel="noreferrer">gomez</a
        >, and
        <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href="https://nlopt.readthedocs.io/"
            target="_blank"
            rel="noreferrer">nlopt</a
        >. Each library has only the algorithms it ships, so coverage varies:
        argmin lines up on GD, NM, and L-BFGS, gomez on derivative-free NM only,
        and nlopt on NM, L-BFGS, and NEWUOA. Most cases pit different
        implementations of the same family against each other; the NEWUOA case
        is the exception: basin and nlopt run the <em>same</em> Powell
        algorithm (matched ρ_beg/ρ_end), and it's the only case off Rosenbrock
        (Styblinski–Tang at n = 5). Because no two implementations share a code
        path, a single mean solve time would hide the differences in path and
        per-iteration cost. Instead each chart plots
        <strong>suboptimality</strong>
        <code class="font-mono">f(x) − f*</code>
        against <strong>wall-clock time</strong> on log–log axes: how far down the
        objective each library gets, and how long it spends getting there. Lower and
        further left is better.
    </p>

    <div class="mt-6 grid gap-6 lg:grid-cols-2">
        {#each COMPETITOR_CASES as c}
            <div
                class="rounded-xl border border-slate-200 dark:border-slate-800 p-5"
            >
                <h3 class="text-sm font-semibold">
                    {SOLVER_LABELS[c.solver]}
                    <span class="text-slate-400 dark:text-slate-500">·</span>
                    {PROBLEM_LABELS[c.problem]}
                </h3>
                <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">
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

    <p class="mt-8 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
        Measured {data.generatedAt} on {data.env.cpu}
        ({data.env.os}/{data.env.arch}). All libraries run on the
        <code class="font-mono">Vec&lt;f64&gt;</code> backend (gomez through its
        own bundled
        <code class="font-mono">nalgebra::DVector&lt;f64&gt;</code>). The GD,
        NM, and L-BFGS cases run from the classic Rosenbrock start to a
        {data.iterations}-iteration cap (a cap: the quasi-Newton case converges
        first, and gomez's NM hits its internal no-progress stop before the
        budget); the NEWUOA case instead runs on Styblinski–Tang (n = 5) from
        the origin to natural ρ-convergence. Each point is the median wall-clock
        time per iteration over repeated runs (nlopt, which exposes no
        per-iteration hook, is sampled per function evaluation as a best-so-far
        curve); the solvers are deterministic, so only the timing varies.
        Absolute times are machine-specific; compare the curves within a chart,
        not across machines.
    </p>

    <p class="mt-6 text-sm text-slate-500 dark:text-slate-400">
        To watch basin's solvers converge interactively, try the <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href={resolve("/visualizer/")}>visualizer</a
        >.
    </p>
</section>
