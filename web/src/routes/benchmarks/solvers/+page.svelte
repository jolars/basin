<script lang="ts">
    import { resolve } from "$app/paths";
    import ConvergenceChart from "$lib/ConvergenceChart.svelte";
    import Seo from "$lib/Seo.svelte";
    import { formatDuration } from "$lib/data/benchmarks";
    import {
        BY_PROBLEM,
        SOLVER_BENCHMARKS as data,
        SOLVER_COLORS,
        SOLVER_LABELS,
        SOLVER_ORDER,
        seriesFor,
    } from "$lib/data/solvers";

    function fmtF0(v: number): string {
        if (!Number.isFinite(v)) return "?";
        if (v >= 100) return v.toFixed(0);
        if (v >= 1) return v.toFixed(2);
        return v.toExponential(2);
    }
</script>

<Seo
    title="Solver Benchmarks – Basin"
    description="Basin's general optimizers (GD, Nelder–Mead, BFGS, L-BFGS, CMA-ES) head-to-head on standard benchmark problems, as suboptimality-vs-time convergence traces under a fixed wall-clock budget."
/>

<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <p class="text-sm text-stone-500 dark:text-stone-400">
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("/benchmarks/")}>Benchmarks</a
        >
        <span class="text-stone-400 dark:text-stone-600">/</span> Solvers
    </p>
    <h1 class="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">
        Solver Benchmarks
    </h1>
    <p class="mt-3 max-w-3xl text-stone-600 dark:text-stone-300">
        This benchmark compares optimizers from Basin across several starting
        points sampled randomly. Each run is capped on a fixed
        {formatDuration(data.budgetNs)} wall-clock budget and stopped early on reaching
        suboptimality
        <code class="font-mono">1e−10</code>. Lines within a panel share the
        same <code class="font-mono">f(x₀)</code>; lower and further left is
        better.
    </p>

    {#each BY_PROBLEM as group}
        <section class="mt-10">
            <h2 class="text-xl md:text-2xl font-semibold tracking-tight">
                {group.label}
            </h2>
            <p class="mt-1 text-sm text-stone-500 dark:text-stone-400">
                n = {group.n}, starts sampled uniformly in
                <code class="font-mono">[−2, 2]<sup>{group.n}</sup></code>.
            </p>

            <!-- Legend, centered above the grid so it sits over the plots
                 it describes. -->
            <div
                class="mt-5 flex flex-wrap justify-center gap-x-5 gap-y-2 text-sm"
            >
                {#each SOLVER_ORDER as solver}
                    <span class="inline-flex items-center gap-2">
                        <span
                            class="inline-block h-2.5 w-2.5 rounded-full"
                            style="background: {SOLVER_COLORS[solver]}"
                        ></span>
                        <span
                            class="font-mono text-stone-600 dark:text-stone-300"
                        >
                            {SOLVER_LABELS[solver]}
                        </span>
                    </span>
                {/each}
            </div>

            <!-- Grid wrapped with shared y-/x-axis labels so individual
                 charts can run in compact mode (no per-panel titles). -->
            <div class="mt-3 flex gap-2">
                <div
                    class="flex items-center justify-center text-sm text-stone-500 dark:text-stone-400 shrink-0"
                    style="writing-mode: vertical-rl; transform: rotate(180deg);"
                >
                    suboptimality f(x) − f*
                </div>
                <div class="flex-1 min-w-0">
                    <div
                        class="grid gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3"
                    >
                        {#each group.panels as s}
                            <div>
                                <h3 class="text-sm font-semibold">
                                    Seed {s.seed}
                                    <span
                                        class="ml-2 font-mono font-normal text-stone-500 dark:text-stone-400"
                                    >
                                        f(x₀) = {fmtF0(s.f0)}
                                    </span>
                                </h3>
                                <div class="mt-1">
                                    <ConvergenceChart
                                        compact
                                        series={seriesFor(s.problem, s.seed)}
                                        ariaLabel={`${group.label} n=${s.n}, seed ${s.seed}, f(x₀)=${fmtF0(s.f0)}: suboptimality vs wall-clock time, one line per solver`}
                                    />
                                </div>
                            </div>
                        {/each}
                    </div>
                    <p
                        class="mt-1 text-center text-sm text-stone-500 dark:text-stone-400"
                    >
                        wall-clock time
                    </p>
                </div>
            </div>
        </section>
    {/each}

    <p class="mt-10 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
        Measured {data.generatedAt} on {data.env.cpu}
        ({data.env.os}/{data.env.arch}). Every solver runs on the
        <code class="font-mono">Vec&lt;f64&gt;</code> backend, capped at
        {formatDuration(data.budgetNs)} per (solver, seed) run. Each per-iteration
        timestamp is the median over 11 repetitions of the same deterministic run;
        absolute times are machine-specific.
    </p>

    <p class="mt-6 text-sm text-stone-500 dark:text-stone-400">
        For the basin-versus-other-libraries view, see the <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("/benchmarks/competitors/")}>competitors</a
        >
        axis; for backend cost on the same solvers, see the
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("/benchmarks/backends/")}>backends</a
        >
        axis. To watch the same solvers converge interactively, try the
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("/visualizer/")}>visualizer</a
        >.
    </p>
</section>
