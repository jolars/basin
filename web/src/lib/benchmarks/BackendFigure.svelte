<script lang="ts">
    import BackendChart from "#lib/BackendChart.svelte";
    import {
        BACKEND_BENCHMARKS as data,
        BACKEND_COLORS,
        BACKEND_LABELS,
        PROBLEM_LABELS,
        SOLVER_LABELS,
        backendsFor,
        type Solver,
    } from "#lib/data/benchmarks.ts";

    let { solver, problem }: { solver: Solver; problem: string } = $props();

    const results = $derived(
        data.results.filter(
            (r) => r.solver === solver && r.problem === problem,
        ),
    );
    const dims = $derived(
        [...new Set(results.map((r) => r.n))].sort((a, b) => a - b),
    );
    const series = $derived(
        backendsFor(solver, problem).map((backend) => ({
            label: BACKEND_LABELS[backend],
            color: BACKEND_COLORS[backend],
            points: results
                .filter((r) => r.backend === backend)
                .map((r) => ({ n: r.n, ns: r.ns })),
        })),
    );
</script>

<figure class="not-prose mx-auto my-8 w-full max-w-lg">
    <BackendChart
        {series}
        {dims}
        ariaLabel={`${SOLVER_LABELS[solver]} on ${PROBLEM_LABELS[problem]}: solve time vs problem size, one line per backend`}
    />
    <figcaption class="mt-3 text-sm text-stone-500 dark:text-stone-400">
        {SOLVER_LABELS[solver]} on {PROBLEM_LABELS[problem]}. Mean time per
        solve across the measured problem sizes; both axes are logarithmic.
        Lower is better.
    </figcaption>
</figure>
