<script lang="ts">
    import ConvergenceChart from "#lib/ConvergenceChart.svelte";
    import ConstraintViolationChart from "#lib/ConstraintViolationChart.svelte";
    import {
        COMPETITOR_BENCHMARKS as data,
        LIBRARY_COLORS,
        LIBRARY_LABELS,
        PROBLEM_LABELS,
        SOLVER_LABELS,
        librariesFor,
        type Solver,
    } from "#lib/data/competitors.ts";

    let { solver, problem }: { solver: Solver; problem: string } = $props();

    const results = $derived(
        data.results.filter(
            (r) => r.solver === solver && r.problem === problem,
        ),
    );
    const series = $derived(
        librariesFor(solver, problem).map((library) => ({
            label: LIBRARY_LABELS[library],
            color: LIBRARY_COLORS[library],
            points: results.find((r) => r.library === library)?.points ?? [],
        })),
    );
    const violations = $derived(
        librariesFor(solver, problem).map((library) => ({
            label: LIBRARY_LABELS[library],
            color: LIBRARY_COLORS[library],
            points:
                results.find((r) => r.library === library)?.violations ?? [],
        })),
    );
</script>

<div class="not-prose my-8">
    {#if results.some((r) => r.summary?.verified === false)}
        <p class="mb-4 text-sm text-red-700 dark:text-red-400">
            A returned solution failed the independent accuracy check. See the
            run details below.
        </p>
    {/if}

    <figure class="mx-auto w-full max-w-lg">
        <ConvergenceChart
            directLabels
            {series}
            ariaLabel={`${SOLVER_LABELS[solver]} on ${PROBLEM_LABELS[problem]}: suboptimality vs wall-clock time, one line per library`}
        />
        <figcaption class="mt-3 text-sm text-stone-500 dark:text-stone-400">
            {SOLVER_LABELS[solver]} on {PROBLEM_LABELS[problem]}, n = {results[0]
                ?.n}. Lower objective gaps and shorter times are better; both
            axes are logarithmic.
        </figcaption>
    </figure>

    {#if solver === "cobyla"}
        <figure class="mx-auto mt-8 w-full max-w-lg">
            <ConstraintViolationChart
                tolerance={1.4901161193847656e-8}
                series={violations}
                ariaLabel="COBYLA on the constrained quadratic: trial constraint violation vs wall-clock time"
            />
            <figcaption class="mt-3 text-sm text-stone-500 dark:text-stone-400">
                Every trial and the returned point, including infeasible trials.
                Time is linear; violation is linear within the shaded
                feasibility band and logarithmic above it.
            </figcaption>
        </figure>
    {/if}

    {#if results.some((r) => r.summary)}
        <details
            class="mt-6 rounded-lg border border-stone-200 px-4 py-3 text-sm text-stone-600 dark:border-stone-800 dark:text-stone-300"
        >
            <summary class="cursor-pointer font-medium">
                Returned solutions and evaluation counts
            </summary>
            {#each results as r (r.library)}
                {#if r.summary}
                    <p class="mt-3">
                        <strong>{LIBRARY_LABELS[r.library]}</strong>:
                        {r.summary.status}. Independent accuracy check
                        {r.summary.verified ? "passed" : "failed"}. Returned ({r.summary.point
                            .map((x) => x?.toPrecision(6) ?? "nonfinite")
                            .join(", ")}), objective
                        {r.summary.cost?.toExponential(3) ?? "nonfinite"}.
                        {#if solver === "lm"}
                            {r.summary.residualCalls} residual and
                            {r.summary.jacobianCalls} Jacobian calls.
                        {:else}
                            Violation
                            {r.summary.violation?.toExponential(3) ??
                                "nonfinite"};
                            {r.summary.objectiveCalls} objective and
                            {r.summary.constraintCalls} constraint calls ({r.library ===
                            "basin"
                                ? "vector callbacks, including bounds"
                                : "scalar nonlinear callbacks; native bounds excluded"}).
                        {/if}
                    </p>
                {/if}
            {/each}
        </details>
    {/if}
</div>
