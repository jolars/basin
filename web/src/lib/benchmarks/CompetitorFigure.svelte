<script lang="ts">
    import ConvergenceChart from "#lib/ConvergenceChart.svelte";
    import ConstraintViolationChart from "#lib/ConstraintViolationChart.svelte";
    import { formatDuration } from "#lib/data/benchmarks.ts";
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
    const seeds = $derived(
        [
            ...new Set(
                results.flatMap((r) => (r.seed === undefined ? [] : [r.seed])),
            ),
        ].sort((a, b) => a - b),
    );
    let seed = $state(0);
    const selected = $derived(
        results.filter((r) => r.seed === undefined || r.seed === seed),
    );
    const seedSummaries = $derived(
        librariesFor(solver, problem).map((library) => {
            const runs = results.filter((r) => r.library === library);
            const times = runs.map((r) => r.points.at(-1)?.tNs ?? 0);
            const calls = runs.map((r) => r.summary?.objectiveCalls ?? 0);
            return {
                library,
                successes: runs.filter((r) => r.summary?.verified).length,
                count: runs.length,
                minTime: Math.min(...times),
                maxTime: Math.max(...times),
                minCalls: Math.min(...calls),
                maxCalls: Math.max(...calls),
            };
        }),
    );
    const series = $derived(
        librariesFor(solver, problem).map((library) => ({
            label: LIBRARY_LABELS[library],
            color: LIBRARY_COLORS[library],
            points: selected.find((r) => r.library === library)?.points ?? [],
        })),
    );
    const violations = $derived(
        librariesFor(solver, problem).map((library) => ({
            label: LIBRARY_LABELS[library],
            color: LIBRARY_COLORS[library],
            points:
                selected.find((r) => r.library === library)?.violations ?? [],
        })),
    );
</script>

<div class="not-prose my-8">
    {#if results.some((r) => r.summary?.verified === false)}
        <p class="mb-4 text-sm text-red-700 dark:text-red-400">
            {seeds.length
                ? "Some seeds did not reach the accuracy target."
                : "A returned solution failed the independent accuracy check."}
            See the run details below.
        </p>
    {/if}

    {#if seeds.length}
        <div class="mb-6 overflow-x-auto text-sm">
            <table class="w-full text-left">
                <caption
                    class="mb-2 text-left text-stone-600 dark:text-stone-300"
                >
                    All {seeds.length} seeds. Ranges cover final times and objective
                    calls, including unsuccessful runs. Each seed's time is a median
                    over {data.repetitions} timing repetitions.
                </caption>
                <thead>
                    <tr>
                        <th class="pr-4 py-2">Library</th>
                        <th class="pr-4 py-2">Targets reached</th>
                        <th class="pr-4 py-2">Final time</th>
                        <th class="py-2">Objective calls</th>
                    </tr>
                </thead>
                <tbody>
                    {#each seedSummaries as s (s.library)}
                        <tr
                            class="border-t border-stone-200 dark:border-stone-800"
                        >
                            <th class="pr-4 py-2"
                                >{LIBRARY_LABELS[s.library]}</th
                            >
                            <td class="pr-4 py-2">{s.successes}/{s.count}</td>
                            <td class="pr-4 py-2"
                                >{formatDuration(s.minTime)}–{formatDuration(
                                    s.maxTime,
                                )}</td
                            >
                            <td class="py-2"
                                >{s.minCalls.toLocaleString(
                                    "en-US",
                                )}–{s.maxCalls.toLocaleString("en-US")}</td
                            >
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
        <label class="mb-4 flex items-center justify-center gap-2 text-sm">
            Seed
            <select
                bind:value={seed}
                class="rounded border border-stone-300 bg-white px-2 py-1 dark:border-stone-700 dark:bg-stone-950"
            >
                {#each seeds as s (s)}
                    <option value={s}>{s}</option>
                {/each}
            </select>
        </label>
    {/if}

    <figure class="mx-auto w-full max-w-lg">
        <ConvergenceChart
            directLabels
            xScale="linear"
            {series}
            ariaLabel={`${SOLVER_LABELS[solver]} on ${PROBLEM_LABELS[problem]}${seeds.length ? `, seed ${seed}` : ""}: suboptimality vs time, one line per library`}
        />
        <figcaption class="mt-3 text-sm text-stone-500 dark:text-stone-400">
            {SOLVER_LABELS[solver]} on {PROBLEM_LABELS[problem]}, n = {results[0]
                ?.n}. Lower objective gaps and shorter times are better. Time is
            linear; objective gap is logarithmic.
            {#if seeds.length}Showing seed {seed}.{/if}
        </figcaption>
    </figure>

    {#if solver === "cobyla" || solver === "slsqp"}
        <figure class="mx-auto mt-8 w-full max-w-lg">
            <ConstraintViolationChart
                tolerance={solver === "cobyla" ? 1.4901161193847656e-8 : 1e-8}
                series={violations}
                ariaLabel={`${SOLVER_LABELS[solver]}: trial constraint violation vs time`}
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
            {#each selected as r (r.library)}
                {#if r.summary}
                    <p class="mt-3">
                        <strong>{LIBRARY_LABELS[r.library]}</strong>:
                        {#if r.seed !== undefined}seed {r.seed};{/if}
                        {r.summary.status}. Independent accuracy check
                        {r.summary.verified ? "passed" : "failed"}. Returned ({r.summary.point
                            .slice(0, 6)
                            .map((x) => x?.toPrecision(6) ?? "nonfinite")
                            .join(", ")}{r.n > 6 ? ", …" : ""}), objective
                        {r.summary.cost?.toExponential(3) ?? "nonfinite"}.
                        {#if solver === "lm"}
                            {r.summary.residualCalls} residual and
                            {r.summary.jacobianCalls} Jacobian calls.
                        {:else if solver === "lbfgsb"}
                            {r.summary.iterations} iterations;
                            {r.summary.objectiveCalls} objective and
                            {r.summary.gradientCalls} gradient calls (fused); bound
                            violation
                            {r.summary.violation?.toExponential(3) ??
                                "nonfinite"}; projected-gradient norm
                            {r.summary.projectedGradient?.toExponential(3) ??
                                "nonfinite"}.
                        {:else if solver === "cmaes"}
                            {r.summary.iterations} sampled generations;
                            {r.summary.objectiveCalls} objective calls.
                        {:else}
                            Violation
                            {r.summary.violation?.toExponential(3) ??
                                "nonfinite"};
                            {#if solver === "slsqp"}
                                {r.summary.gradientCalls} gradient calls;
                            {/if}
                            {r.summary.objectiveCalls} objective and
                            {r.summary.constraintCalls} constraint calls ({r.library ===
                            "basin"
                                ? solver === "cobyla"
                                    ? "vector callbacks, including bounds"
                                    : "vector callbacks"
                                : "scalar nonlinear callbacks; native bounds excluded"}).
                        {/if}
                    </p>
                {/if}
            {/each}
        </details>
    {/if}
</div>
