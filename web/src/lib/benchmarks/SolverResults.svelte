<script lang="ts">
    import ConvergenceChart from "#lib/ConvergenceChart.svelte";
    import { formatDuration } from "#lib/data/benchmarks.ts";
    import {
        SOLVER_BENCHMARKS as data,
        SOLVER_COLORS,
        SOLVER_LABELS,
        SOLVER_ORDER,
        seriesFor,
        type ProblemSection,
    } from "#lib/data/solvers.ts";

    let { group }: { group: ProblemSection } = $props();

    const target = 1e-10;

    function fmtF0(v: number): string {
        if (!Number.isFinite(v)) return "?";
        if (v >= 100) return v.toFixed(0);
        if (v >= 1) return v.toFixed(2);
        return v.toExponential(2);
    }

    function summaryFor(problem: string) {
        return SOLVER_ORDER.map((solver) => {
            const runs = data.results.filter(
                (r) => r.solver === solver && r.problem === problem,
            );
            const times = runs
                .flatMap((r) => {
                    const hit = r.points.find((p) => p.subopt <= target);
                    return hit ? [hit.tNs] : [];
                })
                .sort((a, b) => a - b);
            const median = times.length
                ? (times[Math.floor((times.length - 1) / 2)] +
                      times[Math.floor(times.length / 2)]) /
                  2
                : null;
            return { solver, runs: runs.length, reached: times.length, median };
        });
    }
</script>

<div class="overflow-x-auto">
    <table>
        <caption class="sr-only">
            {group.label}: target attainment and median time to an objective gap
            of 1e-10.
        </caption>
        <thead>
            <tr>
                <th scope="col">Solver</th>
                <th scope="col">Starts reaching target</th>
                <th scope="col">Median time when reached</th>
            </tr>
        </thead>
        <tbody>
            {#each summaryFor(group.problem) as row (row.solver)}
                <tr>
                    <th scope="row">{SOLVER_LABELS[row.solver]}</th>
                    <td>{row.reached} / {row.runs}</td>
                    <td>
                        {row.median === null ? "—" : formatDuration(row.median)}
                    </td>
                </tr>
            {/each}
        </tbody>
    </table>
</div>
<p>
    The median uses the first recorded timestamp at which the target is reached
    and includes only successful starts. Read it together with the count: a
    short conditional median does not describe runs that miss the target. This
    small set of starts is a comparison on this fixture, rather than an estimate
    of performance across all problems.
</p>

<h3 id={`${group.problem}-starting-points`}>Starting Points</h3>
<p>
    Lower curves have smaller objective gaps; curves further left reach that
    accuracy sooner. Each panel has its own logarithmic axes. A curve ending
    above the target can reflect the time budget or another stopping condition,
    so its endpoint alone does not establish convergence.
</p>
<details
    class="not-prose my-6 rounded-lg border border-stone-200 dark:border-stone-800"
>
    <summary class="cursor-pointer px-4 py-3 text-sm font-medium">
        Show all {group.panels.length} starting points
    </summary>
    <div class="px-3 pb-4 sm:px-4">
        <div class="mb-5 flex flex-wrap justify-center gap-x-5 gap-y-2 text-sm">
            {#each SOLVER_ORDER as solver}
                <span class="inline-flex items-center gap-2">
                    <span
                        class="inline-block h-2.5 w-2.5 rounded-full"
                        style:background-color={SOLVER_COLORS[solver]}
                    ></span>
                    <span class="text-stone-600 dark:text-stone-300">
                        {SOLVER_LABELS[solver]}
                    </span>
                </span>
            {/each}
        </div>
        <div class="flex gap-2">
            <div
                class="flex shrink-0 items-center justify-center text-sm text-stone-500 dark:text-stone-400"
                style="writing-mode: vertical-rl; transform: rotate(180deg);"
            >
                suboptimality f(x) − f*
            </div>
            <div class="min-w-0 flex-1">
                <div class="grid gap-x-4 gap-y-5 sm:grid-cols-2">
                    {#each group.panels as s (s.seed)}
                        <figure>
                            <figcaption class="mb-2 text-sm font-medium">
                                Seed {s.seed}
                                <span
                                    class="ml-2 font-mono font-normal text-stone-500 dark:text-stone-400"
                                >
                                    f(x₀) = {fmtF0(s.f0)}
                                </span>
                            </figcaption>
                            <ConvergenceChart
                                compact
                                series={seriesFor(s.problem, s.seed)}
                                ariaLabel={`${group.label} n=${s.n}, seed ${s.seed}, f(x₀)=${fmtF0(s.f0)}: suboptimality vs wall-clock time, one line per solver`}
                            />
                        </figure>
                    {/each}
                </div>
                <p
                    class="mt-2 text-center text-sm text-stone-500 dark:text-stone-400"
                >
                    wall-clock time
                </p>
            </div>
        </div>
    </div>
</details>
