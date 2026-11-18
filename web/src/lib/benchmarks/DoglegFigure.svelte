<script lang="ts">
    import ConvergenceChart from "#lib/ConvergenceChart.svelte";
    import traces from "#lib/data/dogleg-convergence.json";

    const configurations = [
        { id: "basin-dogleg", label: "Basin dogleg", color: "#6366f1" },
        { id: "dogleg-unscaled", label: "dogleg unscaled", color: "#b45309" },
        { id: "dogleg-default", label: "dogleg default", color: "#be185d" },
    ] as const;

    const series = configurations.map(({ id, label, color }) => ({
        label,
        color,
        points: traces[id].map(([evaluation, subopt]) => ({
            evaluation,
            subopt,
        })),
    }));
</script>

<figure class="not-prose mx-auto my-8 w-full max-w-lg">
    <ConvergenceChart
        {series}
        xUnit="evaluations"
        yLabel="relative RSS gap"
        ariaLabel="BoxBOD start 1 convergence, relative residual sum of squares gap by residual evaluation for three dogleg configurations"
    />
    <ul class="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-sm">
        {#each configurations as { id, label, color } (id)}
            <li class="flex items-center gap-2">
                <span class="h-0.5 w-4" style:background-color={color}></span>
                {label}
            </li>
        {/each}
    </ul>
    <figcaption class="mt-3 text-sm text-stone-500 dark:text-stone-400">
        BoxBOD, start 1. Each line tracks the best residual sum of squares seen
        across all trials, including rejected steps. The gap is relative to the
        NIST reference RSS and is floored at 1e-12. Both axes are logarithmic,
        and each line ends at its last residual evaluation.
    </figcaption>
</figure>
