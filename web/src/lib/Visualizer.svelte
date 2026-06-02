<script lang="ts">
    import { onMount } from 'svelte';
    import init, {
        ProblemKind,
        SolverKind,
        Run,
        evalGrid,
    } from '$lib/basin-wasm/basin_wasm';
    import { PROBLEMS, problemByKind, SUBOPT_TARGET } from '$lib/problems';
    import { SOLVERS, defaultOptionValues } from '$lib/solvers';
    import ContourPlot from '$lib/ContourPlot.svelte';
    import CostChart from '$lib/CostChart.svelte';
    import Controls from '$lib/Controls.svelte';
    import { theme } from '$lib/theme.svelte';

    // Wasm boot. The viz waits on this once; everything downstream assumes
    // the module is already loaded.
    let wasmReady = $state(false);

    // Default option values for a solver, with the problem-dependent α
    // default folded in for gradient descent.
    function initOptionValues(
        sk: SolverKind,
        pk: ProblemKind,
    ): Record<string, string | number> {
        const meta = SOLVERS.find((s) => s.kind === sk)!;
        const vals = defaultOptionValues(meta);
        if ('gdAlpha' in vals) vals.gdAlpha = problemByKind(pk).gdAlphaDefault;
        return vals;
    }

    let problemKind: ProblemKind = $state(ProblemKind.Rosenbrock);
    let solverKind: SolverKind = $state(SolverKind.GradientDescent);
    // Solver-specific option values, keyed by option id (see solvers.ts).
    let optionValues = $state<Record<string, string | number>>(
        initOptionValues(SolverKind.GradientDescent, ProblemKind.Rosenbrock),
    );
    let maxIter = $state(500);
    let startPoint = $state({ x: -1.5, y: 2.0 });

    // Heatmap grid. Recomputed when the problem changes (or on first boot).
    const GRID_N = 192;
    // Use the wide `Float64Array<ArrayBufferLike>` type so values returned
    // from wasm-bindgen (which use `ArrayBufferLike`) assign cleanly.
    let grid: Float64Array<ArrayBufferLike> = $state(
        new Float64Array(GRID_N * GRID_N),
    );

    // Animated trajectory and cost log fed to the children.
    let trajectory: Float64Array<ArrayBufferLike> = $state(new Float64Array(0));
    let costs: Float64Array<ArrayBufferLike> = $state(new Float64Array(0));
    // Current-generation population for stochastic solvers (CMA-ES, DE, RS,
    // SSGA). Empty Float64Array for the local solvers — the contour plot
    // skips rendering when length is zero.
    let population: Float64Array<ArrayBufferLike> = $state(new Float64Array(0));
    let reason = $state('');

    let problemMeta = $derived(problemByKind(problemKind));
    let solverMeta = $derived(SOLVERS.find((s) => s.kind === solverKind)!);

    // Plain (non-reactive) handles for the in-flight run + animation
    // frame. We deliberately keep these out of `$state` because the run
    // effect both reads (cleanup) and writes (assignment) them, and a
    // reactive write would re-trigger the effect — Svelte detects that
    // as `effect_update_depth_exceeded` and aborts.
    let activeRun: Run | null = null;
    let frameId: number | null = null;

    // Refresh the heatmap when the problem changes.
    $effect(() => {
        if (!wasmReady) return;
        const d = problemMeta.domain;
        grid = evalGrid(
            problemMeta.kind,
            d.xmin,
            d.xmax,
            d.ymin,
            d.ymax,
            GRID_N,
            GRID_N,
        );
    });

    // Boot a fresh run whenever the inputs change. The reads inside
    // `new Run(...)` track the dependencies; writes to `activeRun` and
    // `frameId` are non-reactive so they don't retrigger this effect.
    $effect(() => {
        if (!wasmReady) return;
        const pk = problemKind;
        const sk = solverKind;
        const mi = maxIter;
        const sx = startPoint.x;
        const sy = startPoint.y;
        // Solver options crossing the wasm boundary (see RunOptions in
        // crates/basin-wasm/src/lib.rs). Each solver reads only what it
        // needs; reading the fields here tracks them as effect deps.
        const d = problemMeta.domain;
        const opts = {
            gdLineSearch: optionValues.gdLineSearch ?? 'constant',
            gdAlpha: optionValues.gdAlpha ?? problemMeta.gdAlphaDefault,
            // β = 0: the visualizer has no momentum knob (the landing-page
            // playground does). Plain steepest descent for the GD solver.
            gdBeta: 0,
            lbfgsM: optionValues.lbfgsM ?? 10,
            // Stochastic solvers (CMA-ES, DE, RandomSearch, SSGA). NaN on
            // cmaSigma → wasm picks a viewport-scaled default.
            seed: optionValues.seed ?? 0,
            cmaSigma: optionValues.cmaSigma ?? NaN,
            cmaLambda: optionValues.cmaLambda ?? 0,
            dePopSize: optionValues.dePopSize ?? 0,
            deF: optionValues.deF ?? 0.8,
            deCr: optionValues.deCr ?? 0.9,
            rsLambda: optionValues.rsLambda ?? 16,
            ssgaPopSize: optionValues.ssgaPopSize ?? 0,
            // Box bounds for DE / SSGA / RandomSearch — they use the visible
            // viewport as the feasible region (the right semantics for a 2D
            // demo). Reading the fields here ties the effect to domain
            // changes when the user switches problems.
            xmin: d.xmin,
            xmax: d.xmax,
            ymin: d.ymin,
            ymax: d.ymax,
        };
        // Stop early once the cost is within SUBOPT_TARGET of the known
        // optimum — the same value the cost chart uses as its log floor.
        const stopAtCost = problemMeta.fStar + SUBOPT_TARGET;

        if (frameId !== null) {
            cancelAnimationFrame(frameId);
            frameId = null;
        }
        if (activeRun !== null) {
            activeRun.free();
            activeRun = null;
        }

        const run = new Run(pk, sk, sx, sy, opts, mi, stopAtCost);
        activeRun = run;
        trajectory = run.trajectoryXy();
        costs = run.costs();
        population = run.populationXy();
        reason = '';

        // Generation-based solvers (CMA-ES / DE / RS) advance one full
        // generation per `next_iter`, so 8/frame races through the run in
        // a flash. The solver's meta gates this. Fractional rates (< 1) mean
        // "step every Nth frame" — for population-based solvers where each
        // generation is a big visible jump, this gives the eye time to read
        // the cloud between updates. SSGA's "iter" is a single offspring
        // evaluation, so it keeps the default chunk of 8.
        const itersPerFrame = solverMeta.itersPerFrame ?? 8;
        // Carry fractional remainder across frames so non-integer rates
        // advance whole iterations on the frames the accumulator overflows.
        let acc = 0;
        const tick = () => {
            // Stale-frame guard: a newer effect run replaces `activeRun`,
            // so a tick from an older closure should bail.
            if (run !== activeRun) return;
            acc += itersPerFrame;
            const n = Math.floor(acc);
            if (n >= 1) {
                acc -= n;
                const result = run.stepMany(n) as {
                    done: boolean;
                    iters_added: number;
                    reason?: string | null;
                };
                trajectory = run.trajectoryXy();
                costs = run.costs();
                population = run.populationXy();
                if (result.done) {
                    reason = result.reason ?? '';
                    frameId = null;
                    return;
                }
            }
            frameId = requestAnimationFrame(tick);
        };
        frameId = requestAnimationFrame(tick);

        return () => {
            if (frameId !== null) {
                cancelAnimationFrame(frameId);
                frameId = null;
            }
            if (activeRun === run) {
                run.free();
                activeRun = null;
            }
        };
    });

    onMount(async () => {
        await init();
        wasmReady = true;
        // Seed start point near a visually interesting corner of the
        // initial problem (Rosenbrock).
        startPoint = { x: -1.5, y: 2.0 };
    });

    function handlePick(p: { x: number; y: number }) {
        startPoint = p;
    }

    function handleControlChange(patch: {
        problemKind?: ProblemKind;
        solverKind?: SolverKind;
        maxIter?: number;
    }) {
        if (patch.problemKind !== undefined && patch.problemKind !== problemKind) {
            problemKind = patch.problemKind;
            // Re-center start and reset the α default for the new problem.
            const d = problemByKind(problemKind).domain;
            startPoint = {
                x: d.xmin + 0.25 * (d.xmax - d.xmin),
                y: d.ymin + 0.75 * (d.ymax - d.ymin),
            };
            if ('gdAlpha' in optionValues) {
                optionValues = {
                    ...optionValues,
                    gdAlpha: problemByKind(problemKind).gdAlphaDefault,
                };
            }
        }
        if (patch.solverKind !== undefined && patch.solverKind !== solverKind) {
            solverKind = patch.solverKind;
            // Reset options to the new solver's schema defaults.
            optionValues = initOptionValues(solverKind, problemKind);
            // Clear the stale population so we don't render the old solver's
            // dots for the one frame before the run-effect reseeds them.
            population = new Float64Array(0);
        }
        if (patch.maxIter !== undefined) maxIter = patch.maxIter;
    }

    function handleOptionChange(id: string, value: string | number) {
        optionValues = { ...optionValues, [id]: value };
    }
</script>

<section
    class="min-h-[calc(100vh-8rem)] max-w-screen-2xl w-full mx-auto px-4 md:px-8 py-6 flex flex-col gap-6"
>
    <header class="flex flex-wrap items-start justify-between gap-4">
        <div>
            <h1 class="text-2xl md:text-3xl font-semibold tracking-tight">
                Solver visualizer
            </h1>
            <p class="text-slate-600 dark:text-slate-400 text-sm mt-1">
                Live wasm-driven 2D trajectories. Click on the contour to reset
                the start point.
            </p>
        </div>
        <p
            class="text-xs text-slate-500 dark:text-slate-500 font-mono hidden md:block self-center"
        >
            {solverMeta.blurb}
        </p>
    </header>

    {#if !wasmReady}
        <p class="text-slate-500 dark:text-slate-400">Loading wasm…</p>
    {:else}
        <div class="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6 flex-1 min-h-0">
            <div
                class="relative bg-slate-100 dark:bg-slate-900 rounded-lg overflow-hidden aspect-square lg:aspect-auto lg:min-h-[360px]"
            >
                <ContourPlot
                    problem={problemMeta}
                    {grid}
                    nx={GRID_N}
                    ny={GRID_N}
                    {trajectory}
                    {population}
                    {startPoint}
                    theme={theme.effective}
                    onPick={handlePick}
                />
            </div>
            <aside class="flex flex-col gap-6 min-w-0">
                <div class="bg-slate-100 dark:bg-slate-900 rounded-lg p-4">
                    <Controls
                        {problemKind}
                        {solverKind}
                        solverOptions={solverMeta.options}
                        {optionValues}
                        {maxIter}
                        {startPoint}
                        onChange={handleControlChange}
                        onOptionChange={handleOptionChange}
                    />
                </div>
                <div
                    class="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 h-56 lg:flex-1"
                >
                    <CostChart
                        {costs}
                        fStar={problemMeta.fStar}
                        {maxIter}
                        {reason}
                        theme={theme.effective}
                    />
                </div>
            </aside>
        </div>
    {/if}
</section>
