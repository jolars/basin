<script lang="ts">
    import { onMount } from "svelte";
    import init, {
        ProblemKind,
        SolverKind,
        Run,
        evalGrid,
    } from "$lib/basin-wasm/basin_wasm";
    import { problemByKind } from "$lib/problems";
    import ContourPlot from "$lib/ContourPlot.svelte";
    import { theme } from "$lib/theme.svelte";
    import type { RunOutput } from "./codegen";

    // Mirrors the playground config that drives the generated snippet, so the
    // animation and the code always show the same run. Fixed to gradient
    // descent (constant step) on Rosenbrock — the playground's one solver.
    type Props = {
        alpha: number;
        beta: number;
        maxIter: number;
        start: [number, number];
        onPick: (p: { x: number; y: number }) => void;
        /** Reports the run's Rust-formatted result for the live output console. */
        onResult: (o: RunOutput) => void;
    };

    let { alpha, beta, maxIter, start, onPick, onResult }: Props = $props();

    // A wider view than the visualizer's square Rosenbrock window: a 6×4
    // domain (3:2) matches the wide plot panel, so the surface fills it
    // without distortion (and shows more of the valley's arms).
    const problem = {
        ...problemByKind(ProblemKind.Rosenbrock),
        domain: { xmin: -3, xmax: 3, ymin: -1, ymax: 3 },
    };
    const GRID_N = 160;

    let wasmReady = $state(false);
    // Wide `ArrayBufferLike` element type matches what wasm-bindgen returns.
    let grid: Float64Array<ArrayBufferLike> = $state(
        new Float64Array(GRID_N * GRID_N),
    );
    let trajectory: Float64Array<ArrayBufferLike> = $state(new Float64Array(0));

    // Non-reactive handles: the run effect both reads (cleanup) and writes
    // them, and reactive writes would retrigger the effect.
    let activeRun: Run | null = null;
    let frameId: number | null = null;

    onMount(async () => {
        await init();
        const d = problem.domain;
        grid = evalGrid(
            problem.kind,
            d.xmin,
            d.xmax,
            d.ymin,
            d.ymax,
            GRID_N,
            GRID_N,
        );
        wasmReady = true;
    });

    // Boot a fresh run whenever the inputs change, animating the trajectory
    // with a requestAnimationFrame loop. Mirrors the visualizer's run effect.
    $effect(() => {
        if (!wasmReady) return;
        const a = alpha;
        const b = beta;
        const mi = maxIter;
        const sx = start[0];
        const sy = start[1];

        if (frameId !== null) {
            cancelAnimationFrame(frameId);
            frameId = null;
        }
        if (activeRun !== null) {
            activeRun.free();
            activeRun = null;
        }

        const run = new Run(
            problem.kind,
            SolverKind.GradientDescent,
            sx,
            sy,
            // Constant-step gradient descent with momentum β; no early-stop
            // (NaN cost target disables it — run the full max_iter).
            { gdLineSearch: "constant", gdAlpha: a, gdBeta: b },
            mi,
            Number.NaN,
        );
        activeRun = run;
        trajectory = run.trajectoryXy();
        // Push the run's result up for the live console. Reading the
        // Rust-formatted values each frame keeps the console exact (it is the
        // program's stdout) and lets it tick toward the converged values.
        const report = (done: boolean) =>
            onResult({
                done,
                paramDebug: run.paramDebug(),
                costDisplay: run.costDisplay(),
            });
        report(false);

        const tick = () => {
            // Stale-frame guard: a newer effect run replaces `activeRun`.
            if (run !== activeRun) return;
            const result = run.stepMany(8) as { done: boolean };
            trajectory = run.trajectoryXy();
            report(result.done);
            if (result.done) {
                frameId = null;
                return;
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
</script>

{#if wasmReady}
    <ContourPlot
        {problem}
        {grid}
        nx={GRID_N}
        ny={GRID_N}
        {trajectory}
        startPoint={{ x: start[0], y: start[1] }}
        theme={theme.effective}
        {onPick}
        monochrome
    />
{:else}
    <div
        class="absolute inset-0 grid place-items-center text-xs text-slate-500 dark:text-slate-400"
    >
        Booting wasm…
    </div>
{/if}
