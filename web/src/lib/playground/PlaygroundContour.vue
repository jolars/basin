<script setup lang="ts">
import { ref, watchEffect, onMounted } from "vue";
import init, {
    ProblemKind,
    SolverKind,
    Run,
    evalGrid,
} from "#lib/basin-wasm/basin_wasm.js";
import { problemByKind } from "#lib/problems.ts";
import ContourPlot from "#lib/ContourPlot.vue";
import { useData } from "vitepress";
const { isDark } = useData();
import type { RunOutput } from "./codegen";

// Mirrors the playground config that drives the generated snippet, so the
// animation and the code always show the same run. Fixed to gradient
// descent (constant step) on Rosenbrock: the playground's one solver.
type Props = {
    alpha: number;
    beta: number;
    maxIter: number;
    start: [number, number];
    onPick: (p: { x: number; y: number }) => void;
    /** Reports the run's Rust-formatted result for the live output console. */
    onResult: (o: RunOutput) => void;
};

let { alpha, beta, maxIter, start, onPick, onResult } = defineProps<Props>();

// Show more of the valley's arms than the visualizer's square Rosenbrock window.
const problem = {
    ...problemByKind(ProblemKind.Rosenbrock),
    domain: { xmin: -3, xmax: 3, ymin: -1, ymax: 3 },
};
const GRID_N = 160;

let wasmReady = ref(false);
// Wide `ArrayBufferLike` element type matches what wasm-bindgen returns.
let grid = ref<Float64Array<ArrayBufferLike>>(
    new Float64Array(GRID_N * GRID_N),
);
let trajectory = ref<Float64Array<ArrayBufferLike>>(new Float64Array(0));

// Non-reactive handles: the run effect both reads (cleanup) and writes
// them, and reactive writes would retrigger the effect.
let activeRun: Run | null = null;
let frameId: number | null = null;

onMounted(async () => {
    await init();
    const d = problem.domain;
    grid.value = evalGrid(
        problem.kind,
        d.xmin,
        d.xmax,
        d.ymin,
        d.ymax,
        GRID_N,
        GRID_N,
    );
    wasmReady.value = true;
});

// Boot a fresh run whenever the inputs change, animating the trajectory
// with a requestAnimationFrame loop. Mirrors the visualizer's run effect.
watchEffect((onCleanup) => {
    if (!wasmReady.value) return;
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
        // (NaN cost target disables it: run the full max_iter).
        { gdLineSearch: "constant", gdAlpha: a, gdBeta: b },
        mi,
        Number.NaN,
    );
    activeRun = run;
    trajectory.value = run.trajectoryXy();
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
        trajectory.value = run.trajectoryXy();
        report(result.done);
        if (result.done) {
            frameId = null;
            return;
        }
        frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);

    onCleanup(() => {
        if (frameId !== null) {
            cancelAnimationFrame(frameId);
            frameId = null;
        }
        if (activeRun === run) {
            run.free();
            activeRun = null;
        }
    });
});
</script>

<template>
    <template v-if="wasmReady">
        <ContourPlot
            :problem="problem"
            :grid="grid"
            :nx="GRID_N"
            :ny="GRID_N"
            :trajectory="trajectory"
            :startPoint="{ x: start[0], y: start[1] }"
            :theme="isDark ? 'dark' : 'light'"
            :onPick="onPick"
            monochrome
        ></ContourPlot> </template
    ><template v-else>
        <div
            class="absolute inset-0 grid place-items-center text-xs text-stone-500 dark:text-stone-400"
        >
            Booting wasm…
        </div>
    </template>
</template>
