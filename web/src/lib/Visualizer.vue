<script setup lang="ts">
import { ref, computed, watchEffect, onMounted } from "vue";
import init, {
    ProblemKind,
    SolverKind,
    Run,
    evalGrid,
} from "#lib/basin-wasm/basin_wasm.js";
import { PROBLEMS, problemByKind, SUBOPT_TARGET } from "#lib/problems.ts";
import { SOLVERS, defaultOptionValues } from "#lib/solvers.ts";
import ContourPlot from "#lib/ContourPlot.vue";
import CostChart from "#lib/CostChart.vue";
import Controls from "#lib/Controls.vue";
import { useData } from "vitepress";
const { isDark } = useData();

// Wasm boot. The viz waits on this once; everything downstream assumes
// the module is already loaded.
let wasmReady = ref(false);

// Default option values for a solver, with the problem-dependent α
// default folded in for gradient descent.
function initOptionValues(
    sk: SolverKind,
    pk: ProblemKind,
): Record<string, string | number> {
    const meta = SOLVERS.find((s) => s.kind === sk)!;
    const vals = defaultOptionValues(meta);
    if ("gdAlpha" in vals) vals.gdAlpha = problemByKind(pk).gdAlphaDefault;
    return vals;
}

let problemKind = ref<ProblemKind>(ProblemKind.Rosenbrock);
let solverKind = ref<SolverKind>(SolverKind.GradientDescent);
// Solver-specific option values, keyed by option id (see solvers.ts).
let optionValues = ref<Record<string, string | number>>(
    initOptionValues(SolverKind.GradientDescent, ProblemKind.Rosenbrock),
);
let maxIter = ref(500);
let startPoint = ref({ x: -1.5, y: 2.0 });

// Heatmap grid. Recomputed when the problem changes (or on first boot).
const GRID_N = 192;
// Use the wide `Float64Array<ArrayBufferLike>` type so values returned
// from wasm-bindgen (which use `ArrayBufferLike`) assign cleanly.
let grid = ref<Float64Array<ArrayBufferLike>>(
    new Float64Array(GRID_N * GRID_N),
);

// Animated trajectory and cost log fed to the children.
let trajectory = ref<Float64Array<ArrayBufferLike>>(new Float64Array(0));
let costs = ref<Float64Array<ArrayBufferLike>>(new Float64Array(0));
// Current-generation population for stochastic solvers (CMA-ES, DE, RS,
// SSGA). Empty Float64Array for the local solvers: the contour plot
// skips rendering when length is zero.
let population = ref<Float64Array<ArrayBufferLike>>(new Float64Array(0));
let reason = ref("");

let problemMeta = computed(() => problemByKind(problemKind.value));
let solverMeta = computed(() =>
    SOLVERS.find((s) => s.kind === solverKind.value)!,
);

// Run and frame handles belong to the current watcher invocation.
let activeRun: Run | null = null;
let frameId: number | null = null;

// Refresh the heatmap when the problem changes.
watchEffect(() => {
    if (!wasmReady.value) return;
    const d = problemMeta.value.domain;
    grid.value = evalGrid(
        problemMeta.value.kind,
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
watchEffect((onCleanup) => {
    if (!wasmReady.value) return;
    const pk = problemKind.value;
    const sk = solverKind.value;
    const mi = maxIter.value;
    const sx = startPoint.value.x;
    const sy = startPoint.value.y;
    // Solver options crossing the wasm boundary (see RunOptions in
    // crates/basin-wasm/src/lib.rs). Each solver reads only what it
    // needs; reading the fields here tracks them as effect deps.
    const d = problemMeta.value.domain;
    const opts = {
        gdLineSearch: optionValues.value.gdLineSearch ?? "constant",
        gdAlpha: optionValues.value.gdAlpha ?? problemMeta.value.gdAlphaDefault,
        // β = 0: the visualizer has no momentum knob (the landing-page
        // playground does). Plain steepest descent for the GD solver.
        gdBeta: 0,
        lbfgsM: optionValues.value.lbfgsM ?? 10,
        // Stochastic solvers (CMA-ES, DE, RandomSearch, SSGA). NaN on
        // cmaSigma → wasm picks a viewport-scaled default.
        seed: optionValues.value.seed ?? 0,
        cmaSigma: optionValues.value.cmaSigma ?? NaN,
        cmaLambda: optionValues.value.cmaLambda ?? 0,
        dePopSize: optionValues.value.dePopSize ?? 0,
        deF: optionValues.value.deF ?? 0.8,
        deCr: optionValues.value.deCr ?? 0.9,
        rsLambda: optionValues.value.rsLambda ?? 16,
        ssgaPopSize: optionValues.value.ssgaPopSize ?? 0,
        // Box bounds for DE, SSGA, or RandomSearch: they use the visible
        // viewport as the feasible region (the right semantics for a 2D
        // demo). Reading the fields here ties the effect to domain
        // changes when the user switches problems.
        xmin: d.xmin,
        xmax: d.xmax,
        ymin: d.ymin,
        ymax: d.ymax,
    };
    // Stop early once the cost is within SUBOPT_TARGET of the known
    // optimum: the same value the cost chart uses as its log floor.
    const stopAtCost = problemMeta.value.fStar + SUBOPT_TARGET;

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
    trajectory.value = run.trajectoryXy();
    costs.value = run.costs();
    population.value = run.populationXy();
    reason.value = "";

    // Generation-based solvers (CMA-ES, DE, or RS) advance one full
    // generation per `next_iter`, so 8/frame races through the run in
    // a flash. The solver's meta gates this. Fractional rates (< 1) mean
    // "step every Nth frame": for population-based solvers where each
    // generation is a big visible jump, this gives the eye time to read
    // the cloud between updates. SSGA's "iter" is a single offspring
    // evaluation, so it keeps the default chunk of 8.
    const itersPerFrame = solverMeta.value.itersPerFrame ?? 8;
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
            trajectory.value = run.trajectoryXy();
            costs.value = run.costs();
            population.value = run.populationXy();
            if (result.done) {
                reason.value = result.reason ?? "";
                frameId = null;
                return;
            }
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

onMounted(async () => {
    await init();
    wasmReady.value = true;
    // Seed start point near a visually interesting corner of the
    // initial problem (Rosenbrock).
    startPoint.value = { x: -1.5, y: 2.0 };
});

function handlePick(p: { x: number; y: number }) {
    startPoint.value = p;
}

function handleControlChange(patch: {
    problemKind?: ProblemKind;
    solverKind?: SolverKind;
    maxIter?: number;
}) {
    if (
        patch.problemKind !== undefined &&
        patch.problemKind !== problemKind.value
    ) {
        problemKind.value = patch.problemKind;
        // Re-center start and reset the α default for the new problem.
        const d = problemByKind(problemKind.value).domain;
        startPoint.value = {
            x: d.xmin + 0.25 * (d.xmax - d.xmin),
            y: d.ymin + 0.75 * (d.ymax - d.ymin),
        };
        if ("gdAlpha" in optionValues.value) {
            optionValues.value = {
                ...optionValues.value,
                gdAlpha: problemByKind(problemKind.value).gdAlphaDefault,
            };
        }
    }
    if (
        patch.solverKind !== undefined &&
        patch.solverKind !== solverKind.value
    ) {
        solverKind.value = patch.solverKind;
        // Reset options to the new solver's schema defaults.
        optionValues.value = initOptionValues(
            solverKind.value,
            problemKind.value,
        );
        // Clear the stale population so we don't render the old solver's
        // dots for the one frame before the run-effect reseeds them.
        population.value = new Float64Array(0);
    }
    if (patch.maxIter !== undefined) maxIter.value = patch.maxIter;
}

function handleOptionChange(id: string, value: string | number) {
    optionValues.value = { ...optionValues.value, [id]: value };
}
</script>

<template>
    <section
        class="min-h-[calc(100dvh-9rem)] visualizer-container [container-type:inline-size] py-6 flex flex-col gap-6"
    >
        <header class="flex flex-wrap items-start justify-between gap-4">
            <div>
                <h1 class="text-2xl md:text-3xl font-semibold tracking-tight">
                    Solver visualizer
                </h1>
                <p class="text-stone-600 dark:text-stone-400 text-sm mt-1">
                    Live wasm-driven 2D trajectories. Click on the contour to
                    reset the start point.
                </p>
            </div>
            <p
                class="text-xs text-stone-500 dark:text-stone-500 font-mono hidden md:block self-center"
            >
                {{ solverMeta.blurb }}
            </p>
        </header>

        <template v-if="!wasmReady">
            <p class="text-stone-500 dark:text-stone-400">
                Loading wasm…
            </p> </template
        ><template v-else>
            <div
                class="grid grid-cols-1 lg:grid-cols-[2fr_1fr] lg:h-[clamp(36rem,calc((100cqw-1.5rem)*2/3),48rem)] lg:grid-rows-[minmax(0,1fr)] gap-6 min-h-0"
            >
                <div
                    class="relative bg-stone-100 dark:bg-stone-900 rounded-lg overflow-hidden w-full aspect-square max-h-[48rem] lg:aspect-auto lg:h-full"
                >
                    <ContourPlot
                        :problem="problemMeta"
                        :grid="grid"
                        :nx="GRID_N"
                        :ny="GRID_N"
                        :trajectory="trajectory"
                        :population="population"
                        :startPoint="startPoint"
                        :theme="isDark ? 'dark' : 'light'"
                        :onPick="handlePick"
                        monochrome
                    ></ContourPlot>
                </div>
                <aside class="flex flex-col gap-6 min-w-0 min-h-0">
                    <div
                        class="bg-stone-100 dark:bg-stone-900 rounded-lg p-4 lg:min-h-0 lg:overflow-y-auto"
                    >
                        <Controls
                            :problemKind="problemKind"
                            :solverKind="solverKind"
                            :solverOptions="solverMeta.options"
                            :optionValues="optionValues"
                            :maxIter="maxIter"
                            :startPoint="startPoint"
                            :onChange="handleControlChange"
                            :onOptionChange="handleOptionChange"
                        ></Controls>
                    </div>
                    <div
                        class="bg-stone-100 dark:bg-stone-900 rounded-lg p-3 h-56 lg:h-auto lg:min-h-40 lg:flex-1 lg:shrink-0"
                    >
                        <CostChart
                            :costs="costs"
                            :fStar="problemMeta.fStar"
                            :maxIter="maxIter"
                            :reason="reason"
                            :theme="isDark ? 'dark' : 'light'"
                        ></CostChart>
                    </div>
                </aside>
            </div>
        </template>
    </section>
</template>
