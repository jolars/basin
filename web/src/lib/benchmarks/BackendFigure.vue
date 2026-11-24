<script setup lang="ts">
import { computed } from "vue";
import BackendChart from "#lib/BackendChart.vue";
import {
    BACKEND_BENCHMARKS as data,
    BACKEND_COLORS,
    BACKEND_LABELS,
    PROBLEM_LABELS,
    SOLVER_LABELS,
    backendsFor,
    type Solver,
} from "#lib/data/benchmarks.ts";

let { solver, problem } = defineProps<{ solver: Solver; problem: string }>();

const results = computed(() =>
    data.results.filter((r) => r.solver === solver && r.problem === problem),
);
const dims = computed(() =>
    [...new Set(results.value.map((r) => r.n))].sort((a, b) => a - b),
);
const series = computed(() =>
    backendsFor(solver, problem).map((backend) => ({
        label: BACKEND_LABELS[backend],
        color: BACKEND_COLORS[backend],
        points: results.value
            .filter((r) => r.backend === backend)
            .map((r) => ({ n: r.n, ns: r.ns })),
    })),
);
</script>

<template>
    <figure class="not-prose mx-auto my-8 w-full max-w-lg">
        <BackendChart
            :series="series"
            :dims="dims"
            :ariaLabel="`${SOLVER_LABELS[solver]} on ${PROBLEM_LABELS[problem]}: solve time vs problem size, one line per backend`"
        ></BackendChart>
        <figcaption class="mt-3 text-sm text-stone-500 dark:text-stone-400">
            {{ SOLVER_LABELS[solver] }} on {{ PROBLEM_LABELS[problem] }}. Mean
            time per solve across the measured problem sizes; both axes are
            logarithmic. Lower is better.
        </figcaption>
    </figure>
</template>
