<script setup lang="ts">
import { computed } from "vue";
import ConvergenceChart from "#lib/ConvergenceChart.vue";
import ConstraintViolationChart from "#lib/ConstraintViolationChart.vue";
import {
    COMPETITOR_BENCHMARKS as data,
    LIBRARY_COLORS,
    LIBRARY_LABELS,
    PROBLEM_LABELS,
    SOLVER_LABELS,
    librariesFor,
    type Solver,
} from "#lib/data/competitors.ts";

let { solver, problem } = defineProps<{ solver: Solver; problem: string }>();

const results = computed(() =>
    data.results.filter((r) => r.solver === solver && r.problem === problem),
);
const series = computed(() =>
    librariesFor(solver, problem).map((library) => ({
        label: LIBRARY_LABELS[library],
        color: LIBRARY_COLORS[library],
        points: results.value.find((r) => r.library === library)?.points ?? [],
    })),
);
const violations = computed(() =>
    librariesFor(solver, problem).map((library) => ({
        label: LIBRARY_LABELS[library],
        color: LIBRARY_COLORS[library],
        points:
            results.value.find((r) => r.library === library)?.violations ?? [],
    })),
);
</script>

<template>
    <div class="not-prose my-8">
        <template v-if="results.some((r) => r.summary?.verified === false)">
            <p class="mb-4 text-sm text-red-700 dark:text-red-400">
                A returned solution failed the independent accuracy check. See
                the run details below.
            </p>
        </template>

        <figure class="mx-auto w-full max-w-lg">
            <ConvergenceChart
                directLabels
                xScale="linear"
                :series="series"
                :ariaLabel="`${SOLVER_LABELS[solver]} on ${PROBLEM_LABELS[problem]}: suboptimality vs time, one line per library`"
            ></ConvergenceChart>
            <figcaption class="mt-3 text-sm text-stone-500 dark:text-stone-400">
                {{ SOLVER_LABELS[solver] }} on {{ PROBLEM_LABELS[problem] }}, n
                = {{ results[0]?.n }}. Lower objective gaps and shorter times
                are better. Time is linear; objective gap is logarithmic.
            </figcaption>
        </figure>

        <template v-if="solver === 'cobyla' || solver === 'slsqp'">
            <figure class="mx-auto mt-8 w-full max-w-lg">
                <ConstraintViolationChart
                    :tolerance="
                        solver === 'cobyla' ? 1.4901161193847656e-8 : 1e-8
                    "
                    :series="violations"
                    :ariaLabel="`${SOLVER_LABELS[solver]}: trial constraint violation vs time`"
                ></ConstraintViolationChart>
                <figcaption
                    class="mt-3 text-sm text-stone-500 dark:text-stone-400"
                >
                    Every trial and the returned point, including infeasible
                    trials. Time is linear; violation is linear within the
                    shaded feasibility band and logarithmic above it.
                </figcaption>
            </figure>
        </template>

        <template v-if="results.some((r) => r.summary)">
            <details
                class="mt-6 rounded-lg border border-stone-200 px-4 py-3 text-sm text-stone-600 dark:border-stone-800 dark:text-stone-300"
            >
                <summary class="cursor-pointer font-medium">
                    Returned solutions and evaluation counts
                </summary>
                <template v-for="r in results" :key="r.library">
                    <template v-if="r.summary">
                        <p class="mt-3">
                            <strong>{{ LIBRARY_LABELS[r.library] }}</strong>:
                            <template v-if="r.seed !== undefined"
                                >seed {{ r.seed }};</template
                            >
                            {{ r.summary.status }}. Independent accuracy check
                            {{ r.summary.verified ? "passed" : "failed" }}.
                            Returned ({{
                                r.summary.point
                                    .slice(0, 6)
                                    .map(
                                        (x) => x?.toPrecision(6) ?? "nonfinite",
                                    )
                                    .join(", ")
                            }}{{ r.n > 6 ? ", …" : "" }}), objective
                            {{
                                r.summary.cost?.toExponential(3) ?? "nonfinite"
                            }}.
                            <template v-if="solver === 'lm'">
                                {{ r.summary.residualCalls }} residual and
                                {{ r.summary.jacobianCalls }} Jacobian calls. </template
                            ><template v-else-if="solver === 'lbfgsb'">
                                {{ r.summary.iterations }} iterations;
                                {{ r.summary.objectiveCalls }} objective and
                                {{ r.summary.gradientCalls }} gradient calls
                                (fused); bound violation
                                {{
                                    r.summary.violation?.toExponential(3) ??
                                    "nonfinite"
                                }}; projected-gradient norm
                                {{
                                    r.summary.projectedGradient?.toExponential(
                                        3,
                                    ) ?? "nonfinite"
                                }}. </template
                            ><template
                                v-else-if="
                                    solver === 'bfgs' ||
                                    solver === 'nonlinear_cg'
                                "
                            >
                                {{ r.summary.iterations }} iterations;
                                {{ r.summary.objectiveCalls }} objective and
                                {{ r.summary.gradientCalls }} gradient calls. </template
                            ><template v-else-if="solver === 'cmaes'">
                                {{ r.summary.iterations }} sampled generations;
                                {{ r.summary.objectiveCalls }} objective calls. </template
                            ><template v-else>
                                Violation
                                {{
                                    r.summary.violation?.toExponential(3) ??
                                    "nonfinite"
                                }};
                                <template v-if="solver === 'slsqp'">
                                    {{ r.summary.gradientCalls }} gradient
                                    calls;
                                </template>
                                {{ r.summary.objectiveCalls }} objective and
                                {{ r.summary.constraintCalls }} constraint calls
                                ({{
                                    r.library === "basin"
                                        ? solver === "cobyla"
                                            ? "vector callbacks, including bounds"
                                            : "vector callbacks"
                                        : "scalar nonlinear callbacks; native bounds excluded"
                                }}).
                            </template>
                        </p>
                    </template>
                </template>
            </details>
        </template>
    </div>
</template>
