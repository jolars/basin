<script setup lang="ts">
import { computed, ref } from "vue";
import { withBase } from "vitepress";

type Problem = "objective" | "scalar" | "least-squares";
type Constraint =
    "bounds" | "linear" | "nonlinear-equalities" | "nonlinear-inequalities";
type Recommendation = {
    name: string;
    path: string;
    reason: string;
};

const problem = ref<Problem>("objective");
const constraints = ref<Constraint[]>([]);
const smoothness = ref<"unknown" | "smooth" | "nonsmooth">("unknown");
const convexity = ref<"unknown" | "convex" | "nonconvex">("unknown");
const objectiveDerivatives = ref(1);
const scalarDerivative = ref(false);
const jacobian = ref(true);
const constraintDerivatives = ref(false);
const globalSearch = ref(false);
const exploreBasins = computed(
    () => convexity.value !== "convex" && globalSearch.value,
);
const canUseObjectiveDerivatives = computed(
    () => smoothness.value !== "nonsmooth" && !exploreBasins.value,
);

const constraintOptions: { value: Constraint; label: string }[] = [
    { value: "bounds", label: "Box bounds" },
    { value: "linear", label: "Linear constraints" },
    { value: "nonlinear-equalities", label: "Nonlinear equalities" },
    { value: "nonlinear-inequalities", label: "Nonlinear inequalities" },
];

const hasNonlinearConstraints = computed(() =>
    constraints.value.some(
        (constraint) =>
            constraint === "nonlinear-equalities" ||
            constraint === "nonlinear-inequalities",
    ),
);
const unconstrained = computed(() => constraints.value.length === 0);
const boundsOnly = computed(
    () =>
        constraints.value.length === 1 && constraints.value.includes("bounds"),
);

const nonlinearInequalitiesOnly = computed(
    () =>
        constraints.value.length === 1 &&
        constraints.value.includes("nonlinear-inequalities"),
);

const recommendations = computed<Recommendation[]>(() => {
    switch (problem.value) {
        case "scalar": {
            const methods = [];
            if (scalarDerivative.value) {
                methods.push({
                    name: "Brent with derivatives",
                    path: "scalar-minimization/brent-derivative",
                    reason: "Use the first derivative to propose steps within finite bounds.",
                });
            }
            return [
                ...methods,
                {
                    name: "Brent minimization",
                    path: "scalar-minimization/brent",
                    reason: "Combine parabolic interpolation and golden-section steps using objective values.",
                },
                {
                    name: "Golden-section search",
                    path: "scalar-minimization/golden-section",
                    reason: "Narrow a finite interval using objective comparisons.",
                },
            ];
        }
        case "least-squares":
            if (!jacobian.value || (!unconstrained.value && !boundsOnly.value))
                return [];
            return boundsOnly.value
                ? [
                      {
                          name: "Trust-region reflective",
                          path: "nonlinear-least-squares/trust-region-reflective",
                          reason: "Use a bound-aware trust region and a rank-aware dense factorization.",
                      },
                      {
                          name: "Bounded Levenberg–Marquardt",
                          path: "nonlinear-least-squares/bounded-levenberg-marquardt",
                          reason: "Use damped steps with box scaling. Some backends also support sparse Jacobians.",
                      },
                  ]
                : [
                      {
                          name: "Levenberg–Marquardt",
                          path: "nonlinear-least-squares/levenberg-marquardt",
                          reason: "Use damping to control steps based on the residual model.",
                      },
                      {
                          name: "Levenberg–Marquardt QR",
                          path: "nonlinear-least-squares/levenberg-marquardt-qr",
                          reason: "Factor the Jacobian directly to avoid forming the Gram matrix.",
                      },
                  ];
        case "objective":
            break;
    }

    if (exploreBasins.value) {
        if (unconstrained.value) {
            return [
                {
                    name: "CMA-ES",
                    path: "global-and-stochastic/cma-es",
                    reason: "Adapt a sampling distribution to explore multiple basins.",
                },
            ];
        }
        if (boundsOnly.value) {
            return [
                {
                    name: "Differential evolution",
                    path: "global-and-stochastic/de",
                    reason: "Explore a finite box using a population of candidate points.",
                },
                {
                    name: "DIRECT",
                    path: "global-and-stochastic/direct",
                    reason: "Explore a finite box by deterministic subdivision.",
                },
            ];
        }
        return [];
    }

    if (smoothness.value === "nonsmooth") {
        if (
            !unconstrained.value &&
            !boundsOnly.value &&
            !nonlinearInequalitiesOnly.value
        ) {
            return [];
        }
        return [
            {
                name: "MADS",
                path: "derivative-free/mads",
                reason: "Poll nearby points without fitting a smooth model. Choose the mode that matches your constraints.",
            },
        ];
    }

    const methods: Recommendation[] = [];
    if (unconstrained.value) {
        if (objectiveDerivatives.value === 2) {
            methods.push({
                name: "Trust-region Newton",
                path: "trust-region/trust-region-newton",
                reason: "Use curvature from a Hessian or Hessian-vector products.",
            });
        }
        if (objectiveDerivatives.value >= 1) {
            return [
                ...methods,
                {
                    name: "L-BFGS",
                    path: "quasi-newton/l-bfgs",
                    reason: "Approximate curvature with a short history of gradients.",
                },
                {
                    name: "BFGS",
                    path: "quasi-newton/bfgs",
                    reason: "Maintain a dense curvature model when the dimension is modest.",
                },
            ];
        }
        return [
            {
                name: "NEWUOA",
                path: "derivative-free/newuoa",
                reason: "Build a quadratic model of a locally smooth objective.",
            },
            {
                name: "Nelder–Mead",
                path: "derivative-free/nelder-mead",
                reason: "Use a simplex search in a modest number of dimensions.",
            },
        ];
    }

    if (boundsOnly.value) {
        return objectiveDerivatives.value >= 1
            ? [
                  {
                      name: "L-BFGS-B",
                      path: "quasi-newton/l-bfgs-b",
                      reason: "Build a limited-memory curvature model while enforcing the box.",
                  },
                  {
                      name: "Projected gradient descent",
                      path: "first-order/projected-gradient-descent",
                      reason: "Take gradient steps and project them onto the box.",
                  },
              ]
            : [
                  {
                      name: "BOBYQA",
                      path: "derivative-free/bobyqa",
                      reason: "Build a quadratic model of a smooth objective within the box.",
                  },
                  {
                      name: "MADS",
                      path: "derivative-free/mads",
                      reason: "Use its bounded polling mode when a smooth model is a poor fit.",
                  },
              ];
    }

    if (
        objectiveDerivatives.value >= 1 &&
        (!hasNonlinearConstraints.value || constraintDerivatives.value)
    ) {
        methods.push({
            name: "SLSQP",
            path: "general-constrained/slsqp",
            reason: "Use objective gradients and any required constraint Jacobians to solve local quadratic subproblems.",
        });
    }
    if (!hasNonlinearConstraints.value) {
        methods.push({
            name: "LINCOA",
            path: "derivative-free/lincoa",
            reason: "Build a quadratic objective model while handling linear constraints.",
        });
    }
    methods.push({
        name: "COBYLA",
        path: "derivative-free/cobyla",
        reason: "Model objective and constraint values without derivatives. Trial points may be infeasible.",
    });
    if (nonlinearInequalitiesOnly.value) {
        methods.push({
            name: "MADS",
            path: "derivative-free/mads",
            reason: "Use progressive-barrier polling for nonlinear inequalities on their own.",
        });
    }
    return methods;
});
</script>

<template>
    <div class="vp-raw solver-finder">
        <div class="finder-controls">
            <div class="finder-selects">
                <label class="finder-select">
                    What are you solving?
                    <select v-model="problem">
                        <option value="objective">
                            Minimize a function of several variables
                        </option>
                        <option value="scalar">
                            Minimize a function of one variable
                        </option>
                        <option value="least-squares">
                            Fit residuals by least squares
                        </option>
                    </select>
                </label>
                <template v-if="problem === 'objective'">
                    <label class="finder-select">
                        Is the objective smooth?
                        <select v-model="smoothness">
                            <option value="unknown">Not sure</option>
                            <option value="smooth">Smooth</option>
                            <option value="nonsmooth">
                                Has kinks or discontinuities
                            </option>
                        </select>
                    </label>
                    <label class="finder-select">
                        Is the problem convex?
                        <select v-model="convexity">
                            <option value="unknown">Not sure</option>
                            <option value="convex">
                                Convex objective and feasible region
                            </option>
                            <option value="nonconvex">Nonconvex</option>
                        </select>
                    </label>
                </template>
                <label
                    v-if="problem === 'objective' && canUseObjectiveDerivatives"
                    class="finder-select"
                >
                    What can you compute?
                    <select v-model.number="objectiveDerivatives">
                        <option :value="0">Objective values</option>
                        <option :value="1">Objective and gradient</option>
                        <option :value="2">
                            Objective, gradient, and Hessian or Hessian-vector
                            products
                        </option>
                    </select>
                </label>
                <label v-else-if="problem === 'scalar'" class="finder-select">
                    What can you compute?
                    <select v-model="scalarDerivative">
                        <option :value="false">Objective values</option>
                        <option :value="true">
                            Objective and first derivative
                        </option>
                    </select>
                </label>
                <label
                    v-else-if="problem === 'least-squares'"
                    class="finder-select"
                >
                    What can you compute?
                    <select v-model="jacobian">
                        <option :value="true">Residuals and Jacobian</option>
                        <option :value="false">Residuals only</option>
                    </select>
                </label>
            </div>

            <fieldset
                v-if="problem === 'objective' || problem === 'least-squares'"
            >
                <legend>What constraints do you have?</legend>
                <div class="finder-checkboxes">
                    <label
                        v-for="option in constraintOptions"
                        :key="option.value"
                    >
                        <input
                            v-model="constraints"
                            type="checkbox"
                            :value="option.value"
                        />
                        {{ option.label }}
                    </label>
                </div>
            </fieldset>

            <label
                v-if="
                    problem === 'objective' &&
                    canUseObjectiveDerivatives &&
                    objectiveDerivatives >= 1 &&
                    hasNonlinearConstraints
                "
            >
                <input v-model="constraintDerivatives" type="checkbox" />
                I can compute Jacobians for my nonlinear constraints
            </label>
            <label v-if="problem === 'objective' && convexity !== 'convex'">
                <input v-model="globalSearch" type="checkbox" />
                Explore multiple local minima
            </label>
        </div>

        <div class="finder-results" aria-live="polite">
            <p
                v-if="problem === 'objective' && convexity === 'convex'"
                class="finder-note"
            >
                Any local minimum is global when the objective and feasible
                region are both convex.
            </p>
            <p
                v-else-if="
                    problem === 'objective' &&
                    convexity === 'nonconvex' &&
                    !exploreBasins
                "
                class="finder-note"
            >
                A local solver can find different solutions from different
                starting points. Explore multiple local minima if you want a
                broader search.
            </p>
            <p
                v-if="
                    problem === 'objective' &&
                    smoothness === 'nonsmooth' &&
                    !exploreBasins &&
                    recommendations.length
                "
                class="finder-note"
            >
                MADS can evaluate discontinuous objectives. Its stationarity
                results require additional regularity.
            </p>
            <div v-if="recommendations.length">
                <h3 id="suggested-solvers" class="finder-results-heading">
                    Suggested solvers
                </h3>
                <ul class="finder-matches" aria-labelledby="suggested-solvers">
                    <li
                        v-for="solver in recommendations"
                        :key="solver.path"
                        class="finder-card"
                    >
                        <h4>
                            <a
                                :href="
                                    withBase(
                                        '/guide/solvers/' + solver.path + '/',
                                    )
                                "
                                >{{ solver.name }}</a
                            >
                        </h4>
                        <p>{{ solver.reason }}</p>
                    </li>
                </ul>
            </div>
            <p
                v-else-if="
                    problem === 'least-squares' && !unconstrained && !boundsOnly
                "
                class="finder-note"
            >
                For linear or nonlinear constraints, minimize the sum of squared
                residuals as an objective and choose a solver that supports
                those constraints.
            </p>
            <p v-else-if="problem === 'least-squares'" class="finder-note">
                These least-squares solvers need a Jacobian. Supply one or
                minimize the sum of squared residuals as a scalar objective
                using a derivative-free solver.
            </p>
            <p v-else-if="exploreBasins" class="finder-note">
                No suggested global solver handles this combination of
                constraints. Clear “Explore multiple local minima” to see
                compatible local methods.
            </p>
            <p v-else class="finder-note">
                No suggested nonsmooth solver supports this combination of
                constraints. MADS supports box bounds alone or nonlinear
                inequalities alone.
            </p>
        </div>
    </div>
</template>

<style scoped>
.solver-finder {
    display: grid;
    gap: 20px;
    margin: 20px 0;
}

.finder-controls,
.finder-results {
    min-width: 0;
}

.finder-controls {
    display: grid;
    align-content: start;
    gap: 16px;
    padding: 16px;
    border: 1px solid var(--vp-c-divider);
    border-radius: 8px;
    background: var(--vp-c-bg-soft);
}

.finder-selects {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr));
    gap: 12px;
}

.finder-select {
    display: grid;
    min-width: 0;
    gap: 6px;
    font-weight: 600;
}

.finder-controls select {
    width: 100%;
    min-width: 0;
    min-height: 40px;
    padding: 6px 10px;
    border: 1px solid var(--vp-c-divider);
    border-radius: 6px;
    color: var(--vp-c-text-1);
    background: var(--vp-c-bg);
    font-weight: 400;
    appearance: auto;
    color-scheme: light dark;
}

.finder-controls fieldset {
    min-width: 0;
    margin: 0;
    padding: 0;
    border: 0;
}

.finder-controls legend {
    margin-bottom: 6px;
    font-weight: 600;
}

.finder-checkboxes {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr));
    gap: 4px 12px;
}

.finder-controls input[type="checkbox"] {
    margin-inline-end: 4px;
    vertical-align: baseline;
    accent-color: var(--vp-c-brand-1);
}

.finder-controls :focus-visible,
.finder-matches a:focus-visible {
    outline: 2px solid var(--vp-c-brand-1);
    outline-offset: 2px;
}

.finder-results {
    display: grid;
    gap: 12px;
}

.finder-results-heading {
    margin: 0 0 12px;
    font-size: 18px;
    font-weight: 600;
}

.finder-matches {
    display: grid;
    gap: 10px;
    margin: 0;
    padding: 0;
    list-style: none;
}

.finder-card {
    padding: 12px 16px;
    border: 1px solid var(--vp-c-divider);
    border-radius: 8px;
}

.finder-card h4 {
    margin: 0;
    font-size: 16px;
}

.finder-matches a {
    color: var(--vp-c-brand-1);
    font-weight: 600;
    text-decoration: underline;
    text-underline-offset: 2px;
}

.finder-card p {
    margin: 4px 0 0;
    color: var(--vp-c-text-2);
}

.finder-note {
    margin: 0;
    color: var(--vp-c-text-2);
}
</style>
