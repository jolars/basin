<script lang="ts">
    import { resolve } from "$app/paths";
    import ConvergenceChart from "#lib/ConvergenceChart.svelte";
    import ConstraintViolationChart from "#lib/ConstraintViolationChart.svelte";
    import Seo from "#lib/Seo.svelte";
    import {
        COMPETITOR_BENCHMARKS as data,
        COMPETITOR_CASES,
        LIBRARY_COLORS,
        LIBRARY_LABELS,
        PROBLEM_LABELS,
        SOLVER_LABELS,
        librariesFor,
        type Solver,
    } from "#lib/data/competitors.ts";

    // One convergence trace per library present for a case: suboptimality vs
    // wall-clock time, the curve each library actually walked.
    function seriesFor(solver: Solver, problem: string) {
        return librariesFor(solver, problem).map((library) => ({
            label: LIBRARY_LABELS[library],
            color: LIBRARY_COLORS[library],
            points:
                data.results.find(
                    (r) =>
                        r.solver === solver &&
                        r.problem === problem &&
                        r.library === library,
                )?.points ?? [],
        }));
    }

    function violationSeriesFor(solver: Solver, problem: string) {
        return librariesFor(solver, problem).map((library) => ({
            label: LIBRARY_LABELS[library],
            color: LIBRARY_COLORS[library],
            points:
                data.results.find(
                    (r) =>
                        r.solver === solver &&
                        r.problem === problem &&
                        r.library === library,
                )?.violations ?? [],
        }));
    }
</script>

<Seo
    title="Competitor Benchmarks – Basin"
    description="Basin versus established Rust optimization crates including levenberg-marquardt, cobyla, argmin, gomez, slsqp, and NLopt on matched problems, as suboptimality-vs-time convergence traces."
/>

<section class="site-container py-16">
    <p class="text-sm text-stone-500 dark:text-stone-400">
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("benchmarks/")}>Benchmarks</a
        >
        <span class="text-stone-400 dark:text-stone-600">/</span> Competitors
    </p>
    <h1 class="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">
        Competitor Benchmarks
    </h1>
    <p class="mt-3 max-w-3xl text-stone-600 dark:text-stone-300">
        This benchmark compares Basin against <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://argmin-rs.org/"
            target="_blank"
            rel="noreferrer">argmin</a
        >,
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://docs.rs/gomez/"
            target="_blank"
            rel="noreferrer">gomez</a
        >,
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://docs.rs/slsqp/1.0.2/slsqp/"
            target="_blank"
            rel="noreferrer">slsqp</a
        >,
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://nlopt.readthedocs.io/"
            target="_blank"
            rel="noreferrer">nlopt</a
        >,
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://docs.rs/levenberg-marquardt/0.15.0/levenberg_marquardt/"
            target="_blank"
            rel="noreferrer">levenberg-marquardt</a
        >, and
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href="https://docs.rs/cobyla/"
            target="_blank"
            rel="noreferrer">cobyla</a
        >. The main plots show <em>suboptimality</em>,
        <code class="font-mono">f(x) − f*</code>, against wall-clock time on
        log–log axes: how far down the objective each library gets, and how long
        it spends getting there. Lower and further left is better.
    </p>

    <div class="mt-6 grid gap-6 lg:grid-cols-2">
        {#each COMPETITOR_CASES as c}
            <div
                class="rounded-xl border border-stone-200 dark:border-stone-800 p-5"
            >
                <h3 class="text-sm font-semibold">
                    {SOLVER_LABELS[c.solver]}
                    <span class="text-stone-400 dark:text-stone-500">·</span>
                    {PROBLEM_LABELS[c.problem]}
                </h3>
                <p class="mt-1 text-xs text-stone-500 dark:text-stone-400">
                    {c.blurb}
                </p>
                {#if data.results.some((r) => r.solver === c.solver && r.problem === c.problem && r.summary?.verified === false)}
                    <p class="mt-2 text-xs text-red-700 dark:text-red-400">
                        A returned solution failed the independent accuracy
                        check. See the run details below.
                    </p>
                {/if}
                <div class="mt-3">
                    <ConvergenceChart
                        directLabels
                        series={seriesFor(c.solver, c.problem)}
                        ariaLabel={`${SOLVER_LABELS[c.solver]} on ${PROBLEM_LABELS[c.problem]}: suboptimality vs wall-clock time, one line per library`}
                    />
                </div>
                {#if c.solver === "cobyla"}
                    <div
                        class="mt-5 border-t border-stone-200 pt-4 dark:border-stone-800"
                    >
                        <ConstraintViolationChart
                            tolerance={1.4901161193847656e-8}
                            series={violationSeriesFor(c.solver, c.problem)}
                            ariaLabel="COBYLA on the constrained quadratic: trial constraint violation vs wall-clock time"
                        />
                    </div>
                {/if}
                {#if c.solver === "lm" || c.solver === "cobyla"}
                    <details
                        class="mt-4 text-xs text-stone-500 dark:text-stone-400"
                    >
                        <summary class="cursor-pointer"
                            >Returned solutions and evaluation counts</summary
                        >
                        {#each data.results.filter((r) => r.solver === c.solver && r.problem === c.problem) as r}
                            {#if r.summary}
                                <p class="mt-3">
                                    <strong>{LIBRARY_LABELS[r.library]}</strong
                                    >:
                                    {r.summary.status}. Independent accuracy
                                    check
                                    {r.summary.verified ? "passed" : "failed"}.
                                    Returned ({r.summary.point
                                        .map(
                                            (x) =>
                                                x?.toPrecision(6) ??
                                                "nonfinite",
                                        )
                                        .join(", ")}), objective {r.summary.cost?.toExponential(
                                        3,
                                    ) ?? "nonfinite"}.
                                    {#if c.solver === "lm"}
                                        {r.summary.residualCalls} residual and
                                        {r.summary.jacobianCalls} Jacobian calls.
                                    {:else}
                                        Violation {r.summary.violation?.toExponential(
                                            3,
                                        ) ?? "nonfinite"};
                                        {r.summary.objectiveCalls} objective and
                                        {r.summary.constraintCalls} constraint calls
                                        ({r.library === "basin"
                                            ? "vector callbacks, including bounds"
                                            : "scalar nonlinear callbacks; native bounds excluded"}).
                                    {/if}
                                </p>
                            {/if}
                        {/each}
                    </details>
                {/if}
            </div>
        {/each}
    </div>

    <p class="mt-8 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
        Measured {data.generatedAt} on {data.env.cpu}
        ({data.env.os}/{data.env.arch}), using {data.env.compiler}. Serial runs
        use one thread. LM uses nalgebra {data.versions.nalgebra}; other cases
        use the
        <code class="font-mono">Vec&lt;f64&gt;</code> backend (gomez through its
        own bundled
        <code class="font-mono">nalgebra::DVector&lt;f64&gt;</code>). The GD,
        NM, and L-BFGS cases run from the classic Rosenbrock start to a
        {data.iterations}-iteration cap. Each point is the median wall-clock
        time per sample over {data.repetitions} repeated runs (nlopt, which exposes
        no per-iteration hook, is sampled per function evaluation as a best-so-far
        curve); the solvers are deterministic, so only the timing varies.
    </p>

    <p class="mt-4 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
        For SLSQP, all three curves record the best objective seen at function
        evaluations, include solve initialization, and end after termination.
        Basin uses Kraft's composite accuracy test at 1e-10; slsqp and NLopt use
        an absolute function-change tolerance of 1e-10, with their other
        optional tolerances disabled. These stopping rules differ. Basin has a
        200-iteration safety cap, and the references have a 200-evaluation cap.
        All three returned solutions pass an independent objective and gradient
        check before the harness emits results.
    </p>

    <p class="mt-4 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
        LM compares Basin {data.versions.basin} with levenberg-marquardt
        {data.versions["levenberg-marquardt"]}. Both use the same observations,
        start, analytic Jacobian, column scaling, and initial scaled step-bound
        factor 100. Function-reduction, orthogonality, and scaled-radius
        tolerances are 30 times f64 machine epsilon (about 6.66e-15). Basin
        disables its absolute-gradient and unscaled-step checks and uses
        trust-region damping with pivoted QR. The reference uses its MINPACK
        implementation. Internal damping searches, acceptance rules, and
        numerical safeguards differ, so these curves do not isolate framework
        overhead. Both have a 300-residual-call budget; Basin also has a
        200-iteration safety cap.
    </p>

    <p class="mt-4 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
        COBYLA compares Basin’s PRIMA implementation with cobyla
        {data.versions.cobyla}, derived from NLopt. Both use initial radius 0.5,
        final radius about 7.45e-9, a 100-objective-call budget, and disabled
        optional objective-change stops. Feasible trials have maximum constraint
        violation at most 1.49e-8. Infeasible trials cannot improve the
        objective curve. The violation plot includes every trial and the
        returned point. Its time axis is linear; violation uses a linear scale
        from zero to the tolerance and a logarithmic scale above it.
    </p>

    <p class="mt-4 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
        LM and COBYLA include construction, initialization, callbacks, and
        solver teardown. Timing excludes fixture preparation and independent
        solution checks. Curves include callback instrumentation; COBYLA
        feasibility is reconstructed afterward from recorded trial points. Each
        comparison warms up once before its repeated runs. Endpoints include
        work after the last evaluation, and independent accuracy checks do not
        relabel termination as convergence. Objective gaps are floored at 1e-16
        for display, including small negative gaps permitted by the feasibility
        tolerance. Source revision {data.env.revision.slice(0, 7)}{data.env
            .dirty
            ? " with local changes"
            : ""}.
    </p>

    <p class="mt-6 text-sm text-stone-500 dark:text-stone-400">
        To watch basin's solvers converge interactively, try the <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("visualizer/")}>visualizer</a
        >.
    </p>
</section>
