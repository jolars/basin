<script lang="ts">
    import { asset, resolve } from "$app/paths";
    import IconGithub from "~icons/simple-icons/github";
    import Seo from "$lib/Seo.svelte";
    import Playground from "$lib/playground/Playground.svelte";

    const features = [
        {
            title: "Pluggable solvers",
            body: "Gradient descent, Nelder–Mead, L-BFGS/L-BFGS-B, Gauss–Newton, Levenberg–Marquardt, CMA-ES, and more, driven by one shared executor loop.",
        },
        {
            title: "Multiple backends",
            body: "Run on plain Vec<f64>, nalgebra, ndarray, or faer. Each backend sits behind a single feature: no per-version feature explosion.",
        },
        {
            title: "First-class constraints",
            body: "Box bounds are part of the problem and enforced at the type level: handing a constrained problem to an unconstrained solver is a compile error.",
        },
        {
            title: "Composable termination",
            body: "Gradient, parameter, and cost tolerances, iteration and time budgets: configured uniformly across solvers, bound to the state each one exposes.",
        },
        {
            title: "Runs in the browser",
            body: "wasm-first by design: the default build pulls in no BLAS/LAPACK or threads, so basin compiles to wasm32 out of the box.",
        },
        {
            title: "Paper-anchored",
            body: "Solvers track published algorithms (Nocedal’s L-BFGS-B, Nielsen’s LM damping, Hansen’s CMA-ES) rather than ad-hoc variants.",
        },
    ];
</script>

<Seo
    title="Basin – Numerical Optimization for Rust"
    description="Basin is a numerical optimization library for Rust: pluggable solvers, multiple linear-algebra backends, first-class constraints, and a wasm-first design."
/>

<!-- Hero. `flow-root` contains the floated logo so it can't bleed into the
     next section. Below `md` the logo stacks on top (centered); at `md`+ it
     floats right and the heading, paragraph, and buttons wrap around it.

     Two logo variants swap on the `.dark` class (the same class the
     Light/Auto/Dark toggle drives, set pre-hydration in app.html so there is
     no flash): the daytime scene in light mode, the moonlit one in dark.
     Tailwind preflight renders <img> as `display: block`, so `dark:hidden` /
     `hidden dark:block` toggle visibility without changing the layout. -->
<section
    class="max-w-screen-2xl mx-auto px-4 md:px-8 pt-16 pb-12 md:pt-24 md:pb-16 flow-root"
>
    <img
        src={asset("/logo.svg")}
        alt="Visualization of optimization trajectories on the Rosenbrock function, a common test problem in optimization."
        class="mx-auto mb-8 w-3/4 max-w-sm h-auto md:float-right md:mx-0 md:mb-4 md:ml-10 md:w-2/5 md:max-w-sm dark:hidden"
    />
    <img
        src={asset("/logo-dark.svg")}
        alt="Visualization of optimization trajectories on the Rosenbrock function, a common test problem in optimization."
        class="hidden dark:block mx-auto mb-8 w-3/4 max-w-sm h-auto md:float-right md:mx-0 md:mb-4 md:ml-10 md:w-2/5 md:max-w-sm"
    />
    <span
        class="inline-block text-xs font-mono uppercase tracking-widest text-stone-500 dark:text-stone-400"
    >
        Rust
    </span>
    <h1
        class="mt-3 text-4xl md:text-6xl font-semibold tracking-tight text-balance"
    >
        Numerical Optimization in Rust
    </h1>
    <p
        class="mt-5 max-w-2xl text-lg md:text-xl text-stone-600 dark:text-stone-300 text-pretty"
    >
        <span class="font-semibold">Basin</span> is a solver framework with a generic
        executor loop over pluggable solvers, multiple linear-algebra backends, first-class
        constraints, and a wasm-first design.
    </p>
    <div class="mt-8 flex flex-wrap gap-3">
        <a
            href={resolve("/docs/getting-started/")}
            class="px-5 py-2.5 rounded-lg bg-stone-900 text-white font-medium hover:bg-stone-700 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-white transition-colors"
        >
            Get started
        </a>
        <a
            href={resolve("/visualizer/")}
            class="px-5 py-2.5 rounded-lg border border-stone-300 dark:border-stone-700 font-medium hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
        >
            Open the visualizer
        </a>
        <a
            href="https://github.com/jolars/basin"
            target="_blank"
            rel="noreferrer"
            class="px-5 py-2.5 rounded-lg border border-stone-300 dark:border-stone-700 font-medium hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors inline-flex items-center gap-2"
        >
            <IconGithub class="w-4 h-4" aria-hidden="true" /> Source Code
        </a>
    </div>
</section>

<!-- Quick taste: interactive code-gen playground -->
<section class="max-w-screen-2xl mx-auto px-4 md:px-8 pb-16">
    <Playground />
    <p class="mt-6 text-sm text-stone-600 dark:text-stone-300">
        Want to see it move? The
        <a
            class="underline decoration-dotted hover:text-stone-900 dark:hover:text-stone-100"
            href={resolve("/visualizer/")}>visualizer</a
        >
        animates these trajectories live, compiled to wasm.
    </p>
</section>

<!-- Features -->
<section
    class="border-t border-stone-200 dark:border-stone-800 bg-stone-50/60 dark:bg-stone-900/40"
>
    <div class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
        <h2 class="text-2xl font-semibold tracking-tight">What's in the box</h2>
        <div class="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {#each features as f}
                <div
                    class="rounded-xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-950 p-5"
                >
                    <h3 class="font-semibold">{f.title}</h3>
                    <p class="mt-2 text-sm text-stone-600 dark:text-stone-300">
                        {f.body}
                    </p>
                </div>
            {/each}
        </div>
    </div>
</section>

<!-- Closing CTA -->
<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <div
        class="rounded-2xl border border-stone-200 dark:border-stone-800 p-8 md:p-12 flex flex-wrap items-center justify-between gap-6"
    >
        <div>
            <h2 class="text-2xl font-semibold tracking-tight">
                Compare solvers on classical problems
            </h2>
            <p class="mt-2 text-stone-600 dark:text-stone-300 max-w-xl">
                Benchmarks and head-to-head solver comparisons are on the way.
                In the meantime, explore the docs or watch solvers converge in
                the visualizer.
            </p>
        </div>
        <div class="flex flex-wrap gap-3">
            <a
                href={resolve("/docs/getting-started/")}
                class="px-5 py-2.5 rounded-lg bg-stone-900 text-white font-medium hover:bg-stone-700 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-white transition-colors"
            >
                Read the docs
            </a>
            <a
                href={resolve("/benchmarks/")}
                class="px-5 py-2.5 rounded-lg border border-stone-300 dark:border-stone-700 font-medium hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
            >
                Benchmarks
            </a>
        </div>
    </div>
</section>
