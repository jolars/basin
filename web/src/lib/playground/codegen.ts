/**
 * Reactive Rust code generation for the landing-page playground.
 *
 * This module is the single source of truth for the snippet the playground
 * shows. It is deliberately **pure** — no wasm, no DOM, no Svelte — so it
 * renders on the server (the landing page prerenders the default snippet)
 * and so the CI compile-check (`scripts/check-snippets.ts`) can import it
 * directly and verify that *every* snippet it can produce actually builds
 * against the real `basin` API. If the API drifts, the check fails — the
 * demo can't silently lie.
 *
 * Scope is intentionally tight: this is a showcase, not the `/visualizer`.
 * One solver — gradient descent with a constant step — on the Rosenbrock
 * valley. The knobs are sliders: step size α, heavy-ball **momentum** β
 * (β = 0 at the far left is plain steepest descent; turn it up and momentum
 * glides along the valley floor), and the iteration budget. The `Gradient`
 * impl is written inline rather than referencing `basin::problems::*` —
 * inline reads as more representative of real usage.
 *
 * Phase 1 is wasm-free. Phase 2 will wire a live contour beside the code.
 */

export interface PlaygroundConfig {
    /** Constant step size for `GradientDescent::new`. */
    alpha: number;
    /** Heavy-ball momentum coefficient; `0` disables it (plain GD). */
    beta: number;
    /** `max_iter` budget. */
    maxIter: number;
    /** Initial point `[x, y]` — set by clicking the live contour. */
    start: [number, number];
}

// Discrete, "nice" values for the sliders so the generated literals stay
// clean (1-2-5 sequences rather than the messy output of a log slider).
// α is capped at 0.002: on Rosenbrock from this start, larger constant
// steps overshoot the valley and the trajectory diverges off the plotted
// box. Momentum (β) is the knob for going faster, not a bigger α.
export const ALPHA_STEPS: readonly number[] = [
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3,
];
// Capped at 500: the live contour runs this on every change, so a modest
// budget keeps the animation snappy (and it's plenty to show the descent).
export const MAXITER_STEPS: readonly number[] = [100, 200, 300, 500];

// Momentum β is a linear 0–1 knob, so it gets a plain fine-grained slider
// (not the 1-2-5 index steps α / max_iter use for their multi-decade
// ranges). β = 0 at the far left is "off" — plain steepest descent.
export const BETA_MIN = 0;
export const BETA_MAX = 0.99;
export const BETA_STEP = 0.01;

export const DEFAULT_CONFIG: PlaygroundConfig = {
    alpha: 1e-3,
    beta: 0,
    maxIter: 500,
    // The canonical Rosenbrock start, matching the hero illustration.
    start: [-1.2, 1.0],
};

const COST_IMPL = `impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
    }
}`;

const GRADIENT_IMPL = `impl Gradient for Rosenbrock {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok(vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ])
    }
}`;

/** Index of the step value closest to `value` (for positioning a slider). */
export function nearestIndex(steps: readonly number[], value: number): number {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < steps.length; i++) {
        const d = Math.abs(steps[i] - value);
        if (d < bestDist) {
            bestDist = d;
            best = i;
        }
    }
    return best;
}

/** Format a number as a valid Rust `f64` literal (always with a point). */
export function rustFloat(n: number): string {
    if (!Number.isFinite(n)) throw new Error(`non-finite float: ${n}`);
    let s = String(n);
    // `String(1)` → "1"; Rust wants "1.0". Scientific/decimal forms already
    // read as floats. (Our slider values never reach JS exponential range.)
    if (!/[.eE]/.test(s)) s += ".0";
    return s;
}

/** Format an integer with `_` thousands separators, Rust-style. */
export function rustInt(n: number): string {
    return Math.round(n)
        .toString()
        .replace(/\B(?=(\d{3})+(?!\d))/g, "_");
}

/**
 * Shape of the single line the snippet prints, shared by the generated
 * `println!` and the live "output" console so the two can never disagree.
 * The snippet fills the slots with Rust format specifiers (`{:?}`, `{}`);
 * the console fills them with the matching Rust-formatted values pulled
 * from the wasm run (`Run.paramDebug()` / `Run.costDisplay()`).
 */
export function buildOutputLine(param: string, cost: string): string {
    return `x = ${param} (f = ${cost})`;
}

/** Live result reported by the contour, used to render the output console. */
export type RunOutput = {
    /** True once the run has finished (`.run()` returns / `println!` fires). */
    done: boolean;
    /** `result.param()` Debug-formatted by Rust (e.g. `[0.99, 0.98]`). */
    paramDebug: string;
    /** `result.cost()` Display-formatted by Rust. */
    costDisplay: string;
};

/**
 * Generate a complete, copy-pasteable, compilable Rust program for the
 * given playground configuration.
 */
export function generateSnippet(cfg: PlaygroundConfig): string {
    let solverExpr = `GradientDescent::new(${rustFloat(cfg.alpha)})`;
    if (cfg.beta > 0) {
        solverExpr += `.with_momentum(${rustFloat(cfg.beta)})`;
    }

    const startVec = `vec![${rustFloat(cfg.start[0])}, ${rustFloat(cfg.start[1])}]`;

    return (
        [
            "use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent};",
            "",
            "struct Rosenbrock;",
            "",
            COST_IMPL,
            "",
            GRADIENT_IMPL,
            "",
            "fn main() {",
            `    let solver = ${solverExpr};`,
            `    let state = BasicState::new(${startVec});`,
            "",
            `    let result = Executor::new(Rosenbrock, solver, state)`,
            `        .max_iter(${rustInt(cfg.maxIter)})`,
            "        .run()",
            "        .unwrap();",
            "",
            `    println!("${buildOutputLine("{:?}", "{}")}", result.param(), result.cost());`,
            "}",
        ].join("\n") + "\n"
    );
}

export interface NamedConfig {
    /** Valid Rust bin/file stem identifying this snippet. */
    name: string;
    config: PlaygroundConfig;
}

/**
 * Both snippets the compile-check should build: plain gradient descent and
 * the momentum variant. (α and max_iter don't change which API is
 * exercised, so the defaults stand in for the whole slider range.)
 */
export function enumerateConfigs(): NamedConfig[] {
    return [
        { name: "rosenbrock_gd", config: { ...DEFAULT_CONFIG, beta: 0 } },
        {
            name: "rosenbrock_gd_momentum",
            config: { ...DEFAULT_CONFIG, beta: 0.9 },
        },
    ];
}
