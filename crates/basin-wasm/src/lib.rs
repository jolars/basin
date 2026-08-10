//! WebAssembly bindings for the Basin optimization library.
//!
//! Exposes a small, JS-friendly surface for the `web/` visualizer:
//!
//! - [`ProblemKind`]/[`SolverKind`]: plain enums marshaled across the
//!   wasm boundary as JS-side enums.
//! - [`eval_grid`]: sample a problem's cost on a uniform `nx × ny` grid
//!   for heatmap rendering. Free function so the heatmap can be rendered
//!   without constructing a [`Run`].
//! - [`Run`]: opaque handle that owns a [`basin::Stepper`] for the
//!   chosen `(problem, solver)` plus an in-wasm log of per-iteration
//!   `(x, y)` and cost. Step it with [`Run::step_many`]; pull the typed
//!   arrays out with [`Run::trajectory_xy`] and [`Run::costs`].
//!
//! The visualizer monomorphizes its concerns (2D problems, `Vec<f64>`
//! params, no nalgebra, ndarray, or faer) so the inner stepper is a single
//! concrete type per solver. That keeps the wasm bundle small and avoids
//! `dyn`-incompatible plumbing on the `Solver` trait.

use basin::problems::{ackley, beale, beale_gradient, booth, booth_gradient};
use basin::problems::{goldstein_price, goldstein_price_gradient};
use basin::problems::{matyas, matyas_gradient, mccormick, mccormick_gradient};
use basin::problems::{
    rastrigin, rosenbrock, rosenbrock_gradient, sphere, sphere_gradient,
};
use basin::problems::{styblinski_tang, styblinski_tang_gradient};
use basin::solver::lbfgs::{Lbfgs, Unbounded as LbfgsUnbounded};
use basin::{
    Backtracking, BasicPopulationState, BasicSimplexState, BasicState,
    BoxConstraints, CmaEs, CmaEsState, Constant, CostFunction, De, DenseMatrix,
    Executor, FiniteDiff, Gradient, GradientDescent, LbfgsState, Mads,
    MadsState, MoreThuente, NelderMead, PopulationState, RandomSearch, Ssga,
    State, StepOutcome, Stepper, TerminationReason,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Set up nicer panic messages in dev. Called automatically the first
/// time `Run::new` runs; idempotent.
fn install_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(console_error_panic_hook::set_once);
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemKind {
    Sphere = 0,
    Rosenbrock = 1,
    Beale = 2,
    Booth = 3,
    Matyas = 4,
    McCormick = 5,
    GoldsteinPrice = 6,
    Rastrigin = 7,
    Ackley = 8,
    StyblinskiTang = 9,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    GradientDescent = 0,
    NelderMead = 1,
    Lbfgs = 2,
    CmaEs = 3,
    De = 4,
    RandomSearch = 5,
    Ssga = 6,
    Mads = 7,
}

/// Solver-specific knobs, marshaled across the wasm boundary as a single
/// plain JS object (`{ gdLineSearch, gdAlpha, ... }`) and deserialized here
/// with serde. Passing one object instead of a growing tail of positional
/// args to [`Run::new`] keeps the constructor stable as solvers gain
/// options; each solver branch reads only the fields it cares about.
/// Missing fields fall back to [`RunOptions::default`].
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct RunOptions {
    /// Gradient-descent step strategy: `"constant"` (fixed `gd_alpha`) or
    /// `"backtracking"` (Armijo line search).
    gd_line_search: String,
    /// Constant step size for `gd_line_search == "constant"`.
    gd_alpha: f64,
    /// Heavy-ball momentum coefficient for the gradient-descent solver;
    /// `0.0` disables it (plain steepest descent).
    gd_beta: f64,
    /// L-Bfgs history capacity `m` (number of stored (s, y) pairs).
    lbfgs_m: usize,
    /// RNG seed shared by every stochastic solver. Fixed seed → reproducible
    /// trajectory; the UI's 🎲 button rerolls it.
    seed: u64,
    /// CMA-ES initial step-size σ. `NaN` (default) lets the wasm pick a
    /// sensible value from the viewport dimensions.
    cma_sigma: f64,
    /// CMA-ES population size λ. `0` (default) → `CmaEs::default_lambda(n)`.
    /// Values `< 4` are treated as `0`; CMA-ES needs at least 4 for sane
    /// recombination weights.
    cma_lambda: usize,
    /// DE population size. `0` (default) → `De::default_pop_size(n)` (= 10·n).
    /// Values `< 4` are treated as `0` (DE/rand/1/bin needs ≥ 4 vectors).
    de_pop_size: usize,
    /// DE differential weight F (Storn–Price scaling factor).
    de_f: f64,
    /// DE crossover probability CR ∈ [0, 1].
    de_cr: f64,
    /// Random-search samples per generation λ.
    rs_lambda: usize,
    /// SSGA population size. `0` → use the solver's own default.
    ssga_pop_size: usize,
    /// Box-bounds for stochastic solvers (DE, SSGA, Random Search). Sourced
    /// from the visualizer viewport: "visible" doubles as "feasible" here.
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            gd_line_search: "constant".to_string(),
            gd_alpha: 0.01,
            gd_beta: 0.0,
            lbfgs_m: 10,
            seed: 0,
            cma_sigma: f64::NAN,
            cma_lambda: 0,
            de_pop_size: 0,
            de_f: 0.8,
            de_cr: 0.9,
            rs_lambda: 16,
            ssga_pop_size: 0,
            xmin: -1.0,
            xmax: 1.0,
            ymin: -1.0,
            ymax: 1.0,
        }
    }
}

/// 2D problem dispatcher. Implements `CostFunction` + `Gradient` once
/// for `Vec<f64>`, delegating to Basin's raw functions. Lets the inner
/// stepper be a single concrete type per solver instead of a forest of
/// monomorphizations.
#[derive(Clone, Copy)]
struct Problem2D(ProblemKind);

impl CostFunction for Problem2D {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(match self.0 {
            ProblemKind::Sphere => sphere(x),
            ProblemKind::Rosenbrock => rosenbrock(x),
            ProblemKind::Beale => beale(x),
            ProblemKind::Booth => booth(x),
            ProblemKind::Matyas => matyas(x),
            ProblemKind::McCormick => mccormick(x),
            ProblemKind::GoldsteinPrice => goldstein_price(x),
            ProblemKind::Rastrigin => rastrigin(x),
            ProblemKind::Ackley => ackley(x),
            ProblemKind::StyblinskiTang => styblinski_tang(x),
        })
    }
}

/// Cost-only twin of [`Problem2D`] for use with [`FiniteDiff`] on the
/// global-opt problems (Rastrigin, Ackley) that ship in the corpus without
/// hand-written gradients. Kept separate so [`Problem2D`]'s `Gradient` impl
/// can dispatch to either the analytic gradient or `FiniteDiff` without
/// infinite recursion.
#[derive(Clone, Copy)]
struct Problem2DCost(ProblemKind);

impl CostFunction for Problem2DCost {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Problem2D(self.0).cost(x)
    }
}

/// 2D problem dispatcher with box bounds attached. DE, SSGA, and
/// Random Search bound their `Solver<P, ...>` impl on `P:
/// BoxConstraints<Param = V>`, so they can't reuse the bare [`Problem2D`].
/// Bounds are sourced from the visualizer viewport: "visible" doubles as
/// "feasible" here, which is the right semantics for a 2D demo.
#[derive(Clone)]
struct Problem2DBounded {
    kind: ProblemKind,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for Problem2DBounded {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Problem2D(self.kind).cost(x)
    }
}

impl BoxConstraints for Problem2DBounded {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

impl Gradient for Problem2D {
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        match self.0 {
            ProblemKind::Sphere => sphere_gradient(x, &mut out),
            ProblemKind::Rosenbrock => rosenbrock_gradient(x, &mut out),
            ProblemKind::Beale => beale_gradient(x, &mut out),
            ProblemKind::Booth => booth_gradient(x, &mut out),
            ProblemKind::Matyas => matyas_gradient(x, &mut out),
            ProblemKind::McCormick => mccormick_gradient(x, &mut out),
            ProblemKind::GoldsteinPrice => {
                goldstein_price_gradient(x, &mut out)
            }
            ProblemKind::StyblinskiTang => {
                styblinski_tang_gradient(x, &mut out)
            }
            // Rastrigin and Ackley ship without an analytic gradient in the
            // corpus; synthesize one via central finite differences. Doubles
            // the per-step cost evals on these two problems, which is invisible
            // at the visualizer's pace.
            ProblemKind::Rastrigin | ProblemKind::Ackley => {
                let g = FiniteDiff::new(Problem2DCost(self.0)).gradient(x)?;
                out.copy_from_slice(&g);
            }
        }
        Ok(out)
    }
}

/// Sample `f(x, y)` on a uniform `nx × ny` grid spanning the rectangle
/// `[xmin, xmax] × [ymin, ymax]`.
///
/// Returns a flat row-major `Float64Array` of length `nx * ny` where
/// `row j` (y-coordinate index) has the `nx` x-samples laid out in
/// increasing x order. `j = 0` is `ymin`, `j = ny - 1` is `ymax`.
///
/// Cheap by design: JS calls this once per problem (or on resize) and
/// renders into a canvas. Intentionally returns a flat array, not a
/// `Vec<Vec<f64>>`, to avoid per-row JS object overhead.
#[wasm_bindgen(js_name = evalGrid)]
pub fn eval_grid(
    problem: ProblemKind,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: u32,
    ny: u32,
) -> Vec<f64> {
    let p = Problem2D(problem);
    let nx = nx as usize;
    let ny = ny as usize;
    let mut out = vec![0.0; nx * ny];
    let dx = if nx > 1 {
        (xmax - xmin) / (nx as f64 - 1.0)
    } else {
        0.0
    };
    let dy = if ny > 1 {
        (ymax - ymin) / (ny as f64 - 1.0)
    } else {
        0.0
    };
    let mut xy = vec![0.0; 2];
    for j in 0..ny {
        xy[1] = ymin + dy * j as f64;
        let row = j * nx;
        for i in 0..nx {
            xy[0] = xmin + dx * i as f64;
            out[row + i] = p.cost(&xy).unwrap();
        }
    }
    out
}

/// Concrete L-Bfgs stepper type. Aliased to keep the [`Inner`] variant
/// readable (the boxed, fully-monomorphized generic otherwise trips
/// `clippy::type_complexity`).
type LbfgsStepper = Stepper<
    Problem2D,
    LbfgsState<Vec<f64>>,
    Lbfgs<LbfgsUnbounded, MoreThuente>,
>;
/// Concrete population-solver stepper aliases. Same motivation as
/// [`LbfgsStepper`]: keep the [`Inner`] variants readable.
type CmaEsStepper = Stepper<
    Problem2D,
    CmaEsState<Vec<f64>, DenseMatrix>,
    CmaEs<Vec<f64>, DenseMatrix>,
>;
type DeStepper = Stepper<Problem2DBounded, BasicPopulationState<Vec<f64>>, De>;
type RandomSearchStepper =
    Stepper<Problem2DBounded, BasicPopulationState<Vec<f64>>, RandomSearch>;
type SsgaStepper =
    Stepper<Problem2DBounded, BasicPopulationState<Vec<f64>>, Ssga>;

/// Inner enum dispatching by `(state shape, solver type)`. Each variant
/// is fully concrete so the resulting wasm is tight and no `dyn Solver`
/// gymnastics are needed.
enum Inner {
    GdConstant(
        Stepper<
            Problem2D,
            BasicState<Vec<f64>>,
            GradientDescent<Constant, Vec<f64>>,
        >,
    ),
    GdBacktracking(
        Stepper<
            Problem2D,
            BasicState<Vec<f64>>,
            GradientDescent<Backtracking, Vec<f64>>,
        >,
    ),
    NelderMead(Stepper<Problem2D, BasicSimplexState<Vec<f64>>, NelderMead>),
    Mads(Stepper<Problem2D, MadsState<Vec<f64>>, Mads>),
    // Boxed: `LbfgsState` carries the limited-memory history buffers, so
    // this variant is several times larger than the others, so boxing keeps
    // `Inner` small (clippy::large_enum_variant). Auto-deref means the
    // `step`/`xy`/`cost` match arms need no `*` and read like the rest.
    Lbfgs(Box<LbfgsStepper>),
    // All four population-based steppers carry per-candidate cost buffers
    // plus solver-side RNG and weight tables, so they also get boxed for the
    // same `large_enum_variant` reason.
    CmaEs(Box<CmaEsStepper>),
    De(Box<DeStepper>),
    RandomSearch(Box<RandomSearchStepper>),
    Ssga(Box<SsgaStepper>),
}

impl Inner {
    fn step(&mut self) -> StepOutcome {
        match self {
            Self::GdConstant(s) => s.step().unwrap(),
            Self::GdBacktracking(s) => s.step().unwrap(),
            Self::NelderMead(s) => s.step().unwrap(),
            Self::Mads(s) => s.step().unwrap(),
            Self::Lbfgs(s) => s.step().unwrap(),
            Self::CmaEs(s) => s.step().unwrap(),
            Self::De(s) => s.step().unwrap(),
            Self::RandomSearch(s) => s.step().unwrap(),
            Self::Ssga(s) => s.step().unwrap(),
        }
    }

    fn xy(&self) -> (f64, f64) {
        // For the population steppers, `state().param()` is defined to
        // return the best-so-far candidate (see `BasicPopulationState`'s
        // `State::param` impl), so the trajectory plot reads the same way
        // it does for the single-iterate solvers.
        let p: &Vec<f64> = match self {
            Self::GdConstant(s) => s.state().param(),
            Self::GdBacktracking(s) => s.state().param(),
            Self::NelderMead(s) => s.state().param(),
            Self::Mads(s) => s.state().param(),
            Self::Lbfgs(s) => s.state().param(),
            Self::CmaEs(s) => s.state().param(),
            Self::De(s) => s.state().param(),
            Self::RandomSearch(s) => s.state().param(),
            Self::Ssga(s) => s.state().param(),
        };
        (p[0], p[1])
    }

    fn cost(&self) -> f64 {
        match self {
            Self::GdConstant(s) => s.state().cost(),
            Self::GdBacktracking(s) => s.state().cost(),
            Self::NelderMead(s) => s.state().cost(),
            Self::Mads(s) => s.state().cost(),
            Self::Lbfgs(s) => s.state().cost(),
            Self::CmaEs(s) => s.state().cost(),
            Self::De(s) => s.state().cost(),
            Self::RandomSearch(s) => s.state().cost(),
            Self::Ssga(s) => s.state().cost(),
        }
    }

    /// Flat `(x, y)` pairs of the current generation's population, for the
    /// four population-based solvers. `None` for the single-iterate solvers
    /// so the JS side can render nothing instead of a stale cloud.
    fn population_xy(&self) -> Option<Vec<f64>> {
        let cands: &[Vec<f64>] = match self {
            Self::CmaEs(s) => s.state().candidates(),
            Self::De(s) => s.state().candidates(),
            Self::RandomSearch(s) => s.state().candidates(),
            Self::Ssga(s) => s.state().candidates(),
            _ => return None,
        };
        let mut out = Vec::with_capacity(cands.len() * 2);
        for c in cands {
            out.push(c[0]);
            out.push(c[1]);
        }
        Some(out)
    }
}

#[wasm_bindgen]
pub struct Run {
    inner: Inner,
    /// Flat (x, y) pairs, one per recorded iterate. Initial point is
    /// included at index 0 so JS doesn't need to track it separately.
    trajectory: Vec<f64>,
    costs: Vec<f64>,
    /// Absolute cost at which to stop early: `f* + target_suboptimality`.
    /// `None` disables the suboptimality stop (run to `max_iter`). This is
    /// a visualizer-level convergence test: it knows each problem's `f*`,
    /// so "stop when essentially at the optimum" replaces a per-solver
    /// gradient or simplex tolerance and matches the suboptimality the cost
    /// chart plots.
    target_cost: Option<f64>,
    /// Stable termination-reason string, or `None` while still running.
    /// `"converged"` is the visualizer's suboptimality stop; everything
    /// else comes from [`reason_str`].
    finished: Option<&'static str>,
}

/// Per-call result returned by `step_many`. Plain serializable shape so
/// JS receives `{ done, iters_added, reason? }` without manual JsValue
/// plumbing.
#[derive(Serialize)]
struct StepResult {
    /// True iff the run is finished (the stepper hit a termination
    /// criterion, including `MaxIter`). Once true, further `step_many`
    /// calls are no-ops.
    done: bool,
    /// Iterations actually completed by this call. May be less than the
    /// requested `n` if the run finished early or was already done.
    iters_added: u32,
    /// Termination reason as a stable string (see `reason_str`). `None`
    /// while still running.
    reason: Option<&'static str>,
}

#[wasm_bindgen]
impl Run {
    /// Construct a new run for the given `(problem, solver)` starting at
    /// `(x0, y0)`. `opts` is a plain JS object of solver-specific knobs
    /// (`{ gdLineSearch, gdAlpha, gdBeta, lbfgsM }`); each solver reads
    /// only the fields it needs and missing fields take their defaults
    /// (see [`RunOptions`]). `max_iter` caps the total number of
    /// iterations; subsequent `step_many` calls cumulatively count against
    /// this cap. `stop_at_cost` is the absolute cost at which to stop
    /// early, typically `f* + target_suboptimality`, since the visualizer knows
    /// each problem's `f*`. Pass a non-finite value (e.g. `NaN`) to disable
    /// the early stop (run to `max_iter`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        problem: ProblemKind,
        solver: SolverKind,
        x0: f64,
        y0: f64,
        opts: JsValue,
        max_iter: u32,
        stop_at_cost: f64,
    ) -> Self {
        install_panic_hook();
        // serde_wasm_bindgen reaches into JS, so deserialization can't
        // happen in the native-testable core; do it here and hand a plain
        // Rust struct to `new_inner`.
        let opts: RunOptions =
            serde_wasm_bindgen::from_value(opts).unwrap_or_default();
        Self::new_inner(problem, solver, x0, y0, opts, max_iter, stop_at_cost)
    }

    /// Advance up to `n` iterations, recording the `(x, y)` and cost
    /// after each. Returns `{ done, iters_added, reason? }` so JS can
    /// append only the new tail of the trajectory and stop the
    /// requestAnimationFrame loop when finished.
    #[wasm_bindgen(js_name = stepMany)]
    pub fn step_many(&mut self, n: u32) -> JsValue {
        let result = self.step_many_inner(n);
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Full trajectory as a flat `Float64Array` of `(x, y)` pairs.
    /// Length is `2 * (iter + 1)` (initial point + one per completed
    /// iteration).
    #[wasm_bindgen(js_name = trajectoryXy)]
    pub fn trajectory_xy(&self) -> Vec<f64> {
        self.trajectory.clone()
    }

    /// Per-iterate cost values, including the cost at the initial point
    /// (so `costs.length === trajectory.length / 2`).
    pub fn costs(&self) -> Vec<f64> {
        self.costs.clone()
    }

    /// Flat `(x, y)` pairs for the current generation's population, or an
    /// empty array for non-population solvers. JS reads this once per frame
    /// to render the search cloud beneath the best-so-far trail.
    #[wasm_bindgen(js_name = populationXy)]
    pub fn population_xy(&self) -> Vec<f64> {
        self.inner.population_xy().unwrap_or_default()
    }

    /// Iteration counter (excludes the initial point).
    pub fn iter(&self) -> u32 {
        self.costs.len().saturating_sub(1) as u32
    }

    /// True iff the stepper has stopped.
    pub fn done(&self) -> bool {
        self.finished.is_some()
    }

    /// Termination reason string, or empty if still running.
    pub fn reason(&self) -> String {
        self.finished.unwrap_or("").to_string()
    }

    /// The current parameter vector, Debug-formatted by Rust exactly as
    /// `println!("{:?}", result.param())` would print it. The landing-page
    /// playground shows this in its live "output" console, so the console
    /// is the program's real stdout (Rust formatting), not a JS guess.
    #[wasm_bindgen(js_name = paramDebug)]
    pub fn param_debug(&self) -> String {
        let n = self.trajectory.len();
        let param: &[f64] = if n >= 2 {
            &self.trajectory[n - 2..n]
        } else {
            &[]
        };
        // Slice Debug matches `Vec<f64>` Debug: both print `[x, y]`.
        format!("{param:?}")
    }

    /// The current cost, Display-formatted by Rust exactly as
    /// `println!("{}", result.cost())` would print it. See [`Self::param_debug`].
    #[wasm_bindgen(js_name = costDisplay)]
    pub fn cost_display(&self) -> String {
        format!("{}", self.costs.last().copied().unwrap_or(f64::NAN))
    }
}

impl Run {
    /// Pure-Rust core of the constructor, callable from native unit
    /// tests without going through `serde_wasm_bindgen` (which calls into
    /// JS APIs that panic on non-wasm targets). The wasm-facing
    /// [`Run::new`] deserializes the JS `opts` object and delegates here.
    fn new_inner(
        problem: ProblemKind,
        solver: SolverKind,
        x0: f64,
        y0: f64,
        opts: RunOptions,
        max_iter: u32,
        stop_at_cost: f64,
    ) -> Self {
        let p = Problem2D(problem);
        let initial = vec![x0, y0];
        let initial_cost = p.cost(&initial).unwrap();
        // The only termination beyond `max_iter` is the suboptimality stop
        // applied in `step_many_inner`; solvers themselves run unbounded.
        let inner = match solver {
            SolverKind::GradientDescent => {
                if opts.gd_line_search == "backtracking" {
                    Inner::GdBacktracking(make_stepper(
                        p,
                        GradientDescent::with_line_search(Backtracking::new())
                            .with_momentum(opts.gd_beta),
                        &initial,
                        max_iter,
                    ))
                } else {
                    Inner::GdConstant(make_stepper(
                        p,
                        GradientDescent::new(opts.gd_alpha)
                            .with_momentum(opts.gd_beta),
                        &initial,
                        max_iter,
                    ))
                }
            }
            SolverKind::NelderMead => {
                let stepper = Executor::new(
                    p,
                    NelderMead::new(),
                    BasicSimplexState::<Vec<f64>>::new(initial.clone()),
                )
                .max_iter(max_iter as u64)
                .into_stepper()
                .unwrap();
                Inner::NelderMead(stepper)
            }
            SolverKind::Mads => {
                let stepper = Executor::new(
                    p,
                    Mads::new(),
                    MadsState::<Vec<f64>>::new(initial.clone()),
                )
                .max_iter(max_iter as u64)
                .into_stepper()
                .unwrap();
                Inner::Mads(stepper)
            }
            SolverKind::Lbfgs => {
                // `m_capacity` asserts `>= 1`; clamp so a stray `0` from
                // the JS side can't panic the constructor.
                let m = opts.lbfgs_m.max(1);
                let stepper = Executor::new(
                    p,
                    Lbfgs::<LbfgsUnbounded>::new().with_m_capacity(m),
                    LbfgsState::new(initial.clone(), m),
                )
                .max_iter(max_iter as u64)
                .into_stepper()
                .unwrap();
                Inner::Lbfgs(Box::new(stepper))
            }
            SolverKind::CmaEs => {
                // σ default: a quarter of the average viewport span. Gives
                // σ ≈ 1.5 on Sphere [-3,3]², σ ≈ 1 on Goldstein-Price [-2,2]²,
                // σ ≈ 5 on Booth [-10,10]², all fine starting points; the slider
                // overrides this whenever the user touches it.
                let sigma = if opts.cma_sigma.is_finite()
                    && opts.cma_sigma > 0.0
                {
                    opts.cma_sigma
                } else {
                    0.25 * 0.5
                        * ((opts.xmax - opts.xmin) + (opts.ymax - opts.ymin))
                };
                let mut solver = CmaEs::<Vec<f64>, DenseMatrix>::new(opts.seed);
                // λ < 4 is invalid for CMA-ES recombination weights; treat
                // small overrides as "auto" and let the solver pick.
                if opts.cma_lambda >= 4 {
                    solver = solver.with_lambda(opts.cma_lambda);
                }
                let stepper = Executor::new(
                    p,
                    solver,
                    CmaEsState::<Vec<f64>, DenseMatrix>::new(
                        initial.clone(),
                        sigma,
                    ),
                )
                .max_iter(max_iter as u64)
                .into_stepper()
                .unwrap();
                Inner::CmaEs(Box::new(stepper))
            }
            SolverKind::De => {
                let pb = Problem2DBounded {
                    kind: problem,
                    lower: vec![opts.xmin, opts.ymin],
                    upper: vec![opts.xmax, opts.ymax],
                };
                let pop = if opts.de_pop_size >= 4 {
                    opts.de_pop_size
                } else {
                    De::<f64>::default_pop_size(2)
                };
                let mut solver = De::<f64>::new(opts.seed)
                    .with_f(opts.de_f)
                    .with_cr(opts.de_cr);
                if opts.de_pop_size >= 4 {
                    solver = solver.with_pop_size(opts.de_pop_size);
                }
                let stepper = Executor::new(
                    pb,
                    solver,
                    BasicPopulationState::<Vec<f64>>::with_size(pop),
                )
                .max_iter(max_iter as u64)
                .into_stepper()
                .unwrap();
                Inner::De(Box::new(stepper))
            }
            SolverKind::RandomSearch => {
                let pb = Problem2DBounded {
                    kind: problem,
                    lower: vec![opts.xmin, opts.ymin],
                    upper: vec![opts.xmax, opts.ymax],
                };
                let lambda = opts.rs_lambda.max(1);
                let stepper = Executor::new(
                    pb,
                    RandomSearch::new(lambda, opts.seed),
                    BasicPopulationState::<Vec<f64>>::with_size(lambda),
                )
                .max_iter(max_iter as u64)
                .into_stepper()
                .unwrap();
                Inner::RandomSearch(Box::new(stepper))
            }
            SolverKind::Ssga => {
                let pb = Problem2DBounded {
                    kind: problem,
                    lower: vec![opts.xmin, opts.ymin],
                    upper: vec![opts.xmax, opts.ymax],
                };
                let mut solver = Ssga::<f64>::new(opts.seed);
                let pop = if opts.ssga_pop_size > 0 {
                    solver = solver.with_pop_size(opts.ssga_pop_size);
                    opts.ssga_pop_size
                } else {
                    // SSGA has no public `default_pop_size`; mirror the
                    // documented default by sizing the state buffer at 20.
                    20
                };
                let stepper = Executor::new(
                    pb,
                    solver,
                    BasicPopulationState::<Vec<f64>>::with_size(pop),
                )
                .max_iter(max_iter as u64)
                .into_stepper()
                .unwrap();
                Inner::Ssga(Box::new(stepper))
            }
        };
        Self {
            inner,
            trajectory: vec![x0, y0],
            costs: vec![initial_cost],
            target_cost: stop_at_cost.is_finite().then_some(stop_at_cost),
            finished: None,
        }
    }

    /// Pure-Rust core of `step_many`, callable from native unit tests
    /// without going through `serde_wasm_bindgen` (which calls into JS
    /// APIs that panic on non-wasm targets).
    fn step_many_inner(&mut self, n: u32) -> StepResult {
        if self.finished.is_some() {
            return StepResult {
                done: true,
                iters_added: 0,
                reason: self.finished,
            };
        }
        let mut iters_added = 0;
        for _ in 0..n {
            match self.inner.step() {
                StepOutcome::Continue => {
                    let (x, y) = self.inner.xy();
                    let cost = self.inner.cost();
                    self.trajectory.push(x);
                    self.trajectory.push(y);
                    self.costs.push(cost);
                    iters_added += 1;
                    // Visualizer-level convergence: stop once the cost is
                    // within the target suboptimality of the known optimum.
                    if let Some(target) = self.target_cost {
                        if cost <= target {
                            self.finished = Some("converged");
                            break;
                        }
                    }
                }
                StepOutcome::Stopped(reason) => {
                    self.finished = Some(reason_str(reason));
                    break;
                }
            }
        }
        StepResult {
            done: self.finished.is_some(),
            iters_added,
            reason: self.finished,
        }
    }
}

fn make_stepper<L>(
    problem: Problem2D,
    solver: GradientDescent<L, Vec<f64>>,
    initial: &[f64],
    max_iter: u32,
) -> Stepper<Problem2D, BasicState<Vec<f64>>, GradientDescent<L, Vec<f64>>>
where
    GradientDescent<L, Vec<f64>>:
        basin::Solver<Problem2D, BasicState<Vec<f64>>>,
{
    Executor::new(problem, solver, BasicState::new(initial.to_vec()))
        .max_iter(max_iter as u64)
        .into_stepper()
        .unwrap_or_else(|_| unreachable!("Problem2D's Error is Infallible"))
}

/// Stable, JS-friendly string for a `TerminationReason`. The wasm
/// boundary discards Rust enum nuance, so we serialize one short tag
/// per variant; the UI can branch on it.
fn reason_str(r: TerminationReason) -> &'static str {
    match r {
        TerminationReason::MaxIter => "max_iter",
        TerminationReason::MaxCostEvals => "max_cost_evals",
        TerminationReason::MaxGradientEvals => "max_gradient_evals",
        TerminationReason::GradientTolerance => "gradient_tolerance",
        TerminationReason::RelativeGradientTolerance => {
            "relative_gradient_tolerance"
        }
        TerminationReason::ProjectedGradientTolerance => {
            "projected_gradient_tolerance"
        }
        TerminationReason::ParamTolerance => "param_tolerance",
        TerminationReason::RelativeParamTolerance => "relative_param_tolerance",
        TerminationReason::CostTolerance => "cost_tolerance",
        TerminationReason::RelativeCostTolerance => "relative_cost_tolerance",
        TerminationReason::TargetCost => "target_cost",
        TerminationReason::NoImprovement => "no_improvement",
        TerminationReason::SimplexTolerance => "simplex_tolerance",
        TerminationReason::MaxTime => "max_time",
        TerminationReason::SolverConverged => "solver_converged",
        TerminationReason::SolverFailed => "solver_failed",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_grid_returns_expected_shape_and_values() {
        let g = eval_grid(ProblemKind::Sphere, -1.0, 1.0, -1.0, 1.0, 3, 3);
        assert_eq!(g.len(), 9);
        // Center sample is f(0, 0) = 0 for sphere.
        assert!(g[4].abs() < 1e-12);
        // Corners are f(±1, ±1) = 2.
        assert!((g[0] - 2.0).abs() < 1e-12);
        assert!((g[8] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn run_records_initial_point_and_progresses() {
        let mut run = Run::new_inner(
            ProblemKind::Rosenbrock,
            SolverKind::GradientDescent,
            -1.2,
            1.0,
            RunOptions {
                gd_alpha: 0.001,
                ..RunOptions::default()
            },
            500,
            f64::NAN, // early stop disabled
        );
        assert_eq!(run.iter(), 0);
        assert_eq!(run.trajectory_xy(), vec![-1.2, 1.0]);
        let r = run.step_many_inner(50);
        assert_eq!(r.iters_added, 50);
        assert!(!r.done);
        assert_eq!(run.iter(), 50);
        assert_eq!(run.trajectory_xy().len(), 2 * 51);
        assert_eq!(run.costs().len(), 51);
    }

    #[test]
    fn run_terminates_on_max_iter() {
        let mut run = Run::new_inner(
            ProblemKind::Sphere,
            SolverKind::GradientDescent,
            1.0,
            1.0,
            RunOptions {
                gd_alpha: 0.5,
                ..RunOptions::default()
            },
            5,
            f64::NAN, // early stop disabled; exercise the max_iter path purely
        );
        let r = run.step_many_inner(100);
        assert!(r.done);
        assert_eq!(r.reason, Some("max_iter"));
        assert!(run.done());
        assert_eq!(run.reason(), "max_iter");
        assert!(run.iter() <= 5);
    }

    #[test]
    fn lbfgs_converges_before_max_iter_on_suboptimality() {
        let mut run = Run::new_inner(
            ProblemKind::Rosenbrock,
            SolverKind::Lbfgs,
            -1.2,
            1.0,
            RunOptions::default(),
            1000,
            1e-10, // f* (0) + target suboptimality (1e-10)
        );
        let r = run.step_many_inner(1000);
        // L-Bfgs drives the Rosenbrock cost below 1e-10 well within the
        // iteration cap, so the suboptimality stop fires first.
        assert!(r.done);
        assert_eq!(r.reason, Some("converged"));
        assert!(run.iter() < 1000);
        // ...and lands near the minimum (1, 1).
        let traj = run.trajectory_xy();
        let n = traj.len();
        assert!((traj[n - 2] - 1.0).abs() < 1e-2);
        assert!((traj[n - 1] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn nelder_mead_stops_on_suboptimality() {
        let mut run = Run::new_inner(
            ProblemKind::Sphere,
            SolverKind::NelderMead,
            2.0,
            2.0,
            RunOptions::default(),
            1000,
            1e-8, // f* (0) + target suboptimality (1e-8)
        );
        let r = run.step_many_inner(1000);
        assert!(r.done);
        assert_eq!(r.reason, Some("converged"));
        assert!(run.iter() < 1000);
    }

    /// Run options sized to the Sphere viewport so DE, Random Search, and
    /// SSGA have plausible box bounds to sample inside.
    fn opts_for_sphere() -> RunOptions {
        RunOptions {
            xmin: -3.0,
            xmax: 3.0,
            ymin: -3.0,
            ymax: 3.0,
            ..RunOptions::default()
        }
    }

    #[test]
    fn cma_es_reduces_cost_on_sphere() {
        let mut run = Run::new_inner(
            ProblemKind::Sphere,
            SolverKind::CmaEs,
            2.0,
            2.0,
            RunOptions {
                cma_sigma: 1.0,
                ..opts_for_sphere()
            },
            300,
            f64::NAN,
        );
        run.step_many_inner(300);
        // Initial cost is 8.0; CMA-ES at σ=1 on the sphere should drop it
        // by several orders of magnitude well within 300 generations.
        assert!(run.costs().last().copied().unwrap() < 1e-6);
    }

    #[test]
    fn de_makes_progress_on_sphere() {
        let mut run = Run::new_inner(
            ProblemKind::Sphere,
            SolverKind::De,
            2.0,
            2.0,
            opts_for_sphere(),
            200,
            f64::NAN,
        );
        let initial = run.costs()[0];
        run.step_many_inner(200);
        // Modest target: DE on 2D Sphere is easy but the test just needs
        // to confirm the box-constraints plumbing works end to end.
        let last = run.costs().last().copied().unwrap();
        assert!(last < 0.1 * initial, "{last} not < 0.1 × {initial}");
    }

    #[test]
    fn random_search_is_reproducible_under_fixed_seed() {
        let mk = || {
            let mut run = Run::new_inner(
                ProblemKind::Sphere,
                SolverKind::RandomSearch,
                2.0,
                2.0,
                RunOptions {
                    seed: 42,
                    rs_lambda: 8,
                    ..opts_for_sphere()
                },
                50,
                f64::NAN,
            );
            run.step_many_inner(50);
            run.trajectory_xy()
        };
        assert_eq!(mk(), mk());
    }

    #[test]
    fn population_xy_is_2_lambda_for_cmaes_and_empty_for_gd() {
        // GD: single iterate, no population.
        let gd = Run::new_inner(
            ProblemKind::Sphere,
            SolverKind::GradientDescent,
            1.0,
            1.0,
            RunOptions::default(),
            10,
            f64::NAN,
        );
        assert!(gd.population_xy().is_empty());

        // CMA-ES: default λ for n=2 is 4 + ⌊3 ln 2⌋ = 6, so 2λ = 12 floats.
        let cma = Run::new_inner(
            ProblemKind::Sphere,
            SolverKind::CmaEs,
            2.0,
            2.0,
            RunOptions {
                cma_sigma: 0.5,
                ..opts_for_sphere()
            },
            10,
            f64::NAN,
        );
        let pop = cma.population_xy();
        assert_eq!(pop.len() % 2, 0);
        assert!(
            pop.len() >= 8,
            "expected at least 4 candidates worth of pairs"
        );
    }

    #[test]
    fn gradient_via_finite_diff_works_on_rastrigin() {
        // GD on Rastrigin would normally stall; we only check that the
        // FiniteDiff-synthesized gradient path doesn't panic and that the
        // run produces finite costs.
        let mut run = Run::new_inner(
            ProblemKind::Rastrigin,
            SolverKind::GradientDescent,
            1.0,
            0.0,
            RunOptions {
                gd_alpha: 0.01,
                ..RunOptions::default()
            },
            20,
            f64::NAN,
        );
        run.step_many_inner(20);
        for c in run.costs() {
            assert!(c.is_finite());
        }
    }
}
