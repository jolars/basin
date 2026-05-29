//! Convergence-trace harness for the *competitor* benchmark axis: basin
//! vs `argmin` on `Vec<f64>`, recording suboptimality `f(x) − f*` against
//! wall-clock time at every iteration. Powers the `/benchmarks/competitors`
//! page (see `web/scripts/collect-competitors.ts`).
//!
//! Unlike the criterion benches (a single mean solve time vs `n`, valid
//! only when the *same* algorithm runs across backends), competitors do
//! not share an implementation: argmin's Nelder-Mead / GD / L-BFGS take a
//! different path and have different per-iteration cost than basin's. A
//! single mean would hide that, so we emit the whole convergence curve and
//! let the chart show both how fast each library drives down the objective
//! and how much wall time it spends.
//!
//! Cases (all Rosenbrock, `n = 2`, classic start `(−1.2, 1.0)`, matched
//! configs mirroring `benches/gd_nm.rs`, a fixed `MAX_ITERS` budget with
//! no early stop on either side):
//!   * GD     — steepest descent + More-Thuente line search.
//!   * NM     — standard coefficients and a bit-identical initial simplex
//!     (basin's `IntoInitialSimplex`, relative step 0.05).
//!   * L-BFGS — limited-memory `m = 10`, More-Thuente line search; argmin's
//!     gradient/cost tolerances are zeroed so it runs the full budget.
//!
//! Timing: the solvers are deterministic, so the cost sequence is identical
//! every run and only timing jitters. We run `REPS` reps per (case, library)
//! and take the *median* elapsed-ns per iteration index, paired with the
//! (rep-invariant) cost at that index. One honest asymmetry: basin is
//! timestamped from the `Stepper` side, argmin from inside an `Arc<Mutex>`
//! observer — negligible against per-iteration cost.
//!
//! Run: `cargo run -p competitor-bench --release --bin trace`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{Error, Executor as ArgminExecutor, State as ArgminState, KV};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;
use argmin::solver::quasinewton::LBFGS as ArgminLBFGS;
use basin::problems::{rosenbrock, rosenbrock_gradient, Rosenbrock};
use basin::{
    BasicSimplexState, BasicState, CountsMirror, Executor, GradientDescent, IntoInitialSimplex,
    LbfgsState, Lbfgsb, MoreThuente, NelderMead, Solver, State as BasinState, StepOutcome,
};
use competitor_bench::ArgminProblem;

/// Fixed iteration budget — matches `benches/gd_nm.rs` and the backend
/// bench, so both libraries do the same nominal work.
const MAX_ITERS: u64 = 200;
/// Repetitions per (case, library) for the per-iteration time median.
const REPS: usize = 11;
/// Rosenbrock's global minimum value.
const F_OPT: f64 = 0.0;
/// Suboptimality floor so the log-scale y-axis stays well-defined.
const FLOOR: f64 = 1e-16;
/// Problem dimension (classic 2-D Rosenbrock).
const N: usize = 2;

/// classic Rosenbrock start.
fn start() -> Vec<f64> {
    vec![-1.2, 1.0]
}

// ---------------------------------------------------------------------
// basin side: step the `Stepper`, timestamping after each iteration.
// ---------------------------------------------------------------------

/// Run a basin solve to the fixed budget, returning `(elapsed_ns, cost)`
/// at iter 0 and after every completed iteration.
fn basin_trace<P, S, So>(exec: Executor<P, S, So>) -> Vec<(u128, f64)>
where
    S: BasinState<Float = f64> + CountsMirror,
    So: Solver<P, S>,
    So::Error: std::fmt::Debug,
{
    let mut stepper = exec.max_iter(MAX_ITERS).into_stepper().unwrap();
    let mut pts = Vec::with_capacity(MAX_ITERS as usize + 1);
    pts.push((0u128, stepper.state().cost()));
    let t0 = Instant::now();
    while stepper.step().unwrap() == StepOutcome::Continue {
        pts.push((t0.elapsed().as_nanos(), stepper.state().cost()));
    }
    pts
}

// ---------------------------------------------------------------------
// argmin side: an observer recording `(elapsed_ns, best_cost)` per iter.
// `observe_init` fires once after init (iter 0) and resets the clock, so
// both libraries measure time from "just after init, at iter 0".
// ---------------------------------------------------------------------

/// Shared `(elapsed_ns, cost)` buffer the observer writes and the caller
/// drains after the run.
type Points = Arc<Mutex<Vec<(u128, f64)>>>;

#[derive(Clone)]
struct TraceObserver {
    start: Option<Instant>,
    points: Points,
}

impl TraceObserver {
    fn record<I: ArgminState<Float = f64>>(&mut self, state: &I) {
        let t = match self.start {
            Some(s) => s.elapsed().as_nanos(),
            None => {
                self.start = Some(Instant::now());
                0
            }
        };
        self.points.lock().unwrap().push((t, state.get_best_cost()));
    }
}

impl<I: ArgminState<Float = f64>> Observe<I> for TraceObserver {
    fn observe_init(&mut self, _name: &str, state: &I, _kv: &KV) -> Result<(), Error> {
        self.record(state);
        Ok(())
    }

    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), Error> {
        self.record(state);
        Ok(())
    }
}

/// Fresh observer plus a handle to drain its points after the run.
fn observer() -> (TraceObserver, Points) {
    let points: Points = Arc::new(Mutex::new(Vec::new()));
    let obs = TraceObserver {
        start: None,
        points: Arc::clone(&points),
    };
    (obs, points)
}

fn drain(points: &Points) -> Vec<(u128, f64)> {
    points.lock().unwrap().clone()
}

/// argmin reports `get_best_cost() == +∞` at `observe_init`, before the
/// first iteration registers a best — but at iter 0 (t = 0) the true best
/// is `f(x0)`, which argmin *did* evaluate during init. Overwrite that
/// leading non-finite cost so both libraries' curves start at `f(x0)`.
fn finite_start(mut pts: Vec<(u128, f64)>, f0: f64) -> Vec<(u128, f64)> {
    if let Some(first) = pts.first_mut() {
        if !first.1.is_finite() {
            *first = (0, f0);
        }
    }
    pts
}

// ---------------------------------------------------------------------
// median over reps
// ---------------------------------------------------------------------

/// Run `run` `REPS` times and median the elapsed time per iteration index,
/// keeping the (deterministic) cost from the first rep.
fn median_reps(mut run: impl FnMut() -> Vec<(u128, f64)>) -> Vec<(u128, f64)> {
    let runs: Vec<Vec<(u128, f64)>> = (0..REPS).map(|_| run()).collect();
    let len = runs.iter().map(Vec::len).min().unwrap_or(0);
    (0..len)
        .map(|i| {
            let mut times: Vec<u128> = runs.iter().map(|r| r[i].0).collect();
            times.sort_unstable();
            (times[REPS / 2], runs[0][i].1)
        })
        .collect()
}

// ---------------------------------------------------------------------
// JSON output (hand-rolled to avoid a serde dependency in this crate)
// ---------------------------------------------------------------------

struct Trace {
    solver: &'static str,
    library: &'static str,
    points: Vec<(u128, f64)>,
}

fn print_traces(traces: &[Trace]) {
    let mut out = String::from("[\n");
    for (ti, t) in traces.iter().enumerate() {
        out.push_str(&format!(
            "  {{\"solver\":\"{}\",\"problem\":\"rosenbrock\",\"n\":{},\"library\":\"{}\",\"points\":[",
            t.solver, N, t.library
        ));
        // Skip any non-finite point so the emitted JSON is always valid
        // (`finite_start` already handles argmin's leading +∞).
        let mut first = true;
        for &(t_ns, cost) in &t.points {
            let diff = cost - F_OPT;
            if !diff.is_finite() {
                continue;
            }
            let subopt = diff.max(FLOOR);
            if !first {
                out.push(',');
            }
            first = false;
            out.push_str(&format!("{{\"tNs\":{t_ns},\"subopt\":{subopt}}}"));
        }
        out.push(']');
        out.push('}');
        out.push_str(if ti + 1 < traces.len() { ",\n" } else { "\n" });
    }
    out.push(']');
    println!("{out}");
}

fn main() {
    // Cost at the shared start, used to give argmin's curve a finite t = 0.
    let f0 = rosenbrock(&start());

    let traces = vec![
        // ---- gradient descent (steepest + More-Thuente) ----
        Trace {
            solver: "gd",
            library: "basin",
            points: median_reps(|| {
                basin_trace(Executor::new(
                    Rosenbrock::<Vec<f64>>::default(),
                    GradientDescent::with_line_search(MoreThuente::new()),
                    BasicState::new(start()),
                ))
            }),
        },
        Trace {
            solver: "gd",
            library: "argmin",
            points: finite_start(
                median_reps(|| {
                    let ls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
                        MoreThuenteLineSearch::new();
                    let (obs, points) = observer();
                    ArgminExecutor::new(
                        ArgminProblem::new(rosenbrock, rosenbrock_gradient),
                        SteepestDescent::new(ls),
                    )
                    .configure(|s| s.param(start()).max_iters(MAX_ITERS))
                    .add_observer(obs, ObserverMode::Always)
                    .run()
                    .unwrap();
                    drain(&points)
                }),
                f0,
            ),
        },
        // ---- Nelder-Mead (standard coeffs, identical initial simplex) ----
        Trace {
            solver: "nm",
            library: "basin",
            points: median_reps(|| {
                basin_trace(Executor::new(
                    Rosenbrock::<Vec<f64>>::default(),
                    NelderMead::standard(),
                    BasicSimplexState::new(start()),
                ))
            }),
        },
        Trace {
            solver: "nm",
            library: "argmin",
            points: finite_start(
                median_reps(|| {
                    let simplex = IntoInitialSimplex::into_initial_simplex(start(), 0.05);
                    let nm = ArgminNelderMead::new(simplex)
                        .with_sd_tolerance(0.0)
                        .unwrap();
                    let (obs, points) = observer();
                    ArgminExecutor::new(ArgminProblem::new(rosenbrock, rosenbrock_gradient), nm)
                        .configure(|s| s.max_iters(MAX_ITERS))
                        .add_observer(obs, ObserverMode::Always)
                        .run()
                        .unwrap();
                    drain(&points)
                }),
                f0,
            ),
        },
        // ---- L-BFGS (limited memory m = 10, More-Thuente) ----
        Trace {
            solver: "lbfgs",
            library: "basin",
            points: median_reps(|| {
                basin_trace(Executor::new(
                    Rosenbrock::<Vec<f64>>::default(),
                    Lbfgsb::new().unbounded(),
                    LbfgsState::new(start(), 10),
                ))
            }),
        },
        Trace {
            solver: "lbfgs",
            library: "argmin",
            points: finite_start(
                median_reps(|| {
                    let ls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
                        MoreThuenteLineSearch::new();
                    let lbfgs: ArgminLBFGS<_, Vec<f64>, Vec<f64>, f64> = ArgminLBFGS::new(ls, 10)
                        .with_tolerance_grad(0.0)
                        .unwrap()
                        .with_tolerance_cost(0.0)
                        .unwrap();
                    let (obs, points) = observer();
                    ArgminExecutor::new(ArgminProblem::new(rosenbrock, rosenbrock_gradient), lbfgs)
                        .configure(|s| s.param(start()).max_iters(MAX_ITERS))
                        .add_observer(obs, ObserverMode::Always)
                        .run()
                        .unwrap();
                    drain(&points)
                }),
                f0,
            ),
        },
    ];

    print_traces(&traces);
}
