//! Convergence-trace harness for the *competitor* benchmark axis: basin
//! vs `argmin` (and `gomez` and `nlopt` where they line up) on `Vec<f64>`,
//! recording suboptimality `f(x) − f*` against wall-clock time. Powers
//! the `/benchmarks/competitors` page (see
//! `web/scripts/collect-competitors.ts`).
//!
//! Unlike the criterion benches (a single mean solve time vs `n`, valid
//! only when the *same* algorithm runs across backends), competitors do
//! not share an implementation: each library's Nelder-Mead, GD, and L-BFGS
//! takes a different path and has a different per-iteration cost than
//! basin's. A single mean would hide that, so we emit the whole
//! convergence curve and let the chart show both how fast each library
//! drives down the objective and how much wall time it spends.
//!
//! GD, NM, and L-BFGS run on Rosenbrock, `n = 2`, classic start `(−1.2, 1.0)`,
//! matched configs mirroring `benches/gd_nm.rs`, a fixed `MAX_ITERS` budget
//! with no early stop on any side:
//!   * GD:      steepest descent + More-Thuente line search. basin vs argmin.
//!   * NM:      standard coefficients and a bit-identical initial simplex
//!     (basin's `IntoInitialSimplex`, relative step 0.05) for basin and argmin;
//!     gomez and nlopt construct their own simplex with their default
//!     coefficients, so they're "out-of-the-box" against the matched
//!     basin and argmin pair.
//!   * L-BFGS:  limited-memory `m = 10`, More-Thuente line search; argmin's
//!     gradient and cost tolerances are zeroed so it runs the full budget.
//!     gomez has no L-BFGS, so the third comparator is nlopt's `Lbfgs`
//!     (NLopt's own L-BFGS: limited memory, no line-search knob exposed).
//!   * NEWUOA:  Powell's model-based derivative-free method: the *same*
//!     algorithm in two implementations, basin vs nlopt's `LN_NEWUOA`. Unlike
//!     the cases above (different implementations of the same *family*), here
//!     ρ_beg/ρ_end and `npt = 2n+1` are matched as closely as the two APIs
//!     allow. Run on Styblinski–Tang at `n = 5` from the origin: a multimodal
//!     problem (2ⁿ local wells) where the quadratic-model method has more to
//!     chew on than on Rosenbrock, and both implementations descend to the
//!     global minimum; both sides converge on ρ rather than running the fixed
//!     iteration budget the others use.
//!
//! Timing: the solvers are deterministic, so the cost sequence is identical
//! every run and only timing jitters. We run `REPS` reps per (case, library)
//! and take the *median* elapsed-ns per sample index, paired with the
//! (rep-invariant) cost at that index. Two honest asymmetries:
//!   * basin and gomez are timestamped from the driving loop, argmin from
//!     inside an `Arc<Mutex>` observer: negligible against per-iteration
//!     cost.
//!   * nlopt exposes no per-iteration hook, only the objective closure, so
//!     its curve is a *best-so-far* trace at function-eval granularity
//!     (matching argmin's `get_best_cost()` semantic monotonically, but
//!     sampled per cost call rather than per iter). The eval budget is set
//!     to `MAX_ITERS`, so nlopt sees no more cost calls than the other
//!     libraries log iterations.
//!
//! Run: `cargo run -p competitor-bench --release --bin trace`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{Error, Executor as ArgminExecutor, KV, State as ArgminState};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;
use argmin::solver::quasinewton::LBFGS as ArgminLBFGS;
use basin::problems::{
    Rosenbrock, StyblinskiTang, rosenbrock, rosenbrock_gradient, styblinski_tang,
};
use basin::{
    BasicSimplexState, BasicState, CountsMirror, Executor, GradientDescent, IntoInitialSimplex,
    LbfgsState, Lbfgsb, MoreThuente, NelderMead, Newuoa, NewuoaState, Solver, State as BasinState,
    StepOutcome,
};
use competitor_bench::{ArgminProblem, GomezProblem};
use gomez::OptimizerDriver;
use gomez::algo::NelderMead as GomezNelderMead;

/// Fixed iteration budget: matches `benches/gd_nm.rs` and the backend
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

/// Dimension for the NEWUOA-vs-NEWUOA Styblinski–Tang case. A modest `n` where
/// both implementations descend to the global minimum and track closely; the
/// multimodal Styblinski–Tang gives the model-based DFO method more to chew on
/// than Rosenbrock does.
const N_ST: usize = 5;

/// Matched trust-region schedule for the NEWUOA case: initial radius ρ_beg
/// (basin's `with_rho_beg` and nlopt's initial step) and final radius ρ_end
/// (basin's `with_rho_end` and nlopt's `xtol_abs`).
const ST_RHO_BEG: f64 = 0.5;
const ST_RHO_END: f64 = 1e-6;
/// Safety budget for the NEWUOA case: basin's iteration cap and nlopt's eval
/// cap (≈ the same on NEWUOA, ~1 eval/iter after the `2n+1` init). Both sides
/// converge on ρ well before this, so it just bounds a pathological run; the
/// case does *not* use the fixed `MAX_ITERS` budget the other cases do.
const ST_BUDGET: u64 = 1000;

/// classic Rosenbrock start.
fn start() -> Vec<f64> {
    vec![-1.2, 1.0]
}

/// Styblinski–Tang start: the origin sits in the global well's basin, so both
/// NEWUOA implementations descend to the global minimum (rather than stalling
/// in one of the `2ⁿ` local wells), giving an honest convergence-to-`f*` curve.
fn st_start() -> Vec<f64> {
    vec![0.0; N_ST]
}

/// Styblinski–Tang optimum `f*` for dimension `n`. The library's `MINIMIZER`
/// (`-2.903534`) is only 6-digit accurate, coarser than the solvers' final
/// accuracy, which would float `f*` above the true minimum and floor the
/// suboptimality tail dishonestly. Newton-refine the per-coordinate minimizer
/// (root of `g'(t) = 2t³ − 16t + 2.5`) to machine precision first.
fn st_fopt(n: usize) -> f64 {
    let mut t = -2.903534_f64;
    for _ in 0..8 {
        let g1 = 2.0 * t * t * t - 16.0 * t + 2.5;
        let g2 = 6.0 * t * t - 16.0;
        t -= g1 / g2;
    }
    styblinski_tang(&vec![t; n])
}

// ---------------------------------------------------------------------
// basin side: step the `Stepper`, timestamping after each iteration.
// ---------------------------------------------------------------------

/// Run a basin solve to `max_iter`, returning `(elapsed_ns, cost)` at iter 0
/// and after every completed iteration. `max_iter` is a cap, not a target:
/// the GD, NM, and L-BFGS cases run the full `MAX_ITERS` budget, but NEWUOA stops
/// earlier on its own ρ-convergence, so it passes a generous cap that only
/// bounds a pathological run.
fn basin_trace<P, S, So>(exec: Executor<P, S, So>, max_iter: u64) -> Vec<(u128, f64)>
where
    S: BasinState<Float = f64> + CountsMirror,
    So: Solver<P, S>,
    So::Error: std::fmt::Debug,
{
    let mut stepper = exec.max_iter(max_iter).into_stepper().unwrap();
    let mut pts = Vec::with_capacity(max_iter as usize + 1);
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
/// first iteration registers a best. But at iter 0 (t = 0) the true best
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
// gomez side: step `OptimizerDriver::next`, timestamping after each call.
// gomez doesn't expose a guaranteed-initialized `fx()` before the first
// step (unlike basin's stepper, whose `init` runs eagerly), so we seed
// (0_ns, f(x0)) explicitly and reset the clock, mirroring how
// `finite_start` handles argmin's leading +∞.
// ---------------------------------------------------------------------

fn gomez_trace_nm(f0: f64) -> Vec<(u128, f64)> {
    let problem = GomezProblem::new(rosenbrock, N);
    let mut optimizer = OptimizerDriver::builder(&problem)
        .with_initial(start())
        .with_algo(GomezNelderMead::new)
        .build();
    let mut pts = Vec::with_capacity(MAX_ITERS as usize + 1);
    pts.push((0u128, f0));
    let t0 = Instant::now();
    for _ in 0..MAX_ITERS {
        match optimizer.next() {
            Ok((_x, fx)) => pts.push((t0.elapsed().as_nanos(), fx)),
            Err(_) => break,
        }
    }
    pts
}

// ---------------------------------------------------------------------
// nlopt side: NLopt has no per-iteration callback hook: the objective
// closure is the only observation point. Record `(elapsed_ns, f_best)`
// inside the closure on every improvement, producing a monotone
// best-so-far curve at function-eval granularity. The clock is started
// at the first cost call (so t = 0 means "just before NLopt asks for
// the first cost", matching argmin's observer-init reset). The eval
// budget is `MAX_ITERS` and the function and parameter tolerances are
// zeroed so NLopt definitely uses the full budget.
// ---------------------------------------------------------------------

struct NloptState {
    start: Option<Instant>,
    best: f64,
    points: Vec<(u128, f64)>,
}

/// Inert gradient hook for derivative-free algorithms: NLopt never
/// asks for a gradient under `Neldermead`, so this is dead code there.
fn noop_grad(_x: &[f64], _g: &mut [f64]) {}

fn nlopt_trace(
    algo: nlopt::Algorithm,
    cost: fn(&[f64]) -> f64,
    grad: fn(&[f64], &mut [f64]),
    f0: f64,
) -> Vec<(u128, f64)> {
    let state = NloptState {
        start: None,
        best: f0,
        points: vec![(0u128, f0)],
    };
    let obj = move |x: &[f64], g: Option<&mut [f64]>, st: &mut NloptState| -> f64 {
        if let Some(g) = g {
            grad(x, g);
        }
        let f = cost(x);
        let t = match st.start {
            Some(t0) => t0.elapsed().as_nanos(),
            None => {
                st.start = Some(Instant::now());
                0
            }
        };
        if f < st.best {
            st.best = f;
            st.points.push((t, st.best));
        }
        f
    };
    let mut opt = nlopt::Nlopt::new(algo, N, obj, nlopt::Target::Minimize, state);
    opt.set_maxeval(MAX_ITERS as u32).unwrap();
    // Zero tolerances so NLopt runs the full eval budget (no early
    // stop), matching the other libraries' policy in this harness.
    let _ = opt.set_ftol_rel(0.0);
    let _ = opt.set_ftol_abs(0.0);
    let _ = opt.set_xtol_rel(0.0);
    let mut x = start();
    let _ = opt.optimize(&mut x);
    opt.recover_user_data().points
}

/// nlopt's `LN_NEWUOA` on the Styblinski–Tang case, configured to match
/// basin's NEWUOA: initial step = ρ_beg and `xtol_abs` = ρ_end (the two
/// libraries' nearest analogs of NEWUOA's initial and final trust-region radius),
/// plus the same generous eval cap, so both sides converge on ρ rather than
/// running a fixed iteration budget. Like `nlopt_trace`, NLopt exposes no
/// per-iteration hook, so the curve is a per-eval best-so-far trace.
fn nlopt_newuoa_trace(f0: f64) -> Vec<(u128, f64)> {
    let state = NloptState {
        start: None,
        best: f0,
        points: vec![(0u128, f0)],
    };
    let obj = move |x: &[f64], _g: Option<&mut [f64]>, st: &mut NloptState| -> f64 {
        let f = styblinski_tang(x);
        let t = match st.start {
            Some(t0) => t0.elapsed().as_nanos(),
            None => {
                st.start = Some(Instant::now());
                0
            }
        };
        if f < st.best {
            st.best = f;
            st.points.push((t, st.best));
        }
        f
    };
    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Newuoa,
        N_ST,
        obj,
        nlopt::Target::Minimize,
        state,
    );
    let _ = opt.set_initial_step1(ST_RHO_BEG);
    let _ = opt.set_xtol_abs1(ST_RHO_END);
    let _ = opt.set_maxeval(ST_BUDGET as u32);
    let mut x = st_start();
    let _ = opt.optimize(&mut x);
    opt.recover_user_data().points
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
    problem: &'static str,
    n: usize,
    f_opt: f64,
    library: &'static str,
    points: Vec<(u128, f64)>,
}

fn print_traces(traces: &[Trace]) {
    let mut out = String::from("[\n");
    for (ti, t) in traces.iter().enumerate() {
        out.push_str(&format!(
            "  {{\"solver\":\"{}\",\"problem\":\"{}\",\"n\":{},\"library\":\"{}\",\"points\":[",
            t.solver, t.problem, t.n, t.library
        ));
        // Skip any non-finite point so the emitted JSON is always valid
        // (`finite_start` already handles argmin's leading +∞).
        let mut first = true;
        for &(t_ns, cost) in &t.points {
            let diff = cost - t.f_opt;
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
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "basin",
            points: median_reps(|| {
                basin_trace(
                    Executor::new(
                        Rosenbrock::<Vec<f64>>::default(),
                        GradientDescent::with_line_search(MoreThuente::new()),
                        BasicState::new(start()),
                    ),
                    MAX_ITERS,
                )
            }),
        },
        Trace {
            solver: "gd",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
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
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "basin",
            points: median_reps(|| {
                basin_trace(
                    Executor::new(
                        Rosenbrock::<Vec<f64>>::default(),
                        NelderMead::new(),
                        BasicSimplexState::new(start()),
                    ),
                    MAX_ITERS,
                )
            }),
        },
        Trace {
            solver: "nm",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
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
        Trace {
            solver: "nm",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "gomez",
            points: median_reps(|| gomez_trace_nm(f0)),
        },
        Trace {
            solver: "nm",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "nlopt",
            points: median_reps(|| {
                nlopt_trace(nlopt::Algorithm::Neldermead, rosenbrock, noop_grad, f0)
            }),
        },
        // ---- L-BFGS (limited memory m = 10, More-Thuente) ----
        Trace {
            solver: "lbfgs",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "basin",
            points: median_reps(|| {
                basin_trace(
                    Executor::new(
                        Rosenbrock::<Vec<f64>>::default(),
                        Lbfgsb::new().unbounded(),
                        LbfgsState::new(start(), 10),
                    ),
                    MAX_ITERS,
                )
            }),
        },
        Trace {
            solver: "lbfgs",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
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
        Trace {
            solver: "lbfgs",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "nlopt",
            points: median_reps(|| {
                nlopt_trace(nlopt::Algorithm::Lbfgs, rosenbrock, rosenbrock_gradient, f0)
            }),
        },
        // ---- NEWUOA (Powell's model-based DFO): same algorithm, two
        //      implementations: basin vs nlopt's `LN_NEWUOA`, matched
        //      ρ_beg/ρ_end and `npt = 2n+1`, on Styblinski–Tang at n = 5 from the
        //      origin. Both run to natural ρ-convergence (not the iter cap). ----
        Trace {
            solver: "newuoa",
            problem: "styblinski",
            n: N_ST,
            f_opt: st_fopt(N_ST),
            library: "basin",
            points: median_reps(|| {
                basin_trace(
                    Executor::new(
                        StyblinskiTang::<Vec<f64>>::default(),
                        Newuoa::new()
                            .with_rho_beg(ST_RHO_BEG)
                            .with_rho_end(ST_RHO_END),
                        NewuoaState::new(st_start()),
                    ),
                    ST_BUDGET,
                )
            }),
        },
        Trace {
            solver: "newuoa",
            problem: "styblinski",
            n: N_ST,
            f_opt: st_fopt(N_ST),
            library: "nlopt",
            points: median_reps(|| nlopt_newuoa_trace(styblinski_tang(&st_start()))),
        },
    ];

    print_traces(&traces);
}
