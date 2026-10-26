//! Published competitor convergence traces on workloads sized for millisecond solves.
//!
//! Run with the serial L-BFGS-B reference feature:
//! `cargo run --release -p competitor-bench --features lbfgsb-reference --bin trace`.
//! Append `-- --nm` to run only the shared-simplex Nelder-Mead comparison.
//! Append `-- --lbfgsb` to run only the bounded L-BFGS-B comparison.
//! Cases, starts, targets, and remaining comparison asymmetries are documented
//! on the website's library-benchmark page. These traces measure solver
//! effectiveness, including differences in trajectories and stopping rules.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{
    Error, Executor as ArgminExecutor, KV, State as ArgminState,
};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;
use argmin::solver::quasinewton::LBFGS as ArgminLBFGS;
use basin::problems::{
    Rosenbrock, StyblinskiTang, rosenbrock, rosenbrock_gradient,
    styblinski_tang,
};
use basin::{
    CountsMirror, Executor, FirstOrderState, GradientDescent, Lbfgsb,
    MoreThuente, NelderMead, Newuoa, PointState, SimplexProgress, Solver,
    State as BasinState, StepOutcome,
};
use competitor_bench::{
    ArgminProblem, GomezProblem, cobyla, convergence, lm, nm, slsqp,
};
use gomez::OptimizerDriver;

/// Safety cap, not a prescription of equal work across libraries.
const MAX_ITERS: u64 = 10_000;
/// Repetitions per (case, library) for the per-iteration time median.
const REPS: usize = 11;
/// Rosenbrock's global minimum value.
const F_OPT: f64 = 0.0;
/// Suboptimality floor so the log-scale y-axis stays well-defined.
const FLOOR: f64 = 1e-16;
/// Dimension for gradient methods on chained Rosenbrock.
const N: usize = 100;
const NM_N: usize = 12;

/// Dimension for the model-based derivative-free comparison.
const N_ST: usize = 30;

/// Matched trust-region schedule for the NEWUOA case: initial radius ρ_beg
/// (basin's `with_initial_radius` and nlopt's initial step) and final radius ρ_end
/// (basin's `with_final_radius` and nlopt's `xtol_abs`).
const ST_RHO_BEG: f64 = 0.5;
const ST_RHO_END: f64 = 1e-6;
/// Safety cap; both implementations normally stop on the final radius.
const ST_BUDGET: u64 = 10_000;

/// Alternating Rosenbrock start, extended to the selected dimension.
fn start() -> Vec<f64> {
    (0..N)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
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

/// Include initialization and the final step, then stop at the target or solver termination.
fn basin_trace<P, S, So>(
    exec: Executor<P, S, So>,
    max_iter: u64,
) -> Vec<(u128, f64)>
where
    S: BasinState<Float = f64> + CountsMirror,
    So: Solver<P, S>,
    So::Error: std::fmt::Debug,
{
    let t0 = Instant::now();
    let mut stepper = exec.max_iter(max_iter).into_stepper().unwrap();
    let mut pts = Vec::with_capacity(max_iter as usize + 1);
    pts.push((0u128, stepper.state().cost()));
    loop {
        let outcome = stepper.step().unwrap();
        let cost = stepper.state().cost();
        pts.push((t0.elapsed().as_nanos(), cost));
        if outcome != StepOutcome::Continue || (0.0..=1e-10).contains(&cost) {
            break;
        }
    }
    pts
}

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
    fn observe_init(
        &mut self,
        _name: &str,
        state: &I,
        _kv: &KV,
    ) -> Result<(), Error> {
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
        start: Some(Instant::now()),
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

fn nm_start() -> Vec<f64> {
    nm::start(NM_N)
}

fn gomez_trace_nm(f0: f64) -> Vec<(u128, f64)> {
    let t0 = Instant::now();
    let problem = GomezProblem::new(rosenbrock, NM_N);
    let mut optimizer = OptimizerDriver::builder(&problem)
        .with_initial(nm_start())
        .with_algo(nm::gomez_solver)
        .build();
    let mut pts = Vec::with_capacity(MAX_ITERS as usize + 1);
    pts.push((0u128, f0));
    for _ in 0..MAX_ITERS {
        match optimizer.next() {
            Ok((_x, fx)) => {
                pts.push((t0.elapsed().as_nanos(), fx));
                if fx <= 1e-10 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    pts
}

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
        start: Some(Instant::now()),
        best: f0,
        points: vec![(0u128, f0)],
    };
    let obj =
        move |x: &[f64], g: Option<&mut [f64]>, st: &mut NloptState| -> f64 {
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
    let mut opt = nlopt::Nlopt::new(
        algo,
        if matches!(algo, nlopt::Algorithm::Neldermead) {
            NM_N
        } else {
            N
        },
        obj,
        nlopt::Target::Minimize,
        state,
    );
    if matches!(algo, nlopt::Algorithm::Lbfgs) {
        opt.set_vector_storage(Some(10)).unwrap();
    }
    if matches!(algo, nlopt::Algorithm::Neldermead) {
        opt.set_initial_step(&nm::initial_steps(NM_N)).unwrap();
    }
    opt.set_maxeval(MAX_ITERS as u32).unwrap();
    opt.set_stopval(1e-10).unwrap();
    // The common objective target is distinct from library-specific tolerances.
    let _ = opt.set_ftol_rel(0.0);
    let _ = opt.set_ftol_abs(0.0);
    let _ = opt.set_xtol_rel(0.0);
    let mut x = if matches!(algo, nlopt::Algorithm::Neldermead) {
        nm_start()
    } else {
        start()
    };
    let _status = opt.optimize(&mut x);
    let mut state = opt.recover_user_data();
    let elapsed = state.start.unwrap().elapsed().as_nanos();
    state.points.push((elapsed, state.best));
    assert!(x.iter().all(|v| v.is_finite()));
    assert!(state.best.is_finite());
    state.points
}

/// nlopt's `LN_NEWUOA` on the Styblinski–Tang case, configured to match
/// basin's NEWUOA: initial step = ρ_beg and `xtol_abs` = ρ_end (the two
/// libraries' nearest analogs of NEWUOA's initial and final trust-region radius),
/// plus the same generous eval cap, so both sides converge on ρ rather than
/// running a fixed iteration budget. Like `nlopt_trace`, NLopt exposes no
/// per-iteration hook, so the curve is a per-eval best-so-far trace.
fn nlopt_newuoa_trace(f0: f64) -> Vec<(u128, f64)> {
    let state = NloptState {
        start: Some(Instant::now()),
        best: f0,
        points: vec![(0u128, f0)],
    };
    let obj =
        move |x: &[f64], _g: Option<&mut [f64]>, st: &mut NloptState| -> f64 {
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
    let _status = opt.optimize(&mut x);
    let mut state = opt.recover_user_data();
    let elapsed = state.start.unwrap().elapsed().as_nanos();
    state.points.push((elapsed, state.best));
    assert!(x.iter().all(|v| v.is_finite()));
    assert!(state.best.is_finite());
    state.points
}

// ---------------------------------------------------------------------
// median over reps
// ---------------------------------------------------------------------

/// Run `run` `REPS` times and median the elapsed time per iteration index,
/// keeping the (deterministic) cost from the first rep.
fn median_reps(mut run: impl FnMut() -> Vec<(u128, f64)>) -> Vec<(u128, f64)> {
    drop(run());
    let runs: Vec<Vec<(u128, f64)>> = (0..REPS).map(|_| run()).collect();
    let len = runs[0].len();
    for candidate in &runs[1..] {
        assert_eq!(candidate.len(), len, "trace length changed between runs");
        assert!(candidate.iter().zip(&runs[0]).all(|(a, b)| a.1 == b.1));
    }
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
    violations: Vec<(u128, f64)>,
    summary: Option<convergence::Summary>,
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
        let stride = t.points.len().div_ceil(512).max(1);
        for (i, &(t_ns, cost)) in t.points.iter().enumerate() {
            if i % stride != 0 && i + 1 != t.points.len() {
                continue;
            }
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
        if !t.violations.is_empty() {
            out.push_str(",\"violations\":[");
            for (i, &(time, violation)) in t.violations.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"tNs\":{time},\"violation\":{violation}}}"
                ));
            }
            out.push(']');
        }
        if let Some(summary) = &t.summary {
            let point = format!(
                "[{}]",
                summary
                    .point
                    .iter()
                    .map(|&x| json_number(x))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            out.push_str(&format!(
                ",\"summary\":{{\"point\":{},\"cost\":{},\"violation\":{},\"objectiveCalls\":{},\"residualCalls\":{},\"jacobianCalls\":{},\"gradientCalls\":{},\"constraintCalls\":{},\"status\":{:?},\"verified\":{}",
                point,
                json_number(summary.cost), json_number(summary.violation),
                summary.objective_calls, summary.residual_calls, summary.jacobian_calls, summary.gradient_calls,
                summary.constraint_calls, summary.status, summary.verified,
            ));
            if let Some(iterations) = summary.iterations {
                out.push_str(&format!(",\"iterations\":{iterations}"));
            }
            if let Some(pg) = summary.projected_gradient {
                out.push_str(&format!(
                    ",\"projectedGradient\":{}",
                    json_number(pg)
                ));
            }
            out.push('}');
        }
        out.push('}');
        out.push_str(if ti + 1 < traces.len() { ",\n" } else { "\n" });
    }
    out.push(']');
    println!("{out}");
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".into()
    }
}

fn published_trace(
    solver: &'static str,
    problem: &'static str,
    f_opt: f64,
    library: &'static str,
    run: convergence::Run,
) -> Trace {
    eprintln!(
        "{solver}/{library}: {:.3} ms, f={:.6e}, violation={:.3e}, calls={}, status={}, verified={}",
        run.points.last().map_or(0.0, |p| p.0 as f64 / 1e6),
        run.summary.cost,
        run.summary.violation,
        run.summary.objective_calls + run.summary.residual_calls,
        run.summary.status,
        run.summary.verified
    );
    Trace {
        solver,
        problem,
        n: run.summary.point.len(),
        f_opt,
        library,
        points: run.points,
        violations: run.violations,
        summary: Some(run.summary),
    }
}

fn nm_traces() -> Vec<Trace> {
    let nm_f0 = rosenbrock(&nm_start());
    vec![
        // ---- Nelder-Mead (standard coeffs, identical initial simplex) ----
        Trace {
            violations: Vec::new(),
            summary: None,
            solver: "nm",
            problem: "rosenbrock",
            n: NM_N,
            f_opt: F_OPT,
            library: "basin",
            points: median_reps(|| {
                basin_trace(
                    Executor::new(
                        Rosenbrock::<Vec<f64>>::default(),
                        NelderMead::new(),
                        SimplexProgress::from_simplex(nm::initial_simplex(
                            NM_N,
                        )),
                    ),
                    MAX_ITERS,
                )
            }),
        },
        Trace {
            violations: Vec::new(),
            summary: None,
            solver: "nm",
            problem: "rosenbrock",
            n: NM_N,
            f_opt: F_OPT,
            library: "argmin",
            points: finite_start(
                median_reps(|| {
                    let simplex = nm::initial_simplex(NM_N);
                    let nm = ArgminNelderMead::new(simplex)
                        .with_sd_tolerance(0.0)
                        .unwrap();
                    let (obs, points) = observer();
                    ArgminExecutor::new(
                        ArgminProblem::new(rosenbrock, rosenbrock_gradient),
                        nm,
                    )
                    .configure(|s| s.max_iters(MAX_ITERS).target_cost(1e-10))
                    .add_observer(obs, ObserverMode::Always)
                    .run()
                    .unwrap();
                    drain(&points)
                }),
                nm_f0,
            ),
        },
        Trace {
            violations: Vec::new(),
            summary: None,
            solver: "nm",
            problem: "rosenbrock",
            n: NM_N,
            f_opt: F_OPT,
            library: "gomez",
            points: median_reps(|| gomez_trace_nm(nm_f0)),
        },
        Trace {
            violations: Vec::new(),
            summary: None,
            solver: "nm",
            problem: "rosenbrock",
            n: NM_N,
            f_opt: F_OPT,
            library: "nlopt",
            points: median_reps(|| {
                nlopt_trace(
                    nlopt::Algorithm::Neldermead,
                    rosenbrock,
                    noop_grad,
                    nm_f0,
                )
            }),
        },
    ]
}

fn main() {
    assert!(
        !cfg!(feature = "parallel") && !cfg!(feature = "basin-latest"),
        "Publish competitor traces with the default, serial nalgebra 0.34 configuration"
    );
    if std::env::args().nth(1).as_deref() == Some("--nm") {
        print_traces(&nm_traces());
        return;
    }
    assert!(
        cfg!(feature = "lbfgsb-reference"),
        "Enable --features lbfgsb-reference to generate the complete published dataset"
    );
    #[cfg(feature = "lbfgsb-reference")]
    if std::env::args().nth(1).as_deref() == Some("--lbfgsb") {
        print_traces(&lbfgsb_traces());
        return;
    }
    // Cost at the shared start, used to give argmin's curve a finite t = 0.
    let f0 = rosenbrock(&start());

    let mut traces = vec![
        // ---- gradient descent (steepest + More-Thuente) ----
        Trace {
            violations: Vec::new(),
            summary: None,
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
                        FirstOrderState::new(start()),
                    ),
                    MAX_ITERS,
                )
            }),
        },
        Trace {
            violations: Vec::new(),
            summary: None,
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
                    .configure(|s| {
                        s.param(start()).max_iters(MAX_ITERS).target_cost(1e-10)
                    })
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
            violations: Vec::new(),
            summary: None,
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
                        FirstOrderState::new(start()),
                    ),
                    MAX_ITERS,
                )
            }),
        },
        Trace {
            violations: Vec::new(),
            summary: None,
            solver: "lbfgs",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "argmin",
            points: finite_start(
                median_reps(|| {
                    let ls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
                        MoreThuenteLineSearch::new();
                    let lbfgs: ArgminLBFGS<_, Vec<f64>, Vec<f64>, f64> =
                        ArgminLBFGS::new(ls, 10)
                            .with_tolerance_grad(0.0)
                            .unwrap()
                            .with_tolerance_cost(0.0)
                            .unwrap();
                    let (obs, points) = observer();
                    ArgminExecutor::new(
                        ArgminProblem::new(rosenbrock, rosenbrock_gradient),
                        lbfgs,
                    )
                    .configure(|s| {
                        s.param(start()).max_iters(MAX_ITERS).target_cost(1e-10)
                    })
                    .add_observer(obs, ObserverMode::Always)
                    .run()
                    .unwrap();
                    drain(&points)
                }),
                f0,
            ),
        },
        Trace {
            violations: Vec::new(),
            summary: None,
            solver: "lbfgs",
            problem: "rosenbrock",
            n: N,
            f_opt: F_OPT,
            library: "nlopt",
            points: median_reps(|| {
                nlopt_trace(
                    nlopt::Algorithm::Lbfgs,
                    rosenbrock,
                    rosenbrock_gradient,
                    f0,
                )
            }),
        },
        // ---- NEWUOA (Powell's model-based DFO): same algorithm, two
        //      implementations: basin vs nlopt's `LN_NEWUOA`, matched
        //      ρ_beg/ρ_end and `npt = 2n+1`, on Styblinski–Tang at n = 30 from the
        //      origin. Both run to natural ρ-convergence (not the iter cap). ----
        Trace {
            violations: Vec::new(),
            summary: None,
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
                            .with_initial_radius(ST_RHO_BEG)
                            .with_final_radius(ST_RHO_END),
                        PointState::new(st_start()),
                    ),
                    ST_BUDGET,
                )
            }),
        },
        Trace {
            violations: Vec::new(),
            summary: None,
            solver: "newuoa",
            problem: "styblinski",
            n: N_ST,
            f_opt: st_fopt(N_ST),
            library: "nlopt",
            points: median_reps(|| {
                nlopt_newuoa_trace(styblinski_tang(&st_start()))
            }),
        },
    ];

    traces.extend(nm_traces());

    for library in [
        slsqp::Library::Basin,
        slsqp::Library::Slsqp,
        slsqp::Library::Nlopt,
    ] {
        let n = 100;
        let run = convergence::median_run(
            || {
                let result = slsqp::run_constrained(library, n);
                let verified = result.verify_constrained();
                let violation = (result.x.iter().map(|v| v * v).sum::<f64>()
                    / n as f64
                    - 0.5625)
                    .max(0.0);
                convergence::Run {
                    points: result.points,
                    violations: result.violations,
                    summary: convergence::Summary {
                        point: result.x,
                        cost: result.cost,
                        violation,
                        iterations: result.iterations,
                        projected_gradient: None,
                        objective_calls: result.cost_evals,
                        residual_calls: 0,
                        jacobian_calls: 0,
                        gradient_calls: result.gradient_evals,
                        constraint_calls: result.constraint_evals,
                        status: result.status,
                        verified,
                    },
                }
            },
            REPS,
        );
        traces.push(published_trace(
            "slsqp",
            "constrained_quadratic",
            n as f64 / 16.0,
            library.name(),
            run,
        ));
    }

    for library in [lm::Library::Basin, lm::Library::Reference] {
        traces.push(published_trace(
            "lm",
            "exp_fit",
            0.0,
            library.name(),
            convergence::median_run(|| lm::run(library), REPS),
        ));
    }
    for library in [
        cobyla::Library::Basin,
        cobyla::Library::Reference,
        cobyla::Library::Nlopt,
    ] {
        traces.push(published_trace(
            "cobyla",
            "quadratic_30d",
            30.0 / 16.0,
            library.name(),
            convergence::median_run(
                || cobyla::Case::QuadraticNd(30).trace(library),
                REPS,
            ),
        ));
    }
    #[cfg(feature = "lbfgsb-reference")]
    traces.extend(lbfgsb_traces());
    print_traces(&traces);
}

#[cfg(feature = "lbfgsb-reference")]
fn lbfgsb_traces() -> Vec<Trace> {
    use competitor_bench::lbfgsb::{self, Library};
    [Library::Basin, Library::Reference]
        .into_iter()
        .map(|library| {
            published_trace(
                "lbfgsb",
                "extended_rosenbrock",
                0.0,
                library.name(),
                convergence::median_run(
                    || lbfgsb::run(library, lbfgsb::DIMENSION),
                    REPS,
                ),
            )
        })
        .collect()
}
