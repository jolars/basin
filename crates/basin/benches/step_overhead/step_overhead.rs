//! Reproduces the small solves in https://github.com/jolars/basin/issues/140.
use std::convert::Infallible;
use std::fmt::Debug;
use std::hint::black_box;
use std::time::Instant;

use basin::core::convergence::{ConfiguredSolver, CostChecks, GradientChecks};
use basin::core::state::CountsMirror;
use basin::solver::lbfgs::Unbounded;
use basin::{
    CostFunction, EvaluationKind, Executor, FirstOrderState, Gradient,
    IncumbentState, InnerExecutor, Jacobian, Lbfgs, Lbfgsb, LevenbergMarquardt,
    LmDamping, MoreThuente, PointState, Problem, Residual, Solver, State,
    Stepper, TerminationReport,
};
use ndarray::{Array1, Array2};

struct Decay {
    t: Vec<f64>,
    y: Vec<f64>,
}

impl Decay {
    fn new(m: usize, truth: &[f64]) -> Self {
        let t: Vec<_> = (0..m).map(|i| 0.1 + i as f64 / m as f64).collect();
        let y = t.iter().map(|&t| Self::model(truth, t)).collect();
        Self { t, y }
    }

    fn model(x: &[f64], t: f64) -> f64 {
        (-x.iter()
            .enumerate()
            .map(|(j, x)| x * t.powi(j as i32 + 1))
            .sum::<f64>())
        .exp()
    }
}

impl Residual for &Decay {
    type Param = Array1<f64>;
    type Output = Array1<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Array1<f64>) -> Result<Array1<f64>, Infallible> {
        let x = x.as_slice().unwrap();
        Ok(self
            .t
            .iter()
            .zip(&self.y)
            .map(|(&t, y)| Decay::model(x, t) - y)
            .collect())
    }
}

impl Jacobian for &Decay {
    type Jacobian = Array2<f64>;

    fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>, Infallible> {
        let xs = x.as_slice().unwrap();
        Ok(Array2::from_shape_fn((self.t.len(), x.len()), |(i, j)| {
            -self.t[i].powi(j as i32 + 1) * Decay::model(xs, self.t[i])
        }))
    }
}

struct Bowl;

impl CostFunction for &Bowl {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, x)| {
                (i + 1) as f64 * (x - 1.0).powi(2) + 0.25 * (x - 1.0).powi(4)
            })
            .sum())
    }
}

impl Gradient for &Bowl {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, x)| 2.0 * (i + 1) as f64 * (x - 1.0) + (x - 1.0).powi(3))
            .collect())
    }
}

type Lm = LevenbergMarquardt<Array1<f64>, Array2<f64>>;
type Polish = ConfiguredSolver<
    Lbfgs<Vec<f64>, f64, Unbounded, MoreThuente>,
    GradientChecks,
    (),
    CostChecks,
>;

fn lm() -> Lm {
    let tol = 30.0 * f64::EPSILON;
    Lm::new()
        .with_damping(LmDamping::TrustRegion)
        .with_absolute_gradient_tolerance(None)
        .with_gradient_orthogonality_tolerance(tol)
        .with_relative_model_reduction_tolerance(tol)
        .with_relative_trust_radius_tolerance(tol)
}

fn polish() -> Polish {
    let search = MoreThuente::new()
        .with_sufficient_decrease_coefficient(1e-4)
        .with_curvature_coefficient(0.9)
        .with_relative_bracket_tolerance(1e-10)
        .stpmin(f64::EPSILON.sqrt())
        .stpmax(f64::INFINITY);
    Lbfgsb::with_line_search(search)
        .with_m_capacity(10)
        .unbounded()
        .with_absolute_gradient_tolerance(f64::EPSILON.sqrt())
        .with_absolute_cost_change_tolerance(f64::EPSILON)
}

// The borrowed-report variant replaces just this body.
fn step<P, S, So, T>(
    stepper: &mut Stepper<P, S, So>,
    read: impl FnOnce(&TerminationReport<f64>) -> T,
) -> Option<T>
where
    S: State<Float = f64> + CountsMirror,
    So: Solver<P, S, Error = Infallible>,
{
    stepper.step().unwrap().report().map(read)
}

fn cases(
    mut decay: impl FnMut(&str, &Decay, &Array1<f64>),
    mut bowl: impl FnMut(&Bowl, &Vec<f64>),
) {
    for (name, m, truth, start) in [
        ("lm-1x9", 9, vec![1.3], vec![0.2]),
        ("lm-1x9-far", 9, vec![1.3], vec![4.0]),
        (
            "lm-5x21",
            21,
            vec![1.3, -0.4, 0.3, 0.1, -0.2],
            vec![0.5, 0.0, 0.0, 0.0, 0.0],
        ),
    ] {
        decay(name, &Decay::new(m, &truth), &Array1::from(start));
    }
    bowl(&Bowl, &vec![3.0, -2.0, 0.5, 4.0]);
}

const LM_CAPS: [u64; 5] = [0, 1, 2, 4, 1000];
const POLISH_CAPS: [u64; 6] = [0, 1, 2, 4, 8, 1000];

fn time(
    name: &str,
    path: &str,
    cap: u64,
    reps: usize,
    mut solve: impl FnMut(),
) {
    for _ in 0..1000 {
        solve();
    }
    let start = Instant::now();
    for _ in 0..reps {
        solve();
    }
    println!(
        "{name},{path},{cap},{:.3}",
        start.elapsed().as_nanos() as f64 / reps as f64
    );
}

fn timing(reps: usize) {
    cases(
        |name, problem, start| {
            for cap in LM_CAPS {
                time(name, "manual", cap, reps, || {
                    let problem = black_box(problem);
                    let start = black_box(start);
                    let mut best = start.clone();
                    let mut stepper = Executor::new(
                        problem,
                        lm(),
                        PointState::new(start.clone()),
                    )
                    .max_iter(cap)
                    .max_evaluations(EvaluationKind::Residual, 100_000)
                    .into_stepper()
                    .unwrap();
                    loop {
                        let cost = stepper.state().best_cost();
                        let stop = step(&mut stepper, TerminationReport::code);
                        if let Some((point, _)) = stepper
                            .state()
                            .best()
                            .filter(|(_, lower)| *lower < cost)
                        {
                            best.clone_from(point);
                        }
                        if let Some(code) = stop {
                            black_box((stepper.iter(), best[0], code));
                            break;
                        }
                    }
                });
            }
        },
        |problem, start| {
            for cap in POLISH_CAPS {
                time("lbfgs-4", "run", cap, reps, || {
                    let result = Executor::new(
                        black_box(problem),
                        polish(),
                        FirstOrderState::new(black_box(start).clone()),
                    )
                    .max_iter(cap)
                    .run()
                    .unwrap();
                    black_box((
                        result.iter(),
                        result.best_cost(),
                        result.report.code(),
                    ));
                });
                time("lbfgs-4", "manual", cap, reps, || {
                    let mut stepper = Executor::new(
                        black_box(problem),
                        polish(),
                        FirstOrderState::new(black_box(start).clone()),
                    )
                    .max_iter(cap)
                    .into_stepper()
                    .unwrap();
                    loop {
                        let stop = step(&mut stepper, TerminationReport::code);
                        black_box(stepper.state().best());
                        if let Some(code) = stop {
                            black_box((
                                stepper.iter(),
                                stepper.state().best_cost(),
                                code,
                            ));
                            break;
                        }
                    }
                });
            }
        },
    );
}

fn verify<P, S, So>(
    name: &str,
    caps: &[u64],
    make_executor: impl Fn() -> Executor<P, S, So>,
    make_problem: impl Fn() -> P,
    seed: impl Fn() -> S,
    make_solver: impl Fn() -> So,
    charge_prefix: impl Fn(&mut Problem<P>),
    objective: impl Fn(&S::Param) -> f64,
) where
    S: IncumbentState<Float = f64> + CountsMirror + PartialEq + Debug,
    S::Param: Debug,
    So: Solver<P, S, Error = Infallible>,
{
    for &cap in caps {
        let reference = make_executor().max_iter(cap).run().unwrap();
        let mut stepper = make_executor().max_iter(cap).into_stepper().unwrap();
        println!(
            "{name}/{cap}/init {:?} {:?}",
            stepper.state(),
            stepper.counts()
        );
        let mut trace = Vec::new();
        loop {
            // Own a printable report only in verification, after reading it.
            let report = step(&mut stepper, |report| format!("{report:?}"));
            let boundary = format!(
                "{:?} {:?} {report:?}",
                stepper.state(),
                stepper.counts()
            );
            println!("{name}/{cap}/step {boundary}");
            trace.push(boundary);
            if report.is_some() {
                // Sticky manual stop and ownership handoff to the result.
                assert_eq!(
                    step(&mut stepper, |report| format!("{report:?}")),
                    report
                );
                break;
            }
        }
        let manual = stepper.run_to_end().unwrap();
        assert_eq!(manual.state, reference.state);
        assert_eq!(manual.counts, reference.counts);
        assert_eq!(manual.report, reference.report);
        let best = manual.incumbent_record().unwrap();
        let expected = objective(best.param);
        assert!(
            (best.cost - expected).abs()
                <= 16.0 * f64::EPSILON * expected.abs()
        );

        // The same solver via InnerExecutor with pre-existing wrapper counts.
        let mut problem = Problem::new(make_problem());
        charge_prefix(&mut problem);
        let baseline = *problem.counts();
        let composed = InnerExecutor::new(make_solver())
            .max_iter(cap)
            .run(&mut problem, seed())
            .unwrap();
        assert_eq!(composed.state, reference.state);
        assert_eq!(composed.report, reference.report);
        assert_eq!(composed.counts, reference.counts);
        assert_eq!(problem.counts().delta_since(&baseline), reference.counts);

        // Compare the full suffix of the trajectory across exact continuation.
        for pause in [0, 2] {
            if pause >= trace.len() {
                continue;
            }
            let mut split =
                make_executor().max_iter(cap).into_stepper().unwrap();
            for _ in 0..pause {
                assert!(step(&mut split, TerminationReport::code).is_none());
            }
            let checkpoint = split.into_checkpoint().unwrap();
            let expected_counts = *checkpoint.counts();
            let mut resumed =
                Executor::resume_from_checkpoint(make_problem(), checkpoint)
                    .max_iter(cap)
                    .into_stepper()
                    .unwrap();
            assert_eq!(*resumed.counts(), expected_counts);
            for expected in &trace[pause..] {
                let report = step(&mut resumed, |report| format!("{report:?}"));
                let boundary = format!(
                    "{:?} {:?} {report:?}",
                    resumed.state(),
                    resumed.counts()
                );
                assert_eq!(&boundary, expected);
            }
            assert_eq!(resumed.run_to_end().unwrap().report, reference.report);
        }
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("--verify") => cases(
            |name, problem, start| {
                verify(
                    name,
                    &LM_CAPS,
                    || {
                        Executor::new(
                            problem,
                            lm(),
                            PointState::new(start.clone()),
                        )
                        .max_evaluations(EvaluationKind::Residual, 100_000)
                    },
                    || problem,
                    || PointState::new(start.clone()),
                    lm,
                    |wrapper| {
                        wrapper.residual(start).unwrap();
                        wrapper.jacobian(start).unwrap();
                    },
                    |x| {
                        let residual = problem.residual(x).unwrap();
                        0.5 * residual.dot(&residual)
                    },
                )
            },
            |problem, start| {
                verify(
                    "lbfgs-4",
                    &POLISH_CAPS,
                    || {
                        Executor::new(
                            problem,
                            polish(),
                            FirstOrderState::new(start.clone()),
                        )
                    },
                    || problem,
                    || FirstOrderState::new(start.clone()),
                    polish,
                    |wrapper| {
                        wrapper.cost(start).unwrap();
                        wrapper.gradient(start).unwrap();
                    },
                    |x| problem.cost(x).unwrap(),
                )
            },
        ),
        Some(reps) => timing(reps.parse().unwrap()),
        None => timing(20_000),
    }
}
