#[path = "support/nonlinear_cg_backend.rs"]
mod backend;
#[path = "support/backend_aliases.rs"]
mod backend_aliases;

use backend::{Kind, Objective};
use basin::{
    Constant, CostFunction, Dot, Executor, FirstOrderState, Gradient,
    LineSearch, LineSearchOutcome, LineSearchResult, MoreThuente, NonlinearCg,
    NormInfinity, Problem, State, TerminationReason,
};
use std::{cell::RefCell, convert::Infallible, rc::Rc};

fn quadratic() -> Objective<Vec<f64>, f64> {
    Objective {
        make: |x| x.to_vec(),
        kind: Kind::Quadratic,
    }
}

#[test]
fn vec_f64() {
    backend::check::<_, f64>(|x| x.to_vec());
}

#[test]
fn optimal_start_and_empty_history() {
    let result = Executor::new(
        quadratic(),
        NonlinearCg::new(),
        FirstOrderState::new(vec![0.0, 0.0]),
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.state.counts().cost_evals, 1);
    assert_eq!(result.state.counts().gradient_evals, 1);
    assert_eq!(result.state.current().unwrap().2, &vec![0.0, 0.0]);
}

#[derive(Clone, Debug)]
struct Call {
    param: Vec<f64>,
    gradient: Vec<f64>,
    direction: Vec<f64>,
}
type Trace = Rc<RefCell<Vec<Call>>>;

struct Recording<L> {
    inner: L,
    trace: Trace,
    failures: Vec<usize>,
}

impl<P, L> LineSearch<P, Vec<f64>> for Recording<L>
where
    P: Gradient<Param = Vec<f64>, Output = f64, Gradient = Vec<f64>>,
    L: LineSearch<P, Vec<f64>, Error = P::Error>,
{
    type Error = P::Error;
    fn next(
        &mut self,
        _: &mut Problem<P>,
        _: &Vec<f64>,
        _: f64,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<f64, Self::Error> {
        panic!("solver must request the accepted evaluation")
    }
    fn next_with_evaluation(
        &mut self,
        problem: &mut Problem<P>,
        param: &Vec<f64>,
        cost: f64,
        gradient: &Vec<f64>,
        direction: &Vec<f64>,
    ) -> Result<LineSearchResult<Vec<f64>, f64>, Self::Error> {
        let call = self.trace.borrow().len();
        assert!(gradient.dot(direction) <= -0.875 * gradient.dot(gradient));
        self.trace.borrow_mut().push(Call {
            param: param.clone(),
            gradient: gradient.clone(),
            direction: direction.clone(),
        });
        if self.failures.contains(&call) {
            // A rejected probe must still appear in the executor's counts.
            problem.cost(param)?;
            Ok(LineSearchResult::new(LineSearchOutcome::Failed))
        } else {
            self.inner
                .next_with_evaluation(problem, param, cost, gradient, direction)
        }
    }
}

struct ExactQuadratic;
impl<P> LineSearch<P, Vec<f64>> for ExactQuadratic
where
    P: CostFunction<Param = Vec<f64>, Output = f64>,
{
    type Error = P::Error;
    fn next(
        &mut self,
        _: &mut Problem<P>,
        _: &Vec<f64>,
        _: f64,
        gradient: &Vec<f64>,
        d: &Vec<f64>,
    ) -> Result<f64, Self::Error> {
        Ok(-gradient.dot(d)
            / (3.0 * d[0] * d[0] + 2.0 * d[0] * d[1] + 2.0 * d[1] * d[1]))
    }
}

#[test]
fn exact_search_produces_conjugate_directions_and_n_step_convergence() {
    let trace = Trace::default();
    let solver = NonlinearCg::with_line_search(Recording {
        inner: ExactQuadratic,
        trace: trace.clone(),
        failures: vec![],
    });
    let result = Executor::from_start(
        quadratic(),
        solver.with_absolute_gradient_tolerance(1e-12),
        vec![2.0, -1.0],
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert_eq!(result.iter(), 2);
    assert!(result.param().norm_infinity() < 1e-12);
    let trace = trace.borrow();
    assert_eq!(trace.len(), 2);
    let a = &trace[0].direction;
    let b = &trace[1].direction;
    let conjugacy = a[0] * (3.0 * b[0] + b[1]) + a[1] * (b[0] + 2.0 * b[1]);
    assert!(conjugacy.abs() < 1e-12);
}

fn recorded_steps(
    interval: Option<u64>,
    failures: Vec<usize>,
) -> (basin::OptimizationResult<FirstOrderState<Vec<f64>>>, Trace) {
    let trace = Trace::default();
    let solver = NonlinearCg::with_line_search(Recording {
        inner: Constant(0.1),
        trace: trace.clone(),
        failures,
    })
    .with_absolute_gradient_tolerance(None::<f64>)
    .with_eta(0.01)
    .with_restart_interval(interval);
    let result = Executor::from_start(quadratic(), solver, vec![2.0, -1.0])
        .max_iter(4)
        .run()
        .unwrap();
    (result, trace)
}

fn is_steepest(call: &Call) -> bool {
    call.gradient
        .iter()
        .zip(&call.direction)
        .all(|(&g, &d)| d == -g)
}

#[test]
fn periodic_restarts_and_builder_forwarding() {
    for interval in [None, Some(1), Some(2)] {
        let (result, trace) = recorded_steps(interval, vec![]);
        assert_eq!(result.iter(), 4);
        let trace = trace.borrow();
        for (i, call) in trace.iter().enumerate() {
            let expected =
                i == 0 || interval.is_some_and(|n| i as u64 % n == 0);
            assert_eq!(
                is_steepest(call),
                expected,
                "interval={interval:?}, call={i}"
            );
        }
    }
}

#[test]
fn failed_conjugate_search_retries_same_point_with_steepest_descent() {
    let (result, trace) = recorded_steps(Some(2), vec![1]);
    assert_eq!(result.iter(), 4);
    assert_eq!(result.state.counts().cost_evals, 6);
    assert_eq!(result.state.counts().gradient_evals, 5);
    let trace = trace.borrow();
    assert_eq!(trace.len(), 5);
    assert!(!is_steepest(&trace[1]));
    assert!(is_steepest(&trace[2]));
    assert_eq!(trace[1].param, trace[2].param);
    assert_eq!(trace[1].gradient, trace[2].gradient);
    assert!(!is_steepest(&trace[3]));
    assert!(is_steepest(&trace[4]));
}

#[test]
fn exhausted_recovery_preserves_current_record_and_counts() {
    let (result, trace) = recorded_steps(None, vec![1, 2]);
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.iter(), 1);
    assert_eq!(result.param(), &vec![1.5, -1.0]);
    let (x, cost, gradient) = result.state.current().unwrap();
    assert_eq!(
        quadratic().cost_and_gradient(x).unwrap(),
        (cost, gradient.clone())
    );
    assert_eq!(result.state.counts().cost_evals, 4);
    assert_eq!(result.state.counts().gradient_evals, 2);
    assert_eq!(trace.borrow().len(), 3);
    let (result, trace) = recorded_steps(None, vec![0]);
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.iter(), 0);
    assert_eq!(trace.borrow().len(), 1);
    assert_eq!(result.state.counts().cost_evals, 2);
}

struct StepOnly<L>(L);
impl<P, V, F, L: LineSearch<P, V, F>> LineSearch<P, V, F> for StepOnly<L> {
    type Error = L::Error;
    fn next(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<F, Self::Error> {
        self.0.next(problem, param, cost, gradient, direction)
    }
}

#[test]
fn retained_evaluations_save_one_fused_call_per_step() {
    let objective = || Objective {
        make: |x: &[f64]| x.to_vec(),
        kind: Kind::Rosenbrock,
    };
    let retained = Executor::from_start(
        objective(),
        NonlinearCg::with_line_search(MoreThuente::new()),
        vec![-1.2, 1.0],
    )
    .max_iter(6)
    .run()
    .unwrap();
    let discarded = Executor::from_start(
        objective(),
        NonlinearCg::with_line_search(StepOnly(MoreThuente::new())),
        vec![-1.2, 1.0],
    )
    .max_iter(6)
    .run()
    .unwrap();
    assert_eq!(retained.iter(), 6);
    assert_eq!(retained.state.current(), discarded.state.current());
    let saved = retained.state.counts();
    let repeated = discarded.state.counts();
    assert_eq!(repeated.cost_evals, saved.cost_evals + 6);
    assert_eq!(repeated.gradient_evals, saved.gradient_evals + 6);
}

#[test]
fn exact_checkpoints_preserve_directions_and_restart_history() {
    let solver = || {
        NonlinearCg::new()
            .with_relative_gradient_tolerance(1e-10)
            .with_absolute_cost_change_tolerance(None::<f64>)
            .with_restart_interval(Some(3))
    };
    let objective = || Objective {
        make: |x: &[f64]| x.to_vec(),
        kind: Kind::Rosenbrock,
    };
    for pause in [0, 1, 2, 3, 5] {
        let mut stepper =
            Executor::from_start(objective(), solver(), vec![-1.2, 1.0])
                .into_stepper()
                .unwrap();
        for _ in 0..pause {
            stepper.step().unwrap();
        }
        let resumed = Executor::resume_from_checkpoint(
            objective(),
            stepper.into_checkpoint().unwrap(),
        )
        .max_iter(8)
        .run_with_solver()
        .unwrap();
        let direct =
            Executor::from_start(objective(), solver(), vec![-1.2, 1.0])
                .max_iter(8)
                .run_with_solver()
                .unwrap();
        assert_eq!(resumed.state, direct.state);
        assert_eq!(resumed.counts, direct.counts);
        let start = resumed.state.param().clone();
        let fresh = Executor::new(objective(), resumed.solver, resumed.state)
            .max_iter(2)
            .run_with_solver()
            .unwrap();
        let expected = Executor::from_start(objective(), solver(), start)
            .max_iter(2)
            .run_with_solver()
            .unwrap();
        assert_eq!(fresh.state, expected.state);
        assert_eq!(fresh.counts, expected.counts);
    }
}

struct NonFinite {
    initial_cost: f64,
    initial_gradient: Vec<f64>,
    invalid_trial: bool,
}
impl CostFunction for NonFinite {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(if self.invalid_trial && x[0] != 1.0 {
            f64::INFINITY
        } else {
            self.initial_cost
        })
    }
}
impl Gradient for NonFinite {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(self.initial_gradient.clone())
    }
}

#[test]
fn nonfinite_initial_or_trial_data_fail_without_publishing_a_step() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (cost, gradient) in [(invalid, vec![0.0]), (1.0, vec![invalid])] {
            let result = Executor::from_start(
                NonFinite {
                    initial_cost: cost,
                    initial_gradient: gradient,
                    invalid_trial: false,
                },
                NonlinearCg::new(),
                vec![1.0],
            )
            .max_iter(2)
            .run()
            .unwrap();
            assert_eq!(result.reason, TerminationReason::SolverFailed);
            assert_eq!(result.iter(), 0);
            assert_eq!(result.param(), &vec![1.0]);
            assert_eq!(result.state.counts().cost_evals, 1);
        }
    }
    let result = Executor::from_start(
        NonFinite {
            initial_cost: 1.0,
            initial_gradient: vec![2.0],
            invalid_trial: true,
        },
        NonlinearCg::with_line_search(Constant(1.0)),
        vec![1.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.state.current(), Some((&vec![1.0], 1.0, &vec![2.0])));
    assert_eq!(result.state.counts().cost_evals, 2);
}

#[test]
fn invalid_steps_are_soft_failures() {
    for alpha in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300] {
        let result = Executor::from_start(
            quadratic(),
            NonlinearCg::with_line_search(Constant(alpha)),
            vec![2.0, -1.0],
        )
        .max_iter(2)
        .run()
        .unwrap();
        assert_eq!(
            result.reason,
            TerminationReason::SolverFailed,
            "alpha={alpha}"
        );
        assert_eq!(result.iter(), 0);
        assert_eq!(result.param(), &vec![2.0, -1.0]);
    }
}

#[derive(Debug, PartialEq)]
struct Aborted;
struct Fallible;
impl CostFunction for Fallible {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Aborted;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Aborted> {
        Ok(x.dot(x))
    }
}
impl Gradient for Fallible {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Aborted> {
        if x[0] < 1.0 {
            Err(Aborted)
        } else {
            Ok(vec![2.0 * x[0]])
        }
    }
}

#[test]
fn typed_gradient_errors_propagate_from_initialization_and_search() {
    for start in [0.5, 1.0] {
        let result =
            Executor::from_start(Fallible, NonlinearCg::new(), vec![start])
                .max_iter(3)
                .run();
        assert!(matches!(result, Err(Aborted)));
    }
}

#[test]
fn invalid_configuration_panics() {
    for eta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(
                || NonlinearCg::<_, Vec<f64>>::new().with_eta(eta)
            )
            .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(
            || NonlinearCg::<_, Vec<f64>>::new().with_restart_interval(Some(0))
        )
        .is_err()
    );
}

#[test]
fn solutions_agree_with_cg_descent_c_1_2() {
    for line in include_str!("fixtures/nonlinear_cg_reference.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let n = fields[1].parse::<usize>().unwrap();
        let (kind, start) = match fields[0] {
            "quadratic" => (Kind::Quadratic, vec![2.0, -1.0]),
            "rosenbrock" => (Kind::Rosenbrock, vec![-1.2, 1.0]),
            "ill_conditioned" => (Kind::IllConditioned, vec![1.0; n]),
            name => panic!("unexpected fixture: {name}"),
        };
        let reference_cost = fields[3].parse::<f64>().unwrap();
        let reference_gradient = fields[4].parse::<f64>().unwrap();
        let reference_point: Vec<f64> = fields[5..]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(fields[2], "0");
        assert_eq!(reference_point.len(), n);
        let objective = || Objective {
            make: |x: &[f64]| x.to_vec(),
            kind,
        };
        let (cost, gradient) =
            objective().cost_and_gradient(&reference_point).unwrap();
        assert!((cost - reference_cost).abs() < 1e-18);
        assert!((gradient.norm_infinity() - reference_gradient).abs() < 1e-12);
        assert!(reference_gradient < 1e-8);
        let result = Executor::from_start(
            objective(),
            NonlinearCg::new().with_absolute_gradient_tolerance(1e-7),
            start,
        )
        .max_iter(10_000)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::GradientTolerance);
        assert!((result.cost() - reference_cost).abs() < 1e-12);
        for (&actual, &expected) in result.param().iter().zip(&reference_point)
        {
            assert!((actual - expected).abs() < 5e-7);
        }
        assert!(result.state.current().unwrap().2.norm_infinity() <= 1e-7);
    }
}

#[test]
fn iteration_zero_record_and_warm_start_are_evaluated() {
    let start = vec![2.0, -1.0];
    let result = Executor::from_start(
        quadratic(),
        NonlinearCg::default(),
        start.clone(),
    )
    .require_evaluated_state()
    .max_iter(0)
    .run()
    .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(result.state.current().unwrap().0, &start);
    assert_eq!(result.state.counts().cost_evals, 1);
    assert_eq!(result.state.counts().gradient_evals, 1);
    assert_eq!(result.state.best().unwrap().0, &start);
}

#[test]
fn underflowing_gradient_norm_is_not_reported_as_exact_zero() {
    let result = Executor::from_start(
        NonFinite {
            initial_cost: 1.0,
            initial_gradient: vec![1e-200],
            invalid_trial: false,
        },
        NonlinearCg::new(),
        vec![1.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.iter(), 0);
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra() {
    backend::check::<_, f64>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
    backend::check::<_, f32>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray() {
    backend::check::<_, f64>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
    backend::check::<_, f32>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
}

#[cfg(feature = "faer_all")]
#[test]
fn faer() {
    backend::check::<_, f64>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
    backend::check::<_, f32>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
}
