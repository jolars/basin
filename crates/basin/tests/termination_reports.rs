use basin::{
    ConvergenceTest, DenseMatrix, EvaluationKind, Executor, Jacobian,
    LevenbergMarquardtQr, PointState, Residual, Termination, TerminationStage,
};
use std::convert::Infallible;

struct Fit;
impl Residual for Fit {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0] - 1.])
    }
}
impl Jacobian for Fit {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[1.]))
    }
}

#[test]
fn ordinary_result_owns_native_criteria_measurements_and_counts() {
    let result = Executor::new(
        Fit,
        LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new()
            .with_absolute_gradient_tolerance(0.)
            .with_gradient_orthogonality_tolerance(0.),
        PointState::new(vec![1.]),
    )
    .run()
    .unwrap();
    let Termination::Converged(convergence) = &result.report.termination else {
        panic!("expected convergence");
    };
    assert_eq!(convergence.criteria().len(), 2);
    assert_eq!(
        convergence.criteria()[0].test,
        ConvergenceTest::AbsoluteGradientInfinity
    );
    assert_eq!(
        convergence.criteria()[1].test,
        ConvergenceTest::GradientOrthogonality
    );
    assert_eq!(result.counts.residual_evals, 1);
    assert_eq!(result.counts.jacobian_evals, 1);
    assert_eq!(
        result.report.stage,
        TerminationStage::Step { completed: false }
    );
}

#[test]
fn exact_continuation_creates_a_new_report_before_touching_the_solver() {
    let first = Executor::new(
        Fit,
        LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new(),
        PointState::new(vec![1.]),
    )
    .run_with_solver()
    .unwrap();
    assert!(first.report.termination.is_converged());
    let counts = first.counts;
    let resumed =
        Executor::resume_from_checkpoint(Fit, first.into_checkpoint())
            .max_evaluations(EvaluationKind::Residual, 1)
            .run()
            .unwrap();
    assert!(matches!(resumed.report.termination, Termination::Limit(_)));
    assert_eq!(resumed.report.stage, TerminationStage::Boundary);
    assert_eq!(resumed.counts, counts);
}

use basin::{
    ApplicationStop, ConvergenceEvidence, CostFunction, EvalCounts,
    ExecutionLimit, Gradient, GradientDescent, NumericalFailure,
    ObservationEvent, ObserverMode, PartialResultPolicy, Problem, Solver,
    SolverStep, State, StepOutcome, TerminationReport,
};
use std::{cell::RefCell, rc::Rc};

impl CostFunction for Fit {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(0.5 * (x[0] - 1.).powi(2))
    }
}
impl Gradient for Fit {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0] - 1.])
    }
}

#[test]
fn shared_checks_retain_simultaneous_predicates_and_reference_values() {
    let result = Executor::from_start(
        Fit,
        GradientDescent::new(0.1)
            .with_absolute_gradient_tolerance(2.)
            .with_relative_gradient_tolerance(1.),
        vec![3.],
    )
    .run()
    .unwrap();
    let Termination::Converged(convergence) = result.report.termination else {
        panic!("expected gradient convergence");
    };
    assert_eq!(convergence.criteria().len(), 2);
    assert_eq!(
        convergence.criteria()[0].test,
        ConvergenceTest::AbsoluteGradientSquared
    );
    assert_eq!(
        convergence.criteria()[1].evidence,
        ConvergenceEvidence::UpperBound {
            value: 4.,
            bound: 4.,
            tolerance: 1.,
            reference: Some(4.),
        }
    );
    assert_eq!(
        result.counts,
        EvalCounts {
            cost_evals: 1,
            gradient_evals: 1,
            ..EvalCounts::default()
        }
    );
}

struct LastStep {
    completed: bool,
    abort: bool,
}
impl Solver<Fit, PointState<Vec<f64>>> for LastStep {
    type Error = &'static str;
    fn init(
        &mut self,
        p: &mut Problem<Fit>,
        mut s: PointState<Vec<f64>>,
    ) -> Result<PointState<Vec<f64>>, Self::Error> {
        s.replace(s.param().clone(), p.cost(s.param()).unwrap());
        Ok(s)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<Fit>,
        mut s: PointState<Vec<f64>>,
    ) -> Result<SolverStep<PointState<Vec<f64>>>, Self::Error> {
        let x = vec![1.];
        let cost = p.cost(&x).unwrap();
        if self.abort {
            return Err("callback aborted");
        }
        s.replace(x, cost);
        Ok(SolverStep {
            state: s,
            completed: self.completed,
            termination: Some(Termination::custom(
                "example.exact",
                "The exact minimizer was evaluated.",
                vec![],
            )),
        })
    }
}

#[test]
fn completion_controls_iteration_observation_and_final_report_is_sticky() {
    for completed in [false, true] {
        let events = Rc::new(RefCell::new(Vec::new()));
        let output = events.clone();
        let mut stepper = Executor::new(
            Fit,
            LastStep {
                completed,
                abort: false,
            },
            PointState::new(vec![3.]),
        )
        .observe_solver(
            move |_, _, event| {
                output.borrow_mut().push(match event {
                    ObservationEvent::Init => "init",
                    ObservationEvent::Iter => "iter",
                    ObservationEvent::Final(report) => {
                        assert_eq!(
                            report.stage,
                            TerminationStage::Step { completed }
                        );
                        "final"
                    }
                    _ => unreachable!(),
                });
            },
            ObserverMode::Always,
        )
        .into_stepper()
        .unwrap();
        let first = stepper.step().unwrap();
        assert_eq!(stepper.step().unwrap(), first);
        let StepOutcome::Stopped(report) = first else {
            panic!("expected stop")
        };
        assert_eq!(report.iteration, u64::from(completed));
        assert_eq!(stepper.state().best_param(), &vec![1.]);
        assert_eq!(stepper.counts().cost_evals, 2);
        assert_eq!(
            &*events.borrow(),
            if completed {
                &["init", "iter", "final"][..]
            } else {
                &["init", "final"][..]
            }
        );
        let saved = stepper.into_checkpoint().unwrap();
        let result = Executor::resume_from_checkpoint(Fit, saved)
            .max_iter(0)
            .run()
            .unwrap();
        assert_eq!(result.report.stage, TerminationStage::Boundary);
    }
}

#[test]
fn all_exhausted_budgets_are_reported_before_numerical_checks() {
    let result = Executor::from_start(
        Fit,
        GradientDescent::new(0.1).with_absolute_gradient_tolerance(2.),
        vec![1.],
    )
    .max_iter(0)
    .max_evaluations(EvaluationKind::Cost, 1)
    .max_evaluations(EvaluationKind::Gradient, 0)
    .run()
    .unwrap();
    assert_eq!(
        result.report.termination,
        Termination::Limit(vec![
            ExecutionLimit::Iterations {
                observed: 0,
                limit: 0
            },
            ExecutionLimit::Evaluations {
                kind: EvaluationKind::Cost,
                observed: 1,
                limit: 1
            },
            ExecutionLimit::Evaluations {
                kind: EvaluationKind::Gradient,
                observed: 1,
                limit: 0
            },
        ])
    );
    assert!(
        result
            .report
            .termination
            .native_convergence_tests()
            .is_empty()
    );
}

#[test]
fn hard_abort_preserves_work_counts_without_a_report_or_checkpoint() {
    let mut stepper = Executor::new(
        Fit,
        LastStep {
            completed: true,
            abort: true,
        },
        PointState::new(vec![3.]),
    )
    .into_stepper()
    .unwrap();
    assert_eq!(stepper.step(), Err("callback aborted"));
    assert_eq!(stepper.counts().cost_evals, 2);
    assert!(stepper.finished().is_none());
    assert!(stepper.into_checkpoint().is_none());
}

#[test]
fn inner_routing_requires_explicit_partial_policy_and_retains_failures() {
    let report = TerminationReport::<f32> {
        termination: Termination::Limit(vec![ExecutionLimit::Iterations {
            observed: 7,
            limit: 7,
        }]),
        stage: TerminationStage::Boundary,
        iteration: 7,
    };
    assert!(
        report
            .clone()
            .into_outer_termination(PartialResultPolicy::Consume)
            .is_none()
    );
    let outer = report
        .clone()
        .into_outer_termination(PartialResultPolicy::RequireConvergence)
        .unwrap();
    assert_eq!(
        outer,
        Termination::Failed(NumericalFailure::Inner {
            report: Box::new(report.clone())
        })
    );
    for termination in [
        Termination::Cancelled,
        Termination::Application(ApplicationStop::new("service_shutdown")),
    ] {
        let report = TerminationReport {
            termination: termination.clone(),
            ..report.clone()
        };
        assert_eq!(
            report.into_outer_termination(PartialResultPolicy::Consume),
            Some(termination)
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn f32_report_serialization_preserves_evidence() {
    let report = TerminationReport::<f32> {
        termination: Termination::upper_bound(
            ConvergenceTest::RelativeTrialStep,
            0.125,
            0.25,
            0.5,
            Some(0.5),
        ),
        stage: TerminationStage::Step { completed: true },
        iteration: 1,
    };
    let bytes = postcard::to_allocvec(&report).unwrap();
    let restored: TerminationReport<f32> =
        postcard::from_bytes(&bytes).unwrap();
    assert_eq!(restored, report);
}

#[cfg(feature = "serde")]
#[test]
fn serialized_convergence_requires_a_satisfied_predicate() {
    let empty: Vec<basin::ConvergenceCriterion<f32>> = vec![];
    let bytes = postcard::to_allocvec(&empty).unwrap();
    assert!(postcard::from_bytes::<basin::Convergence<f32>>(&bytes).is_err());
}

struct PartialThenComplete(bool);
impl Solver<Fit, PointState<Vec<f64>>> for PartialThenComplete {
    type Error = Infallible;
    fn init(
        &mut self,
        p: &mut Problem<Fit>,
        mut s: PointState<Vec<f64>>,
    ) -> Result<PointState<Vec<f64>>, Self::Error> {
        s.replace(s.param().clone(), p.cost(s.param())?);
        Ok(s)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<Fit>,
        mut s: PointState<Vec<f64>>,
    ) -> Result<SolverStep<PointState<Vec<f64>>>, Self::Error> {
        let x = vec![if self.0 { 1. } else { 2. }];
        s.replace(x.clone(), p.cost(&x)?);
        let completed = self.0;
        self.0 = true;
        Ok(SolverStep {
            state: s,
            completed,
            termination: None,
        })
    }
}

#[test]
fn continuing_partial_steps_publish_work_without_iteration_observation() {
    let iterations = Rc::new(RefCell::new(Vec::new()));
    let output = iterations.clone();
    let mut stepper = Executor::new(
        Fit,
        PartialThenComplete(false),
        PointState::new(vec![3.]),
    )
    .observe_solver(
        move |s, _, event| {
            if matches!(event, ObservationEvent::Iter) {
                output.borrow_mut().push(s.iter());
            }
        },
        ObserverMode::Always,
    )
    .max_iter(1)
    .into_stepper()
    .unwrap();
    assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    assert_eq!(stepper.state().param(), &vec![2.]);
    assert_eq!(stepper.state().iter(), 0);
    assert_eq!(stepper.counts().cost_evals, 2);
    assert!(iterations.borrow().is_empty());
    assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    assert_eq!(stepper.state().iter(), 1);
    assert_eq!(*iterations.borrow(), vec![1]);
    assert!(matches!(stepper.step().unwrap(), StepOutcome::Stopped(_)));
}
