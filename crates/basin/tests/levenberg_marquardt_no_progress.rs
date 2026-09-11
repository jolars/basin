//! Callback routing and composition of the LM numerical safeguard.

use basin::{
    DenseMatrix, Executor, Jacobian, LevenbergMarquardt, LmDamping, NllsState,
    Residual, Solver, TerminationReason,
};
use std::{cell::Cell, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq)]
struct CallbackError;

struct TrialProblem {
    calls: Rc<Cell<usize>>,
    response: Result<f64, CallbackError>,
}

impl Residual for TrialProblem {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = CallbackError;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, CallbackError> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(x, &[1.]);
        if self.calls.get() == 1 {
            Ok(vec![-1.])
        } else {
            self.response.map(|value| vec![value])
        }
    }
}

impl Jacobian for TrialProblem {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, CallbackError> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[1.]))
    }
}

fn check_callback_routing<S>(solver: S, response: Result<f64, CallbackError>)
where
    S: Solver<TrialProblem, NllsState<Vec<f64>>, Error = CallbackError>,
{
    let calls = Rc::new(Cell::new(0));
    let result = Executor::new(
        TrialProblem {
            calls: calls.clone(),
            response,
        },
        solver,
        NllsState::new(vec![1.]),
    )
    .max_iter(1)
    .run();
    assert_eq!(calls.get(), 2);
    match response {
        Err(error) => assert_eq!(result.err(), Some(error)),
        Ok(value) => {
            let result = result.unwrap();
            assert_eq!(result.reason, TerminationReason::MaxIter);
            assert_eq!(result.param(), &[1.]);
            assert_eq!(result.cost(), if value == 0. { 0. } else { 0.5 });
            assert_eq!(result.cost_evals(), 2);
        }
    }
}

#[test]
fn unchanged_trial_preserves_errors_nonfinite_rejections_and_acceptance() {
    for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
        for response in [
            Err(CallbackError),
            Ok(f64::INFINITY),
            Ok(f64::NEG_INFINITY),
            Ok(f64::NAN),
            // A stateful callback can improve cost even at identical coordinates.
            Ok(0.),
        ] {
            let solver = || {
                LevenbergMarquardt::new()
                    .with_damping(damping)
                    .with_tau(1e20)
                    .with_initial_step_bound(1e-20)
                    .with_absolute_gradient_tolerance(None)
            };
            check_callback_routing(solver(), response);
            check_callback_routing(solver().with_pivoted_qr(), response);
        }
    }
}

#[test]
fn trust_radius_preserves_callback_errors_and_nonfinite_rejections() {
    for response in [
        Err(CallbackError),
        Ok(f64::INFINITY),
        Ok(f64::NEG_INFINITY),
        Ok(f64::NAN),
    ] {
        let solver = || {
            LevenbergMarquardt::new()
                .with_damping(LmDamping::TrustRegion)
                .with_initial_step_bound(1e-20)
                .with_absolute_gradient_tolerance(None)
                .with_relative_trust_radius_tolerance(1.)
        };
        check_callback_routing(solver(), response);
        check_callback_routing(solver().with_pivoted_qr(), response);
    }
}

fn check_radius_configuration<S>(solver: S)
where
    S: Solver<TrialProblem, NllsState<Vec<f64>>, Error = CallbackError>,
{
    let result = Executor::new(
        TrialProblem {
            calls: Rc::new(Cell::new(0)),
            response: Ok(-1.),
        },
        solver,
        NllsState::new(vec![1.]),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.cost_evals(), 2);
    assert_eq!(result.param(), &[1.]);
    assert_eq!(result.cost(), 0.5);
}

#[test]
fn radius_tolerance_survives_qr_conversion_and_convergence_builder_order() {
    let solver = || {
        LevenbergMarquardt::new()
            .with_damping(LmDamping::TrustRegion)
            .with_initial_step_bound(1e-20)
            .with_absolute_gradient_tolerance(None)
    };
    check_radius_configuration(
        solver()
            .with_relative_trust_radius_tolerance(1.)
            .with_pivoted_qr(),
    );
    check_radius_configuration(
        solver()
            .with_pivoted_qr()
            .with_relative_trust_radius_tolerance(1.),
    );
    check_radius_configuration(
        solver()
            .with_absolute_step_tolerance(None)
            .with_relative_trust_radius_tolerance(1.),
    );
    check_radius_configuration(
        solver()
            .with_relative_trust_radius_tolerance(1.)
            .with_absolute_step_tolerance(None),
    );
    check_radius_configuration(
        solver()
            .with_pivoted_qr()
            .with_absolute_step_tolerance(None)
            .with_relative_trust_radius_tolerance(1.),
    );
}

#[test]
fn trust_radius_tolerance_requires_finite_nonnegative_values() {
    for invalid in [-1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(|| {
                LevenbergMarquardt::<Vec<f64>, DenseMatrix>::new()
                    .with_relative_trust_radius_tolerance(invalid)
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                LevenbergMarquardt::<Vec<f64>, DenseMatrix>::new()
                    .with_pivoted_qr()
                    .with_relative_trust_radius_tolerance(invalid)
            })
            .is_err()
        );
    }
}

#[test]
fn trust_radius_does_not_bypass_model_solve_failure() {
    let solver = || {
        LevenbergMarquardt::new()
            .with_damping(LmDamping::TrustRegion)
            .with_initial_step_bound(1e-20)
            .with_max_inner_attempts(1)
            .with_absolute_gradient_tolerance(None)
            .with_relative_trust_radius_tolerance(1.)
    };
    fn check<S>(solver: S)
    where
        S: Solver<TrialProblem, NllsState<Vec<f64>>, Error = CallbackError>,
    {
        let calls = Rc::new(Cell::new(0));
        let result = Executor::new(
            TrialProblem {
                calls: calls.clone(),
                response: Err(CallbackError),
            },
            solver,
            NllsState::new(vec![1.]),
        )
        .max_iter(1)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverFailed);
        assert_eq!(calls.get(), 1);
    }
    check(solver());
    check(solver().with_pivoted_qr());
}

fn check_disabled<S>(solver: S)
where
    S: Solver<TrialProblem, NllsState<Vec<f64>>, Error = CallbackError>,
{
    let result = Executor::new(
        TrialProblem {
            calls: Rc::new(Cell::new(0)),
            response: Ok(-1.),
        },
        solver,
        NllsState::new(vec![1.]),
    )
    .max_iter(4)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.cost_evals(), 5);
}

#[test]
fn opt_out_survives_qr_conversion_and_convergence_builder_order() {
    let solver = || {
        LevenbergMarquardt::new()
            .with_tau(1e20)
            .with_absolute_gradient_tolerance(None)
    };
    check_disabled(solver().with_no_progress_check(false).with_pivoted_qr());
    check_disabled(solver().with_pivoted_qr().with_no_progress_check(false));
    check_disabled(
        solver()
            .with_absolute_step_tolerance(None)
            .with_no_progress_check(false),
    );
    check_disabled(
        solver()
            .with_no_progress_check(false)
            .with_absolute_step_tolerance(None),
    );
    check_disabled(
        solver()
            .with_absolute_step_tolerance(None)
            .with_no_progress_check(false)
            .with_pivoted_qr(),
    );
    check_disabled(
        solver()
            .with_pivoted_qr()
            .with_absolute_step_tolerance(None)
            .with_no_progress_check(false),
    );
    check_disabled(
        solver()
            .with_no_progress_check(true)
            .with_no_progress_check(false),
    );
}

#[test]
fn execution_budget_precedes_a_trial_and_its_numerical_stop() {
    let calls = Rc::new(Cell::new(0));
    let result = Executor::from_start(
        TrialProblem {
            calls: calls.clone(),
            response: Err(CallbackError),
        },
        LevenbergMarquardt::new().with_absolute_gradient_tolerance(None),
        vec![1.],
    )
    .max_cost_evals(1)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert_eq!(calls.get(), 1);
}

#[test]
fn numerical_stop_precedes_the_next_observed_change_check() {
    for enabled in [true, false] {
        let result = Executor::from_start(
            TrialProblem {
                calls: Rc::new(Cell::new(0)),
                response: Ok(-1.),
            },
            LevenbergMarquardt::new()
                .with_tau(1e20)
                .with_absolute_gradient_tolerance(None)
                .with_absolute_step_tolerance(0.)
                .with_no_progress_check(enabled),
            vec![1.],
        )
        .max_iter(4)
        .run()
        .unwrap();
        assert_eq!(
            result.reason,
            if enabled {
                TerminationReason::NumericalNoProgress
            } else {
                TerminationReason::ParamTolerance
            }
        );
        assert_eq!(result.iter(), if enabled { 0 } else { 1 });
        assert_eq!(result.cost_evals(), 2);
    }
}
