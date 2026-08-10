#![cfg(feature = "faer")]

use basin::problems::BoothBoxed;
use basin::{
    Backtracking, BasicState, Executor, ProjectedGradientDescent,
    ProjectedGradientTolerance, TerminationReason,
};
use faer::Col;

fn col(values: [f64; 2]) -> Col<f64> {
    Col::<f64>::from_fn(2, |i| values[i])
}

#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let problem =
        BoothBoxed::<Col<f64>>::new(col([-5.0, -5.0]), col([5.0, 5.0]));
    let initial = col([0.0, 0.0]);

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(2000)
    .run()
    .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 3.0).abs() < 1e-4,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn tight_bounds_converge_to_box_corner() {
    let problem =
        BoothBoxed::<Col<f64>>::new(col([-1.0, -1.0]), col([1.0, 1.0]));
    let initial = col([0.0, 0.0]);

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(2000)
    .run()
    .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-6,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-6,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn infeasible_initial_param_is_projected_at_init() {
    let problem =
        BoothBoxed::<Col<f64>>::new(col([-1.0, -1.0]), col([1.0, 1.0]));
    let initial = col([10.0, 10.0]);

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::new(0.01),
        BasicState::new(initial),
    )
    .max_iter(0)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.param()[0], 1.0);
    assert_eq!(result.param()[1], 1.0);
}

#[test]
fn projected_gradient_tolerance_triggers_at_corner_minimum() {
    let lower = col([-1.0, -1.0]);
    let upper = col([1.0, 1.0]);
    let problem = BoothBoxed::<Col<f64>>::new(lower.clone(), upper.clone());
    let initial = col([0.0, 0.0]);

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(2000)
    .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-7))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::ProjectedGradientTolerance);
}
