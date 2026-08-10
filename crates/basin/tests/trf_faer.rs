#![cfg(feature = "faer")]

use basin::problems::BoothBoxedResiduals;
use basin::{Executor, MaxIter, NllsState, TerminationReason, Trf};
use faer::Col;

#[test]
fn trf_with_slack_bounds_reaches_unconstrained_min() {
    let problem = BoothBoxedResiduals::<Col<f64>>::new(
        Col::<f64>::from_fn(2, |i| [-5.0, -5.0][i]),
        Col::<f64>::from_fn(2, |i| [5.0, 5.0][i]),
    );
    let initial = Col::<f64>::from_fn(2, |i| [0.0, 0.0][i]);

    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-5,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 3.0).abs() < 1e-5,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn trf_with_tight_bounds_converges_to_box_corner() {
    let problem = BoothBoxedResiduals::<Col<f64>>::new(
        Col::<f64>::from_fn(2, |i| [-1.0, -1.0][i]),
        Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]),
    );
    let initial = Col::<f64>::from_fn(2, |i| [0.0, 0.0][i]);

    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-3,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-3,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn trf_init_projects_infeasible_start_strictly_inside_box() {
    let problem = BoothBoxedResiduals::<Col<f64>>::new(
        Col::<f64>::from_fn(2, |i| [-1.0, -1.0][i]),
        Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]),
    );
    let initial = Col::<f64>::from_fn(2, |i| [10.0, 10.0][i]);

    let mut executor =
        Executor::new(problem, Trf::new(), NllsState::new(initial));
    executor = executor.terminate_on(MaxIter(0));
    let result = executor.run().unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    let x = result.param();
    assert!(
        x[0] < 1.0,
        "x[0] = {} should be < 1.0 (strictly inside)",
        x[0]
    );
    assert!(
        x[1] < 1.0,
        "x[1] = {} should be < 1.0 (strictly inside)",
        x[1]
    );
    assert!(x[0] > -1.0, "x[0] = {} should be > -1.0", x[0]);
    assert!(x[1] > -1.0, "x[1] = {} should be > -1.0", x[1]);
}

#[test]
fn trf_emits_solver_converged_via_scaled_first_order_optimality() {
    let problem = BoothBoxedResiduals::<Col<f64>>::new(
        Col::<f64>::from_fn(2, |i| [-1.0, -1.0][i]),
        Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]),
    );
    let initial = Col::<f64>::from_fn(2, |i| [0.0, 0.0][i]);

    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}
