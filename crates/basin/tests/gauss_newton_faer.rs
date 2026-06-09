#![cfg(feature = "faer")]

use basin::problems::{PowellSingular, RosenbrockResiduals};
use basin::{Executor, GaussNewton, NllsState, TerminationReason};
use faer::Col;

#[test]
fn gauss_newton_converges_on_rosenbrock_residuals() {
    let problem = RosenbrockResiduals::<Col<f64>>::new();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result = Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
        .max_iter(20)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-20, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-9,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-9,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn gauss_newton_single_step_matches_normal_equation_solution() {
    // Mirrors the nalgebra single-step verification — see that file for the
    // hand-computed derivation of x_new = (1.0, −3.84).
    let problem = RosenbrockResiduals::<Col<f64>>::new();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result = Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
        .max_iter(1)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 1);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-9,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - (-3.84)).abs() < 1e-9,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn gauss_newton_emits_solver_converged_via_first_order_optimality() {
    let problem = RosenbrockResiduals::<Col<f64>>::new();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result = Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[test]
fn gauss_newton_fails_on_rank_deficient_powell_singular_jacobian() {
    // See nalgebra suite for the rationale — at x = (1, 2, 1, 1) Powell's
    // J has rank 2 < 4 so J^T J is singular and Cholesky fails. Pure GN
    // returns SolverFailed; LM's damping in S4 will recover.
    let problem = PowellSingular::<Col<f64>>::new();
    let initial = Col::from_fn(4, |i| match i {
        0 => 1.0,
        1 => 2.0,
        2 => 1.0,
        _ => 1.0,
    });

    let result = Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
        .max_iter(100)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverFailed);
}
