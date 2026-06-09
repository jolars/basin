#![cfg(feature = "faer")]

use basin::problems::{ExponentialFit, PowellSingular, RosenbrockResiduals};
use basin::{Executor, LevenbergMarquardt, NllsState, TerminationReason};
use faer::Col;

#[test]
fn levenberg_marquardt_converges_on_rosenbrock_residuals() {
    let problem = RosenbrockResiduals::<Col<f64>>::new();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-15, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-7,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-7,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn levenberg_marquardt_converges_fast_on_poorly_scaled_exponential_fit() {
    // Regression guard for Marquardt diagonal damping (issue #6), faer
    // mirror of the nalgebra test. The exponential model ŷ = a·exp(b·t)
    // has Jacobian columns ~1e5× apart in scale; Marquardt scaling
    // reaches (1e5, −1) in a handful of iterations where isotropic μI
    // damping needs ~27.
    let problem = ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 });

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] + 1.0).abs() < 1e-6,
        "b = {} (expected −1)",
        result.param()[1]
    );
    assert!(
        result.iter() <= 15,
        "Marquardt scaling should reach the optimum in ≤15 iters; took {} \
         (isotropic μI damping needs ~27)",
        result.iter()
    );
}

#[test]
fn levenberg_marquardt_converges_via_relative_gradient_tolerance() {
    // MINPACK `gtol` cosine test (issue #6), faer mirror: disable the
    // absolute ‖Jᵀr‖∞ check so only the relative gradient test can stop
    // the run, and confirm it reaches the optimum.
    let problem = ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 });

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new()
            .with_tol_grad(0.0)
            .with_tol_grad_rel(1e-10),
        NllsState::new(initial),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
}

#[test]
fn levenberg_marquardt_converges_via_ftol() {
    // MINPACK `ftol` test (issue #8), faer mirror: disable both gradient
    // tests so SolverConverged implies the relative-cost `ftol` fired,
    // and confirm it reaches the optimum of the poorly-scaled fit.
    let problem = ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 });

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new()
            .with_tol_grad(0.0)
            .with_tol_grad_rel(0.0)
            .with_tol_cost_rel(1e-10),
        NllsState::new(initial),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
}

#[test]
fn levenberg_marquardt_converges_via_xtol() {
    // MINPACK `xtol` test (issue #8), faer mirror: ‖h‖ ≤ xtol·‖x‖.
    // Disable the other tests so SolverConverged implies `xtol` fired.
    let problem = ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 });

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new()
            .with_tol_grad(0.0)
            .with_tol_grad_rel(0.0)
            .with_tol_step_rel(1e-10),
        NllsState::new(initial),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
}

#[test]
fn levenberg_marquardt_recovers_on_rank_deficient_powell_singular() {
    // Mirror of the nalgebra "why LM" test. At (1, 2, 1, 1) GN's
    // Cholesky fails on the singular JᵀJ; LM's damping recovers and
    // drives Powell toward x* = 0.
    let problem = PowellSingular::<Col<f64>>::new();
    let initial = Col::from_fn(4, |i| match i {
        0 => 1.0,
        1 => 2.0,
        2 => 1.0,
        _ => 1.0,
    });

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.cost() < 1e-10,
        "cost = {} (LM should drive Powell to the origin)",
        result.cost()
    );
    for i in 0..4 {
        let xi = result.param()[i];
        assert!(xi.abs() < 1e-2, "x[{}] = {}", i, xi);
    }
}

#[test]
fn levenberg_marquardt_converges_on_powell_singular_classical_start() {
    let problem = PowellSingular::<Col<f64>>::new();
    let initial = Col::from_fn(4, |i| match i {
        0 => 3.0,
        1 => -1.0,
        2 => 0.0,
        _ => 1.0,
    });

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(100)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.cost() < 1e-10,
        "cost = {} (Powell from classical start should reach near-zero)",
        result.cost()
    );
}

#[test]
fn levenberg_marquardt_emits_solver_converged_via_first_order_optimality() {
    let problem = RosenbrockResiduals::<Col<f64>>::new();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(100)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}
