//! Levenberg-Marquardt over the default `Vec<f64>` backend (no feature gate).
//!
//! Mirrors `tests/levenberg_marquardt_nalgebra.rs` to confirm the generic
//! `Solver` impl runs on the hand-rolled [`DenseMatrix`](basin::DenseMatrix)
//! Jacobian: `GramMatrix` forms `JᵀJ`, `AddDiagonalVectorInPlace` applies the
//! Marquardt damping `μ·D`, and the pure-Rust Cholesky `LinearSolveSpd`
//! (`dense_chol`) solves the damped normal equations. The load-bearing case is
//! `PowellSingular`: LM's damping makes `JᵀJ + μI` SPD where bare Gauss-Newton
//! fails the Cholesky (see `gauss_newton_vec.rs`).

use basin::problems::{PowellSingular, RosenbrockResiduals};
use basin::{Executor, LevenbergMarquardt, NllsState, TerminationReason};

#[test]
fn levenberg_marquardt_converges_on_rosenbrock_residuals() {
    // LM converges on Rosenbrock-as-residuals from the classical start.
    // It takes a few more iterations than GN's exact two-step convergence
    // because the damping starts non-zero, but reaches the optimum cleanly
    // and emits SolverConverged.
    let problem = RosenbrockResiduals::<Vec<f64>>::new();
    let initial = vec![-1.2, 1.0];

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
fn levenberg_marquardt_recovers_on_rank_deficient_powell_singular() {
    // Load-bearing "why LM" test, mirror of GN's failure at the same
    // point. At x = (1, 2, 1, 1) Powell's quadratic-residual rows r₂, r₃
    // have vanishing Jacobian rows (J has rank 2 < 4), so JᵀJ is singular
    // and pure GN fails Cholesky. LM's damping makes (JᵀJ + μI) SPD by
    // construction, so it converges cleanly, the canonical demonstration
    // that LM strictly subsumes GN, here on the pure-Rust Cholesky path.
    let problem = PowellSingular::<Vec<f64>>::new();
    let initial = vec![1.0, 2.0, 1.0, 1.0];

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
    // Powell's optimum is x* = 0; check each component drifted toward it.
    for (i, &xi) in result.param().iter().enumerate() {
        assert!(xi.abs() < 1e-2, "x[{}] = {} (expected near 0)", i, xi);
    }
}

#[test]
fn levenberg_marquardt_converges_on_powell_singular_classical_start() {
    // Classical start (3, −1, 0, 1), where the rank deficiency only bites
    // at the optimum. LM with default Nielsen damping converges to the
    // origin in a comparable iteration count to GN.
    let problem = PowellSingular::<Vec<f64>>::new();
    let initial = vec![3.0, -1.0, 0.0, 1.0];

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
    // Convergence path lands SolverConverged (not MaxIter): LM's internal
    // ‖Jᵀr‖_∞ ≤ tol_grad check fires once the iterate is at the optimum.
    let problem = RosenbrockResiduals::<Vec<f64>>::new();
    let initial = vec![-1.2, 1.0];

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(100)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}
