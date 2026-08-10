#![cfg(feature = "ndarray")]

//! Gauss-Newton over the `ndarray` backend (`Array1<f64>`/`Array2<f64>`).
//!
//! Mirrors `tests/gauss_newton_nalgebra.rs`: `Array2`'s `GramMatrix` +
//! `LinearSolveSpd` route through the same pure-Rust Cholesky (`dense_chol`)
//! as the `DenseMatrix` backend, so the normal-equation step, eval counts,
//! and the rank-deficient failure are backend-independent; the assertions
//! match the nalgebra mirror exactly.

use basin::problems::{PowellSingular, RosenbrockResiduals};
use basin::{Executor, GaussNewton, NllsState, TerminationReason};
use ndarray::Array1;

#[test]
fn gauss_newton_converges_on_rosenbrock_residuals() {
    // GN converges on Rosenbrock-as-residuals from the classical start in
    // 2 iterations exactly (the residual is linear in y at fixed x, so the
    // linear model is exact along that axis).
    let problem = RosenbrockResiduals::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![-1.2, 1.0]);

    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
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
    // Smallest credible end-to-end test: one iteration must reproduce the
    // hand-computed normal-equation step. Guards the inner-step code
    // against sign or transpose mistakes that the convergence test would
    // mask. δ from the S2a verification at (-1.2, 1.0) is
    // (J^T J)^{-1}·(J^T r) = [-2.2, 4.84]; the GN update is x ← x − δ,
    // so x_new = (-1.2 + 2.2, 1.0 − 4.84) = (1.0, −3.84).
    let problem = RosenbrockResiduals::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![-1.2, 1.0]);

    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
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
    // Convergence path lands SolverConverged (not MaxIter): GN's internal
    // ‖J^T r‖_∞ ≤ tol_grad check fires once the iterate is at the
    // optimum. The previous test happens to land here too; this one
    // tightens the assertion to just the termination reason so the contract
    // is documented in isolation.
    let problem = RosenbrockResiduals::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![-1.2, 1.0]);

    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
            .max_iter(50)
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[test]
fn gauss_newton_fails_on_rank_deficient_powell_singular_jacobian() {
    // Load-bearing "why LM" test for S4. At x = (1, 2, 1, 1) two of
    // Powell's residuals (r₂, r₃) have vanishing Jacobian rows because
    // both `x₁ − 2x₂` and `x₀ − x₃` are zero, so J has rank 2 < 4 and
    // J^T J is exactly singular. Pure GN's Cholesky fails and the
    // solver returns SolverFailed. This is the case Levenberg-Marquardt's
    // damping is designed to recover.
    let problem = PowellSingular::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![1.0, 2.0, 1.0, 1.0]);

    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
            .max_iter(100)
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverFailed);
}

#[test]
fn gauss_newton_caches_residual_and_jacobian_across_iterations() {
    // Regression test for the GN caching contract. Every iter accepts
    // a full Newton step, so:
    //   - the post-step residual `r(x_new)` is stashed and reused as
    //     the start-of-iter `r` next time (no re-eval at the same x);
    //   - `init` seeds `J(x₀)` which carries iter 1; afterwards the
    //     Jacobian is recomputed on each iter (x moves every step).
    // For K completed iters with the run terminating on MaxIter
    // (avoiding the in-`next_iter` convergence check that also evaluates
    // J on the early-exit path):
    //   - cost_evals = 1 + K
    //   - jacobian_evals = K
    // Disable the internal tol_grad check so termination is purely by
    // MaxIter; keeps the assertion deterministic regardless of how
    // close Rosenbrock has driven `‖Jᵀr‖_∞` to zero.
    let problem = RosenbrockResiduals::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(
        problem,
        GaussNewton::new().with_tol_grad(0.0),
        NllsState::new(initial),
    )
    .max_iter(3)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 3);
    assert_eq!(
        result.cost_evals(),
        4,
        "expected init (1) + one post-step residual per iter (3) = 4"
    );
    assert_eq!(
        result.state.jacobian_evals(),
        3,
        "expected init's J reused for iter 1, then one J recompute per subsequent iter \
         (3 total)"
    );
}
