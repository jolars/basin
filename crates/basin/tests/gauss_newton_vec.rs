//! Gauss-Newton over the default `Vec<f64>` backend (no feature gate).
//!
//! Mirrors `tests/gauss_newton_nalgebra.rs` to confirm the generic `Solver`
//! impl runs on the hand-rolled [`DenseMatrix`](basin::DenseMatrix) Jacobian,
//! whose `GramMatrix` + `LinearSolveSpd` are the pure-Rust Cholesky path
//! (`dense_chol`). The numbers (normal-equation step, eval counts, the
//! rank-deficient failure) are backend-independent, so the assertions match
//! the nalgebra mirror exactly.

#![cfg(feature = "problems")]

use basin::problems::{PowellSingular, RosenbrockResiduals};
use basin::{
    DenseMatrix, Executor, GaussNewton, GramMatrix, Jacobian, LinearSolveError,
    LinearSolveSpd, MatTransposeVec, PointState, Residual, TerminationCode,
};
use std::convert::Infallible;

#[test]
fn gauss_newton_converges_on_rosenbrock_residuals() {
    // GN converges on Rosenbrock-as-residuals from the classical start in
    // 2 iterations exactly (the residual is linear in y at fixed x, so the
    // linear model is exact along that axis).
    let problem = RosenbrockResiduals::<Vec<f64>>::new();
    let initial = vec![-1.2, 1.0];

    let result =
        Executor::new(problem, GaussNewton::new(), PointState::new(initial))
            .max_iter(20)
            .run()
            .unwrap();

    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert!(
        result.current_record().unwrap().cost.unwrap() < 1e-20,
        "cost = {}",
        result.current_record().unwrap().cost.unwrap()
    );
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
    // One iteration must reproduce the hand-computed normal-equation step.
    // δ = (JᵀJ)⁻¹·(Jᵀr) at (−1.2, 1.0) is [−2.2, 4.84]; the GN update is
    // x ← x − δ, so x_new = (1.0, −3.84). Guards the Cholesky solve path
    // against sign or transpose mistakes the convergence test would mask.
    let problem = RosenbrockResiduals::<Vec<f64>>::new();
    let initial = vec![-1.2, 1.0];

    let result =
        Executor::new(problem, GaussNewton::new(), PointState::new(initial))
            .max_iter(1)
            .run()
            .unwrap();

    assert_eq!(result.report.code(), TerminationCode::MaxIter);
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
fn gauss_newton_fails_on_rank_deficient_powell_singular_jacobian() {
    // Load-bearing "why LM" test. At x = (1, 2, 1, 1) two of Powell's
    // residuals (r₂, r₃) have vanishing Jacobian rows, so J has rank 2 < 4
    // and JᵀJ is exactly singular. The pure-Rust Cholesky hits a
    // non-positive pivot → `NotPositiveDefinite`, which GN surfaces as
    // SolverFailed, the case Levenberg-Marquardt's damping is built to
    // recover (see `levenberg_marquardt_vec.rs`).
    let problem = PowellSingular::<Vec<f64>>::new();
    let initial = vec![1.0, 2.0, 1.0, 1.0];

    let result =
        Executor::new(problem, GaussNewton::new(), PointState::new(initial))
            .max_iter(100)
            .run()
            .unwrap();

    assert_eq!(result.report.code(), TerminationCode::SolverFailed);
}

#[test]
fn gauss_newton_caches_residual_and_jacobian_across_iterations() {
    // Regression test for the GN caching contract (mirror of the nalgebra
    // case). For K completed iters terminating on MaxIter:
    //   - residual_evals = 1 (init) + K
    //   - jacobian_evals = K (init's J reused for iter 1, then one recompute
    //     per subsequent iter). Disable the internal tol_grad check so
    //     termination is purely by MaxIter.
    let problem = RosenbrockResiduals::<Vec<f64>>::new();
    let initial = vec![-1.2, 1.0];

    let result = Executor::new(
        problem,
        GaussNewton::new().with_absolute_gradient_tolerance(None),
        PointState::new(initial),
    )
    .max_iter(3)
    .run()
    .unwrap();

    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    assert_eq!(result.iter(), 3);
    assert_eq!(
        result.state.counts().residual_evals,
        4,
        "expected init (1) + one post-step residual per iter (3) = 4"
    );
    assert_eq!(
        result.state.counts().jacobian_evals,
        3,
        "expected init's J reused for iter 1, then one J recompute per subsequent iter \
         (3 total)"
    );
}

struct Fit;
struct CustomJacobian(f64);

impl Residual for Fit {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![2.0 * x[0] - 4.0])
    }
}

impl Jacobian for Fit {
    type Jacobian = CustomJacobian;
    fn jacobian(&self, _: &Vec<f64>) -> Result<CustomJacobian, Infallible> {
        Ok(CustomJacobian(2.0))
    }
}

impl MatTransposeVec<Vec<f64>> for CustomJacobian {
    fn mat_transpose_vec(&self, rhs: &Vec<f64>) -> Vec<f64> {
        vec![self.0 * rhs[0]]
    }
}

impl GramMatrix for CustomJacobian {
    fn gram(&self) -> Self {
        Self(self.0 * self.0)
    }
}

impl GramMatrix<DenseMatrix> for CustomJacobian {
    fn gram(&self) -> DenseMatrix {
        DenseMatrix::from_row_slice(1, 1, &[self.0 * self.0])
    }
}

impl LinearSolveSpd<Vec<f64>> for CustomJacobian {
    fn solve_spd(&self, rhs: &Vec<f64>) -> Result<Vec<f64>, LinearSolveError> {
        Ok(vec![rhs[0] / self.0])
    }
}

#[test]
fn gauss_newton_infers_the_legacy_gram_type() {
    let result = Executor::from_start(Fit, GaussNewton::new(), vec![0.0])
        .run()
        .unwrap();
    assert_eq!(result.param(), &[2.0]);

    let result = Executor::from_start(
        Fit,
        GaussNewton::<_, _, _, _, DenseMatrix>::new_with_gram(),
        vec![0.0],
    )
    .run()
    .unwrap();
    assert_eq!(result.param(), &[2.0]);
}
