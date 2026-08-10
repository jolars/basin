//! LAPACK-backed nalgebra factorizations (`nalgebra-lapack` feature).
//!
//! These exercise the `#[cfg(feature = "nalgebra-lapack")]` impls in
//! `core::math::nalgebra_backend` (LAPACK Cholesky for `LinearSolveSpd`, LAPACK
//! `dsyev` for `SymmetricEigen`), mirroring the pure-Rust unit tests in that
//! module so both code paths are checked against the same expectations.
//!
//! Running this test *links* LAPACK, so it needs a provider supplied at link
//! time (basin's `nalgebra-lapack` feature uses `lapack-custom`). CI installs a
//! system OpenBLAS and links it via `RUSTFLAGS`; see `.github/workflows/ci.yml`.
//! Locally (devenv exposes an LP64 OpenBLAS as `$OPENBLAS_LP64_LIB`):
//!
//! ```sh
//! RUSTFLAGS="-L $OPENBLAS_LP64_LIB -l openblas" \
//!   cargo test -p basin --features nalgebra-lapack --test lapack_nalgebra
//! ```
#![cfg(all(feature = "nalgebra", feature = "nalgebra-lapack"))]

use basin::{GramMatrix, LinearSolveError, LinearSolveSpd, SymmetricEigen};
use nalgebra::{DMatrix, DVector};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn solve_spd_happy_path() {
    // A = [[4, 1], [1, 3]], b = [1, 2]; det = 11, x = [1/11, 7/11].
    let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    let b = DVector::from_vec(vec![1.0, 2.0]);
    let x = a.solve_spd(&b).expect("SPD system must solve");
    assert!(approx_eq(x[0], 1.0 / 11.0, 1e-12));
    assert!(approx_eq(x[1], 7.0 / 11.0, 1e-12));
}

#[test]
fn solve_spd_indefinite_returns_error() {
    // A = [[1, 2], [2, 1]] is symmetric but indefinite (det = −3): LAPACK's
    // Cholesky must report a failed factorization, mapped to NotPositiveDefinite.
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    let b = DVector::from_vec(vec![1.0, 1.0]);
    let err = a.solve_spd(&b).expect_err("indefinite must fail");
    assert_eq!(err, LinearSolveError::NotPositiveDefinite);
}

#[test]
fn solve_spd_rank_deficient_gram_is_singular() {
    // Rank-1 A → AᵀA is rank-1, singular, fails Cholesky.
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    let g = a.gram();
    let b = DVector::from_vec(vec![1.0, 1.0]);
    let err = g.solve_spd(&b).expect_err("rank-deficient gram must fail");
    assert_eq!(err, LinearSolveError::NotPositiveDefinite);
}

#[test]
fn symmetric_eigen_recovers_factorization() {
    // C = [[2, 1], [1, 2]] has eigenvalues 1, 3. Verify B Λ Bᵀ ≈ C regardless
    // of the (unspecified) eigenvalue ordering LAPACK returns.
    let c = DMatrix::<f64>::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let (b, lambda) = c.try_eigh().expect("eigendecomposition");
    let mut lambda_diag = DMatrix::<f64>::zeros(2, 2);
    for i in 0..2 {
        lambda_diag[(i, i)] = lambda[i];
    }
    let recomposed = &b * &lambda_diag * b.transpose();
    for r in 0..2 {
        for col in 0..2 {
            assert!(approx_eq(recomposed[(r, col)], c[(r, col)], 1e-10));
        }
    }
}

#[test]
fn gauss_newton_converges_through_lapack_cholesky() {
    // End-to-end wiring check: Gauss-Newton's normal-equation solve routes
    // through `LinearSolveSpd` on the nalgebra backend, i.e. the LAPACK
    // Cholesky impl under this feature. Mirrors `gauss_newton_nalgebra.rs`.
    use basin::problems::RosenbrockResiduals;
    use basin::{Executor, GaussNewton, NllsState, TerminationReason};

    let problem = RosenbrockResiduals::<DVector<f64>>::new();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
            .max_iter(20)
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.param();
    assert!(approx_eq(x[0], 1.0, 1e-6));
    assert!(approx_eq(x[1], 1.0, 1e-6));
}
