//! Cholesky solve of an SPD system for the `Vec<F>` backend.
//!
//! [`DenseMatrix`](super::DenseMatrix) is `Vec<F>`'s dense matrix companion,
//! but it has no external linear-algebra crate behind it. The least-squares
//! solvers form the normal equations `(JᵀJ + μD) x = −g` and solve them by
//! Cholesky every iteration (the [`LinearSolveSpd`](super::LinearSolveSpd) op),
//! so to run Gauss-Newton/Levenberg-Marquardt on the default backend we need
//! an honest, pure-Rust SPD solve: one with no BLAS/LAPACK dependency, so it
//! stays `wasm`-clean.
//!
//! This module implements the classic **Cholesky–Banachiewicz** factorization
//! `A = L Lᵀ` (Golub & Van Loan, *Matrix Computations* §4.2) followed by a
//! forward solve `L y = b` and a back solve `Lᵀ x = y`. It is the simplest
//! correct SPD solver: a single `O(n³)` factorization and two `O(n²)`
//! triangular solves, with no pivoting (SPD matrices need none). It is the
//! solve-layer sibling of the cyclic Jacobi eigensolver in
//! [`dense_eig`](super::dense_eig); like that module, `Vec<F>` is the
//! convenience backend; callers wanting speed at large `n` reach for faer.

use super::Scalar;

/// Solve the SPD system `A x = b` via a Cholesky factorization `A = L Lᵀ`.
///
/// `a` is the row-major `n × n` matrix; only the lower triangle (and diagonal)
/// is read, matching the [`LinearSolveSpd`](super::LinearSolveSpd) and
/// [`SymmetricEigen`](super::SymmetricEigen) lower-triangle-authoritative
/// contract. `b` has length `n`. Returns the solution `x` of length `n`.
///
/// Returns `None` when the factorization hits a non-positive pivot: the matrix
/// is not positive definite (e.g. `A = JᵀJ` with rank-deficient `J`). The
/// caller maps this to
/// [`LinearSolveError::NotPositiveDefinite`](super::LinearSolveError::NotPositiveDefinite).
pub(super) fn cholesky_solve_spd<F: Scalar>(a: &[F], n: usize, b: &[F]) -> Option<Vec<F>> {
    debug_assert_eq!(a.len(), n * n, "cholesky_solve_spd: expected an n×n buffer");
    debug_assert_eq!(b.len(), n, "cholesky_solve_spd: expected a length-n rhs");

    let zero = F::zero();

    // Factor: L is lower-triangular, row-major. L[i][j] for j ≤ i; the strict
    // upper triangle stays zero.
    let mut l = vec![zero; n * n];
    for i in 0..n {
        for j in 0..=i {
            // s ← A[i][j] − Σ_{k<j} L[i][k] · L[j][k]
            let mut s = a[i * n + j];
            for k in 0..j {
                s = s - l[i * n + k] * l[j * n + k];
            }
            if i == j {
                // A non-positive pivot means `A` is not positive definite.
                if s <= zero {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }

    // Forward solve L y = b (overwrite into `y`).
    let mut y = vec![zero; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s = s - l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }

    // Back solve Lᵀ x = y (overwrite into `x`).
    let mut x = vec![zero; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s = s - l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multiply a row-major `n × n` matrix by a length-`n` vector.
    fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < tol, "‖{x} − {y}‖ ≥ {tol}");
        }
    }

    #[test]
    fn solves_known_2x2() {
        // [[4, 1], [1, 3]] x = (1, 2); solution x = (1/11, 7/11).
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = cholesky_solve_spd::<f64>(&a, 2, &b).unwrap();
        assert_close(&x, &[1.0 / 11.0, 7.0 / 11.0], 1e-12);
        // Residual must vanish.
        assert_close(&matvec(&a, &x, 2), &b, 1e-12);
    }

    #[test]
    fn solves_spd_5x5_round_trip() {
        // SPD A = MᵀM + 5I; pick a known x, form b = A x, recover x.
        let n = 5;
        let mraw = [
            1.0, 0.3, -0.2, 0.5, 0.1, 0.0, 1.2, 0.4, -0.1, 0.2, 0.3, 0.0, 0.9, 0.6, -0.3, -0.4,
            0.1, 0.2, 1.1, 0.0, 0.2, -0.5, 0.3, 0.1, 1.3,
        ];
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += mraw[k * n + i] * mraw[k * n + j];
                }
                a[i * n + j] = s + if i == j { 5.0 } else { 0.0 };
            }
        }
        let x_true = vec![1.0, -2.0, 0.5, 3.0, -1.5];
        let b = matvec(&a, &x_true, n);
        let x = cholesky_solve_spd::<f64>(&a, n, &b).unwrap();
        assert_close(&x, &x_true, 1e-9);
    }

    #[test]
    fn lower_triangle_is_authoritative() {
        // Garbage in the upper triangle must be ignored.
        let clean = vec![4.0, 1.0, 1.0, 3.0];
        let dirty = vec![4.0, 99.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let x_clean = cholesky_solve_spd::<f64>(&clean, 2, &b).unwrap();
        let x_dirty = cholesky_solve_spd::<f64>(&dirty, 2, &b).unwrap();
        assert_close(&x_clean, &x_dirty, 1e-14);
    }

    #[test]
    fn rejects_non_positive_definite() {
        // Indefinite: [[1, 2], [2, 1]] has a negative eigenvalue, so the
        // second pivot goes non-positive.
        let a = vec![1.0, 2.0, 2.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!(cholesky_solve_spd::<f64>(&a, 2, &b).is_none());
    }

    #[test]
    fn rejects_positive_semidefinite_singular() {
        // Rank-1 PSD (not PD): [[1, 1], [1, 1]]—the second pivot is exactly 0.
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!(cholesky_solve_spd::<f64>(&a, 2, &b).is_none());
    }

    #[test]
    fn single_element() {
        let x = cholesky_solve_spd::<f64>(&[4.0], 1, &[8.0]).unwrap();
        assert_close(&x, &[2.0], 1e-12);
    }

    #[test]
    fn single_element_rejects_non_positive() {
        assert!(cholesky_solve_spd::<f64>(&[0.0], 1, &[1.0]).is_none());
        assert!(cholesky_solve_spd::<f64>(&[-3.0], 1, &[1.0]).is_none());
    }

    /// f32 sanity check: the factorization should solve at any precision.
    #[test]
    fn f32_known_2x2() {
        let a: Vec<f32> = vec![4.0, 1.0, 1.0, 3.0];
        let b: Vec<f32> = vec![1.0, 2.0];
        let x = cholesky_solve_spd::<f32>(&a, 2, &b).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-6, "x[1] = {}", x[1]);
    }
}
