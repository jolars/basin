//! Cyclic Jacobi symmetric eigendecomposition for the `Vec<F>` backend.
//!
//! [`DenseMatrix`](super::DenseMatrix) is `Vec<F>`'s dense matrix companion,
//! but it has no external linear-algebra crate behind it. CMA-ES factors its
//! covariance every iteration (the [`SymmetricEigen`](super::SymmetricEigen)
//! op), so to run CMA-ES on the default backend we need an honest, pure-Rust
//! symmetric eigensolver — one with no BLAS/LAPACK dependency, so it stays
//! `wasm`-clean.
//!
//! This module implements the classic **cyclic Jacobi** algorithm (Golub & Van
//! Loan, *Matrix Computations* §8.5; Numerical Recipes §11.1). It is the
//! simplest correct symmetric eigensolver: a sequence of plane rotations that
//! drive the off-diagonal mass to zero, accumulating the eigenvectors in an
//! orthogonal matrix. It is slower than the tridiagonal-QR method the
//! nalgebra/faer backends use, but `Vec<F>` is the convenience backend —
//! callers wanting speed at large `n` reach for faer.

use super::Scalar;

/// Frobenius norm `‖m‖_F = sqrt(Σ mᵢⱼ²)` of a row-major `n × n` matrix.
fn frobenius_norm<F: Scalar>(m: &[F]) -> F {
    m.iter().map(|&x| x * x).sum::<F>().sqrt()
}

/// Sum of squares of the strictly-upper-triangular (off-diagonal) entries of a
/// symmetric row-major `n × n` matrix.
fn off_diagonal_sum_sq<F: Scalar>(m: &[F], n: usize) -> F {
    let mut s = F::zero();
    for p in 0..n {
        for q in (p + 1)..n {
            let v = m[p * n + q];
            s = s + v * v;
        }
    }
    s
}

/// Compute a symmetric eigendecomposition via cyclic Jacobi rotations.
///
/// `a` is the row-major `n × n` symmetric matrix; the lower triangle is
/// authoritative (the upper triangle is mirrored from it before factoring),
/// matching the [`SymmetricEigen`](super::SymmetricEigen) contract. Returns
/// `(eigenvalues, eigenvectors_row_major)` where column `k` of the eigenvector
/// matrix (entry `eigenvectors[i * n + k]`) is the unit eigenvector paired with
/// `eigenvalues[k]`. Eigenvalues are returned unsorted (in whatever order the
/// rotations leave them on the diagonal).
///
/// Returns `None` if the sweep budget is exhausted before the off-diagonal mass
/// falls below tolerance — a failure the caller maps to
/// [`SymmetricEigenError::Failed`](super::SymmetricEigenError::Failed).
pub(super) fn jacobi_eigen<F: Scalar>(a: &[F], n: usize) -> Option<(Vec<F>, Vec<F>)> {
    debug_assert_eq!(a.len(), n * n, "jacobi_eigen: expected an n×n buffer");

    let zero = F::zero();
    let one = F::one();
    let two = F::from_f64(2.0).unwrap();

    // Working symmetric matrix, row-major; mirror the lower triangle into the
    // upper so the input's upper triangle is ignored (per the contract).
    let mut m = vec![zero; n * n];
    for i in 0..n {
        for j in 0..=i {
            let val = a[i * n + j];
            m[i * n + j] = val;
            m[j * n + i] = val;
        }
    }

    // Eigenvector accumulator V = I (row-major). Columns become eigenvectors.
    let mut v = vec![zero; n * n];
    for i in 0..n {
        v[i * n + i] = one;
    }

    // n ≤ 1 is already diagonal; nothing to rotate.
    if n <= 1 {
        let eigvals = (0..n).map(|i| m[i * n + i]).collect();
        return Some((eigvals, v));
    }

    // Converged once the off-diagonal Frobenius mass is negligible relative to
    // the whole matrix. The Frobenius norm is invariant under the orthogonal
    // rotations, so the threshold is effectively fixed across sweeps.
    // The threshold scales with `F::epsilon()` (≈ 2.22e-16 at f64,
    // ≈ 1.19e-7 at f32) so the same code converges to backend-appropriate
    // precision at any scalar width.
    const MAX_SWEEPS: usize = 100;
    let tol = F::epsilon() * frobenius_norm(&m).max(F::min_positive_value());

    for _ in 0..MAX_SWEEPS {
        if off_diagonal_sum_sq(&m, n).sqrt() <= tol {
            let eigvals = (0..n).map(|i| m[i * n + i]).collect();
            return Some((eigvals, v));
        }

        // One sweep over every strict-upper pivot (p, q).
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq == zero {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];

                // Rotation angle that zeros m[p][q] (Golub & Van Loan eq. 8.5.2).
                let theta = (aqq - app) / (two * apq);
                let t = if theta == zero {
                    one
                } else {
                    let sign = if theta > zero { one } else { -one };
                    // For huge |theta| this underflows to ~0 (near-identity
                    // rotation) rather than producing a NaN.
                    sign / (theta.abs() + (theta * theta + one).sqrt())
                };
                let c = one / (t * t + one).sqrt();
                let s = t * c;
                let tau = s / (one + c);

                // Diagonal updates; zero the pivot in both symmetric slots.
                m[p * n + p] = app - t * apq;
                m[q * n + q] = aqq + t * apq;
                m[p * n + q] = zero;
                m[q * n + p] = zero;

                // Rotate the remaining entries of rows/cols p and q, keeping
                // the matrix symmetric.
                for i in 0..n {
                    if i != p && i != q {
                        let aip = m[i * n + p];
                        let aiq = m[i * n + q];
                        let new_ip = aip - s * (aiq + tau * aip);
                        let new_iq = aiq + s * (aip - tau * aiq);
                        m[i * n + p] = new_ip;
                        m[p * n + i] = new_ip;
                        m[i * n + q] = new_iq;
                        m[q * n + i] = new_iq;
                    }
                }

                // Accumulate the rotation into the eigenvector columns p, q.
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = vip - s * (viq + tau * vip);
                    v[i * n + q] = viq + s * (vip - tau * viq);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstruct `A = B · diag(λ) · Bᵀ` from the returned factors, into a
    /// row-major buffer, so tests can compare against the original matrix.
    fn reconstruct(eigvals: &[f64], eigvecs: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += eigvecs[i * n + k] * eigvals[k] * eigvecs[j * n + k];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < tol, "‖{x} − {y}‖ ≥ {tol}");
        }
    }

    /// Eigenvectors must be orthonormal: `BᵀB ≈ I`.
    fn assert_orthonormal(eigvecs: &[f64], n: usize) {
        for p in 0..n {
            for q in 0..n {
                let mut dot = 0.0;
                for i in 0..n {
                    dot += eigvecs[i * n + p] * eigvecs[i * n + q];
                }
                let expected = if p == q { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "BᵀB[{p}][{q}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn known_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues {1, 3}.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (eigs, vecs) = jacobi_eigen::<f64>(&a, 2).unwrap();
        assert_orthonormal(&vecs, 2);
        // Reconstruct from the *paired* (unsorted) factors before sorting.
        assert_close(&reconstruct(&eigs, &vecs, 2), &a, 1e-12);
        let mut sorted = eigs.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-12, "λ₀ = {}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1e-12, "λ₁ = {}", sorted[1]);
    }

    #[test]
    fn reconstructs_symmetric_3x3() {
        // A symmetric (indefinite) matrix.
        let a = vec![4.0, 1.0, -2.0, 1.0, 2.0, 0.0, -2.0, 0.0, 3.0];
        let (eigs, vecs) = jacobi_eigen::<f64>(&a, 3).unwrap();
        assert_orthonormal(&vecs, 3);
        assert_close(&reconstruct(&eigs, &vecs, 3), &a, 1e-10);
    }

    #[test]
    fn diagonal_input_passthrough() {
        // diag(3, 5, 7): eigenvalues are the diagonal, eigenvectors orthonormal.
        let a = vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0];
        let (mut eigs, vecs) = jacobi_eigen::<f64>(&a, 3).unwrap();
        assert_orthonormal(&vecs, 3);
        eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_close(&eigs, &[3.0, 5.0, 7.0], 1e-12);
    }

    #[test]
    fn lower_triangle_is_authoritative() {
        // Garbage in the upper triangle must be ignored; only the lower
        // triangle (and diagonal) define the matrix.
        let clean = vec![2.0, 1.0, 1.0, 2.0];
        let dirty = vec![2.0, 99.0, 1.0, 2.0]; // upper entry differs
        let (mut e_clean, _) = jacobi_eigen::<f64>(&clean, 2).unwrap();
        let (mut e_dirty, _) = jacobi_eigen::<f64>(&dirty, 2).unwrap();
        e_clean.sort_by(|x, y| x.partial_cmp(y).unwrap());
        e_dirty.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_close(&e_clean, &e_dirty, 1e-14);
    }

    #[test]
    fn reconstructs_spd_5x5() {
        // Build an SPD matrix A = MᵀM + 5I (well-conditioned, like a covariance).
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
        let (eigs, vecs) = jacobi_eigen::<f64>(&a, n).unwrap();
        assert_orthonormal(&vecs, n);
        for &lam in &eigs {
            assert!(
                lam > 0.0,
                "SPD matrix must have positive eigenvalues, got {lam}"
            );
        }
        assert_close(&reconstruct(&eigs, &vecs, n), &a, 1e-9);
    }

    #[test]
    fn single_element() {
        let (eigs, vecs) = jacobi_eigen::<f64>(&[42.0], 1).unwrap();
        assert_eq!(eigs, vec![42.0]);
        assert_eq!(vecs, vec![1.0]);
    }

    /// f32 sanity check: the Jacobi sweep should converge at any precision.
    /// Tolerances loosened to f32-appropriate scale.
    #[test]
    fn f32_known_2x2() {
        let a: Vec<f32> = vec![2.0, 1.0, 1.0, 2.0];
        let (mut eigs, _vecs) = jacobi_eigen::<f32>(&a, 2).unwrap();
        eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-5, "λ₀ = {}", eigs[0]);
        assert!((eigs[1] - 3.0).abs() < 1e-5, "λ₁ = {}", eigs[1]);
    }
}
