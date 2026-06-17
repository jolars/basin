//! Test-only brute-force oracle for the NEWUOA `H = W⁻¹` algebra.
//!
//! The §3 initialization and §4 update maintain `H` in closed form / by rank-2
//! updates. To check that machinery we assemble the full KKT matrix `W`
//! (Powell 2006, eqs. 3.10–3.12) densely from the interpolation geometry,
//! invert it with a plain Gauss-Jordan solver, and assert that the model's
//! stored `Ω`/`Ξ`/`Υ` blocks equal the corresponding blocks of `inv(W)`.
//!
//! Index convention for the order-`(m+n+1)` system: indices `0..m` are the
//! interpolation/`A`-block components `λ`, index `m` is the suppressed constant
//! term `c` (Powell 2006, §4 — its row/column is *not* stored by the model),
//! and indices `m+1..m+n+1` are the gradient components `g`.

use crate::core::math::DenseMatrix;

use super::model::QuadraticModel;

/// Assemble the full dense KKT matrix `W` of order `m + n + 1` from the model's
/// interpolation geometry (Powell 2006, eqs. 3.10, 3.11, 3.12):
///
/// ```text
/// W = [[A, P], [Pᵀ, 0]] ,   A_{ij} = ½((xᵢ−x0)·(xⱼ−x0))² ,
/// ```
///
/// where `P` is `m × (n+1)` with `P[i,0] = 1` (the `c` column) and
/// `P[i,1+k] = (xᵢ−x0)_k` (the `g` columns).
pub(crate) fn build_w_dense(model: &QuadraticModel<f64>) -> DenseMatrix<f64> {
    let n = model.n();
    let m = model.m();
    let dim = m + n + 1;
    let mut w = DenseMatrix::from_fn(dim, dim, |_, _| 0.0);

    // A block (top-left m×m).
    for i in 0..m {
        let xi = model.xpt_row(i);
        for j in 0..m {
            let xj = model.xpt_row(j);
            let d: f64 = xi.iter().zip(xj).map(|(a, b)| a * b).sum();
            w.set(i, j, 0.5 * d * d);
        }
    }

    // P / Pᵀ blocks. Column/row `m` is the constant term `c`; columns/rows
    // `m+1..m+n+1` are the gradient components.
    for i in 0..m {
        let xi = model.xpt_row(i);
        // c coupling.
        w.set(i, m, 1.0);
        w.set(m, i, 1.0);
        // g coupling.
        for k in 0..n {
            w.set(i, m + 1 + k, xi[k]);
            w.set(m + 1 + k, i, xi[k]);
        }
    }
    // Bottom-right (n+1)×(n+1) block stays zero.
    w
}

/// Invert a square matrix by Gauss-Jordan elimination with partial pivoting.
/// Returns `None` if the matrix is singular to working precision. Test-only;
/// intended for the small (`m + n + 1 ≤ ~12`) systems the oracle builds.
pub(crate) fn invert_dense(a: &DenseMatrix<f64>) -> Option<DenseMatrix<f64>> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "invert_dense: matrix must be square");
    // Augmented [A | I] in a flat row-major buffer of width 2n.
    let w = 2 * n;
    let mut aug = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n {
            aug[i * w + j] = a.get(i, j);
        }
        aug[i * w + n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivot: largest |entry| in this column at or below the diagonal.
        let mut piv = col;
        let mut best = aug[col * w + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * w + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != col {
            for j in 0..w {
                aug.swap(col * w + j, piv * w + j);
            }
        }
        // Normalize pivot row.
        let pivval = aug[col * w + col];
        for j in 0..w {
            aug[col * w + j] /= pivval;
        }
        // Eliminate this column from all other rows.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r * w + col];
            if factor != 0.0 {
                for j in 0..w {
                    aug[r * w + j] -= factor * aug[col * w + j];
                }
            }
        }
    }

    let mut inv = DenseMatrix::from_fn(n, n, |_, _| 0.0);
    for i in 0..n {
        for j in 0..n {
            inv.set(i, j, aug[i * w + n + j]);
        }
    }
    Some(inv)
}

/// Reconstruct the dense `Ω = Σ_k sₖ zₖ zₖᵀ` (`m × m`) from the model's stored
/// factorization (columns of `zmat`, signs `zsign`).
pub(crate) fn omega_from_factorization(model: &QuadraticModel<f64>) -> DenseMatrix<f64> {
    let m = model.m();
    let n = model.n();
    let rank = m - n - 1;
    let mut omega = DenseMatrix::from_fn(m, m, |_, _| 0.0);
    for k in 0..rank {
        let s = model.zsign[k];
        for i in 0..m {
            let zi = model.zmat.get(i, k);
            for j in 0..m {
                let zj = model.zmat.get(j, k);
                omega.set(i, j, omega.get(i, j) + s * zi * zj);
            }
        }
    }
    omega
}

/// Assert that the model's stored `Ω`/`Ξ`/`Υ` blocks equal the corresponding
/// blocks of `inv(W)`, where `W` is assembled from the current interpolation
/// geometry. This is the master KKT-identity check for both INIT and UPDATE.
///
/// `Ξ` (`bmat_xi`, `n × m`) maps to the gradient-rows × `λ`-cols block of
/// `inv(W)` (rows `m+1..`, cols `0..m`); `Υ` (`bmat_ups`, `n × n`) to the
/// gradient × gradient block (rows/cols `m+1..`); `Ω` to the leading `m × m`
/// block. The suppressed `c` row/column (index `m`) is never compared.
///
/// `tol` is a *mixed* tolerance: the threshold for each entry is
/// `tol·(1 + |want|)`. A purely absolute tolerance is unfair when ill-conditioned
/// (near-coincident) interpolation points blow `H` up to large magnitudes.
pub(crate) fn assert_h_matches_inverse(model: &QuadraticModel<f64>, tol: f64) {
    let n = model.n();
    let m = model.m();
    let w = build_w_dense(model);
    let h = invert_dense(&w).expect("KKT matrix W must be nonsingular");

    let check = |got: f64, want: f64, label: &str| {
        let thresh = tol * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= thresh,
            "{label}: stored {got} vs inv(W) {want} (Δ={:e}, thresh={thresh:e})",
            (got - want).abs()
        );
    };

    // Ω block.
    let omega = omega_from_factorization(model);
    for i in 0..m {
        for j in 0..m {
            check(omega.get(i, j), h.get(i, j), &format!("Ω[{i},{j}]"));
        }
    }

    // Ξ block (gradient rows × λ cols).
    for r in 0..n {
        for j in 0..m {
            check(
                model.bmat_xi.get(r, j),
                h.get(m + 1 + r, j),
                &format!("Ξ[{r},{j}]"),
            );
        }
    }

    // Υ block (gradient × gradient).
    for r in 0..n {
        for s in 0..n {
            check(
                model.bmat_ups.get(r, s),
                h.get(m + 1 + r, m + 1 + s),
                &format!("Υ[{r},{s}]"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Gauss-Jordan inverse must satisfy `A · inv(A) = I` and agree with
    /// the crate's SPD solver column by column on an SPD matrix.
    #[test]
    fn invert_dense_round_trips() {
        use crate::core::math::LinearSolveSpd;
        let a = DenseMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0]);
        let inv = invert_dense(&a).unwrap();

        // A · inv = I.
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += a.get(i, k) * inv.get(k, j);
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((acc - want).abs() < 1e-12, "A·inv[{i},{j}] = {acc}");
            }
        }

        // Columns of inv solve A x = eⱼ — cross-check against Cholesky solve.
        for j in 0..3 {
            let mut e = vec![0.0; 3];
            e[j] = 1.0;
            let x = a.solve_spd(&e).unwrap();
            for i in 0..3 {
                assert!((x[i] - inv.get(i, j)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn invert_dense_reports_singular() {
        // Rank-1 (singular) matrix.
        let a = DenseMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(invert_dense(&a).is_none());
    }
}
