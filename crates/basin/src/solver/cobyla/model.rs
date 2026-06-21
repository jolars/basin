//! Linear interpolation models for COBYLA.
//!
//! At each iteration COBYLA reads the gradient of the linear model of the
//! objective, `g`, and of the constraints, `A` (n × m, one column of gradients
//! per constraint), from the simplex displacements via `simi = inv(sim[:,0..n])`.
//! These are PRIMA's `g = matprod(fval(1:n)−fval(n+1), simi)` and
//! `A = transpose((conmat(:,1:n)−conmat(:,n+1)) · simi)` (with `m_lcon = 0`,
//! since basin's COBYLA carries no separate linear-constraint block).

use crate::core::math::Scalar;

/// Objective-model gradient `g` (length `n`):
/// `g[l] = Σ_i (fval[i] − fval[n]) · simi[i, l]`.
pub(crate) fn build_g<F: Scalar>(fval: &[F], simi: &[F], n: usize) -> Vec<F> {
    let fn_pole = fval[n];
    (0..n)
        .map(|l| {
            (0..n)
                .map(|i| (fval[i] - fn_pole) * simi[i + l * n])
                .sum::<F>()
        })
        .collect()
}

/// Constraint-model gradients `A` (n × m, column-major; column `i` is the
/// gradient of constraint `i`):
/// `A[l, i] = Σ_j (conmat[i, j] − conmat[i, n]) · simi[j, l]`.
pub(crate) fn build_a<F: Scalar>(conmat: &[F], simi: &[F], n: usize, m: usize) -> Vec<F> {
    let mut a = vec![F::zero(); n * m];
    for i in 0..m {
        let pole = conmat[i + n * m];
        for l in 0..n {
            let mut s = F::zero();
            for j in 0..n {
                s = s + (conmat[i + j * m] - pole) * simi[j + l * n];
            }
            a[l + i * n] = s;
        }
    }
    a
}
