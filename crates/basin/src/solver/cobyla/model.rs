//! Linear interpolation models for COBYLA.
//!
//! At each iteration COBYLA reads the gradient of the linear model of the
//! objective, `g`, and of the constraints, `A` (n × m, one column of gradients
//! per constraint), from the simplex displacements via `simi = inv(sim[:,0..n])`.
//! These are PRIMA's `g = matprod(fval(1:n)−fval(n+1), simi)` and
//! `A = transpose((conmat(:,1:n)−conmat(:,n+1)) · simi)` (with `m_lcon = 0`,
//! since basin's COBYLA carries no separate linear-constraint block).

use crate::core::math::Scalar;

/// Reusable interpolation-model storage.
pub(crate) struct ModelWork<F> {
    pub(crate) g: Vec<F>,
    pub(crate) a: Vec<F>,
    pub(crate) b: Vec<F>,
    difference: Vec<F>,
}

impl<F: Scalar> ModelWork<F> {
    pub(crate) fn new(n: usize, m: usize) -> Self {
        Self {
            g: vec![F::zero(); n],
            a: vec![F::zero(); n * m],
            b: vec![F::zero(); m],
            difference: vec![F::zero(); n],
        }
    }

    pub(crate) fn build(&mut self, fval: &[F], conmat: &[F], simi: &[F]) {
        let n = self.g.len();
        let m = self.b.len();
        build_g_into(fval, simi, n, &mut self.g);
        build_a_into(conmat, simi, n, m, &mut self.a, &mut self.difference);
        for i in 0..m {
            self.b[i] = -conmat[i + n * m];
        }
    }
}

/// Objective-model gradient `g` (length `n`):
/// `g[l] = Σ_i (fval[i] − fval[n]) · simi[i, l]`.
fn build_g_into<F: Scalar>(fval: &[F], simi: &[F], n: usize, g: &mut [F]) {
    let fn_pole = fval[n];
    for l in 0..n {
        g[l] = (0..n).map(|i| (fval[i] - fn_pole) * simi[i + l * n]).sum();
    }
}

/// Constraint-model gradients `A` (n × m, column-major; column `i` is the
/// gradient of constraint `i`):
/// `A[l, i] = Σ_j (conmat[i, j] − conmat[i, n]) · simi[j, l]`.
fn build_a_into<F: Scalar>(
    conmat: &[F],
    simi: &[F],
    n: usize,
    m: usize,
    a: &mut [F],
    difference: &mut [F],
) {
    for i in 0..m {
        let pole = conmat[i + n * m];
        for j in 0..n {
            difference[j] = conmat[i + j * m] - pole;
        }
        for l in 0..n {
            let mut s = F::zero();
            for j in 0..n {
                s = s + difference[j] * simi[j + l * n];
            }
            a[l + i * n] = s;
        }
    }
}
