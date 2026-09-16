//! Portable one-sided cyclic Jacobi SVD for small dense least-squares models.
//!
//! Orthogonalize columns directly instead of squaring their condition number.
//! See Drmač, "Implementation of Jacobi rotations for accurate singular value
//! computation in floating point arithmetic", SIAM J. Sci. Comput. 18 (1997),
//! 1200–1222, <https://doi.org/10.1137/S1064827594265095>, and LAPACK's DGESVJ.
//! This is a cyclic implementation, not DGESVJ's pivoted/blocked algorithm.

use super::Scalar;

pub(crate) fn norm<F: Scalar>(values: &[F]) -> F {
    values.iter().fold(F::zero(), |s, &x| s.hypot(x))
}

#[derive(Clone, Debug)]
pub(crate) struct DenseSvd<F> {
    pub(crate) singular: Vec<F>,
    u: Vec<F>,
    v: Vec<F>,
    rows: usize,
}

impl<F: Scalar> DenseSvd<F> {
    /// Input and factors use row-major storage; zero singular vectors in U
    /// remain zero because least-squares solves never use their coefficients.
    pub(crate) fn factor(
        rows: usize,
        cols: usize,
        mut a: Vec<F>,
    ) -> Option<Self> {
        assert!(rows >= cols && cols > 0);
        assert_eq!(a.len(), rows * cols);
        if a.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let scale = a.iter().fold(F::zero(), |s, x| s.max(x.abs()));
        if scale > F::zero() {
            for x in &mut a {
                *x = *x / scale;
            }
        }
        let mut v = vec![F::zero(); cols * cols];
        for i in 0..cols {
            v[i * cols + i] = F::one();
        }
        let tol = F::epsilon() * F::from_usize(rows).unwrap().sqrt();
        let mut converged = false;
        for _ in 0..100 {
            let mut changed = false;
            for p in 0..cols {
                for q in p + 1..cols {
                    let pn = (0..rows)
                        .fold(F::zero(), |s, i| s.hypot(a[i * cols + p]));
                    let qn = (0..rows)
                        .fold(F::zero(), |s, i| s.hypot(a[i * cols + q]));
                    if pn == F::zero() || qn == F::zero() {
                        continue;
                    }
                    let corr: F = (0..rows)
                        .map(|i| {
                            (a[i * cols + p] / pn) * (a[i * cols + q] / qn)
                        })
                        .sum();
                    if corr.abs() <= tol {
                        continue;
                    }
                    // Normalize before computing the rotation to avoid squaring
                    // large norms or dividing by an unrepresentable tangent.
                    let mx = pn.max(qn);
                    let ap = pn / mx;
                    let aq = qn / mx;
                    let cross = corr * ap * aq;
                    if cross == F::zero() {
                        continue;
                    }
                    let delta = (aq * aq - ap * ap) / (F::one() + F::one());
                    let t = if delta == F::zero() {
                        F::one()
                    } else {
                        cross / (delta + delta.hypot(cross).copysign(delta))
                    };
                    let c = F::one() / F::one().hypot(t);
                    let s = c * t;
                    if s == F::zero() {
                        continue;
                    }
                    for i in 0..rows {
                        let (x, y) = (a[i * cols + p], a[i * cols + q]);
                        a[i * cols + p] = c * x - s * y;
                        a[i * cols + q] = s * x + c * y;
                    }
                    for i in 0..cols {
                        let (x, y) = (v[i * cols + p], v[i * cols + q]);
                        v[i * cols + p] = c * x - s * y;
                        v[i * cols + q] = s * x + c * y;
                    }
                    changed = true;
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        if !converged {
            return None;
        }
        let mut singular = vec![F::zero(); cols];
        for j in 0..cols {
            let length =
                (0..rows).fold(F::zero(), |s, i| s.hypot(a[i * cols + j]));
            if length > F::zero() {
                for i in 0..rows {
                    a[i * cols + j] = a[i * cols + j] / length;
                }
            }
            singular[j] = length * scale;
        }
        if singular.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(Self {
            singular,
            u: a,
            v,
            rows,
        })
    }

    /// Minimum-norm interior step or a boundary step from a safeguarded
    /// secular solve. Damping is measured in units of the largest singular
    /// value squared, so that neither normal equations nor squared large
    /// singular values are necessary.
    pub(crate) fn trust_step(
        &self,
        rhs: &[F],
        radius: F,
        rank_tolerance: Option<F>,
        max_iter: usize,
    ) -> Option<Vec<F>> {
        assert_eq!(rhs.len(), self.rows);
        let n = self.singular.len();
        let largest = self.singular.iter().copied().fold(F::zero(), F::max);
        if !radius.is_finite()
            || radius <= F::zero()
            || rhs.iter().any(|x| !x.is_finite())
        {
            return None;
        }
        if largest == F::zero() {
            return Some(vec![F::zero(); n]);
        }
        let cutoff = rank_tolerance
            .unwrap_or(F::epsilon() * F::from_usize(self.rows.max(n)).unwrap());
        let mut s = vec![F::zero(); n];
        let mut coeff = vec![F::zero(); n];
        let mut z = vec![F::zero(); n];
        for j in 0..n {
            s[j] = self.singular[j] / largest;
            if s[j] <= cutoff {
                s[j] = F::zero();
                continue;
            }
            let uf: F =
                (0..self.rows).map(|i| self.u[i * n + j] * rhs[i]).sum();
            coeff[j] = s[j] * (uf / largest);
            z[j] = -uf / self.singular[j];
        }
        if coeff.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let length = norm(&z);
        if length <= radius {
            return self.rotate(&z);
        }
        let mut high = norm(&coeff) / radius;
        if !high.is_finite() || high <= F::zero() {
            return None;
        }
        let derivative = |z: &[F], length: F, alpha: F| -> F {
            (0..n)
                .filter(|&j| coeff[j] != F::zero())
                .map(|j| (z[j] / length).powi(2) / (s[j] * s[j] + alpha))
                .sum()
        };
        // The step norm is decreasing and convex in damping, so its Newton
        // step at zero bounds the root from below on the retained subspace.
        let mut low = F::zero();
        if length.is_finite() {
            let lower = (F::one() - radius / length)
                / derivative(&z, length, F::zero());
            if lower.is_finite() {
                low = lower.max(low).min(high);
            }
        }
        // Keep a fraction of the upper bound so that large roots also start
        // within a few Newton steps of the solution.
        let mut alpha =
            (low.sqrt() * high.sqrt()).max(F::from_f64(0.001).unwrap() * high);
        let half = F::from_f64(0.5).unwrap();
        for _ in 0..max_iter {
            for j in 0..n {
                z[j] = if coeff[j] == F::zero() {
                    F::zero()
                } else {
                    -coeff[j] / (s[j] * s[j] + alpha)
                };
            }
            let length = norm(&z);
            if !length.is_finite() || length == F::zero() {
                return None;
            }
            if (length / radius - F::one()).abs() <= F::from_f64(0.01).unwrap()
            {
                for x in &mut z {
                    *x = (*x / length) * radius;
                }
                return self.rotate(&z);
            }
            if length > radius {
                low = alpha;
            } else {
                high = alpha;
            }
            let newton = alpha
                + (F::one() - radius / length) / derivative(&z, length, alpha);
            alpha = if newton.is_finite() && newton > low && newton < high {
                newton
            } else if low > F::zero() {
                // A geometric midpoint closes brackets spanning many orders
                // of magnitude without halving all the way to a tiny root.
                low.sqrt() * high.sqrt()
            } else {
                half * low + half * high
            };
        }
        None
    }

    fn rotate(&self, z: &[F]) -> Option<Vec<F>> {
        let n = z.len();
        let result: Vec<F> = (0..n)
            .map(|i| (0..n).map(|j| self.v[i * n + j] * z[j]).sum())
            .collect();
        result.iter().all(|x| x.is_finite()).then_some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_deficient_models_reconstruct_and_solve() {
        for n in 2..12 {
            let m = n + 3;
            let a: Vec<f64> = (0..m)
                .flat_map(|i| {
                    (0..n).map(move |j| {
                        (i as f64).sin() * (j as f64).cos()
                            + (0.3 * i as f64).cos() * (0.9 * j as f64).sin()
                    })
                })
                .collect();
            let svd = DenseSvd::factor(m, n, a.clone())
                .expect("rank-deficient SVD must converge");
            let rhs: Vec<f64> =
                a.chunks(n).map(|row| -row.iter().sum::<f64>()).collect();
            let p = svd.trust_step(&rhs, 100.0, None, 50).unwrap();
            for (i, row) in a.chunks(n).enumerate() {
                let residual: f64 =
                    row.iter().zip(&p).map(|(a, p)| a * p).sum::<f64>()
                        + rhs[i];
                assert!(residual.abs() < 1e-12, "n={n}, residual={residual}");
                for (j, &entry) in row.iter().enumerate() {
                    let got: f64 = (0..n)
                        .map(|k| {
                            svd.u[i * n + k]
                                * svd.singular[k]
                                * svd.v[j * n + k]
                        })
                        .sum();
                    assert!((got - entry).abs() < 1e-13);
                }
            }
        }
    }

    #[test]
    fn reconstruction_orthogonality_and_small_singular_values() {
        for a in [
            vec![1.0, 1.0, 0.0, 1e-10, 0.0, 0.0],
            vec![1e150, 1e150, 0.0, 1e140, 0.0, 0.0],
        ] {
            let svd = DenseSvd::factor(3, 2, a.clone()).unwrap();
            let scale = a[0];
            for i in 0..3 {
                for j in 0..2 {
                    let got: f64 = (0..2)
                        .map(|k| {
                            svd.u[i * 2 + k]
                                * svd.singular[k]
                                * svd.v[j * 2 + k]
                        })
                        .sum();
                    assert!((got - a[i * 2 + j]).abs() / scale < 1e-14);
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    let dot: f64 = (0..3)
                        .map(|k| svd.u[k * 2 + i] * svd.u[k * 2 + j])
                        .sum();
                    assert!((dot - f64::from(i == j)).abs() < 1e-14);
                }
            }
            let smallest =
                svd.singular.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                (smallest / scale / (1e-10 / 2.0_f64.sqrt()) - 1.0).abs()
                    < 1e-6
            );
        }
    }

    #[test]
    fn rank_deficient_minimum_norm_and_boundary_kkt() {
        let svd =
            DenseSvd::<f64>::factor(2, 2, vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let p = svd.trust_step(&[-1.0, -2.0], 10.0, None, 50).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-14 && (p[1] - 0.5).abs() < 1e-14);
        let p = svd.trust_step(&[-1.0, -2.0], 0.1, None, 50).unwrap();
        assert!((norm(&p) - 0.1).abs() < 1e-14);
        assert!((p[0] - p[1]).abs() < 1e-14);
        let svd =
            DenseSvd::<f64>::factor(2, 2, vec![1.0, 0.0, 0.0, 3.0]).unwrap();
        let p = svd.trust_step(&[1.0, 1.0], 0.2, None, 50).unwrap();
        let alpha = (-1.0 - p[0]) / p[0];
        assert!(alpha > 0.0);
        assert!((3.0 + (9.0 + alpha) * p[1]).abs() < 0.03);
    }

    #[test]
    fn small_secular_roots_reach_the_boundary() {
        for small in [1e-4_f64, 1e-8, 1e-10, 1e-14] {
            for deficient in [false, true] {
                let last = if deficient { 0.0 } else { 1.0 };
                let svd = DenseSvd::factor(
                    3,
                    3,
                    vec![1.0, 0.0, 0.0, 0.0, small, 0.0, 0.0, 0.0, last],
                )
                .unwrap();
                let p = svd
                    .trust_step(&[-1.0, -10.0 * small, 0.0], 2.0, None, 50)
                    .expect("small retained singular values must converge");
                assert!((norm(&p) - 2.0).abs() < 1e-14);
                // The first coordinate is nearly one, leaving sqrt(3) of
                // the radius for the weak singular direction.
                assert!((p[0] - 1.0).abs() < 0.02, "{small}: {p:?}");
                assert!((p[1] - 3.0_f64.sqrt()).abs() < 0.02);
                assert_eq!(p[2], 0.0);
                let alpha = small * small * (10.0 / p[1] - 1.0);
                assert!(alpha > 0.0);
                assert!(((1.0 + alpha) * p[0] - 1.0).abs() < 0.02);
            }
        }
    }

    #[test]
    fn large_secular_roots_reach_the_boundary() {
        let svd = DenseSvd::factor(1, 1, vec![1.0_f64]).unwrap();
        for (rhs, radius) in [(-1e100, 1.0), (-1.0, 1e-100)] {
            let p = svd.trust_step(&[rhs], radius, None, 50).unwrap();
            assert!((p[0] / radius - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn zero_nonfinite_and_exhaustion() {
        let svd = DenseSvd::factor(2, 2, vec![0.0; 4]).unwrap();
        assert_eq!(
            svd.trust_step(&[1.0, 1.0], 1.0, None, 50).unwrap(),
            vec![0.0; 2]
        );
        assert!(DenseSvd::factor(1, 1, vec![f64::NAN]).is_none());
        let svd = DenseSvd::factor(2, 2, vec![1.0, 0.0, 0.0, 3.0]).unwrap();
        assert!(svd.trust_step(&[1.0, 1.0], 0.001, None, 1).is_none());
    }
}
