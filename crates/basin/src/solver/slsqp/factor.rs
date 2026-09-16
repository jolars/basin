//! Packed LDLᵀ factors and the Fletcher–Powell rank-one modification used
//! by Kraft's damped BFGS update. See the notices in `COPYRIGHT`.

#![allow(clippy::needless_range_loop)]

use super::least_squares::{Matrix, dot, number};
use crate::core::math::Scalar;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct Factor<F> {
    n: usize,
    packed: Vec<F>,
}

impl<F: Scalar> Factor<F> {
    pub fn identity(n: usize) -> Self {
        let mut result = Self {
            n,
            packed: vec![F::zero(); n * (n + 1) / 2],
        };
        for i in 0..n {
            let p = result.start(i);
            result.packed[p] = F::one();
        }
        result
    }
    fn start(&self, j: usize) -> usize {
        j * self.n - j * j.saturating_sub(1) / 2
    }
    fn lower(&self, i: usize, j: usize) -> F {
        self.packed[self.start(j) + i - j]
    }
    pub fn product(&self, s: &[F], v: &mut [F]) {
        for i in 0..self.n {
            v[i] = (s[i]
                + (i + 1..self.n).map(|j| self.lower(j, i) * s[j]).sum::<F>())
                * self.lower(i, i);
        }
        // Descending rows preserve the inputs still needed by lower columns.
        for i in (0..self.n).rev() {
            v[i] = v[i] + (0..i).map(|j| self.lower(i, j) * v[j]).sum::<F>();
        }
    }
    pub fn least_squares(
        &self,
        g: &[F],
        slack: Option<F>,
        e: &mut Matrix<F>,
        f: &mut Vec<F>,
    ) {
        let n = self.n + usize::from(slack.is_some());
        e.resize_zeroed(n, n);
        f.resize(n, F::zero());
        f.fill(F::zero());
        for i in 0..self.n {
            let diag = self.lower(i, i).sqrt();
            e.set(i, i, diag);
            for j in i + 1..self.n {
                e.set(i, j, diag * self.lower(j, i));
            }
            f[i] =
                (-g[i] - (0..i).map(|j| e.get(j, i) * f[j]).sum::<F>()) / diag;
        }
        if let Some(weight) = slack {
            e.set(n - 1, n - 1, weight);
        }
    }
    pub fn update(
        &mut self,
        s: &[F],
        y: &mut [F],
        bs: &mut Vec<F>,
        w: &mut Vec<F>,
    ) -> bool {
        bs.resize(self.n, F::zero());
        self.product(s, bs);
        let mut sy = dot(s, y);
        let sbs = dot(s, bs);
        let threshold = number::<F>(0.2) * sbs;
        if sy < threshold {
            let theta = (sbs - threshold) / (sbs - sy);
            for i in 0..self.n {
                y[i] = theta * y[i] + (F::one() - theta) * bs[i];
            }
            sy = threshold;
        }
        if !sy.is_finite()
            || !sbs.is_finite()
            || sy <= F::zero()
            || sbs <= F::zero()
        {
            return false;
        }
        self.rank_one(y, F::one() / sy, w);
        self.rank_one(bs, -F::one() / sbs, w);
        (0..self.n).all(|i| self.lower(i, i) > F::zero())
            && self.packed.iter().all(|v| v.is_finite())
    }
    fn rank_one(&mut self, z: &mut [F], sigma: F, w: &mut Vec<F>) {
        let mut t = F::one() / sigma;
        let mut ij = 0;
        if sigma < F::zero() {
            w.clear();
            w.extend_from_slice(z);
            for i in 0..self.n {
                let v = w[i];
                t = t + v * v / self.packed[ij];
                for j in i + 1..self.n {
                    ij += 1;
                    w[j] = w[j] - v * self.packed[ij];
                }
                ij += 1;
            }
            if t >= F::zero() {
                t = F::epsilon() / sigma;
            }
            for j in (0..self.n).rev() {
                ij -= self.n - j;
                let u = w[j];
                w[j] = t;
                t = t - u * u / self.packed[ij];
            }
        }
        for i in 0..self.n {
            let v = z[i];
            let delta = v / self.packed[ij];
            let tp = if sigma < F::zero() {
                w[i]
            } else {
                t + delta * v
            };
            let alpha = tp / t;
            self.packed[ij] = alpha * self.packed[ij];
            let beta = delta / tp;
            for j in i + 1..self.n {
                ij += 1;
                let old = self.packed[ij];
                let original = z[j];
                z[j] = z[j] - v * old;
                self.packed[ij] = if alpha > number(4.0) {
                    // This ordering avoids cancellation when the diagonal grows.
                    (t / tp) * old + beta * original
                } else {
                    old + beta * z[j]
                };
            }
            ij += 1;
            t = tp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn damped_update_stays_positive_and_satisfies_modified_secant() {
        let mut factor = Factor::<f64>::identity(2);
        let mut bs = Vec::new();
        assert!(factor.update(
            &[1., 0.],
            &mut [-1., 1.],
            &mut bs,
            &mut Vec::new()
        ));
        factor.product(&[1., 0.], &mut bs);
        assert!((bs[0] - 0.2).abs() < 1e-12);
        assert!((bs[1] - 0.4).abs() < 1e-12);
        let mut e = Matrix::default();
        factor.least_squares(&[0., 0.], None, &mut e, &mut Vec::new());
        for v in [[1., 2.], [-3., 1.], [0., 1.]] {
            let mut b = [0.0; 2];
            factor.product(&v, &mut b);
            let ev: Vec<f64> = (0..2)
                .map(|i| (0..2).map(|j| e.get(i, j) * v[j]).sum())
                .collect();
            assert!((dot(&v, &b) - dot(&ev, &ev)).abs() < 1e-12);
        }
    }
}
