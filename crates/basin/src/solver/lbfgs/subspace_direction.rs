//! Dimension-dependent solutions of the existing reduced BFGS model.
//!
//! The fully free inverse identity and small free-coordinate solve follow the
//! approach in sisyfuzz's Apache-2.0 `lbfgsb/model.rs`. Basin retains its own
//! compact-factor validation and projected-step safeguard.

use super::{backend::AsFloatSlice, compact};
use crate::core::math::Scalar;

pub(super) struct Workspace<F: Scalar> {
    pub direction: Vec<F>,
    full: Vec<F>,
    alpha: Vec<F>,
    column: Vec<F>,
    product: Vec<F>,
}

impl<F: Scalar> Default for Workspace<F> {
    fn default() -> Self {
        Self {
            direction: Vec::new(),
            full: Vec::new(),
            alpha: Vec::new(),
            column: Vec::new(),
            product: Vec::new(),
        }
    }
}

impl<F: Scalar> Workspace<F> {
    pub fn reset(&mut self) {
        self.direction.clear();
        self.full.clear();
        self.alpha.clear();
        self.column.clear();
        self.product.clear();
    }

    /// Solve B_FF d = r without changing the caller's reduced right-hand side.
    /// Decline unusable candidates so the caller can run the reference solve.
    #[allow(clippy::too_many_arguments)]
    pub fn try_direction<V: AsFloatSlice<F>>(
        &mut self,
        n: usize,
        ind: &[usize],
        r: &[F],
        ws: &[V],
        wy: &[V],
        sy: &[F],
        wt: &[F],
        m: usize,
        theta: F,
    ) -> bool {
        let col = ws.len();
        let nf = ind.len();
        if nf == 0
            || col == 0
            || (nf != n && nf != 1)
            || !theta.is_finite()
            || theta <= F::zero()
            || !r[..nf].iter().all(|v| v.is_finite())
            || !(0..col)
                .all(|j| sy[j * m + j].is_finite() && sy[j * m + j] > F::zero())
        {
            return false;
        }
        self.direction.resize(nf, F::zero());
        if nf == n {
            // r = -Z'(g + B(z-x)). Apply H to the actual reduced RHS,
            // including the Cauchy displacement, in coordinate order.
            self.full.resize(n, F::zero());
            for (j, &i) in ind.iter().enumerate() {
                self.full[i] = r[j];
            }
            if self.alpha.capacity() < m {
                self.alpha.reserve_exact(m - self.alpha.len());
            }
            self.alpha.resize(col, F::zero());
            for j in (0..col).rev() {
                let s = ws[j].as_float_slice();
                let y = wy[j].as_float_slice();
                let mut dot = F::zero();
                for (s, q) in s.iter().zip(&self.full) {
                    dot = dot + *s * *q;
                }
                let a = dot / sy[j * m + j];
                self.alpha[j] = a;
                for (q, y) in self.full.iter_mut().zip(y) {
                    *q = *q - a * *y;
                }
            }
            for q in &mut self.full {
                *q = *q / theta;
            }
            for j in 0..col {
                let s = ws[j].as_float_slice();
                let y = wy[j].as_float_slice();
                let mut dot = F::zero();
                for (y, q) in y.iter().zip(&self.full) {
                    dot = dot + *y * *q;
                }
                let coefficient = self.alpha[j] - dot / sy[j * m + j];
                for (q, s) in self.full.iter_mut().zip(s) {
                    *q = *q + coefficient * *s;
                }
            }
            for (j, &i) in ind.iter().enumerate() {
                self.direction[j] = self.full[i];
            }
        } else if nf == 1 {
            // One model column and a scalar solve replace the 2m-dimensional
            // solve. The caller selects a history-size crossover separately.
            self.column.resize(2 * m, F::zero());
            self.product.resize(2 * m, F::zero());
            let coordinate = ind[0];
            for a in 0..col {
                self.column[a] = wy[a].as_float_slice()[coordinate];
                self.column[col + a] =
                    theta * ws[a].as_float_slice()[coordinate];
            }
            if compact::bmv(sy, wt, col, m, &self.column, &mut self.product)
                .is_err()
            {
                return false;
            }
            let mut diagonal = theta;
            let mut scale = theta.abs();
            for a in 0..col {
                let first = self.column[a] * self.product[a];
                let second = self.column[col + a] * self.product[col + a];
                diagonal = diagonal - first - second;
                scale = scale + first.abs() + second.abs();
            }
            // Decline curvature dominated by cancellation, not just negative
            // or non-finite diagonals. The reference solve remains available.
            if !diagonal.is_finite()
                || !scale.is_finite()
                || diagonal <= F::epsilon().sqrt() * scale
            {
                return false;
            }
            self.direction[0] = r[0] / diagonal;
        } else {
            return false;
        }

        let mut descent = F::zero();
        for (d, r) in self.direction.iter().zip(r) {
            if !d.is_finite() {
                return false;
            }
            descent = descent + *d * *r;
        }
        descent.is_finite() && descent > F::zero()
    }
}
