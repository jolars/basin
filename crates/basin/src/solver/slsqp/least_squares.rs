//! Lawson–Hanson constrained least squares, following Kraft's LSEI/LSI/LDP
//! reduction and the original NNLS active-set algorithm. Derived from the
//! Williams 1.6.1 implementation; notices are retained in `COPYRIGHT`.

#![allow(clippy::needless_range_loop)]

use super::SlsqpFailure;
use crate::core::math::Scalar;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct Matrix<F> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<F>,
}

impl<F: Scalar> Matrix<F> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![F::zero(); rows * cols],
        }
    }
    pub fn get(&self, i: usize, j: usize) -> F {
        self.data[i * self.cols + j]
    }
    pub fn set(&mut self, i: usize, j: usize, value: F) {
        self.data[i * self.cols + j] = value;
    }
    fn row(&self, i: usize) -> &[F] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    fn row_mut(&mut self, i: usize) -> &mut [F] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }
    fn column_norm(&self, j: usize, start: usize, end: usize) -> F {
        (start..end).fold(F::zero(), |length, i| length.hypot(self.get(i, j)))
    }
}

pub(super) fn dot<F: Scalar>(x: &[F], y: &[F]) -> F {
    x.iter().zip(y).map(|(&a, &b)| a * b).sum()
}
pub(super) fn norm<F: Scalar>(x: &[F]) -> F {
    x.iter().fold(F::zero(), |a, &b| a.hypot(b))
}
pub(super) fn number<F: Scalar>(x: f64) -> F {
    F::from_f64(x).unwrap()
}

struct Reflection<F> {
    start: usize,
    v: Vec<F>,
    beta: F,
    factor: F,
}
impl<F: Scalar> Reflection<F> {
    fn new(x: &[F], start: usize) -> Self {
        Self::from_tail(x[start..].to_vec(), start)
    }
    fn from_column(a: &Matrix<F>, j: usize, start: usize) -> Self {
        Self::from_tail((start..a.rows).map(|i| a.get(i, j)).collect(), start)
    }
    fn from_tail(mut v: Vec<F>, start: usize) -> Self {
        let scale = v.iter().fold(F::zero(), |a, &b| a.max(b.abs()));
        if scale == F::zero() {
            return Self {
                start,
                v,
                beta: F::zero(),
                factor: F::zero(),
            };
        }
        for value in &mut v {
            *value = *value / scale;
        }
        let length = norm(&v);
        let beta = if v[0] > F::zero() { -length } else { length };
        v[0] = v[0] - beta;
        let factor = F::one() / (beta * v[0]);
        Self {
            start,
            v,
            beta: beta * scale,
            factor,
        }
    }
    fn apply(&self, x: &mut [F]) {
        if self.factor == F::zero() {
            return;
        }
        let alpha = dot(&self.v, &x[self.start..]) * self.factor;
        for (value, &v) in x[self.start..].iter_mut().zip(&self.v) {
            *value = *value + alpha * v;
        }
    }
    fn apply_column(&self, a: &mut Matrix<F>, j: usize) {
        if self.factor == F::zero() {
            return;
        }
        let alpha = self
            .v
            .iter()
            .enumerate()
            .map(|(i, &v)| v * a.get(self.start + i, j))
            .sum::<F>()
            * self.factor;
        for (i, &v) in self.v.iter().enumerate() {
            let row = self.start + i;
            a.set(row, j, a.get(row, j) + alpha * v);
        }
    }
}

/// Original NNLS: maintain a triangular passive set, adding with Householder
/// reflections and deleting with Givens rotations. Re-solving normal equations
/// would square the conditioning in the dual least-distance problem.
fn nnls<F: Scalar>(
    mut a: Matrix<F>,
    mut b: Vec<F>,
    limit: Option<usize>,
) -> Result<(Vec<F>, F), SlsqpFailure> {
    let (m, n) = (a.rows, a.cols);
    let mut x = vec![F::zero(); n];
    let mut index: Vec<_> = (0..n).collect();
    let mut active = 0;
    let mut iterations = 0;
    let limit = limit.unwrap_or(3 * n);
    while active < n && active < m {
        let mut dual = vec![F::zero(); n];
        for &j in &index[active..] {
            dual[j] = (active..m).map(|i| a.get(i, j) * b[i]).sum();
        }
        let mut z;
        loop {
            let mut selected = active;
            let mut largest = F::zero();
            for k in active..n {
                if dual[index[k]] > largest {
                    largest = dual[index[k]];
                    selected = k;
                }
            }
            if largest <= F::zero() {
                return Ok((x, norm(&b[active..])));
            }
            let j = index[selected];
            let h = Reflection::from_column(&a, j, active);
            let upper = a.column_norm(j, 0, active);
            z = b.clone();
            if h.beta.abs() * number(0.01) >= upper * F::epsilon()
                && h.beta != F::zero()
            {
                h.apply(&mut z);
                if z[active] / h.beta > F::zero() {
                    b.clone_from(&z);
                    index.swap(selected, active);
                    for &other in &index[active + 1..] {
                        h.apply_column(&mut a, other);
                    }
                    a.set(active, j, h.beta);
                    for i in active + 1..m {
                        a.set(i, j, F::zero());
                    }
                    active += 1;
                    break;
                }
            }
            dual[j] = F::zero();
        }
        loop {
            for i in (0..active).rev() {
                let rhs = z[i]
                    - (i + 1..active)
                        .map(|k| a.get(i, index[k]) * z[k])
                        .sum::<F>();
                z[i] = rhs / a.get(i, index[i]);
            }
            iterations += 1;
            if iterations > limit {
                return Err(SlsqpFailure::SubproblemIterationLimit);
            }
            let mut alpha = number(2.0);
            let mut remove = 0;
            for k in 0..active {
                let j = index[k];
                if z[k] <= F::zero() {
                    let t = -x[j] / (z[k] - x[j]);
                    if t < alpha {
                        alpha = t;
                        remove = k;
                    }
                }
            }
            if alpha == number(2.0) {
                for k in 0..active {
                    x[index[k]] = z[k];
                }
                break;
            }
            for k in 0..active {
                let j = index[k];
                x[j] = x[j] + alpha * (z[k] - x[j]);
            }
            loop {
                let removed = index[remove];
                x[removed] = F::zero();
                for k in remove + 1..active {
                    let j = index[k];
                    index[k - 1] = j;
                    let (aa, bb) = (a.get(k - 1, j), a.get(k, j));
                    let r = aa.hypot(bb);
                    let (c, s) = if r == F::zero() {
                        (F::zero(), F::one())
                    } else {
                        (aa / r, bb / r)
                    };
                    for col in 0..n {
                        let (u, v) = (a.get(k - 1, col), a.get(k, col));
                        a.set(k - 1, col, c * u + s * v);
                        a.set(k, col, -s * u + c * v);
                    }
                    a.set(k, j, F::zero());
                    let (u, v) = (b[k - 1], b[k]);
                    b[k - 1] = c * u + s * v;
                    b[k] = -s * u + c * v;
                }
                active -= 1;
                index[active] = removed;
                if let Some(k) = (0..active).find(|&k| x[index[k]] <= F::zero())
                {
                    remove = k;
                } else {
                    break;
                }
            }
            z.clone_from(&b);
        }
    }
    Ok((x, norm(&b[active..])))
}

fn ldp<F: Scalar>(
    g: &Matrix<F>,
    h: &[F],
    limit: Option<usize>,
) -> Result<(Vec<F>, Vec<F>), SlsqpFailure> {
    let (m, n) = (g.rows, g.cols);
    if m == 0 {
        return Ok((vec![F::zero(); n], vec![]));
    }
    let mut a = Matrix::zeros(n + 1, m);
    for j in 0..m {
        for i in 0..n {
            a.set(i, j, g.get(j, i));
        }
        a.set(n, j, h[j]);
    }
    let mut b = vec![F::zero(); n + 1];
    b[n] = F::one();
    let (mut dual, residual) = nnls(a, b, limit)?;
    let factor = F::one() - dot(h, &dual);
    if !factor.is_finite() || factor < F::epsilon() || residual <= F::zero() {
        return Err(SlsqpFailure::IncompatibleConstraints);
    }
    for value in &mut dual {
        *value = *value / factor;
    }
    let x = (0..n)
        .map(|i| (0..m).map(|j| g.get(j, i) * dual[j]).sum())
        .collect();
    Ok((x, dual))
}

fn qr<F: Scalar>(e: &mut Matrix<F>, f: &mut [F], pivot: bool) -> Vec<usize> {
    let n = e.cols;
    let mut permutation: Vec<_> = (0..n).collect();
    for k in 0..n {
        if pivot {
            let mut selected = k;
            let mut largest = F::zero();
            for j in k..n {
                let length = e.column_norm(j, k, e.rows);
                if length > largest {
                    largest = length;
                    selected = j;
                }
            }
            permutation.swap(k, selected);
            for i in 0..e.rows {
                let value = e.get(i, k);
                e.set(i, k, e.get(i, selected));
                e.set(i, selected, value);
            }
        }
        let reflection = Reflection::from_column(e, k, k);
        for j in k + 1..n {
            reflection.apply_column(e, j);
        }
        reflection.apply(f);
        e.set(k, k, reflection.beta);
        for i in k + 1..e.rows {
            e.set(i, k, F::zero());
        }
    }
    permutation
}

fn lsi<F: Scalar>(
    mut e: Matrix<F>,
    mut f: Vec<F>,
    mut g: Matrix<F>,
    mut h: Vec<F>,
    limit: Option<usize>,
) -> Result<(Vec<F>, Vec<F>), SlsqpFailure> {
    let n = e.cols;
    let permutation = qr(&mut e, &mut f, g.rows == 0);
    let rank_threshold = if g.rows == 0 {
        F::epsilon().sqrt()
    } else {
        F::epsilon()
    };
    for i in 0..n {
        if !e.get(i, i).is_finite() || e.get(i, i).abs() < rank_threshold {
            return Err(SlsqpFailure::SingularSubproblem);
        }
    }
    for i in 0..g.rows {
        for j in 0..n {
            let previous = (0..j).map(|k| g.get(i, k) * e.get(k, j)).sum::<F>();
            g.set(i, j, (g.get(i, j) - previous) / e.get(j, j));
        }
        h[i] = h[i] - dot(g.row(i), &f[..n]);
    }
    let (mut x, multipliers) = ldp(&g, &h, limit)?;
    for i in (0..n).rev() {
        let previous = (i + 1..n).map(|j| e.get(i, j) * x[j]).sum::<F>();
        x[i] = (x[i] + f[i] - previous) / e.get(i, i);
    }
    let mut ordered = vec![F::zero(); n];
    for i in 0..n {
        ordered[permutation[i]] = x[i];
    }
    Ok((ordered, multipliers))
}

/// Minimize `||E x - f||`, subject to `C x = d` and `G x >= h`.
pub(super) fn lsei<F: Scalar>(
    mut c: Matrix<F>,
    d: &[F],
    mut e: Matrix<F>,
    f: &[F],
    mut g: Matrix<F>,
    h: &[F],
    limit: Option<usize>,
) -> Result<(Vec<F>, Vec<F>), SlsqpFailure> {
    let (n, meq) = (e.cols, c.rows);
    if meq > n {
        return Err(SlsqpFailure::TooManyEqualities);
    }
    if meq == 0 && n > 0 {
        // With no equality elimination, LSI can consume the original matrices.
        // Copying a reduced problem and recovering equality multipliers would
        // only recreate the same problem and compute an unused residual.
        let (x, multipliers) = lsi(e, f.to_vec(), g, h.to_vec(), limit)?;
        if x.iter().chain(&multipliers).any(|v| !v.is_finite()) {
            return Err(SlsqpFailure::SingularSubproblem);
        }
        return Ok((x, multipliers));
    }
    let mut reflections = Vec::with_capacity(meq);
    for i in 0..meq {
        let reflection = Reflection::new(c.row(i), i);
        for j in i + 1..meq {
            reflection.apply(c.row_mut(j));
        }
        for j in 0..e.rows {
            reflection.apply(e.row_mut(j));
        }
        for j in 0..g.rows {
            reflection.apply(g.row_mut(j));
        }
        c.set(i, i, reflection.beta);
        reflections.push(reflection);
    }
    let mut x = vec![F::zero(); n];
    for i in 0..meq {
        let diag = c.get(i, i);
        if !diag.is_finite() || diag.abs() < F::epsilon() {
            return Err(SlsqpFailure::RankDeficientEqualities);
        }
        x[i] = (d[i] - dot(&c.row(i)[..i], &x[..i])) / diag;
    }
    let mut multipliers = vec![F::zero(); meq + g.rows];
    if meq < n {
        let free = n - meq;
        let mut er = Matrix::zeros(e.rows, free);
        let mut gr = Matrix::zeros(g.rows, free);
        let mut fr = f.to_vec();
        let mut hr = h.to_vec();
        for i in 0..e.rows {
            fr[i] = fr[i] - dot(&e.row(i)[..meq], &x[..meq]);
            for j in 0..free {
                er.set(i, j, e.get(i, meq + j));
            }
        }
        for i in 0..g.rows {
            hr[i] = hr[i] - dot(&g.row(i)[..meq], &x[..meq]);
            for j in 0..free {
                gr.set(i, j, g.get(i, meq + j));
            }
        }
        let (xr, lambda) = lsi(er, fr, gr, hr, limit)?;
        x[meq..].copy_from_slice(&xr);
        multipliers[meq..].copy_from_slice(&lambda);
    } else {
        // The equality-determined point must also satisfy the inequalities.
        // Skipping this test can turn an inconsistent QP into false success.
        for i in 0..g.rows {
            let gx = dot(g.row(i), &x);
            let roundoff = number::<F>(100.0)
                * F::epsilon()
                * (F::one() + gx.abs() + h[i].abs());
            if gx < h[i] - roundoff {
                return Err(SlsqpFailure::IncompatibleConstraints);
            }
        }
    }
    let residual: Vec<_> =
        (0..e.rows).map(|i| dot(e.row(i), &x) - f[i]).collect();
    for i in (0..meq).rev() {
        let rhs = (0..e.rows).map(|j| e.get(j, i) * residual[j]).sum::<F>()
            - (0..g.rows)
                .map(|j| g.get(j, i) * multipliers[meq + j])
                .sum::<F>()
            - (i + 1..meq)
                .map(|j| c.get(j, i) * multipliers[j])
                .sum::<F>();
        multipliers[i] = rhs / c.get(i, i);
    }
    for reflection in reflections.iter().rev() {
        reflection.apply(&mut x);
    }
    if x.iter().chain(&multipliers).any(|v| !v.is_finite()) {
        return Err(SlsqpFailure::SingularSubproblem);
    }
    Ok((x, multipliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_equalities_preserves_active_inequality_multipliers() {
        let e = Matrix::<f64> {
            rows: 2,
            cols: 2,
            data: vec![1.0, 0.0, 0.0, 1.0],
        };
        let g = Matrix {
            rows: 1,
            cols: 2,
            data: vec![1.0, 0.0],
        };
        let (x, multipliers) =
            lsei(Matrix::zeros(0, 2), &[], e, &[-1.0, 2.0], g, &[0.0], None)
                .unwrap();
        assert!(x[0].abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
        assert!((multipliers[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_equalities_rejects_singular_and_nonfinite_subproblems() {
        for (diagonal, rhs) in [(0.0, 1.0), (1.0, f64::NAN)] {
            let e = Matrix {
                rows: 1,
                cols: 1,
                data: vec![diagonal],
            };
            assert_eq!(
                lsei(
                    Matrix::zeros(0, 1),
                    &[],
                    e,
                    &[rhs],
                    Matrix::zeros(0, 1),
                    &[],
                    None,
                )
                .unwrap_err(),
                SlsqpFailure::SingularSubproblem,
            );
        }
    }

    #[test]
    fn nnls_deletes_a_passive_variable() {
        let a = Matrix::<f64> {
            rows: 3,
            cols: 2,
            data: vec![1., 1., 1., 2., 1., 3.],
        };
        let (x, _) = nnls(a, vec![1., 0., 0.], None).unwrap();
        assert!((x[0] - 1. / 3.).abs() < 1e-12);
        assert_eq!(x[1], 0.);
    }
    #[test]
    fn least_distance_infeasible() {
        let g = Matrix {
            rows: 2,
            cols: 1,
            data: vec![1., -1.],
        };
        assert_eq!(
            ldp(&g, &[1., 1.], None).unwrap_err(),
            SlsqpFailure::IncompatibleConstraints
        );
    }
    #[test]
    fn equality_and_active_inequality_multipliers() {
        let c = Matrix::<f64> {
            rows: 1,
            cols: 2,
            data: vec![1., 1.],
        };
        let e = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1., 0., 0., 1.],
        };
        let g = Matrix {
            rows: 1,
            cols: 2,
            data: vec![1., 0.],
        };
        let (x, lambda) = lsei(c, &[1.], e, &[0., 2.], g, &[0.], None).unwrap();
        assert!(x[0].abs() < 1e-12 && (x[1] - 1.).abs() < 1e-12);
        assert!(
            (lambda[0] + 1.).abs() < 1e-12 && (lambda[1] - 1.).abs() < 1e-12
        );
    }
}
