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

impl<F> Default for Matrix<F> {
    fn default() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }
}

impl<F: Scalar> Matrix<F> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![F::zero(); rows * cols],
        }
    }
    pub fn resize_zeroed(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.data.resize(rows * cols, F::zero());
        self.data.fill(F::zero());
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
        norm_iter((start..end).map(|i| self.get(i, j)))
    }
}

pub(super) fn dot<F: Scalar>(x: &[F], y: &[F]) -> F {
    x.iter().zip(y).map(|(&a, &b)| a * b).sum()
}
pub(super) fn norm<F: Scalar>(x: &[F]) -> F {
    norm_iter(x.iter().copied())
}
fn norm_iter<F: Scalar>(mut values: impl Iterator<Item = F>) -> F {
    // The first hypot(0, x) is exactly |x|, including subnormals and infinities.
    let first = values.next().unwrap_or_else(F::zero).abs();
    values.fold(first, |a, b| a.hypot(b))
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
    fn from_column(
        a: &Matrix<F>,
        j: usize,
        start: usize,
        mut v: Vec<F>,
    ) -> Self {
        v.clear();
        v.extend((start..a.rows).map(|i| a.get(i, j)));
        Self::from_tail(v, start)
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
        if v.len() == 1 && scale.is_finite() {
            // A scalar reflection only changes sign. These are exactly the
            // scaled coefficients, without normalization or division.
            let beta = -v[0];
            v[0] = if v[0] > F::zero() {
                number(2.0)
            } else {
                number(-2.0)
            };
            return Self {
                start,
                v,
                beta,
                factor: number(-0.5),
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
            let h = Reflection::from_column(&a, j, active, Vec::new());
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

#[derive(Clone, Debug)]
pub(super) struct Workspace<F> {
    rhs: Vec<F>,
    bounds: Vec<F>,
    permutation: Vec<usize>,
    reflection: Vec<F>,
}

impl<F> Default for Workspace<F> {
    fn default() -> Self {
        Self {
            rhs: Vec::new(),
            bounds: Vec::new(),
            permutation: Vec::new(),
            reflection: Vec::new(),
        }
    }
}

impl<F: Scalar> Workspace<F> {
    fn set_rhs(&mut self, f: &[F], h: &[F]) {
        self.rhs.clear();
        self.rhs.extend_from_slice(f);
        self.bounds.clear();
        self.bounds.extend_from_slice(h);
    }
}

fn qr<F: Scalar>(
    e: &mut Matrix<F>,
    f: &mut [F],
    pivot: bool,
    permutation: &mut Vec<usize>,
    buffer: &mut Vec<F>,
) {
    let n = e.cols;
    permutation.clear();
    permutation.extend(0..n);
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
        let reflection =
            Reflection::from_column(e, k, k, std::mem::take(buffer));
        for j in k + 1..n {
            reflection.apply_column(e, j);
        }
        reflection.apply(f);
        e.set(k, k, reflection.beta);
        for i in k + 1..e.rows {
            e.set(i, k, F::zero());
        }
        *buffer = reflection.v;
    }
}

fn lsi<F: Scalar>(
    e: &mut Matrix<F>,
    g: &mut Matrix<F>,
    limit: Option<usize>,
    workspace: &mut Workspace<F>,
) -> Result<(Vec<F>, Vec<F>), SlsqpFailure> {
    let n = e.cols;
    let Workspace {
        rhs,
        bounds,
        permutation,
        reflection,
    } = workspace;
    let (f, h) = (rhs, bounds);
    qr(e, f, g.rows == 0, permutation, reflection);
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
    let (mut x, multipliers) = ldp(g, h, limit)?;
    for i in (0..n).rev() {
        let previous = (i + 1..n).map(|j| e.get(i, j) * x[j]).sum::<F>();
        x[i] = (x[i] + f[i] - previous) / e.get(i, i);
    }
    for i in 0..n {
        // Permutation cycles restore variable order without a second solution.
        while permutation[i] != i {
            let j = permutation[i];
            x.swap(i, j);
            permutation.swap(i, j);
        }
    }
    Ok((x, multipliers))
}

/// Minimize `||E x - f||`, subject to `C x = d` and `G x >= h`.
// Keep the owning entry point for standalone kernel tests and the benchmark probe.
#[allow(dead_code)]
pub(super) fn lsei<F: Scalar>(
    mut c: Matrix<F>,
    d: &[F],
    mut e: Matrix<F>,
    f: &[F],
    mut g: Matrix<F>,
    h: &[F],
    limit: Option<usize>,
) -> Result<(Vec<F>, Vec<F>), SlsqpFailure> {
    Lsei {
        c: &mut c,
        d,
        e: &mut e,
        f,
        g: &mut g,
        h,
    }
    .solve(limit, &mut Workspace::default())
}

pub(super) struct Lsei<'a, F> {
    pub c: &'a mut Matrix<F>,
    pub d: &'a [F],
    pub e: &'a mut Matrix<F>,
    pub f: &'a [F],
    pub g: &'a mut Matrix<F>,
    pub h: &'a [F],
}

impl<F: Scalar> Lsei<'_, F> {
    pub fn solve(
        self,
        limit: Option<usize>,
        workspace: &mut Workspace<F>,
    ) -> Result<(Vec<F>, Vec<F>), SlsqpFailure> {
        let Self { c, d, e, f, g, h } = self;
        let (n, meq) = (e.cols, c.rows);
        if meq > n {
            return Err(SlsqpFailure::TooManyEqualities);
        }
        if meq == 0 && n > 0 {
            // With no equality elimination, LSI can consume the original matrices.
            // Copying a reduced problem and recovering equality multipliers would
            // only recreate the same problem and compute an unused residual.
            workspace.set_rhs(f, h);
            let (x, multipliers) = lsi(e, g, limit, workspace)?;
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
            workspace.set_rhs(f, h);
            let (fr, hr) = (&mut workspace.rhs, &mut workspace.bounds);
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
            let (xr, lambda) = lsi(&mut er, &mut gr, limit, workspace)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norms_preserve_extreme_scales_and_nonfinite_values() {
        for values in [
            vec![],
            vec![-0.0],
            vec![-3.0, 4.0],
            vec![3e200, -4e200],
            vec![3e-200, -4e-200],
            vec![f64::from_bits(1)],
            vec![-f64::MAX],
            vec![f64::NAN],
            vec![f64::NAN, f64::INFINITY],
            vec![f64::NEG_INFINITY, f64::NAN],
        ] {
            let expected = values.iter().fold(0.0_f64, |a, &b| a.hypot(b));
            let mut matrix = Matrix::zeros(values.len() + 2, 2);
            matrix.set(0, 1, f64::NAN);
            matrix.set(values.len() + 1, 1, f64::NAN);
            for (i, &value) in values.iter().enumerate() {
                matrix.set(i + 1, 1, value);
            }
            for result in
                [norm(&values), matrix.column_norm(1, 1, values.len() + 1)]
            {
                if expected.is_nan() {
                    assert!(result.is_nan());
                } else {
                    assert_eq!(result.to_bits(), expected.to_bits());
                }
            }
        }
        assert_eq!(norm(&[-3.0_f32, 4.0]), 5.0);
        assert_eq!(norm(&[f32::from_bits(1)]), f32::from_bits(1));
        assert_eq!(norm(&[-f32::MAX]), f32::MAX);
    }

    #[test]
    fn scalar_reflections_preserve_extreme_scales() {
        for value in [1.0, 1e-200, 1e200, f64::from_bits(1), f64::MAX] {
            for value in [value, -value] {
                let reflection = Reflection::new(&[0.0, value], 1);
                assert_eq!(reflection.beta.to_bits(), (-value).to_bits());
                assert_eq!(reflection.factor, -0.5);
                assert_eq!(reflection.v, vec![2.0 * value.signum()]);
                let mut rhs = [7.0, -3.0];
                reflection.apply(&mut rhs);
                assert_eq!(rhs, [7.0, 3.0]);
            }
        }
        for value in [0.0, -0.0] {
            let reflection = Reflection::new(&[value], 0);
            assert_eq!(reflection.factor, 0.0);
            let mut rhs = [-3.0];
            reflection.apply(&mut rhs);
            assert_eq!(rhs, [-3.0]);
        }
        for value in [f32::from_bits(1), -f32::MAX] {
            let reflection = Reflection::new(&[value], 0);
            assert_eq!(reflection.beta, -value);
            assert_eq!(reflection.factor, -0.5);
            assert_eq!(reflection.v, vec![2.0 * value.signum()]);
        }
    }

    #[test]
    fn reused_workspace_reorders_pivots_and_recovers_after_failure() {
        let mut workspace = Workspace::default();
        for n in [4, 1, 3, 2, 4] {
            let mut e = Matrix::zeros(n, n);
            let expected: Vec<_> = (0..n).map(|i| i as f64 - 2.0).collect();
            for i in 0..n {
                for j in i..n {
                    e.set(i, j, if i == j { (i + 1) as f64 } else { 0.25 });
                }
            }
            let f: Vec<_> = (0..n).map(|i| dot(e.row(i), &expected)).collect();
            let (x, multipliers) = Lsei {
                c: &mut Matrix::zeros(0, n),
                d: &[],
                e: &mut e,
                f: &f,
                g: &mut Matrix::zeros(0, n),
                h: &[],
            }
            .solve(None, &mut workspace)
            .unwrap();
            assert!(multipliers.is_empty());
            for (x, expected) in x.iter().zip(expected) {
                assert!((x - expected).abs() < 1e-12);
            }
            let result = Lsei {
                c: &mut Matrix::zeros(0, n),
                d: &[],
                e: &mut Matrix::zeros(n, n),
                f: &f,
                g: &mut Matrix::zeros(0, n),
                h: &[],
            }
            .solve(None, &mut workspace);
            assert_eq!(result.unwrap_err(), SlsqpFailure::SingularSubproblem);
        }
    }

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
        for (diagonal, rhs) in [
            (0.0, 1.0),
            (1.0, f64::NAN),
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
        ] {
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
