//! Shared Householder factorization and diagonal regularization for dense QR.

use super::{QrSolveError, RegularizedQrSolve, Scalar};

/// Retained triangular factor, pivot permutation, and transformed RHS.
///
/// Obtain this through [`FactorizePivotedQr`](super::FactorizePivotedQr).
/// Backends use their native Householder QR where available, then share this
/// regularization kernel. Storage is `O(n²)` after factorizing an `m × n`
/// Jacobian; no full square `Q` or Gram matrix is retained.
///
/// Diagonal regularization follows the Givens-rotation construction in
/// MINPACK's [`qrsolv`](https://netlib.org/minpack/qrsolv.f), with an explicit
/// numerical-rank check instead of its truncated-solution policy.
#[derive(Clone, Debug)]
pub struct QrFactorization<F: Scalar = f64> {
    rows: usize,
    r: Vec<F>,
    qtb: Vec<F>,
    permutation: Vec<usize>,
    norms: Vec<F>,
}

pub(super) fn column_norms<F: Scalar>(
    rows: usize,
    cols: usize,
    mut entry: impl FnMut(usize, usize) -> F,
) -> Result<Vec<F>, QrSolveError> {
    assert!(
        rows > 0 && cols > 0,
        "pivoted QR requires a nonempty matrix"
    );
    let mut norms = vec![F::zero(); cols];
    for (j, norm) in norms.iter_mut().enumerate() {
        // Scaled accumulation retains tiny components without a hypot per entry.
        *norm = super::norm_l2((0..rows).map(|i| entry(i, j)));
        // LM's unchanged scaling and gradient tests use squared norms.
        if !(*norm * *norm).is_finite() {
            return Err(QrSolveError::NonFinite);
        }
    }
    Ok(norms)
}

impl<F: Scalar> QrFactorization<F> {
    pub(super) fn from_parts(
        rows: usize,
        r: Vec<F>,
        qtb: Vec<F>,
        permutation: Vec<usize>,
        norms: Vec<F>,
    ) -> Result<Self, QrSolveError> {
        if r.iter().chain(&qtb).any(|x| !x.is_finite()) {
            return Err(QrSolveError::NonFinite);
        }
        Ok(Self {
            rows,
            r,
            qtb,
            permutation,
            norms,
        })
    }

    pub(super) fn norms_squared(&self) -> Vec<F> {
        self.norms.iter().map(|&x| x * x).collect()
    }

    pub(super) fn solve(
        &self,
        mu: F,
        diagonal: &[F],
        rank_tolerance: Option<F>,
    ) -> Result<Vec<F>, QrSolveError> {
        let n = self.norms.len();
        assert_eq!(diagonal.len(), n, "QR damping diagonal length mismatch");
        let tolerance = rank_tolerance.unwrap_or_else(|| {
            F::epsilon() * F::from_usize(self.rows + n).unwrap()
        });
        if !mu.is_finite()
            || !tolerance.is_finite()
            || diagonal.iter().any(|x| !x.is_finite())
        {
            return Err(QrSolveError::NonFinite);
        }
        if mu < F::zero()
            || tolerance < F::zero()
            || tolerance >= F::one()
            || diagonal.iter().any(|&x| x < F::zero())
        {
            return Err(QrSolveError::InvalidRegularization);
        }
        if mu == F::zero() {
            return self.solve_undamped(tolerance);
        }
        let mut r = self.r.clone();
        let mut rhs = self.qtb.clone();
        let mut scales = vec![F::zero(); n];
        let mut damping = vec![F::zero(); n];
        for j in 0..n {
            let original = self.permutation[j];
            let reg = mu.sqrt() * diagonal[original].sqrt();
            let scale = self.norms[original].hypot(reg);
            if !scale.is_finite() {
                return Err(QrSolveError::NonFinite);
            }
            if scale == F::zero() {
                return Err(QrSolveError::RankDeficient);
            }
            scales[j] = scale;
            damping[j] = reg / scale;
            for i in 0..=j {
                r[i * n + j] = r[i * n + j] / scale;
            }
        }
        // Equilibration changes variables, not the regularization objective.
        // Eliminating one diagonal row at a time avoids an (m+n)-row refactor.
        let mut row = vec![F::zero(); n];
        for j in 0..n {
            row.fill(F::zero());
            row[j] = damping[j];
            let mut tail = F::zero();
            for k in j..n {
                if row[k] == F::zero() {
                    continue;
                }
                let norm = r[k * n + k].hypot(row[k]);
                let c = r[k * n + k] / norm;
                let s = row[k] / norm;
                r[k * n + k] = norm;
                let old = rhs[k];
                rhs[k] = c * old + s * tail;
                tail = -s * old + c * tail;
                for i in k + 1..n {
                    let old = r[k * n + i];
                    r[k * n + i] = c * old + s * row[i];
                    row[i] = -s * old + c * row[i];
                }
            }
        }
        if r.iter().chain(&rhs).any(|x| !x.is_finite()) {
            return Err(QrSolveError::NonFinite);
        }
        if (0..n).any(|j| r[j * n + j].abs() <= tolerance) {
            return Err(QrSolveError::RankDeficient);
        }
        let mut x = vec![F::zero(); n];
        for j in (0..n).rev() {
            let mut value = rhs[j];
            for k in j + 1..n {
                value = value - r[j * n + k] * rhs[k];
            }
            rhs[j] = value / r[j * n + j];
            x[self.permutation[j]] = rhs[j] / scales[j];
        }
        if x.iter().any(|x| !x.is_finite()) {
            return Err(QrSolveError::NonFinite);
        }
        Ok(x)
    }

    fn solve_undamped(&self, tolerance: F) -> Result<Vec<F>, QrSolveError> {
        let n = self.norms.len();
        if self.norms.contains(&F::zero())
            || (0..n).any(|j| {
                (self.r[j * n + j] / self.norms[self.permutation[j]]).abs()
                    <= tolerance
            })
        {
            return Err(QrSolveError::RankDeficient);
        }
        // The undamped factor already is triangular. Apply the same column
        // scaling during substitution without copying or rotating the factor.
        let mut x = vec![F::zero(); n];
        for j in (0..n).rev() {
            let original = self.permutation[j];
            let mut value = self.qtb[j];
            for k in j + 1..n {
                let column = self.permutation[k];
                value = value
                    - (self.r[j * n + k] / self.norms[column]) * x[column];
            }
            x[original] = value / (self.r[j * n + j] / self.norms[original]);
        }
        for (value, &scale) in x.iter_mut().zip(&self.norms) {
            *value = *value / scale;
        }
        if x.iter().any(|x| !x.is_finite()) {
            return Err(QrSolveError::NonFinite);
        }
        Ok(x)
    }
}

impl<F: Scalar> RegularizedQrSolve<Vec<F>, F> for QrFactorization<F> {
    fn column_norms_squared(&self) -> Vec<F> {
        self.norms_squared()
    }
    fn solve_regularized(
        &self,
        mu: F,
        diagonal: &Vec<F>,
        rank_tolerance: Option<F>,
    ) -> Result<Vec<F>, QrSolveError> {
        self.solve(mu, diagonal, rank_tolerance)
    }
}

pub(super) fn factorize<F: Scalar>(
    rows: usize,
    cols: usize,
    mut a: Vec<F>,
    b: &[F],
) -> Result<QrFactorization<F>, QrSolveError> {
    assert_eq!(b.len(), rows, "pivoted QR right-hand side length mismatch");
    let norms = column_norms(rows, cols, |i, j| a[i * cols + j])?;
    if b.iter().any(|x| !x.is_finite()) {
        return Err(QrSolveError::NonFinite);
    }
    // Equilibrating before pivoting prevents parameter units from deciding the
    // Householder order; unscale R afterward to retain the original objective.
    for i in 0..rows {
        for j in 0..cols {
            if norms[j] > F::zero() {
                a[i * cols + j] = a[i * cols + j] / norms[j];
            }
        }
    }
    let mut rhs = b.to_vec();
    let mut permutation: Vec<_> = (0..cols).collect();
    let mut reflector = vec![F::zero(); rows];
    for k in 0..rows.min(cols) {
        // Recomputing trailing norms avoids unreliable downdates near rank loss.
        let mut pivot = k;
        let mut best = F::zero();
        for j in k..cols {
            let norm = super::norm_l2((k..rows).map(|i| a[i * cols + j]));
            if norm > best {
                best = norm;
                pivot = j;
            }
        }
        if pivot != k {
            for i in 0..rows {
                a.swap(i * cols + k, i * cols + pivot);
            }
            permutation.swap(k, pivot);
        }
        if best == F::zero() {
            continue;
        }
        let sign = if a[k * cols + k] < F::zero() {
            -F::one()
        } else {
            F::one()
        };
        for i in k..rows {
            reflector[i] = a[i * cols + k] / best;
        }
        reflector[k] = reflector[k] + sign;
        let beta = F::one() / (F::one() + sign * a[k * cols + k] / best);
        for j in k + 1..cols {
            let dot =
                (k..rows).map(|i| reflector[i] * a[i * cols + j]).sum::<F>()
                    * beta;
            for i in k..rows {
                a[i * cols + j] = a[i * cols + j] - reflector[i] * dot;
            }
        }
        let dot = (k..rows).map(|i| reflector[i] * rhs[i]).sum::<F>() * beta;
        for i in k..rows {
            rhs[i] = rhs[i] - reflector[i] * dot;
        }
        a[k * cols + k] = -sign * best;
        for i in k + 1..rows {
            a[i * cols + k] = F::zero();
        }
    }
    let mut r = vec![F::zero(); cols * cols];
    for i in 0..rows.min(cols) {
        for j in i..cols {
            r[i * cols + j] = a[i * cols + j] * norms[permutation[j]];
        }
    }
    rhs.resize(cols, F::zero());
    QrFactorization::from_parts(rows, r, rhs, permutation, norms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_column_norms<F: Scalar>() {
        for scale in [
            F::one(),
            F::min_positive_value() / F::from_f64(16.).unwrap(),
            F::max_value().sqrt() / F::from_f64(8.).unwrap(),
        ] {
            for values in [[3., 4., 0.], [0., 4., 3.]] {
                let values = values.map(|x| F::from_f64(x).unwrap() * scale);
                let norms = column_norms(3, 2, |i, j| {
                    if j == 0 { values[i] } else { F::zero() }
                })
                .unwrap();
                let expected = F::from_f64(5.).unwrap() * scale;
                assert!(norms[0] > F::zero());
                assert!(
                    (norms[0] / expected - F::one()).abs()
                        <= F::from_f64(8.).unwrap() * F::epsilon()
                );
                assert_eq!(norms[1], F::zero());
            }
        }
        for invalid in [F::nan(), F::infinity(), F::neg_infinity()] {
            assert_eq!(
                column_norms(2, 1, |i, _| {
                    if i == 0 { F::one() } else { invalid }
                }),
                Err(QrSolveError::NonFinite)
            );
        }
        // A finite norm still cannot feed LM's squared column scaling.
        assert_eq!(
            column_norms(2, 1, |_, _| F::max_value().sqrt()),
            Err(QrSolveError::NonFinite)
        );
    }

    #[test]
    fn f32_column_norms_preserve_scale_and_reject_invalid_models() {
        check_column_norms::<f32>();
    }

    #[test]
    fn f64_column_norms_preserve_scale_and_reject_invalid_models() {
        check_column_norms::<f64>();
    }

    fn check_undamped_solve<F: Scalar>() {
        let norms = vec![
            F::min_positive_value(),
            F::from_f64(3.).unwrap(),
            F::max_value().sqrt() / F::from_f64(4.).unwrap(),
        ];
        let permutation = vec![2, 0, 1];
        let upper = [1., 0.6, 0., 0., 0.8, 0.8, 0., 0., 0.6];
        let r = upper
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                F::from_f64(value).unwrap() * norms[permutation[i % 3]]
            })
            .collect();
        // The equilibrated solution in original column order is (1, -2, 3).
        let qr = QrFactorization::from_parts(
            3,
            r,
            [3.6, -0.8, -1.2].map(|x| F::from_f64(x).unwrap()).to_vec(),
            permutation,
            norms,
        )
        .unwrap();
        let d = vec![F::one(); 3];
        let x = qr.solve(F::zero(), &d, None).unwrap();
        for ((&value, &scale), expected) in
            x.iter().zip(&qr.norms).zip([1., -2., 3.])
        {
            assert!(
                (value * scale - F::from_f64(expected).unwrap()).abs()
                    <= F::from_f64(16.).unwrap() * F::epsilon()
            );
        }
        qr.solve(F::one(), &d, None).unwrap();
        assert_eq!(qr.solve(F::zero(), &d, None).unwrap(), x);

        for diagonal in [F::nan(), F::infinity(), -F::one()] {
            assert!(qr.solve(F::zero(), &[diagonal; 3], None).is_err());
        }
        assert_eq!(
            qr.solve(F::zero(), &d, Some(F::from_f64(0.9).unwrap())),
            Err(QrSolveError::RankDeficient)
        );

        let tiny = F::min_positive_value() * F::epsilon();
        let overflow = QrFactorization::from_parts(
            1,
            vec![tiny],
            vec![F::one()],
            vec![0],
            vec![tiny],
        )
        .unwrap();
        assert_eq!(
            overflow.solve(F::zero(), &[F::one()], None),
            Err(QrSolveError::NonFinite)
        );
    }

    #[test]
    fn f32_undamped_solve_preserves_scaling_rank_and_validation() {
        check_undamped_solve::<f32>();
    }

    #[test]
    fn f64_undamped_solve_preserves_scaling_rank_and_validation() {
        check_undamped_solve::<f64>();
    }
}
