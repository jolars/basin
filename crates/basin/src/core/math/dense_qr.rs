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
        for i in 0..rows {
            let value = entry(i, j);
            if !value.is_finite() {
                return Err(QrSolveError::NonFinite);
            }
            *norm = norm.hypot(value);
        }
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
            let norm =
                (k..rows).fold(F::zero(), |acc, i| acc.hypot(a[i * cols + j]));
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
