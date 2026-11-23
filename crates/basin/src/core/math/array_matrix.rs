//! Dense math on row-major native arrays.

use super::{
    AddDiagonalVectorInPlace, DenseBackend, FactorizePivotedQr,
    GeneralRankOneUpdate, GramMatrix, LinearSolveError, LinearSolveSpd,
    MatDiagonal, MatTransposeVec, MatVec, MatrixFinite, MatrixFromDiagonal,
    MatrixIdentity, MaxDiagonal, QrFactorization, QrSolveError, RankOneUpdate,
    RegularizedQrSolve, Scalar, ScaleInPlace, ScaleRowsInPlace,
    SquareMatrixFromFn, SymmetricEigen, SymmetricEigenError,
};
use std::array::from_fn;

impl<F: Scalar, const N: usize> DenseBackend<F> for [F; N] {
    type Matrix = [[F; N]; N];
}

impl<F: Scalar, const N: usize> SquareMatrixFromFn<F> for [[F; N]; N] {
    fn square_from_fn(
        n: usize,
        mut entry: impl FnMut(usize, usize) -> F,
    ) -> Self {
        assert_eq!(n, N, "matrix dimension must match the array length");
        from_fn(|i| from_fn(|j| entry(i, j)))
    }
}

impl<F: Scalar, const N: usize> MatrixIdentity for [[F; N]; N] {
    fn identity(n: usize) -> Self {
        assert_eq!(n, N, "identity dimension must match the array length");
        Self::from_diagonal(&[F::one(); N])
    }
}

impl<F: Scalar, const N: usize> MatrixFromDiagonal<[F; N]> for [[F; N]; N] {
    fn from_diagonal(diag: &[F; N]) -> Self {
        from_fn(|i| from_fn(|j| if i == j { diag[i] } else { F::zero() }))
    }
}

impl<F: Scalar, const R: usize, const C: usize> MatVec<[F; C], [F; R]>
    for [[F; C]; R]
{
    fn matvec(&self, x: &[F; C]) -> [F; R] {
        from_fn(|i| self[i].iter().zip(x).map(|(&a, &b)| a * b).sum())
    }
}

impl<F: Scalar, const R: usize, const C: usize> MatTransposeVec<[F; C], [F; R]>
    for [[F; C]; R]
{
    fn mat_transpose_vec(&self, x: &[F; R]) -> [F; C] {
        from_fn(|j| {
            self.iter().zip(x).map(|(row, &value)| row[j] * value).sum()
        })
    }
}

impl<F: Scalar, const R: usize, const C: usize> GramMatrix<[[F; C]; C]>
    for [[F; C]; R]
{
    fn gram(&self) -> [[F; C]; C] {
        from_fn(|i| from_fn(|j| self.iter().map(|row| row[i] * row[j]).sum()))
    }
}

impl<F: Scalar, const R: usize, const C: usize> MatrixFinite for [[F; C]; R] {
    fn matrix_is_finite(&self) -> bool {
        self.as_flattened().iter().all(|x| x.is_finite())
    }
}

impl<F: Scalar, const R: usize, const C: usize> ScaleInPlace<F>
    for [[F; C]; R]
{
    fn scale_in_place(&mut self, factor: F) {
        for row in self {
            row.scale_in_place(factor);
        }
    }
}

impl<F: Scalar, const R: usize, const C: usize> ScaleRowsInPlace<F>
    for [[F; C]; R]
{
    fn scale_rows_in_place(&mut self, factors: &[F]) {
        assert_eq!(factors.len(), R, "expected one scale factor per row");
        for (row, &factor) in self.iter_mut().zip(factors) {
            row.scale_in_place(factor);
        }
    }
}

impl<F: Scalar, const N: usize> MatDiagonal<[F; N]> for [[F; N]; N] {
    fn diagonal(&self) -> [F; N] {
        from_fn(|i| self[i][i])
    }
}

impl<F: Scalar, const N: usize> MaxDiagonal<F> for [[F; N]; N] {
    fn max_diagonal(&self) -> F {
        self.iter()
            .enumerate()
            .map(|(i, row)| row[i])
            .fold(F::neg_infinity(), F::max)
    }
}

impl<F: Scalar, const N: usize> AddDiagonalVectorInPlace<[F; N]>
    for [[F; N]; N]
{
    fn add_diagonal_vector_in_place(&mut self, diag: &[F; N]) {
        for (i, (row, &value)) in self.iter_mut().zip(diag).enumerate() {
            row[i] = row[i] + value;
        }
    }
}

impl<F: Scalar, const N: usize> GeneralRankOneUpdate<[F; N], F>
    for [[F; N]; N]
{
    fn general_rank_one_update(&mut self, alpha: F, u: &[F; N], v: &[F; N]) {
        for (row, &left) in self.iter_mut().zip(u) {
            for (entry, &right) in row.iter_mut().zip(v) {
                *entry = *entry + alpha * left * right;
            }
        }
    }
}

impl<F: Scalar, const N: usize> RankOneUpdate<[F; N], F> for [[F; N]; N] {
    fn rank_one_update(&mut self, alpha: F, v: &[F; N]) {
        self.general_rank_one_update(alpha, v, v);
    }
}

impl<F: Scalar, const N: usize> LinearSolveSpd<[F; N]> for [[F; N]; N] {
    fn solve_spd(&self, b: &[F; N]) -> Result<[F; N], LinearSolveError> {
        let mut l = *self;
        let mut x = *b;
        super::dense_chol::cholesky_solve_spd_in_place(
            l.as_flattened_mut(),
            &mut x,
        )
        .ok_or(LinearSolveError::NotPositiveDefinite)?;
        Ok(x)
    }
}

/// Retained pivoted QR factorization with a fixed solution dimension.
#[derive(Clone, Debug)]
pub struct ArrayQrFactorization<const N: usize, F: Scalar = f64>(
    QrFactorization<F>,
);

impl<F: Scalar, const R: usize, const C: usize>
    FactorizePivotedQr<[F; C], F, [F; R]> for [[F; C]; R]
{
    type Factorization = ArrayQrFactorization<C, F>;

    fn factorize_pivoted_qr(
        &self,
        b: &[F; R],
    ) -> Result<Self::Factorization, QrSolveError> {
        super::dense_qr::factorize(R, C, self.as_flattened().to_vec(), b)
            .map(ArrayQrFactorization)
    }
}

impl<F: Scalar, const N: usize> RegularizedQrSolve<[F; N], F>
    for ArrayQrFactorization<N, F>
{
    fn column_norms_squared(&self) -> [F; N] {
        self.0
            .norms_squared()
            .try_into()
            .expect("fixed QR column count")
    }

    fn solve_regularized(
        &self,
        mu: F,
        diagonal: &[F; N],
        tolerance: Option<F>,
    ) -> Result<[F; N], QrSolveError> {
        self.0
            .solve(mu, diagonal, tolerance)
            .map(|x| x.try_into().expect("fixed QR solution dimension"))
    }
}

impl<F: Scalar, const N: usize> SymmetricEigen<[F; N]> for [[F; N]; N] {
    fn try_eigh(&self) -> Result<(Self, [F; N]), SymmetricEigenError> {
        let (values, vectors) =
            super::dense_eig::jacobi_eigen(self.as_flattened(), N)
                .ok_or(SymmetricEigenError::Failed)?;
        if values.iter().chain(&vectors).any(|x| !x.is_finite()) {
            return Err(SymmetricEigenError::Failed);
        }
        Ok((
            from_fn(|i| from_fn(|j| vectors[i * N + j])),
            values.try_into().expect("fixed eigensystem dimension"),
        ))
    }
}
