//! Sparse impls of the `linalg` tier for
//! [`nalgebra_sparse::CscMatrix<f64>`] (CSC) over
//! [`nalgebra::DVector<f64>`]. Lands in S2b alongside the dense
//! nalgebra backend.
//!
//! nalgebra-sparse covers four of the five `linalg` traits: SpMV /
//! Aᵀ-SpMV via [`spmm_csc_dense`], Gram via the `&CscMatrix *
//! &CscMatrix` operator overload (composing transpose + spmm under
//! the hood), and SPD solve via
//! [`CscCholesky`](nalgebra_sparse::factorization::CscCholesky).
//! The fifth — `LinearSolveLstsq` — is **deliberately not
//! implemented** here: nalgebra-sparse 0.10 doesn't ship a sparse QR.
//! Reach for the faer-sparse backend if you need least-squares on
//! sparse `J`.

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::factorization::CscCholesky;
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::ops::serial::spmm_csc_dense;
use nalgebra_sparse::{CscMatrix, SparseEntryMut};

use super::Scalar;
use super::linalg::{
    AddDiagonalVectorInPlace, GramMatrix, LinearSolveError, LinearSolveSpd, MatDiagonal,
    MatTransposeVec, MatVec, MaxDiagonal,
};

// Bound stack mirrors the dense nalgebra backend: matvec / gram routes go
// through nalgebra's `Closed*Assign + Zero + One` BLAS-2 tier, the
// diagonal-walk impls stay on basic `Scalar`, and the Cholesky factor
// step bounds on `RealField`. f32 and f64 satisfy every variant.

impl<F> MatVec<DVector<F>> for CscMatrix<F>
where
    F: Scalar + nalgebra::ClosedAddAssign + nalgebra::ClosedMulAssign,
{
    fn matvec(&self, x: &DVector<F>) -> DVector<F> {
        assert_eq!(
            self.ncols(),
            x.len(),
            "matvec: A.ncols ({}) != x.len ({})",
            self.ncols(),
            x.len()
        );
        // Could call `&self * x` (the operator overload), but that adds
        // ClosedSubAssign + ClosedDivAssign + Neg to the bound list. Going
        // through spmm_csc_dense directly keeps the bound set to the two
        // accumulator traits, mirroring mat_transpose_vec below.
        let mut y = DMatrix::<F>::zeros(self.nrows(), 1);
        let x_mat = DMatrix::from_column_slice(x.len(), 1, x.as_slice());
        spmm_csc_dense(
            F::zero(),
            &mut y,
            F::one(),
            Op::NoOp(self),
            Op::NoOp(&x_mat),
        );
        DVector::from_column_slice(y.column(0).as_slice())
    }
}

impl<F> MatTransposeVec<DVector<F>> for CscMatrix<F>
where
    F: Scalar + nalgebra::ClosedAddAssign + nalgebra::ClosedMulAssign,
{
    fn mat_transpose_vec(&self, x: &DVector<F>) -> DVector<F> {
        assert_eq!(
            self.nrows(),
            x.len(),
            "mat_transpose_vec: A.nrows ({}) != x.len ({})",
            self.nrows(),
            x.len()
        );
        // spmm_csc_dense with Op::Transpose lets us avoid materializing
        // Aᵀ. Output dimension is `ncols(self) × 1`; the helper takes
        // dense RHS as a `DMatrixView`, so we wrap `x` as a 1-column
        // DMatrix.
        let mut y = DMatrix::<F>::zeros(self.ncols(), 1);
        let x_mat = DMatrix::from_column_slice(x.len(), 1, x.as_slice());
        spmm_csc_dense(
            F::zero(),
            &mut y,
            F::one(),
            Op::Transpose(self),
            Op::NoOp(&x_mat),
        );
        DVector::from_column_slice(y.column(0).as_slice())
    }
}

impl<F> GramMatrix for CscMatrix<F>
where
    F: Scalar
        + nalgebra::ClosedAddAssign
        + nalgebra::ClosedSubAssign
        + nalgebra::ClosedMulAssign
        + nalgebra::ClosedDivAssign
        + std::ops::Neg<Output = F>,
{
    fn gram(&self) -> Self {
        // The `&CscMatrix * &CscMatrix` operator overload composes
        // pattern construction + spmm. Aᵀ A → CSC of shape
        // `(ncols, ncols)`; transpose() materializes Aᵀ as CSC. The
        // overload's bound list is fatter than the MatVec / MatTranspose
        // ones because nalgebra-sparse uses it for the general
        // (in)equality-checked matrix algebra surface, not just SPD ops.
        &self.transpose() * self
    }
}

impl<F: Scalar> MaxDiagonal<F> for CscMatrix<F> {
    fn max_diagonal(&self) -> F {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "max_diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        // Implicit-zero entries contribute 0.0 to the comparison.
        (0..self.nrows())
            .map(|i| {
                self.get_entry(i, i)
                    .expect("max_diagonal: index in bounds")
                    .into_value()
            })
            .fold(F::neg_infinity(), F::max)
    }
}

impl<F: Scalar> MatDiagonal<DVector<F>> for CscMatrix<F> {
    fn diagonal(&self) -> DVector<F> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        // Diagonal entries missing from the CSC pattern are the implicit
        // zero — same contract as `max_diagonal`.
        DVector::from_iterator(
            self.nrows(),
            (0..self.nrows()).map(|i| {
                self.get_entry(i, i)
                    .expect("diagonal: index in bounds")
                    .into_value()
            }),
        )
    }
}

impl<F: Scalar> AddDiagonalVectorInPlace<DVector<F>> for CscMatrix<F> {
    fn add_diagonal_vector_in_place(&mut self, diag: &DVector<F>) {
        let n = self.nrows();
        assert_eq!(
            n,
            self.ncols(),
            "add_diagonal_vector_in_place: matrix must be square, got {}x{}",
            n,
            self.ncols()
        );
        assert_eq!(
            n,
            diag.len(),
            "add_diagonal_vector_in_place: matrix is {}x{} but diag has length {}",
            n,
            self.ncols(),
            diag.len()
        );
        for i in 0..n {
            match self
                .get_entry_mut(i, i)
                .expect("add_diagonal_vector_in_place: index in bounds")
            {
                SparseEntryMut::NonZero(v) => *v = *v + diag[i],
                SparseEntryMut::Zero => panic!(
                    "add_diagonal_vector_in_place: diagonal entry ({i}, {i}) missing from CSC pattern"
                ),
            }
        }
    }
}

impl<F> LinearSolveSpd<DVector<F>> for CscMatrix<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn solve_spd(&self, b: &DVector<F>) -> Result<DVector<F>, LinearSolveError> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "solve_spd: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            b.len(),
            "solve_spd: A.nrows ({}) != b.len ({})",
            self.nrows(),
            b.len()
        );
        // CscCholesky::solve takes a `DMatrixView` and returns a
        // `DMatrix`, so we round-trip the DVector through a 1-column
        // dense matrix. One small allocation per solve; the Cholesky
        // factorization itself dominates cost.
        let chol = CscCholesky::factor(self).map_err(|_| LinearSolveError::NotPositiveDefinite)?;
        let b_mat = DMatrix::from_column_slice(b.len(), 1, b.as_slice());
        let x_mat = chol.solve(&b_mat);
        Ok(DVector::from_column_slice(x_mat.column(0).as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::CooMatrix;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// 2×2 dense matrix expressed as CSC via COO triplets.
    fn csc2(row0: [f64; 2], row1: [f64; 2]) -> CscMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, row0[0]);
        coo.push(0, 1, row0[1]);
        coo.push(1, 0, row1[0]);
        coo.push(1, 1, row1[1]);
        CscMatrix::from(&coo)
    }

    #[test]
    fn matvec_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.matvec(&x);
        assert_eq!(y.len(), 2);
        assert!(approx_eq(y[0], 17.0, 1e-12));
        assert!(approx_eq(y[1], 39.0, 1e-12));
    }

    #[test]
    fn mat_transpose_vec_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.mat_transpose_vec(&x);
        assert_eq!(y.len(), 2);
        // Aᵀ x = [1·5 + 3·6, 2·5 + 4·6] = [23, 34]
        assert!(approx_eq(y[0], 23.0, 1e-12));
        assert!(approx_eq(y[1], 34.0, 1e-12));
    }

    #[test]
    fn gram_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let g = a.gram();
        // AᵀA = [[10, 14], [14, 20]]
        assert_eq!(g.nrows(), 2);
        assert_eq!(g.ncols(), 2);
        let e0 = DVector::from_vec(vec![1.0, 0.0]);
        let e1 = DVector::from_vec(vec![0.0, 1.0]);
        let col0 = g.matvec(&e0);
        let col1 = g.matvec(&e1);
        assert!(approx_eq(col0[0], 10.0, 1e-12));
        assert!(approx_eq(col0[1], 14.0, 1e-12));
        assert!(approx_eq(col1[0], 14.0, 1e-12));
        assert!(approx_eq(col1[1], 20.0, 1e-12));
    }

    #[test]
    fn solve_spd_happy_path() {
        let a = csc2([4.0, 1.0], [1.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x = a.solve_spd(&b).expect("SPD system must solve");
        // Same hand-computed answer as dense: x = [1/11, 7/11].
        assert!(approx_eq(x[0], 1.0 / 11.0, 1e-12));
        assert!(approx_eq(x[1], 7.0 / 11.0, 1e-12));
    }

    #[test]
    fn solve_spd_indefinite_returns_error() {
        let a = csc2([1.0, 2.0], [2.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = a.solve_spd(&b).expect_err("indefinite must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn gram_of_rank_deficient_is_singular() {
        let a = csc2([1.0, 2.0], [2.0, 4.0]);
        let g = a.gram();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = g.solve_spd(&b).expect_err("rank-deficient gram must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn add_diagonal_regularizes_singular_gram() {
        let a = csc2([1.0, 2.0], [2.0, 4.0]);
        let mut g = a.gram();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        assert!(g.clone().solve_spd(&b).is_err());
        g.add_diagonal_vector_in_place(&DVector::from_vec(vec![1e-3, 1e-3]));
        let x = g.solve_spd(&b).expect("damped gram must be SPD");
        assert_eq!(x.len(), 2);
    }

    #[test]
    fn add_diagonal_vector_in_place_adds_per_index() {
        let mut a = csc2([1.0, 2.0], [3.0, 4.0]);
        a.add_diagonal_vector_in_place(&DVector::from_vec(vec![10.0, 100.0]));
        // Original [[1,2],[3,4]] + diag(10, 100) → [[11,2],[3,104]].
        let e0 = DVector::from_vec(vec![1.0, 0.0]);
        let e1 = DVector::from_vec(vec![0.0, 1.0]);
        let col0 = a.matvec(&e0);
        let col1 = a.matvec(&e1);
        assert!(approx_eq(col0[0], 11.0, 1e-12));
        assert!(approx_eq(col0[1], 3.0, 1e-12));
        assert!(approx_eq(col1[0], 2.0, 1e-12));
        assert!(approx_eq(col1[1], 104.0, 1e-12));
    }
}
