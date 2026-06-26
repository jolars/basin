//! Hand-rolled dense matrix for the `Vec<F>` backend.
//!
//! basin's default param backend is `Vec<F>` (no external crate). The
//! matrix-capable backends (nalgebra, faer) bring their own dense matrix
//! types, but `Vec<F>` had none, so the linear-constraint solvers
//! ([`BarrierMethod`](crate::solver::BarrierMethod),
//! [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)),
//! which only need `A x` and `Aᵀ v`, were a compile-time error on the default
//! backend (tenet 5).
//!
//! [`DenseMatrix`] closes that gap with the two matvec ops plus the handful of
//! dense ops BFGS needs ([`MatrixIdentity`], [`ScaleInPlace`], and the
//! rank-one Hessian update [`GeneralRankOneUpdate`]), so BFGS also runs on the
//! default backend. CMA-ES runs too: [`DenseMatrix`] additionally implements
//! [`RankOneUpdate`](super::RankOneUpdate),
//! [`MatrixFromDiagonal`](super::MatrixFromDiagonal),
//! [`MatDiagonal`](super::MatDiagonal), and (the load-bearing op) a pure-Rust
//! symmetric eigendecomposition [`SymmetricEigen`](super::SymmetricEigen) via a
//! cyclic Jacobi solver (`dense_eig`). The SPD *solve* layer is also here:
//! [`GramMatrix`](super::GramMatrix) (`JᵀJ`),
//! [`AddDiagonalVectorInPlace`](super::AddDiagonalVectorInPlace) (Marquardt
//! damping), and the load-bearing [`LinearSolveSpd`](super::LinearSolveSpd) via
//! a pure-Rust Cholesky (`dense_chol`), so the normal-equations least-squares
//! solvers all run on `Vec<f64>`: Gauss-Newton, Levenberg-Marquardt, and
//! (with the diagonal-scaling op [`MaxDiagonal`](super::MaxDiagonal)) the
//! trust-region-reflective TRF. The whole family is normal-equations
//! (`JᵀJ` + Cholesky), so none of them touch the QR least-squares solve
//! [`LinearSolveLstsq`](super::LinearSolveLstsq); that op stays *not yet*
//! implemented for `DenseMatrix`, but that is a separate, currently-unmotivated
//! "not yet" (tenet 5), not what gated TRF. No solver has motivated a pure-Rust
//! `DenseMatrix` QR yet, but (like the Cholesky and Jacobi solvers above)
//! one would be welcome if it can be done honestly (pure-Rust, wasm-clean, no
//! BLAS/LAPACK).
//!
//! The scalar `F` defaults to `f64` so existing `DenseMatrix` references keep
//! resolving to `DenseMatrix<f64>` unchanged.

use super::Scalar;
use super::{
    AddDiagonalVectorInPlace, GeneralRankOneUpdate, GramMatrix, LinearSolveError, LinearSolveSpd,
    MatDiagonal, MatTransposeVec, MatVec, MatrixFromDiagonal, MatrixIdentity, MaxDiagonal,
    RankOneUpdate, ScaleInPlace, SymmetricEigen, SymmetricEigenError,
};

/// Row-major dense matrix, the matrix companion to `Vec<F>` as the param
/// vector. `F` defaults to `f64` so the type name `DenseMatrix` keeps
/// resolving to `DenseMatrix<f64>` unchanged.
///
/// Storage is row-major (`data[i * cols + j] = A[i, j]`), the natural layout
/// for a linear-constraint matrix where each row is one constraint
/// `aᵢᵀ x ≤ bᵢ`. This is also what makes [`from_row_slice`](Self::from_row_slice)
/// a transparent mirror of `nalgebra::DMatrix::from_row_slice`.
///
/// The type implements [`MatVec`] (`A x`) and [`MatTransposeVec`] (`Aᵀ v`)
/// for the linear-constraint solvers, plus [`MatrixIdentity`],
/// [`ScaleInPlace`], and `GeneralRankOneUpdate` for BFGS, and
/// `RankOneUpdate`, [`MatrixFromDiagonal`], `MatDiagonal`, and a Jacobi
/// [`SymmetricEigen`] for CMA-ES, and the SPD-solve quartet [`GramMatrix`],
/// `AddDiagonalVectorInPlace`, `MaxDiagonal`, and a Cholesky
/// [`LinearSolveSpd`] for the normal-equations least-squares family
/// (Gauss-Newton, Levenberg-Marquardt, TRF); see the module docs for why the
/// QR least-squares ops are deliberately absent.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix<F = f64> {
    /// Row-major entries: `data[i * cols + j] = A[i, j]`.
    data: Vec<F>,
    rows: usize,
    cols: usize,
}

impl<F: Scalar> DenseMatrix<F> {
    /// Build a `rows × cols` matrix from a **row-major** slice (`data[i * cols
    /// + j] = A[i, j]`). Mirrors `nalgebra::DMatrix::from_row_slice`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    pub fn from_row_slice(rows: usize, cols: usize, data: &[F]) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "DenseMatrix::from_row_slice: expected {} entries for a {}×{} matrix, got {}",
            rows * cols,
            rows,
            cols,
            data.len()
        );
        Self {
            data: data.to_vec(),
            rows,
            cols,
        }
    }

    /// Build a `rows × cols` matrix from a per-entry closure `(i, j) -> A[i,
    /// j]`. Mirrors `faer::Mat::from_fn`. The closure is called once per entry
    /// in row-major order.
    pub fn from_fn<G: FnMut(usize, usize) -> F>(rows: usize, cols: usize, mut f: G) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { data, rows, cols }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Read entry `A[i, j]`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows()` or `j >= ncols()`.
    pub fn get(&self, i: usize, j: usize) -> F {
        assert!(
            i < self.rows && j < self.cols,
            "DenseMatrix::get: index ({i}, {j}) out of bounds for a {}×{} matrix",
            self.rows,
            self.cols
        );
        self.data[i * self.cols + j]
    }

    /// Write entry `A[i, j] = value`.
    ///
    /// Companion to [`get`](Self::get) for the in-place, element-level fills the
    /// NEWUOA model core performs on its small dense matrices (the `from_fn` /
    /// `get`-only API can't express a single-entry write).
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows()` or `j >= ncols()`.
    pub(crate) fn set(&mut self, i: usize, j: usize, value: F) {
        assert!(
            i < self.rows && j < self.cols,
            "DenseMatrix::set: index ({i}, {j}) out of bounds for a {}×{} matrix",
            self.rows,
            self.cols
        );
        self.data[i * self.cols + j] = value;
    }

    /// Shared view of row `i` as a contiguous `&[F]` of length `ncols()`.
    ///
    /// Row-major storage makes a row a contiguous slice; the NEWUOA model core
    /// reads interpolation-point displacements (rows of `xpt`) this way.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows()`.
    pub(crate) fn row(&self, i: usize) -> &[F] {
        assert!(
            i < self.rows,
            "DenseMatrix::row: row {i} out of bounds for a {}×{} matrix",
            self.rows,
            self.cols
        );
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

impl<F: Scalar> MatVec<Vec<F>> for DenseMatrix<F> {
    fn matvec(&self, x: &Vec<F>) -> Vec<F> {
        assert_eq!(
            x.len(),
            self.cols,
            "matvec: x has length {} but the matrix has {} columns",
            x.len(),
            self.cols
        );
        let mut y = vec![F::zero(); self.rows];
        for (i, yi) in y.iter_mut().enumerate() {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            *yi = row.iter().zip(x.iter()).map(|(a, xj)| *a * *xj).sum();
        }
        y
    }
}

impl<F: Scalar> MatTransposeVec<Vec<F>> for DenseMatrix<F> {
    fn mat_transpose_vec(&self, x: &Vec<F>) -> Vec<F> {
        assert_eq!(
            x.len(),
            self.rows,
            "mat_transpose_vec: x has length {} but the matrix has {} rows",
            x.len(),
            self.rows
        );
        let mut y = vec![F::zero(); self.cols];
        for (i, &xi) in x.iter().enumerate() {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            for (yj, a) in y.iter_mut().zip(row.iter()) {
                *yj = *yj + *a * xi;
            }
        }
        y
    }
}

impl<F: Scalar> MatrixIdentity for DenseMatrix<F> {
    fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |i, j| if i == j { F::one() } else { F::zero() })
    }
}

impl<F: Scalar> ScaleInPlace<F> for DenseMatrix<F> {
    fn scale_in_place(&mut self, scalar: F) {
        for entry in &mut self.data {
            *entry = *entry * scalar;
        }
    }
}

impl<F: Scalar> GeneralRankOneUpdate<Vec<F>, F> for DenseMatrix<F> {
    fn general_rank_one_update(&mut self, alpha: F, u: &Vec<F>, v: &Vec<F>) {
        assert_eq!(
            self.rows, self.cols,
            "general_rank_one_update: matrix must be square, got {}x{}",
            self.rows, self.cols
        );
        assert_eq!(
            self.rows,
            u.len(),
            "general_rank_one_update: matrix is {}x{} but u has length {}",
            self.rows,
            self.cols,
            u.len()
        );
        assert_eq!(
            self.cols,
            v.len(),
            "general_rank_one_update: matrix is {}x{} but v has length {}",
            self.rows,
            self.cols,
            v.len()
        );
        // self[i, j] ← self[i, j] + α · u[i] · v[j].
        for (i, &ui) in u.iter().enumerate() {
            let au = alpha * ui;
            let row = &mut self.data[i * self.cols..(i + 1) * self.cols];
            for (entry, &vj) in row.iter_mut().zip(v.iter()) {
                *entry = *entry + au * vj;
            }
        }
    }
}

impl<F: Scalar> RankOneUpdate<Vec<F>, F> for DenseMatrix<F> {
    fn rank_one_update(&mut self, alpha: F, v: &Vec<F>) {
        // The symmetric `α·v·vᵀ` case of the general `α·u·vᵀ` update.
        self.general_rank_one_update(alpha, v, v);
    }
}

impl<F: Scalar> MatrixFromDiagonal<Vec<F>> for DenseMatrix<F> {
    fn from_diagonal(diag: &Vec<F>) -> Self {
        let n = diag.len();
        Self::from_fn(n, n, |i, j| if i == j { diag[i] } else { F::zero() })
    }
}

impl<F: Scalar> MatDiagonal<Vec<F>> for DenseMatrix<F> {
    fn diagonal(&self) -> Vec<F> {
        assert_eq!(
            self.rows, self.cols,
            "diagonal: matrix must be square, got {}x{}",
            self.rows, self.cols
        );
        (0..self.rows)
            .map(|i| self.data[i * self.cols + i])
            .collect()
    }
}

impl<F: Scalar> MaxDiagonal<F> for DenseMatrix<F> {
    fn max_diagonal(&self) -> F {
        assert_eq!(
            self.rows, self.cols,
            "max_diagonal: matrix must be square, got {}x{}",
            self.rows, self.cols
        );
        (0..self.rows)
            .map(|i| self.data[i * self.cols + i])
            .fold(F::neg_infinity(), F::max)
    }
}

impl<F: Scalar> SymmetricEigen<Vec<F>> for DenseMatrix<F> {
    fn try_eigh(&self) -> Result<(Self, Vec<F>), SymmetricEigenError> {
        assert_eq!(
            self.rows, self.cols,
            "try_eigh: matrix must be square, got {}x{}",
            self.rows, self.cols
        );
        let n = self.rows;
        let (eigenvalues, eigenvectors) =
            super::dense_eig::jacobi_eigen(&self.data, n).ok_or(SymmetricEigenError::Failed)?;
        // `jacobi_eigen` returns the eigenvectors row-major with column `k` the
        // eigenvector for `eigenvalues[k]`—exactly `DenseMatrix`'s layout.
        let b = Self {
            data: eigenvectors,
            rows: n,
            cols: n,
        };
        Ok((b, eigenvalues))
    }
}

impl<F: Scalar> GramMatrix for DenseMatrix<F> {
    fn gram(&self) -> Self {
        // G = AᵀA: an `n × n` matrix with G[i][j] = Σ_k A[k][i] · A[k][j].
        let n = self.cols;
        let mut data = vec![F::zero(); n * n];
        for col in self.data.chunks_exact(self.cols) {
            // Each row `col` of `A` contributes the rank-one outer product
            // colᵀ·col to `G`.
            for (i, &ci) in col.iter().enumerate() {
                let grow = &mut data[i * n..(i + 1) * n];
                for (gij, &cj) in grow.iter_mut().zip(col.iter()) {
                    *gij = *gij + ci * cj;
                }
            }
        }
        Self {
            data,
            rows: n,
            cols: n,
        }
    }
}

impl<F: Scalar> AddDiagonalVectorInPlace<Vec<F>> for DenseMatrix<F> {
    fn add_diagonal_vector_in_place(&mut self, diag: &Vec<F>) {
        assert_eq!(
            self.rows, self.cols,
            "add_diagonal_vector_in_place: matrix must be square, got {}x{}",
            self.rows, self.cols
        );
        assert_eq!(
            self.rows,
            diag.len(),
            "add_diagonal_vector_in_place: matrix is {}x{} but diag has length {}",
            self.rows,
            self.cols,
            diag.len()
        );
        for (i, &di) in diag.iter().enumerate() {
            let entry = &mut self.data[i * self.cols + i];
            *entry = *entry + di;
        }
    }
}

impl<F: Scalar> LinearSolveSpd<Vec<F>> for DenseMatrix<F> {
    fn solve_spd(&self, b: &Vec<F>) -> Result<Vec<F>, LinearSolveError> {
        assert_eq!(
            self.rows, self.cols,
            "solve_spd: matrix must be square, got {}x{}",
            self.rows, self.cols
        );
        assert_eq!(
            self.rows,
            b.len(),
            "solve_spd: matrix is {}x{} but rhs has length {}",
            self.rows,
            self.cols,
            b.len()
        );
        super::dense_chol::cholesky_solve_spd(&self.data, self.rows, b)
            .ok_or(LinearSolveError::NotPositiveDefinite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-square `2×3` matrix exercises both ops with distinct input and
    /// output lengths and a non-trivial transpose.
    ///
    /// ```text
    /// A = [1 2 3]   x = (1, 1, 1)ᵀ   ⇒ A x = (6, 15)ᵀ
    ///     [4 5 6]
    /// ```
    fn fixture() -> DenseMatrix {
        DenseMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }

    #[test]
    fn shape_and_entry_access() {
        let a = fixture();
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 3);
        assert_eq!(a.get(0, 2), 3.0);
        assert_eq!(a.get(1, 0), 4.0);
    }

    #[test]
    fn set_writes_single_entry() {
        let mut a = fixture();
        a.set(1, 2, 60.0);
        assert_eq!(a.get(1, 2), 60.0);
        // Neighbors untouched.
        assert_eq!(a.get(1, 1), 5.0);
        assert_eq!(a.get(0, 2), 3.0);
    }

    #[test]
    #[should_panic(expected = "DenseMatrix::set")]
    fn set_rejects_out_of_bounds() {
        let mut a = fixture(); // 2×3
        a.set(2, 0, 1.0);
    }

    #[test]
    fn row_exposes_contiguous_row() {
        let a = fixture();
        assert_eq!(a.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(a.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "DenseMatrix::row")]
    fn row_rejects_out_of_bounds() {
        let a = fixture(); // 2 rows
        let _ = a.row(2);
    }

    #[test]
    fn from_fn_matches_from_row_slice() {
        let by_fn = DenseMatrix::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
        assert_eq!(by_fn, fixture());
    }

    #[test]
    fn matvec_computes_a_times_x() {
        let a = fixture();
        let y = a.matvec(&vec![1.0, 1.0, 1.0]);
        assert_eq!(y, vec![6.0, 15.0]);
    }

    #[test]
    fn mat_transpose_vec_computes_a_transpose_times_x() {
        let a = fixture();
        // Aᵀ v with v = (1, 1)ᵀ sums the two rows column-wise: (5, 7, 9).
        let y = a.mat_transpose_vec(&vec![1.0, 1.0]);
        assert_eq!(y, vec![5.0, 7.0, 9.0]);
    }

    /// The two ops must agree on the implicit transpose: `(A x)·v = x·(Aᵀ v)`.
    #[test]
    fn matvec_and_transpose_are_consistent() {
        let a = fixture();
        let x = vec![0.5, -1.0, 2.0];
        let v = vec![3.0, -2.0];

        let ax = a.matvec(&x);
        let atv = a.mat_transpose_vec(&v);

        let lhs: f64 = ax.iter().zip(&v).map(|(p, q)| p * q).sum();
        let rhs: f64 = x.iter().zip(&atv).map(|(p, q)| p * q).sum();
        assert!((lhs - rhs).abs() < 1e-12, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    #[should_panic(expected = "from_row_slice")]
    fn from_row_slice_rejects_wrong_length() {
        let _ = DenseMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "matvec")]
    fn matvec_rejects_length_mismatch() {
        let a = fixture();
        let _ = a.matvec(&vec![1.0, 1.0]); // needs length 3 (ncols)
    }

    #[test]
    #[should_panic(expected = "mat_transpose_vec")]
    fn mat_transpose_vec_rejects_length_mismatch() {
        let a = fixture();
        let _ = a.mat_transpose_vec(&vec![1.0, 1.0, 1.0]); // needs length 2 (nrows)
    }

    #[test]
    fn identity_is_square_with_unit_diagonal() {
        let id: DenseMatrix = MatrixIdentity::identity(3);
        assert_eq!(id.nrows(), 3);
        assert_eq!(id.ncols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(id.get(i, j), if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn scale_in_place_multiplies_every_entry() {
        let mut a = fixture();
        a.scale_in_place(2.0);
        // Original (1,2,3,4,5,6) doubled.
        assert_eq!(
            a,
            DenseMatrix::from_row_slice(2, 3, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        );
    }

    #[test]
    fn general_rank_one_update_symmetric_case() {
        // 2×2 identity + 1·v·vᵀ with v = (1, 2)ᵀ ⇒ [[2, 2], [2, 5]].
        let mut a: DenseMatrix = MatrixIdentity::identity(2);
        let v = vec![1.0, 2.0];
        a.general_rank_one_update(1.0, &v, &v);
        assert_eq!(a, DenseMatrix::from_row_slice(2, 2, &[2.0, 2.0, 2.0, 5.0]));
    }

    #[test]
    fn general_rank_one_update_asymmetric_case() {
        // α·u·vᵀ with α = 2, u = (1, 0)ᵀ, v = (3, 4)ᵀ touches only row 0:
        // [[6, 8], [0, 0]] added to the zero matrix.
        let mut a = DenseMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        a.general_rank_one_update(2.0, &vec![1.0, 0.0], &vec![3.0, 4.0]);
        assert_eq!(a, DenseMatrix::from_row_slice(2, 2, &[6.0, 8.0, 0.0, 0.0]));
    }

    #[test]
    #[should_panic(expected = "general_rank_one_update")]
    fn general_rank_one_update_rejects_non_square() {
        let mut a = fixture(); // 2×3
        a.general_rank_one_update(1.0, &vec![1.0, 1.0], &vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn gram_computes_a_transpose_a() {
        // A = [[1, 2, 3], [4, 5, 6]] ⇒ AᵀA is the 3×3 column-Gram:
        //   [[17, 22, 27], [22, 29, 36], [27, 36, 45]].
        let g = fixture().gram();
        assert_eq!(g.nrows(), 3);
        assert_eq!(g.ncols(), 3);
        assert_eq!(
            g,
            DenseMatrix::from_row_slice(
                3,
                3,
                &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]
            )
        );
    }

    #[test]
    fn gram_is_symmetric_positive_semidefinite_diagonal() {
        // Diagonal entries Gⱼⱼ = ‖A·,ⱼ‖² must be the column norms-squared.
        let g = fixture().gram();
        // column 0 of A is (1, 4) ⇒ 17; column 2 is (3, 6) ⇒ 45.
        assert_eq!(g.get(0, 0), 17.0);
        assert_eq!(g.get(2, 2), 45.0);
        // Symmetry.
        assert_eq!(g.get(0, 2), g.get(2, 0));
    }

    #[test]
    fn add_diagonal_vector_adds_to_diagonal_only() {
        let mut a = DenseMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        a.add_diagonal_vector_in_place(&vec![10.0, 20.0]);
        assert_eq!(
            a,
            DenseMatrix::from_row_slice(2, 2, &[11.0, 2.0, 3.0, 24.0])
        );
    }

    #[test]
    #[should_panic(expected = "add_diagonal_vector_in_place")]
    fn add_diagonal_vector_rejects_non_square() {
        let mut a = fixture(); // 2×3
        a.add_diagonal_vector_in_place(&vec![1.0, 1.0]);
    }

    #[test]
    fn solve_spd_round_trips() {
        // [[4, 1], [1, 3]] x = (1, 2) ⇒ x = (1/11, 7/11).
        let a: DenseMatrix = DenseMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let x = a.solve_spd(&vec![1.0, 2.0]).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-12, "x[1] = {}", x[1]);
    }

    #[test]
    fn solve_spd_rejects_non_positive_definite() {
        // Indefinite matrix ⇒ NotPositiveDefinite.
        let a: DenseMatrix = DenseMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert_eq!(
            a.solve_spd(&vec![1.0, 1.0]),
            Err(LinearSolveError::NotPositiveDefinite)
        );
    }

    /// The damped normal-equations path LM exercises: `gram()`, then
    /// `add_diagonal_vector_in_place`, then `solve_spd`, end to end.
    #[test]
    fn gram_damp_solve_pipeline() {
        // A 3×2 Jacobian with rank 2 ⇒ AᵀA is 2×2 SPD.
        let j: DenseMatrix = DenseMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let mut g = j.gram(); // [[2, 1], [1, 2]]
        assert_eq!(g, DenseMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]));
        g.add_diagonal_vector_in_place(&vec![1.0, 1.0]); // [[3, 1], [1, 3]]
        let x = g.solve_spd(&vec![1.0, 0.0]).unwrap();
        // [[3, 1], [1, 3]] x = (1, 0) ⇒ x = (3/8, -1/8).
        assert!((x[0] - 3.0 / 8.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] + 1.0 / 8.0).abs() < 1e-12, "x[1] = {}", x[1]);
    }
}
