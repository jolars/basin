// Sparse impls of the `linalg` tier for
// [`faer::sparse::SparseColMat<usize, f64>`] (CSC) over
// [`faer::Col<f64>`]. Lands in S2b alongside the dense faer backend.
//
// Faer's sparse stack covers all five `linalg` traits: SpMV via
// [`sparse_dense_matmul`], Aᵀ-SpMV via a CSC column walk, Gram via
// [`sparse_sparse_matmul`], SPD solve via supernodal/simplicial
// Cholesky ([`SparseColMat::sp_cholesky`]), and least-squares solve
// via sparse QR ([`SparseColMat::sp_qr`]). The QR path is the only
// `LinearSolveLstsq` implementor in basin today; nalgebra-sparse
// doesn't ship sparse QR.

use faer::linalg::solvers::{Solve, SolveLstsq};
use faer::sparse::SparseColMat;
use faer::sparse::linalg::LltError;
use faer::sparse::linalg::matmul::{sparse_dense_matmul, sparse_sparse_matmul};
use faer::{Accum, Col, Par, Side};

use super::Scalar;
use super::linalg::{
    AddDiagonalVectorInPlace, GramMatrix, LinearSolveError, LinearSolveLstsq,
    LinearSolveSpd, MatDiagonal, MatTransposeVec, MatVec, MaxDiagonal,
};

// As with the dense faer backend, sparse matmul/Cholesky/QR all
// require `faer_traits::ComplexField`, while the diagonal-walk impls
// stay on plain `Scalar`. f32 and f64 satisfy both halves; the sparse
// SPD/lstsq factorizations additionally bound on `RealField`-like
// behavior through `ComplexField` (faer's hierarchy folds the real-
// scalar case into ComplexField directly).

impl<F> MatVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn matvec(&self, x: &Col<F>) -> Col<F> {
        assert_eq!(
            self.ncols(),
            x.nrows(),
            "matvec: A.ncols ({}) != x.nrows ({})",
            self.ncols(),
            x.nrows()
        );
        let mut y = Col::<F>::zeros(self.nrows());
        sparse_dense_matmul(
            y.as_mat_mut(),
            Accum::Replace,
            self.as_ref(),
            x.as_mat(),
            F::one(),
            Par::Seq,
        );
        y
    }
}

impl<F> MatTransposeVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn mat_transpose_vec(&self, x: &Col<F>) -> Col<F> {
        assert_eq!(
            self.nrows(),
            x.nrows(),
            "mat_transpose_vec: A.nrows ({}) != x.nrows ({})",
            self.nrows(),
            x.nrows()
        );
        let row_idx = self.row_idx();
        let vals = self.val();

        Col::from_fn(self.ncols(), |j| {
            self.col_range(j).map(|k| vals[k] * x[row_idx[k]]).sum()
        })
    }
}

impl<F> GramMatrix for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn gram(&self) -> Self {
        // Aᵀ · A: `transpose()` gives a SparseRowMatRef view; faer's
        // sparse_sparse_matmul wants two CSC operands, so materialize
        // the transpose into CSC. The resulting Gram is square (n×n).
        let at_csc = self
            .as_ref()
            .transpose()
            .to_col_major()
            .expect("gram: out of memory while transposing");
        sparse_sparse_matmul(at_csc.as_ref(), self.as_ref(), F::one(), Par::Seq)
            .expect("gram: out of memory while multiplying")
    }
}

impl<F: Scalar> MaxDiagonal<F> for SparseColMat<usize, F> {
    fn max_diagonal(&self) -> F {
        let n = self.ncols();
        assert_eq!(
            self.nrows(),
            n,
            "max_diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            n
        );
        // Walk each column and search its row-index slice for the
        // diagonal entry. Missing diagonals contribute the implicit
        // zero; this matches MaxDiagonal's contract.
        let row_idx = self.row_idx();
        let vals = self.val();
        let mut best = F::neg_infinity();
        for j in 0..n {
            let v = self
                .col_range(j)
                .find_map(|k| (row_idx[k] == j).then_some(vals[k]))
                .unwrap_or(F::zero());
            if v > best {
                best = v;
            }
        }
        best
    }
}

impl<F: Scalar> MatDiagonal<Col<F>> for SparseColMat<usize, F> {
    fn diagonal(&self) -> Col<F> {
        let n = self.ncols();
        assert_eq!(
            self.nrows(),
            n,
            "diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            n
        );
        // Same per-column walk as `max_diagonal`: search each column's
        // row-index slice for the diagonal entry; a missing diagonal is
        // the implicit zero.
        let row_idx = self.row_idx();
        let vals = self.val();
        Col::from_fn(n, |j| {
            self.col_range(j)
                .find_map(|k| (row_idx[k] == j).then_some(vals[k]))
                .unwrap_or(F::zero())
        })
    }
}

impl<F: Scalar> AddDiagonalVectorInPlace<Col<F>> for SparseColMat<usize, F> {
    fn add_diagonal_vector_in_place(&mut self, diag: &Col<F>) {
        let n = self.ncols();
        assert_eq!(
            self.nrows(),
            n,
            "add_diagonal_vector_in_place: matrix must be square, got {}x{}",
            self.nrows(),
            n
        );
        assert_eq!(
            n,
            diag.nrows(),
            "add_diagonal_vector_in_place: matrix is {}x{} but diag has length {}",
            n,
            n,
            diag.nrows()
        );
        let (symbolic, vals) = self.parts_mut();
        let row_idx = symbolic.row_idx();
        for j in 0..n {
            let mut found = false;
            for k in symbolic.col_range(j) {
                if row_idx[k] == j {
                    vals[k] = vals[k] + diag[j];
                    found = true;
                    break;
                }
            }
            assert!(
                found,
                "add_diagonal_vector_in_place: diagonal entry ({j}, {j}) missing from CSC pattern"
            );
        }
    }
}

impl<F> LinearSolveSpd<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn solve_spd(&self, b: &Col<F>) -> Result<Col<F>, LinearSolveError> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "solve_spd: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            b.nrows(),
            "solve_spd: A.nrows ({}) != b.nrows ({})",
            self.nrows(),
            b.nrows()
        );
        // Symbolic + numeric Cholesky in one shot via the high-level
        // wrapper. LltError::Numeric is the rank-deficient/non-PSD
        // case; LltError::Generic covers OOM/index overflow, which
        // we surface as NotPositiveDefinite (the trait's only
        // backend-portable failure for this path).
        let llt =
            Self::sp_cholesky(self, Side::Lower).map_err(|e| match e {
                LltError::Numeric(_) => LinearSolveError::NotPositiveDefinite,
                LltError::Generic(_) => LinearSolveError::NotPositiveDefinite,
            })?;
        let mut x = b.clone();
        llt.solve_in_place(&mut x);
        Ok(x)
    }
}

impl<F> LinearSolveLstsq<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn solve_lstsq(&self, b: &Col<F>) -> Result<Col<F>, LinearSolveError> {
        assert_eq!(
            self.nrows(),
            b.nrows(),
            "solve_lstsq: A.nrows ({}) != b.nrows ({})",
            self.nrows(),
            b.nrows()
        );
        let qr = Self::sp_qr(self).map_err(|_| LinearSolveError::Singular)?;
        // High-level solve_lstsq allocates an m-row buffer, copies b
        // in, calls solve_lstsq_in_place, then truncates to n rows.
        Ok(qr.solve_lstsq(b))
    }
}

impl<F: Scalar> super::ScaleRowsInPlace<F> for SparseColMat<usize, F> {
    fn scale_rows_in_place(&mut self, factors: &[F]) {
        assert_eq!(self.nrows(), factors.len(), "row scale shape mismatch");
        let (symbolic, values) = self.parts_mut();
        for col in 0..symbolic.ncols() {
            // Uncompressed CSC columns can contain unused capacity.
            for entry in symbolic.col_range(col) {
                values[entry] =
                    values[entry] * factors[symbolic.row_idx()[entry]];
            }
        }
    }
}

impl<F: Scalar> super::MatrixShape for SparseColMat<usize, F> {
    fn matrix_rows(&self) -> usize {
        self.nrows()
    }
    fn matrix_cols(&self) -> usize {
        self.ncols()
    }
}
impl<F: Scalar> super::MatrixFinite for SparseColMat<usize, F> {
    fn matrix_is_finite(&self) -> bool {
        (0..self.ncols())
            .all(|j| self.val_of_col(j).iter().all(|v| v.is_finite()))
    }
}
impl<F: Scalar, V> super::MatVecInto<V, F> for SparseColMat<usize, F>
where
    V: super::VectorLen + std::ops::Index<usize, Output = F>,
{
    fn matvec_into(&self, x: &V, out: &mut [F]) {
        assert_eq!(x.vec_len(), self.ncols());
        assert_eq!(out.len(), self.nrows());
        out.fill(F::zero());
        let rows = self.row_idx();
        let values = self.val();
        for j in 0..self.ncols() {
            for k in self.col_range(j) {
                out[rows[k]] = out[rows[k]] + values[k] * x[j];
            }
        }
    }
}
impl<F: Scalar, V> super::MatTransposeVecInto<V, F> for SparseColMat<usize, F>
where
    V: super::VectorLen + std::ops::IndexMut<usize, Output = F>,
{
    fn mat_transpose_vec_into(&self, x: &[F], out: &mut V) {
        assert_eq!(x.len(), self.nrows());
        assert_eq!(out.vec_len(), self.ncols());
        let rows = self.row_idx();
        let values = self.val();
        for j in 0..self.ncols() {
            out[j] = self.col_range(j).map(|k| values[k] * x[rows[k]]).sum();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
    use super::*;
    use crate::{MatTransposeVecInto, MatVecInto, MatrixFinite};

    fn uncompressed_storage<F: Scalar + faer_traits::ComplexField>() {
        for row in [0, usize::MAX] {
            for padding in [F::from_f64(99.0).unwrap(), F::nan()] {
                let symbolic = SymbolicSparseColMat::new_checked(
                    1,
                    1,
                    vec![0, 2],
                    Some(vec![1]),
                    vec![0, row],
                );
                let mut a = SparseColMat::new(
                    symbolic,
                    vec![F::from_f64(2.0).unwrap(), padding],
                );
                let x = vec![F::from_f64(3.0).unwrap()];
                let expected = F::from_f64(6.0).unwrap();
                let mut out = vec![F::nan()];
                a.matvec_into(&x, &mut out);
                assert_eq!(out, [expected]);
                out[0] = F::nan();
                a.mat_transpose_vec_into(&x, &mut out);
                assert_eq!(out, [expected]);
                let col = Col::from_fn(1, |_| x[0]);
                assert_eq!(a.matvec(&col)[0], expected);
                assert_eq!(a.mat_transpose_vec(&col)[0], expected);
                assert!(a.matrix_is_finite());
                a.add_diagonal_vector_in_place(&Col::from_fn(1, |_| F::one()));
                assert_eq!(a.diagonal()[0], F::from_f64(3.0).unwrap());
                assert_eq!(a.max_diagonal(), F::from_f64(3.0).unwrap());
                assert_eq!(a.val()[1].is_nan(), padding.is_nan());
                if !padding.is_nan() {
                    assert_eq!(a.val()[1], padding);
                }
                a.val_mut()[0] = F::nan();
                assert!(!a.matrix_is_finite());
                a.val_mut()[0] = F::infinity();
                assert!(!a.matrix_is_finite());
            }
        }

        let symbolic = SymbolicSparseColMat::new_checked(
            2,
            3,
            vec![0, 2, 4, 5],
            Some(vec![1, 0, 1]),
            vec![0, usize::MAX, usize::MAX, usize::MAX, 1],
        );
        let a = SparseColMat::new(
            symbolic,
            vec![
                F::from_f64(2.0).unwrap(),
                F::nan(),
                F::nan(),
                F::nan(),
                F::from_f64(5.0).unwrap(),
            ],
        );
        let x = vec![
            F::from_f64(3.0).unwrap(),
            F::one(),
            F::from_f64(4.0).unwrap(),
        ];
        let mut product = vec![F::nan(); 2];
        a.matvec_into(&x, &mut product);
        assert_eq!(
            product,
            [F::from_f64(6.0).unwrap(), F::from_f64(20.0).unwrap()]
        );
        let mut transpose_product = vec![F::nan(); 3];
        a.mat_transpose_vec_into(&product, &mut transpose_product);
        assert_eq!(
            transpose_product,
            [
                F::from_f64(12.0).unwrap(),
                F::zero(),
                F::from_f64(100.0).unwrap()
            ]
        );
        assert!(a.matrix_is_finite());
    }

    #[test]
    fn uncompressed_products_and_finiteness_skip_padding() {
        uncompressed_storage::<f32>();
        uncompressed_storage::<f64>();
    }

    #[test]
    fn unused_diagonal_entries_remain_implicit_zeros() {
        let symbolic = SymbolicSparseColMat::new_checked(
            1,
            1,
            vec![0, 1],
            Some(vec![0]),
            vec![0],
        );
        let a = SparseColMat::new(symbolic, vec![99.0_f64]);
        assert_eq!(a.diagonal()[0], 0.0);
        assert_eq!(a.max_diagonal(), 0.0);
        assert!(a.matrix_is_finite());
    }

    #[test]
    #[should_panic(expected = "diagonal entry (0, 0) missing")]
    fn diagonal_updates_reject_unused_entries() {
        let symbolic = SymbolicSparseColMat::new_checked(
            1,
            1,
            vec![0, 1],
            Some(vec![0]),
            vec![0],
        );
        let mut a = SparseColMat::new(symbolic, vec![99.0_f64]);
        a.add_diagonal_vector_in_place(&Col::from_fn(1, |_| 1.0));
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// 2×2 dense matrix expressed as CSC triplets (no zeros so the
    /// pattern matches the dense case for direct comparison).
    fn csc2(row0: [f64; 2], row1: [f64; 2]) -> SparseColMat<usize, f64> {
        let triplets = [
            Triplet::new(0_usize, 0_usize, row0[0]),
            Triplet::new(1, 0, row1[0]),
            Triplet::new(0, 1, row0[1]),
            Triplet::new(1, 1, row1[1]),
        ];
        SparseColMat::try_new_from_triplets(2, 2, &triplets)
            .expect("triplets must build")
    }

    #[test]
    fn matvec_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let x = Col::<f64>::from_fn(2, |i| [5.0, 6.0][i]);
        let y = a.matvec(&x);
        assert_eq!(y.nrows(), 2);
        assert!(approx_eq(y[0], 17.0, 1e-12));
        assert!(approx_eq(y[1], 39.0, 1e-12));
    }

    #[test]
    fn mat_transpose_vec_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let x = Col::<f64>::from_fn(2, |i| [5.0, 6.0][i]);
        let y = a.mat_transpose_vec(&x);
        assert_eq!(y.nrows(), 2);
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
        // Materialize via SpMV-on-Identity: gram · eᵢ recovers column i.
        let e0 = Col::<f64>::from_fn(2, |i| if i == 0 { 1.0 } else { 0.0 });
        let e1 = Col::<f64>::from_fn(2, |i| if i == 1 { 1.0 } else { 0.0 });
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
        let b = Col::<f64>::from_fn(2, |i| [1.0, 2.0][i]);
        let x = a.solve_spd(&b).expect("SPD system must solve");
        // Same hand-computed answer as dense: x = [1/11, 7/11].
        assert!(approx_eq(x[0], 1.0 / 11.0, 1e-12));
        assert!(approx_eq(x[1], 7.0 / 11.0, 1e-12));
    }

    #[test]
    fn solve_spd_indefinite_returns_error() {
        let a = csc2([1.0, 2.0], [2.0, 1.0]);
        let b = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
        let err = a.solve_spd(&b).expect_err("indefinite must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn gram_of_rank_deficient_is_singular() {
        let a = csc2([1.0, 2.0], [2.0, 4.0]);
        let g = a.gram();
        let b = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
        let err = g.solve_spd(&b).expect_err("rank-deficient gram must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn add_diagonal_regularizes_singular_gram() {
        let a = csc2([1.0, 2.0], [2.0, 4.0]);
        let mut g = a.gram();
        let b = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
        assert!(g.clone().solve_spd(&b).is_err());
        g.add_diagonal_vector_in_place(&Col::<f64>::from_fn(2, |_| 1e-3));
        let x = g.solve_spd(&b).expect("damped gram must be SPD");
        assert_eq!(x.nrows(), 2);
    }

    #[test]
    fn add_diagonal_vector_in_place_adds_per_index() {
        let mut a = csc2([1.0, 2.0], [3.0, 4.0]);
        a.add_diagonal_vector_in_place(&Col::<f64>::from_fn(2, |i| {
            [10.0, 100.0][i]
        }));
        // Original [[1,2],[3,4]] + diag(10, 100) → [[11,2],[3,104]].
        let e0 = Col::<f64>::from_fn(2, |i| if i == 0 { 1.0 } else { 0.0 });
        let e1 = Col::<f64>::from_fn(2, |i| if i == 1 { 1.0 } else { 0.0 });
        let col0 = a.matvec(&e0);
        let col1 = a.matvec(&e1);
        assert!(approx_eq(col0[0], 11.0, 1e-12));
        assert!(approx_eq(col0[1], 3.0, 1e-12));
        assert!(approx_eq(col1[0], 2.0, 1e-12));
        assert!(approx_eq(col1[1], 104.0, 1e-12));
    }

    #[test]
    fn solve_lstsq_square_matches_direct_solve() {
        // Square non-singular system: the least-squares solution
        // coincides with the exact solve.
        // A = [[1, 2], [3, 5]], b = [3, 8]. Exact x = [1, 1].
        let a = csc2([1.0, 2.0], [3.0, 5.0]);
        let b = Col::<f64>::from_fn(2, |i| [3.0, 8.0][i]);
        let x = a.solve_lstsq(&b).expect("least-squares solve must succeed");
        assert_eq!(x.nrows(), 2);
        assert!(approx_eq(x[0], 1.0, 1e-10));
        assert!(approx_eq(x[1], 1.0, 1e-10));
    }

    #[test]
    fn solve_lstsq_overdetermined_matches_normal_equations() {
        // 3×2 overdetermined system. A = [[1,0],[0,1],[1,1]], b = [1,2,4].
        // Normal equations: AᵀA = [[2,1],[1,2]], Aᵀb = [5,6].
        // Solution: x = (1/3) [4, 7] ≈ [1.333…, 2.333…].
        let triplets = [
            Triplet::new(0_usize, 0_usize, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 0, 1.0),
            Triplet::new(2, 1, 1.0),
        ];
        let a =
            SparseColMat::<usize, f64>::try_new_from_triplets(3, 2, &triplets)
                .expect("triplets must build");
        let b = Col::<f64>::from_fn(3, |i| [1.0, 2.0, 4.0][i]);
        let x = a.solve_lstsq(&b).expect("least-squares solve must succeed");
        assert_eq!(x.nrows(), 2);
        assert!(approx_eq(x[0], 4.0 / 3.0, 1e-10));
        assert!(approx_eq(x[1], 7.0 / 3.0, 1e-10));
    }
}
