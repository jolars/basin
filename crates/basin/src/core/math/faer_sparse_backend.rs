//! Sparse impls of the `linalg` tier for
//! [`faer::sparse::SparseColMat<usize, f64>`] (CSC) over
//! [`faer::Col<f64>`]. Lands in S2b alongside the dense faer backend.
//!
//! Faer's sparse stack covers all five `linalg` traits: SpMV / Aᵀ-SpMV
//! via [`sparse_dense_matmul`], Gram via
//! [`sparse_sparse_matmul`], SPD solve via supernodal/simplicial
//! Cholesky ([`SparseColMat::sp_cholesky`]), and least-squares solve
//! via sparse QR ([`SparseColMat::sp_qr`]). The QR path is the only
//! `LinearSolveLstsq` implementor in basin today; nalgebra-sparse
//! doesn't ship sparse QR.

use faer::linalg::solvers::{Solve, SolveLstsq};
use faer::sparse::linalg::matmul::{sparse_dense_matmul, sparse_sparse_matmul};
use faer::sparse::linalg::LltError;
use faer::sparse::SparseColMat;
use faer::{Accum, Col, Par, Side};

use super::linalg::{
    AddDiagonalInPlace, AddDiagonalVectorInPlace, GramMatrix, LinearSolveError, LinearSolveLstsq,
    LinearSolveSpd, MatDiagonal, MatTransposeVec, MatVec, MaxDiagonal,
};

impl MatVec<Col<f64>> for SparseColMat<usize, f64> {
    fn matvec(&self, x: &Col<f64>) -> Col<f64> {
        assert_eq!(
            self.ncols(),
            x.nrows(),
            "matvec: A.ncols ({}) != x.nrows ({})",
            self.ncols(),
            x.nrows()
        );
        let mut y = Col::<f64>::zeros(self.nrows());
        sparse_dense_matmul(
            y.as_mat_mut(),
            Accum::Replace,
            self.as_ref(),
            x.as_mat(),
            1.0,
            Par::Seq,
        );
        y
    }
}

impl MatTransposeVec<Col<f64>> for SparseColMat<usize, f64> {
    fn mat_transpose_vec(&self, x: &Col<f64>) -> Col<f64> {
        assert_eq!(
            self.nrows(),
            x.nrows(),
            "mat_transpose_vec: A.nrows ({}) != x.nrows ({})",
            self.nrows(),
            x.nrows()
        );
        let mut y = Col::<f64>::zeros(self.ncols());
        // SparseRowMatRef impls SparseDenseMatMul, so transposing the
        // CSC view (giving a CSR view of Aᵀ) lets us reuse the same
        // entry point without materializing the transpose.
        sparse_dense_matmul(
            y.as_mat_mut(),
            Accum::Replace,
            self.as_ref().transpose(),
            x.as_mat(),
            1.0,
            Par::Seq,
        );
        y
    }
}

impl GramMatrix for SparseColMat<usize, f64> {
    fn gram(&self) -> Self {
        // Aᵀ · A: `transpose()` gives a SparseRowMatRef view; faer's
        // sparse_sparse_matmul wants two CSC operands, so materialize
        // the transpose into CSC. The resulting Gram is square (n×n).
        let at_csc = self
            .as_ref()
            .transpose()
            .to_col_major()
            .expect("gram: out of memory while transposing");
        sparse_sparse_matmul(at_csc.as_ref(), self.as_ref(), 1.0, Par::Seq)
            .expect("gram: out of memory while multiplying")
    }
}

impl MaxDiagonal for SparseColMat<usize, f64> {
    fn max_diagonal(&self) -> f64 {
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
        let col_ptr = self.col_ptr();
        let row_idx = self.row_idx();
        let vals = self.val();
        let mut best = f64::NEG_INFINITY;
        for j in 0..n {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
            let v = (start..end)
                .find_map(|k| (row_idx[k] == j).then_some(vals[k]))
                .unwrap_or(0.0);
            if v > best {
                best = v;
            }
        }
        best
    }
}

impl MatDiagonal<Col<f64>> for SparseColMat<usize, f64> {
    fn diagonal(&self) -> Col<f64> {
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
        let col_ptr = self.col_ptr();
        let row_idx = self.row_idx();
        let vals = self.val();
        Col::from_fn(n, |j| {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
            (start..end)
                .find_map(|k| (row_idx[k] == j).then_some(vals[k]))
                .unwrap_or(0.0)
        })
    }
}

impl AddDiagonalInPlace for SparseColMat<usize, f64> {
    fn add_diagonal_in_place(&mut self, scalar: f64) {
        let n = self.ncols();
        assert_eq!(
            self.nrows(),
            n,
            "add_diagonal_in_place: matrix must be square, got {}x{}",
            self.nrows(),
            n
        );
        // col_ptr / row_idx are immutable views of the symbolic part;
        // val_mut needs &mut self. Faer 0.24 doesn't expose a single
        // accessor that hands out (symbolic borrow, mutable val borrow)
        // on the owned SparseColMat without a reborrow dance, so clone
        // the symbolic vectors. Cost is O(nnz) of usize copies vs. the
        // O(n³) Cholesky that follows in the LM loop — negligible.
        let col_ptr: Vec<usize> = self.col_ptr().to_vec();
        let row_idx: Vec<usize> = self.row_idx().to_vec();
        let vals = self.val_mut();
        for j in 0..n {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
            let mut found = false;
            for k in start..end {
                if row_idx[k] == j {
                    vals[k] += scalar;
                    found = true;
                    break;
                }
            }
            assert!(
                found,
                "add_diagonal_in_place: diagonal entry ({j}, {j}) missing from CSC pattern"
            );
        }
    }
}

impl AddDiagonalVectorInPlace<Col<f64>> for SparseColMat<usize, f64> {
    fn add_diagonal_vector_in_place(&mut self, diag: &Col<f64>) {
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
        let col_ptr: Vec<usize> = self.col_ptr().to_vec();
        let row_idx: Vec<usize> = self.row_idx().to_vec();
        let vals = self.val_mut();
        for j in 0..n {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
            let mut found = false;
            for k in start..end {
                if row_idx[k] == j {
                    vals[k] += diag[j];
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

impl LinearSolveSpd<Col<f64>> for SparseColMat<usize, f64> {
    fn solve_spd(&self, b: &Col<f64>) -> Result<Col<f64>, LinearSolveError> {
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
        // case; LltError::Generic covers OOM / index overflow, which
        // we surface as NotPositiveDefinite (the trait's only
        // backend-portable failure for this path).
        let llt = Self::sp_cholesky(self, Side::Lower).map_err(|e| match e {
            LltError::Numeric(_) => LinearSolveError::NotPositiveDefinite,
            LltError::Generic(_) => LinearSolveError::NotPositiveDefinite,
        })?;
        let mut x = b.clone();
        llt.solve_in_place(&mut x);
        Ok(x)
    }
}

impl LinearSolveLstsq<Col<f64>> for SparseColMat<usize, f64> {
    fn solve_lstsq(&self, b: &Col<f64>) -> Result<Col<f64>, LinearSolveError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};

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
        SparseColMat::try_new_from_triplets(2, 2, &triplets).expect("triplets must build")
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
    fn add_diagonal_in_place_adds_to_diagonal_only() {
        let mut a = csc2([1.0, 2.0], [3.0, 4.0]);
        a.add_diagonal_in_place(0.5);
        // Original [[1,2],[3,4]] + 0.5·I = [[1.5,2],[3,4.5]].
        let e0 = Col::<f64>::from_fn(2, |i| if i == 0 { 1.0 } else { 0.0 });
        let e1 = Col::<f64>::from_fn(2, |i| if i == 1 { 1.0 } else { 0.0 });
        let col0 = a.matvec(&e0);
        let col1 = a.matvec(&e1);
        assert!(approx_eq(col0[0], 1.5, 1e-12));
        assert!(approx_eq(col0[1], 3.0, 1e-12));
        assert!(approx_eq(col1[0], 2.0, 1e-12));
        assert!(approx_eq(col1[1], 4.5, 1e-12));
    }

    #[test]
    fn add_diagonal_regularizes_singular_gram() {
        let a = csc2([1.0, 2.0], [2.0, 4.0]);
        let mut g = a.gram();
        let b = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
        assert!(g.clone().solve_spd(&b).is_err());
        g.add_diagonal_in_place(1e-3);
        let x = g.solve_spd(&b).expect("damped gram must be SPD");
        assert_eq!(x.nrows(), 2);
    }

    #[test]
    fn add_diagonal_vector_in_place_adds_per_index() {
        let mut a = csc2([1.0, 2.0], [3.0, 4.0]);
        a.add_diagonal_vector_in_place(&Col::<f64>::from_fn(2, |i| [10.0, 100.0][i]));
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
        // Square non-singular system — least-squares solution
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
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 2, &triplets)
            .expect("triplets must build");
        let b = Col::<f64>::from_fn(3, |i| [1.0, 2.0, 4.0][i]);
        let x = a.solve_lstsq(&b).expect("least-squares solve must succeed");
        assert_eq!(x.nrows(), 2);
        assert!(approx_eq(x[0], 4.0 / 3.0, 1e-10));
        assert!(approx_eq(x[1], 7.0 / 3.0, 1e-10));
    }
}
