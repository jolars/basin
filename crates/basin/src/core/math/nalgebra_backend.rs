use nalgebra::{DMatrix, DVector, Dim, Matrix, Storage, StorageMut};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

use super::Scalar;
use super::cl_scaling::{
    BoxAffineScaling, cl_scaling_pair, max_feasible_step_component,
    project_strictly_inside_component,
};
use super::linalg::{
    AddDiagonalInPlace, AddDiagonalVectorInPlace, DenseMatrixFromFn, GeneralRankOneUpdate,
    GramMatrix, LinearSolveError, LinearSolveSpd, MatDiagonal, MatTransposeVec, MatVec,
    MatrixFromDiagonal, MatrixIdentity, MaxDiagonal, RankOneUpdate, SymmetricEigen,
    SymmetricEigenError,
};
use super::sample::{SampleStandardNormal, SampleUniformBox};
use super::{
    ClampInPlace, ComponentDivAssign, ComponentMaxAssign, ComponentMulAssign, Dot,
    FloorZerosInPlace, NegInPlace, NormInfinity, NormSquared, ScaleInPlace, ScaledAdd, VectorIndex,
    VectorLen,
};

// `F: Scalar` (basin's alias) bundles `Float + FromPrimitive + Sum + Debug +
// Default + 'static`, which transitively satisfies `nalgebra::Scalar`
// (`Clone + PartialEq + Debug + 'static`) via its blanket impl — no separate
// bound needed.

impl<F, R, C, S> ScaledAdd<F> for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn scaled_add(&mut self, scalar: F, other: &Self) {
        assert_eq!(self.shape(), other.shape(), "scaled_add: shape mismatch");
        self.zip_apply(other, |x, y| *x = *x + scalar * y);
    }
}

impl<F, R, C, S> NormSquared<F> for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<F, R, C>,
{
    fn norm_squared(&self) -> F {
        self.iter().map(|x| *x * *x).sum()
    }
}

impl<F, R, C, S> NormInfinity<F> for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<F, R, C>,
{
    fn norm_infinity(&self) -> F {
        self.iter().map(|x| x.abs()).fold(F::zero(), F::max)
    }
}

impl<F, R, C, S> Dot<F> for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<F, R, C>,
{
    fn dot(&self, other: &Self) -> F {
        assert_eq!(self.shape(), other.shape(), "dot: shape mismatch");
        self.iter().zip(other.iter()).map(|(a, b)| *a * *b).sum()
    }
}

impl<F, R, C, S> NegInPlace for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn neg_in_place(&mut self) {
        self.apply(|x| *x = -*x);
    }
}

impl<F: Scalar + SampleUniform> SampleUniformBox for DVector<F> {
    fn sample_uniform_box<G: Rng + ?Sized>(lower: &Self, upper: &Self, rng: &mut G) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "sample_uniform_box: bounds length mismatch"
        );
        Self::from_fn(lower.len(), |i, _| rng.random_range(lower[i]..=upper[i]))
    }
}

impl<F: Scalar> VectorLen for DVector<F> {
    fn vec_len(&self) -> usize {
        self.len()
    }
}

impl<F: Scalar> VectorIndex<F> for DVector<F> {
    fn get_scalar(&self, i: usize) -> F {
        self[i]
    }
    fn set_scalar(&mut self, i: usize, value: F) {
        self[i] = value;
    }
}

impl<F: Scalar> SampleStandardNormal for DVector<F>
where
    StandardNormal: Distribution<F>,
{
    fn sample_standard_normal<G: Rng + ?Sized>(template: &Self, rng: &mut G) -> Self {
        Self::from_fn(template.len(), |_, _| StandardNormal.sample(rng))
    }
}

impl<F, R, C, S> ScaleInPlace<F> for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn scale_in_place(&mut self, scalar: F) {
        // nalgebra's `*=` allocates intermediate; iterate manually.
        self.apply(|x| *x = *x * scalar);
    }
}

impl<F, R, C, S> ComponentMulAssign for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn component_mul_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_mul_assign: shape mismatch"
        );
        self.zip_apply(other, |x, y| *x = *x * y);
    }
}

impl<F, R, C, S> ComponentMaxAssign for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn component_max_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_max_assign: shape mismatch"
        );
        self.zip_apply(other, |x, y| *x = x.max(y));
    }
}

impl<F, R, C, S> FloorZerosInPlace<F> for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn floor_zeros_in_place(&mut self, value: F) {
        self.apply(|x| {
            if *x <= F::zero() {
                *x = value;
            }
        });
    }
}

impl<F, R, C, S> ComponentDivAssign for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn component_div_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_div_assign: shape mismatch"
        );
        self.zip_apply(other, |x, y| *x = *x / y);
    }
}

impl<F, R, C, S> ClampInPlace for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        assert_eq!(
            self.shape(),
            lower.shape(),
            "clamp_in_place: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "clamp_in_place: upper shape mismatch"
        );
        // `iter_mut` and `iter` both traverse in column-major order, so
        // zipping is consistent across self / lower / upper.
        for ((x, &lo), &hi) in self.iter_mut().zip(lower.iter()).zip(upper.iter()) {
            // `Float` has no `clamp`; `max(lo).min(hi)` matches the
            // `f64::clamp` result on finite, ordered bounds.
            *x = (*x).max(lo).min(hi);
        }
    }
}

impl<F, R, C, S> BoxAffineScaling<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar + Copy,
    R: Dim,
    C: Dim,
    S: StorageMut<F, R, C>,
{
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    ) {
        let shape = self.shape();
        assert_eq!(
            shape,
            gradient.shape(),
            "compute_cl_scaling: gradient shape mismatch"
        );
        assert_eq!(
            shape,
            lower.shape(),
            "compute_cl_scaling: lower shape mismatch"
        );
        assert_eq!(
            shape,
            upper.shape(),
            "compute_cl_scaling: upper shape mismatch"
        );
        assert_eq!(
            shape,
            d_sq.shape(),
            "compute_cl_scaling: d_sq shape mismatch"
        );
        assert_eq!(
            shape,
            c_diag.shape(),
            "compute_cl_scaling: c_diag shape mismatch"
        );
        // Column-major iteration order matches the ClampInPlace impl;
        // consistent across all six operands.
        for (((((&x, &g), &l), &u), d), c) in self
            .iter()
            .zip(gradient.iter())
            .zip(lower.iter())
            .zip(upper.iter())
            .zip(d_sq.iter_mut())
            .zip(c_diag.iter_mut())
        {
            let (d_sq_i, c_i) = cl_scaling_pair::<F>(x, g, l, u);
            *d = d_sq_i;
            *c = c_i;
        }
    }

    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> F {
        let shape = self.shape();
        assert_eq!(
            shape,
            step.shape(),
            "max_feasible_step: step shape mismatch"
        );
        assert_eq!(
            shape,
            lower.shape(),
            "max_feasible_step: lower shape mismatch"
        );
        assert_eq!(
            shape,
            upper.shape(),
            "max_feasible_step: upper shape mismatch"
        );
        let mut tau = F::infinity();
        for (((&x, &s), &l), &u) in self
            .iter()
            .zip(step.iter())
            .zip(lower.iter())
            .zip(upper.iter())
        {
            let t = max_feasible_step_component::<F>(x, s, l, u);
            if t < tau {
                tau = t;
            }
        }
        tau
    }

    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> F {
        assert_eq!(
            self.shape(),
            d_sq.shape(),
            "cl_kkt_inf_norm: shape mismatch"
        );
        self.iter()
            .zip(d_sq.iter())
            .map(|(&v, &d)| <F as num_traits::Float>::abs(v) / d)
            .fold(F::zero(), |a, b| if b > a { b } else { a })
    }

    fn weighted_norm_squared(&self, weights: &Self) -> F {
        assert_eq!(
            self.shape(),
            weights.shape(),
            "weighted_norm_squared: shape mismatch"
        );
        self.iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * v * w)
            .sum()
    }

    fn project_strictly_inside(&mut self, lower: &Self, upper: &Self, rstep: F) {
        let shape = self.shape();
        assert_eq!(
            shape,
            lower.shape(),
            "project_strictly_inside: lower shape mismatch"
        );
        assert_eq!(
            shape,
            upper.shape(),
            "project_strictly_inside: upper shape mismatch"
        );
        for ((x, &l), &u) in self.iter_mut().zip(lower.iter()).zip(upper.iter()) {
            *x = project_strictly_inside_component::<F>(*x, l, u, rstep);
        }
    }
}

// ----------------------------------------------------------------------
// linalg tier — dense ops on DMatrix<F> with V = DVector<F>.
// Per tenet 5, this is dense-only; sparse comes in S2b.
//
// The bound `F: Scalar` (basin's alias) is paired here with nalgebra's
// `Closed*Assign` traits — basin's `Float`-based `Scalar` doesn't subsume
// the `+= / *=` requirements that nalgebra's BLAS-2 ops (`Mul`, `tr_mul`,
// `ger`, …) accumulate into. `f64` and `f32` both satisfy both halves, so
// the migration covers every scalar a real consumer will use today; the
// heavyweight factorization impls (`LinearSolveSpd`, `SymmetricEigen`)
// additionally need `nalgebra::ComplexField` / `RealField`.
// ----------------------------------------------------------------------

impl<F> MatVec<DVector<F>> for DMatrix<F>
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
        self * x
    }
}

impl<F> MatTransposeVec<DVector<F>> for DMatrix<F>
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
        self.tr_mul(x)
    }
}

impl<F> GramMatrix for DMatrix<F>
where
    F: Scalar + nalgebra::ClosedAddAssign + nalgebra::ClosedMulAssign,
{
    fn gram(&self) -> Self {
        // tr_mul(self) computes Aᵀ A in one pass without an explicit transpose.
        self.tr_mul(self)
    }
}

impl<F: Scalar> MaxDiagonal<F> for DMatrix<F> {
    fn max_diagonal(&self) -> F {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "max_diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        (0..self.nrows())
            .map(|i| self[(i, i)])
            .fold(F::neg_infinity(), F::max)
    }
}

impl<F: Scalar> MatDiagonal<DVector<F>> for DMatrix<F> {
    fn diagonal(&self) -> DVector<F> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        DVector::from_iterator(self.nrows(), (0..self.nrows()).map(|i| self[(i, i)]))
    }
}

impl<F> AddDiagonalInPlace<F> for DMatrix<F>
where
    F: Scalar + nalgebra::ClosedAddAssign,
{
    fn add_diagonal_in_place(&mut self, scalar: F) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "add_diagonal_in_place: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        for i in 0..self.nrows() {
            self[(i, i)] += scalar;
        }
    }
}

impl<F> AddDiagonalVectorInPlace<DVector<F>> for DMatrix<F>
where
    F: Scalar + nalgebra::ClosedAddAssign,
{
    fn add_diagonal_vector_in_place(&mut self, diag: &DVector<F>) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "add_diagonal_vector_in_place: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            diag.len(),
            "add_diagonal_vector_in_place: matrix is {}x{} but diag has length {}",
            self.nrows(),
            self.ncols(),
            diag.len()
        );
        for i in 0..self.nrows() {
            self[(i, i)] += diag[i];
        }
    }
}

impl<F: Scalar> MatrixIdentity for DMatrix<F> {
    fn identity(n: usize) -> Self {
        Self::identity(n, n)
    }
}

impl<F: Scalar> MatrixFromDiagonal<DVector<F>> for DMatrix<F> {
    fn from_diagonal(diag: &DVector<F>) -> Self {
        Self::from_diagonal(diag)
    }
}

impl<F: Scalar> DenseMatrixFromFn<F> for DVector<F> {
    type Matrix = DMatrix<F>;
    fn dense_from_fn<G: FnMut(usize, usize) -> F>(rows: usize, cols: usize, f: G) -> DMatrix<F> {
        DMatrix::from_fn(rows, cols, f)
    }
}

impl<F> SymmetricEigen<DVector<F>> for DMatrix<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn try_eigh(&self) -> Result<(Self, DVector<F>), SymmetricEigenError> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "try_eigh: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        // `try_new` is the bounded-iteration form. Convergence epsilon
        // matches nalgebra's `symmetric_eigen` default; the iteration
        // cap is the standard `n × 30` heuristic.
        let n = self.nrows();
        let max_iter = n.saturating_mul(30).max(64);
        let eps = <F as num_traits::FromPrimitive>::from_f64(1e-10).unwrap();
        nalgebra::SymmetricEigen::try_new(self.clone(), eps, max_iter)
            .map(|eig| (eig.eigenvectors, eig.eigenvalues))
            .ok_or(SymmetricEigenError::Failed)
    }
}

impl<F> RankOneUpdate<DVector<F>, F> for DMatrix<F>
where
    F: Scalar + nalgebra::ClosedAddAssign + nalgebra::ClosedMulAssign,
{
    fn rank_one_update(&mut self, alpha: F, v: &DVector<F>) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "rank_one_update: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            v.len(),
            "rank_one_update: matrix is {}x{} but v has length {}",
            self.nrows(),
            self.ncols(),
            v.len()
        );
        // self ← self + α · v · vᵀ via ger (nalgebra's BLAS-2 rank-1 update).
        // ger(α, v, w, β) computes self ← α v wᵀ + β self.
        self.ger(alpha, v, v, F::one());
    }
}

impl<F> GeneralRankOneUpdate<DVector<F>, F> for DMatrix<F>
where
    F: Scalar + nalgebra::ClosedAddAssign + nalgebra::ClosedMulAssign,
{
    fn general_rank_one_update(&mut self, alpha: F, u: &DVector<F>, v: &DVector<F>) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "general_rank_one_update: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            u.len(),
            "general_rank_one_update: matrix is {}x{} but u has length {}",
            self.nrows(),
            self.ncols(),
            u.len()
        );
        assert_eq!(
            self.ncols(),
            v.len(),
            "general_rank_one_update: matrix is {}x{} but v has length {}",
            self.nrows(),
            self.ncols(),
            v.len()
        );
        // self ← self + α · u · vᵀ via ger: ger(α, u, v, β) ⇒ α u vᵀ + β self.
        self.ger(alpha, u, v, F::one());
    }
}

impl<F> LinearSolveSpd<DVector<F>> for DMatrix<F>
where
    F: Scalar + nalgebra::ComplexField,
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
        // nalgebra's `cholesky` consumes the matrix — clone is unavoidable
        // without a separate factorize/solve split.
        self.clone()
            .cholesky()
            .ok_or(LinearSolveError::NotPositiveDefinite)
            .map(|chol| chol.solve(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn matvec_known_values() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.matvec(&x);
        assert_eq!(y.len(), 2);
        assert!(approx_eq(y[0], 17.0, 1e-12));
        assert!(approx_eq(y[1], 39.0, 1e-12));
    }

    #[test]
    fn mat_transpose_vec_known_values() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.mat_transpose_vec(&x);
        assert_eq!(y.len(), 2);
        // Aᵀ x = [1·5 + 3·6, 2·5 + 4·6] = [23, 34]
        assert!(approx_eq(y[0], 23.0, 1e-12));
        assert!(approx_eq(y[1], 34.0, 1e-12));
    }

    #[test]
    fn gram_known_values() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let g = a.gram();
        // AᵀA = [[1·1+3·3, 1·2+3·4], [2·1+4·3, 2·2+4·4]] = [[10, 14], [14, 20]]
        assert_eq!(g.shape(), (2, 2));
        assert!(approx_eq(g[(0, 0)], 10.0, 1e-12));
        assert!(approx_eq(g[(0, 1)], 14.0, 1e-12));
        assert!(approx_eq(g[(1, 0)], 14.0, 1e-12));
        assert!(approx_eq(g[(1, 1)], 20.0, 1e-12));
    }

    #[test]
    fn solve_spd_happy_path() {
        // A = [[4, 1], [1, 3]], b = [1, 2].
        // det = 11, x = (1/11) [3·1 − 1·2, −1·1 + 4·2] = [1/11, 7/11].
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x = a.solve_spd(&b).expect("SPD system must solve");
        assert!(approx_eq(x[0], 1.0 / 11.0, 1e-12));
        assert!(approx_eq(x[1], 7.0 / 11.0, 1e-12));
    }

    #[test]
    fn solve_spd_indefinite_returns_error() {
        // A = [[1, 2], [2, 1]] is symmetric but indefinite (det = −3).
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = a.solve_spd(&b).expect_err("indefinite must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn gram_of_rank_deficient_is_singular() {
        // Rank-1 matrix → AᵀA is rank-1, singular, fails Cholesky.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let g = a.gram();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = g.solve_spd(&b).expect_err("rank-deficient gram must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn add_diagonal_in_place_adds_to_diagonal_only() {
        let mut a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        a.add_diagonal_in_place(0.5);
        // Diagonal: 1.5, 5.5, 9.5; off-diagonal untouched.
        assert!(approx_eq(a[(0, 0)], 1.5, 1e-12));
        assert!(approx_eq(a[(1, 1)], 5.5, 1e-12));
        assert!(approx_eq(a[(2, 2)], 9.5, 1e-12));
        assert!(approx_eq(a[(0, 1)], 2.0, 1e-12));
        assert!(approx_eq(a[(1, 0)], 4.0, 1e-12));
        assert!(approx_eq(a[(2, 1)], 8.0, 1e-12));
    }

    #[test]
    fn add_diagonal_vector_in_place_adds_per_index() {
        let mut a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        a.add_diagonal_vector_in_place(&DVector::from_vec(vec![10.0, 100.0, 1000.0]));
        // Diagonal: 11, 105, 1009; off-diagonal untouched.
        assert!(approx_eq(a[(0, 0)], 11.0, 1e-12));
        assert!(approx_eq(a[(1, 1)], 105.0, 1e-12));
        assert!(approx_eq(a[(2, 2)], 1009.0, 1e-12));
        assert!(approx_eq(a[(0, 1)], 2.0, 1e-12));
        assert!(approx_eq(a[(2, 1)], 8.0, 1e-12));
    }

    #[test]
    fn matrix_identity_is_diagonal_ones() {
        let i: DMatrix<f64> = MatrixIdentity::identity(3);
        assert_eq!(i.shape(), (3, 3));
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!(approx_eq(i[(r, c)], want, 1e-12));
            }
        }
    }

    #[test]
    fn matrix_from_diagonal_places_vector_on_diagonal() {
        let d = DVector::from_vec(vec![2.0, 3.0, 5.0]);
        let m: DMatrix<f64> = MatrixFromDiagonal::from_diagonal(&d);
        assert_eq!(m.shape(), (3, 3));
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { d[r] } else { 0.0 };
                assert!(approx_eq(m[(r, c)], want, 1e-12));
            }
        }
    }

    #[test]
    fn rank_one_update_outer_product() {
        // self = 0; α = 2; v = (1, 2, 3) → α v vᵀ = 2 [[1,2,3],[2,4,6],[3,6,9]].
        let mut a = DMatrix::<f64>::zeros(3, 3);
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        a.rank_one_update(2.0, &v);
        assert!(approx_eq(a[(0, 0)], 2.0, 1e-12));
        assert!(approx_eq(a[(0, 1)], 4.0, 1e-12));
        assert!(approx_eq(a[(0, 2)], 6.0, 1e-12));
        assert!(approx_eq(a[(1, 1)], 8.0, 1e-12));
        assert!(approx_eq(a[(2, 2)], 18.0, 1e-12));
    }

    #[test]
    fn symmetric_eigen_recovers_factorization() {
        // C = [[2, 1], [1, 2]] has eigenvalues 1, 3 and eigenvectors
        // ([1, -1]/√2, [1, 1]/√2). Verify B Λ Bᵀ ≈ C.
        let c = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let (b, lambda) = c.try_eigh().expect("eigendecomposition");
        // B Λ Bᵀ
        let mut lambda_diag = DMatrix::<f64>::zeros(2, 2);
        for i in 0..2 {
            lambda_diag[(i, i)] = lambda[i];
        }
        let recomposed = &b * &lambda_diag * b.transpose();
        for r in 0..2 {
            for c_idx in 0..2 {
                assert!(approx_eq(recomposed[(r, c_idx)], c[(r, c_idx)], 1e-10));
            }
        }
    }

    #[test]
    fn add_diagonal_regularizes_singular_gram() {
        // The "why LM works" property: a rank-deficient Gram becomes
        // SPD once you add μI, and Cholesky succeeds.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let mut g = a.gram();
        assert!(
            g.clone()
                .solve_spd(&DVector::from_vec(vec![1.0, 1.0]))
                .is_err()
        );
        g.add_diagonal_in_place(1e-3);
        let x = g
            .solve_spd(&DVector::from_vec(vec![1.0, 1.0]))
            .expect("damped gram must be SPD");
        assert_eq!(x.len(), 2);
    }
}
