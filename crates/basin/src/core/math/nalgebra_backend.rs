use nalgebra::{ClosedAddAssign, ClosedMulAssign};
use nalgebra::{
    Const, DMatrix, DVector, Dim, Dyn, Matrix, OMatrix, SMatrix, SVector,
    Storage, StorageMut,
};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

use super::Scalar;
use super::cl_scaling::{
    BoxAffineScaling, cl_scaling_pair, max_feasible_step_component,
    project_strictly_inside_component,
};
use super::linalg::{
    AddDiagonalVectorInPlace, DenseMatrixFromFn, GeneralRankOneUpdate,
    GramMatrix, LinearSolveError, LinearSolveSpd, MatDiagonal, MatTransposeVec,
    MatVec, MatrixFromDiagonal, MatrixIdentity, MaxDiagonal, RankOneUpdate,
    SymmetricEigen, SymmetricEigenError,
};
use super::sample::{
    SampleStandardNormal, SampleUniformBox, assert_finite_box,
};
use super::{
    ClampInPlace, ComponentDivAssign, ComponentMaxAssign, ComponentMulAssign,
    ComponentZip, Dot, FloorZerosInPlace, NegInPlace, NormInfinity,
    NormSquared, ScaleInPlace, ScaledAdd, VectorIndex, VectorLen,
};

// `F: Scalar` (basin's alias) bundles `Float + FromPrimitive + Sum + Debug +
// Default + 'static`, which transitively satisfies `nalgebra::Scalar`
// (`Clone + PartialEq + Debug + 'static`) via its blanket impl; no separate
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

    fn scaled_add_with_change(
        &mut self,
        scalar: F,
        other: &Self,
    ) -> Option<bool> {
        let mut changed = false;
        assert_eq!(self.shape(), other.shape(), "scaled_add: shape mismatch");
        self.zip_apply(other, |x, y| {
            let next = *x + scalar * y;
            changed |= next != *x;
            *x = next;
        });
        Some(changed)
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

    fn norm_l2(&self) -> F {
        super::norm_l2(self.iter().copied())
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
        super::norm_infinity(self.iter().copied())
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
    fn sample_uniform_box<G: Rng + ?Sized>(
        lower: &Self,
        upper: &Self,
        rng: &mut G,
    ) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "sample_uniform_box: bounds length mismatch"
        );
        assert_finite_box(lower, upper);
        Self::from_fn(lower.len(), |i, _| rng.random_range(lower[i]..=upper[i]))
    }
}

impl<F: Scalar> VectorLen for DVector<F> {
    fn vec_len(&self) -> usize {
        self.len()
    }

    fn residual_buffer() -> Option<Self> {
        Some(Self::zeros(0))
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

impl<F: Scalar, const N: usize> VectorLen for SVector<F, N> {
    fn vec_len(&self) -> usize {
        N
    }

    fn residual_buffer() -> Option<Self> {
        Some(Self::zeros())
    }
}

impl<F: Scalar, const N: usize> VectorIndex<F> for SVector<F, N> {
    fn get_scalar(&self, i: usize) -> F {
        self[i]
    }

    fn set_scalar(&mut self, i: usize, value: F) {
        self[i] = value;
    }
}

impl<F: Scalar + SampleUniform, const N: usize> SampleUniformBox
    for SVector<F, N>
{
    fn sample_uniform_box<G: Rng + ?Sized>(
        lower: &Self,
        upper: &Self,
        rng: &mut G,
    ) -> Self {
        assert_finite_box(lower, upper);
        Self::from_fn(|i, _| rng.random_range(lower[i]..=upper[i]))
    }
}

impl<F: Scalar, const N: usize> SampleStandardNormal for SVector<F, N>
where
    StandardNormal: Distribution<F>,
{
    fn sample_standard_normal<G: Rng + ?Sized>(_: &Self, rng: &mut G) -> Self {
        Self::from_fn(|_, _| StandardNormal.sample(rng))
    }
}

impl<F: Scalar, const N: usize> super::DenseBackend<F> for SVector<F, N> {
    type Matrix = SMatrix<F, N, N>;
}

impl<F: Scalar, const N: usize> super::SquareMatrixFromFn<F>
    for SMatrix<F, N, N>
{
    fn square_from_fn(n: usize, entry: impl FnMut(usize, usize) -> F) -> Self {
        assert_eq!(n, N, "Hessian dimension must match the fixed matrix");
        Self::from_fn(entry)
    }
}

impl<F: Scalar, const N: usize> MatrixIdentity for SMatrix<F, N, N> {
    fn identity(n: usize) -> Self {
        assert_eq!(n, N, "identity dimension does not match the fixed matrix");
        Self::identity()
    }
}

impl<F: Scalar, const R: usize, const C: usize>
    MatVec<SVector<F, C>, SVector<F, R>> for SMatrix<F, R, C>
{
    fn matvec(&self, x: &SVector<F, C>) -> SVector<F, R> {
        SVector::from_fn(|i, _| (0..C).map(|j| self[(i, j)] * x[j]).sum())
    }
}

impl<F: Scalar, const R: usize, const C: usize>
    MatVec<SVector<F, C>, SVector<F, R>> for FixedColumns<F, C>
{
    fn matvec(&self, x: &SVector<F, C>) -> SVector<F, R> {
        assert_eq!(self.nrows(), R, "matvec: shape mismatch");
        SVector::from_fn(|i, _| (0..C).map(|j| self[(i, j)] * x[j]).sum())
    }
}

impl<F: Scalar, const R: usize, const C: usize>
    MatTransposeVec<SVector<F, C>, SVector<F, R>> for SMatrix<F, R, C>
{
    fn mat_transpose_vec(&self, x: &SVector<F, R>) -> SVector<F, C> {
        SVector::from_fn(|j, _| (0..R).map(|i| self[(i, j)] * x[i]).sum())
    }
}

impl<F: Scalar, const R: usize, const C: usize>
    MatTransposeVec<SVector<F, C>, SVector<F, R>> for FixedColumns<F, C>
{
    fn mat_transpose_vec(&self, x: &SVector<F, R>) -> SVector<F, C> {
        assert_eq!(self.nrows(), R, "mat_transpose_vec: shape mismatch");
        SVector::from_fn(|j, _| (0..R).map(|i| self[(i, j)] * x[i]).sum())
    }
}

impl<F: Scalar, const C: usize> MatTransposeVec<SVector<F, C>, DVector<F>>
    for FixedColumns<F, C>
{
    fn mat_transpose_vec(&self, x: &DVector<F>) -> SVector<F, C> {
        let mut out = SVector::zeros();
        super::MatTransposeVecInto::mat_transpose_vec_into(
            self,
            x.as_slice(),
            &mut out,
        );
        out
    }
}

impl<F: Scalar, D: Dim, S: Storage<F, D, Const<C>>, const C: usize>
    GramMatrix<SMatrix<F, C, C>> for Matrix<F, D, Const<C>, S>
{
    fn gram(&self) -> SMatrix<F, C, C> {
        SMatrix::from_fn(|i, j| {
            (0..self.nrows()).map(|k| self[(k, i)] * self[(k, j)]).sum()
        })
    }
}

/// Pivoted QR with a compile-time solution dimension.
pub struct FixedQrFactorization<F: Scalar, const N: usize>(
    super::QrFactorization<F>,
);

impl<F: Scalar, const R: usize, const C: usize>
    super::FactorizePivotedQr<SVector<F, C>, F, SVector<F, R>>
    for SMatrix<F, R, C>
{
    type Factorization = FixedQrFactorization<F, C>;

    fn factorize_pivoted_qr(
        &self,
        b: &SVector<F, R>,
    ) -> Result<Self::Factorization, super::QrSolveError> {
        let rows = (0..R)
            .flat_map(|i| (0..C).map(move |j| self[(i, j)]))
            .collect();
        super::dense_qr::factorize(R, C, rows, b.as_slice())
            .map(FixedQrFactorization)
    }
}

impl<F: Scalar, const R: usize, const C: usize>
    super::FactorizePivotedQr<SVector<F, C>, F, SVector<F, R>>
    for FixedColumns<F, C>
{
    type Factorization = FixedQrFactorization<F, C>;

    fn factorize_pivoted_qr(
        &self,
        b: &SVector<F, R>,
    ) -> Result<Self::Factorization, super::QrSolveError> {
        assert_eq!(self.nrows(), R, "pivoted QR shape mismatch");
        let rows = (0..R)
            .flat_map(|i| (0..C).map(move |j| self[(i, j)]))
            .collect();
        super::dense_qr::factorize(R, C, rows, b.as_slice())
            .map(FixedQrFactorization)
    }
}

impl<F: Scalar, const C: usize>
    super::FactorizePivotedQr<SVector<F, C>, F, DVector<F>>
    for FixedColumns<F, C>
{
    type Factorization = FixedQrFactorization<F, C>;

    fn factorize_pivoted_qr(
        &self,
        b: &DVector<F>,
    ) -> Result<Self::Factorization, super::QrSolveError> {
        let rows = (0..self.nrows())
            .flat_map(|i| (0..C).map(move |j| self[(i, j)]))
            .collect();
        super::dense_qr::factorize(self.nrows(), C, rows, b.as_slice())
            .map(FixedQrFactorization)
    }
}

impl<F: Scalar, const N: usize> super::RegularizedQrSolve<SVector<F, N>, F>
    for FixedQrFactorization<F, N>
{
    fn column_norms_squared(&self) -> SVector<F, N> {
        SVector::from_column_slice(&self.0.norms_squared())
    }

    fn solve_regularized(
        &self,
        mu: F,
        diagonal: &SVector<F, N>,
        tolerance: Option<F>,
    ) -> Result<SVector<F, N>, super::QrSolveError> {
        self.0
            .solve(mu, diagonal.as_slice(), tolerance)
            .map(|x| SVector::from_column_slice(&x))
    }
}

impl<F: Scalar, const N: usize> GeneralRankOneUpdate<SVector<F, N>, F>
    for SMatrix<F, N, N>
{
    fn general_rank_one_update(
        &mut self,
        alpha: F,
        u: &SVector<F, N>,
        v: &SVector<F, N>,
    ) {
        for j in 0..N {
            for i in 0..N {
                self[(i, j)] = self[(i, j)] + alpha * u[i] * v[j];
            }
        }
    }
}

impl<F: Scalar, const N: usize> RankOneUpdate<SVector<F, N>, F>
    for SMatrix<F, N, N>
{
    fn rank_one_update(&mut self, alpha: F, v: &SVector<F, N>) {
        self.general_rank_one_update(alpha, v, v);
    }
}

impl<F: Scalar, const N: usize> MatrixFromDiagonal<SVector<F, N>>
    for SMatrix<F, N, N>
{
    fn from_diagonal(diag: &SVector<F, N>) -> Self {
        Self::from_diagonal(diag)
    }
}

impl<F: Scalar, const N: usize> MatDiagonal<SVector<F, N>>
    for SMatrix<F, N, N>
{
    fn diagonal(&self) -> SVector<F, N> {
        SVector::from_fn(|i, _| self[(i, i)])
    }
}

impl<F: Scalar, const N: usize> MaxDiagonal<F> for SMatrix<F, N, N> {
    fn max_diagonal(&self) -> F {
        (0..N).map(|i| self[(i, i)]).fold(F::neg_infinity(), F::max)
    }
}

impl<F: Scalar, const N: usize> AddDiagonalVectorInPlace<SVector<F, N>>
    for SMatrix<F, N, N>
{
    fn add_diagonal_vector_in_place(&mut self, diag: &SVector<F, N>) {
        for i in 0..N {
            self[(i, i)] = self[(i, i)] + diag[i];
        }
    }
}

impl<F: Scalar, const N: usize> LinearSolveSpd<SVector<F, N>>
    for SMatrix<F, N, N>
{
    fn solve_spd(
        &self,
        b: &SVector<F, N>,
    ) -> Result<SVector<F, N>, LinearSolveError> {
        // Factor the lower triangle of the inline copy in place. Keep the
        // pivot test and summation order of the shared dense Cholesky kernel.
        let mut l = *self;
        for i in 0..N {
            for j in 0..=i {
                let mut pivot = l[(i, j)];
                for k in 0..j {
                    pivot = pivot - l[(i, k)] * l[(j, k)];
                }
                l[(i, j)] = if i == j {
                    if pivot <= F::zero() {
                        return Err(LinearSolveError::NotPositiveDefinite);
                    }
                    pivot.sqrt()
                } else {
                    pivot / l[(j, j)]
                };
            }
        }

        // Reuse the inline right-hand side for both triangular solves.
        let mut x = *b;
        for i in 0..N {
            let mut value = x[i];
            for k in 0..i {
                value = value - l[(i, k)] * x[k];
            }
            x[i] = value / l[(i, i)];
        }
        for i in (0..N).rev() {
            let mut value = x[i];
            for k in (i + 1)..N {
                value = value - l[(k, i)] * x[k];
            }
            x[i] = value / l[(i, i)];
        }
        Ok(x)
    }
}

impl<F: Scalar, const N: usize> SymmetricEigen<SVector<F, N>>
    for SMatrix<F, N, N>
{
    fn try_eigh(&self) -> Result<(Self, SVector<F, N>), SymmetricEigenError> {
        let rows = (0..N)
            .flat_map(|i| (0..N).map(move |j| self[(i, j)]))
            .collect::<Vec<_>>();
        super::dense_eig::jacobi_eigen(&rows, N)
            .map(|(values, vectors)| {
                (
                    Self::from_row_slice(&vectors),
                    SVector::from_column_slice(&values),
                )
            })
            .ok_or(SymmetricEigenError::Failed)
    }
}

// Numerical differentiation knows the parameter dimension statically but
// discovers the number of residual rows at runtime.
type FixedColumns<F, const C: usize> = OMatrix<F, Dyn, Const<C>>;

impl<F: Scalar, const N: usize> DenseMatrixFromFn<F> for SVector<F, N> {
    type Matrix = FixedColumns<F, N>;

    fn dense_from_fn<G: FnMut(usize, usize) -> F>(
        rows: usize,
        cols: usize,
        f: G,
    ) -> Self::Matrix {
        assert_eq!(
            cols, N,
            "matrix column count must match the parameter dimension"
        );
        OMatrix::from_fn_generic(Dyn(rows), Const, f)
    }

    fn dense_from_fn_into<G: FnMut(usize, usize) -> F>(
        rows: usize,
        cols: usize,
        mut f: G,
        out: &mut Self::Matrix,
    ) {
        assert_eq!(
            cols, N,
            "matrix column count must match the parameter dimension"
        );
        if out.nrows() != rows {
            *out = Self::dense_from_fn(rows, cols, f);
            return;
        }
        for j in 0..N {
            for i in 0..rows {
                out[(i, j)] = f(i, j);
            }
        }
    }
}

impl<F: Scalar> SampleStandardNormal for DVector<F>
where
    StandardNormal: Distribution<F>,
{
    fn sample_standard_normal<G: Rng + ?Sized>(
        template: &Self,
        rng: &mut G,
    ) -> Self {
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

impl<F, R, C, S> ComponentZip<F> for Matrix<F, R, C, S>
where
    F: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<F, R, C>,
{
    fn all_zip(
        &self,
        other: &Self,
        mut predicate: impl FnMut(F, F) -> bool,
    ) -> bool {
        assert_eq!(self.shape(), other.shape(), "all_zip: shape mismatch");
        self.iter()
            .zip(other.iter())
            .all(|(x, y)| predicate(*x, *y))
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
        // zipping is consistent across self/lower/upper.
        for ((x, &lo), &hi) in
            self.iter_mut().zip(lower.iter()).zip(upper.iter())
        {
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

    fn project_strictly_inside(
        &mut self,
        lower: &Self,
        upper: &Self,
        rstep: F,
    ) {
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
        for ((x, &l), &u) in self.iter_mut().zip(lower.iter()).zip(upper.iter())
        {
            *x = project_strictly_inside_component::<F>(*x, l, u, rstep);
        }
    }
}

// Dense linear algebra on DMatrix<F> with V = DVector<F>.
//
// `F: Scalar` is paired here with nalgebra's
// `Closed*Assign` traits; basin's `Float`-based `Scalar` doesn't subsume
// the `+=`/`*=` requirements that nalgebra's BLAS-2 ops (`Mul`, `tr_mul`,
// `ger`, …) accumulate into. `f64` and `f32` both satisfy both halves, so
// f64 and f32 satisfy both sets of bounds. Factorizations additionally
// require `nalgebra::ComplexField` or `RealField`.

impl<F> MatVec<DVector<F>> for DMatrix<F>
where
    F: Scalar + ClosedAddAssign + ClosedMulAssign,
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
    F: Scalar + ClosedAddAssign + ClosedMulAssign,
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
    F: Scalar + ClosedAddAssign + ClosedMulAssign,
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
        DVector::from_iterator(
            self.nrows(),
            (0..self.nrows()).map(|i| self[(i, i)]),
        )
    }
}

impl<F> AddDiagonalVectorInPlace<DVector<F>> for DMatrix<F>
where
    F: Scalar + ClosedAddAssign,
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
    fn dense_from_fn<G: FnMut(usize, usize) -> F>(
        rows: usize,
        cols: usize,
        f: G,
    ) -> DMatrix<F> {
        DMatrix::from_fn(rows, cols, f)
    }
    fn dense_from_fn_into<G: FnMut(usize, usize) -> F>(
        rows: usize,
        cols: usize,
        mut f: G,
        out: &mut Self::Matrix,
    ) {
        if out.shape() != (rows, cols) {
            out.resize_mut(rows, cols, F::zero());
        }
        for j in 0..cols {
            for i in 0..rows {
                out[(i, j)] = f(i, j);
            }
        }
    }
}

// Pure-Rust symmetric eigendecomposition (default), generic over every
// `F: RealField`. Swapped for the LAPACK impl below under `nalgebra-lapack`;
// the two are mutually exclusive (`#[cfg]`).
if_not_nalgebra_lapack! {
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
}

// LAPACK-backed symmetric eigendecomposition (`nalgebra-lapack` feature).
// Behaviorally interchangeable with the pure-Rust impl above for the LAPACK
// scalar set: same square assertion, same `Failed` mapping on non-convergence,
// same `(eigenvectors, eigenvalues)` return shape (eigenvalues unsorted, as the
// trait contract permits). Concrete per-scalar (f64/f32) rather than generic
// because `nalgebra_lapack`'s scalar trait `SymmetricEigenScalar` is not
// re-exported (only the `SymmetricEigen` struct is), so it cannot be named in a
// bound, unlike `CholeskyScalar` above. f64 and f32 are the only scalars
// LAPACK's `dsyev`/`ssyev` cover; see the `nalgebra-lapack` feature note in
// `Cargo.toml`.
if_nalgebra_lapack! {
macro_rules! lapack_symmetric_eigen_impl {
    ($scalar:ty) => {
        impl SymmetricEigen<DVector<$scalar>> for DMatrix<$scalar> {
            fn try_eigh(
                &self,
            ) -> Result<(Self, DVector<$scalar>), SymmetricEigenError> {
                assert_eq!(
                    self.nrows(),
                    self.ncols(),
                    "try_eigh: matrix must be square, got {}x{}",
                    self.nrows(),
                    self.ncols()
                );
                // LAPACK's symmetric eigensolver (`dsyev`/`ssyev`) is direct,
                // not iterated; `try_new` returns `None` only on internal
                // non-convergence.
                nalgebra_lapack::SymmetricEigen::try_new(self.clone())
                    .map(|eig| (eig.eigenvectors, eig.eigenvalues))
                    .ok_or(SymmetricEigenError::Failed)
            }
        }
    };
}
}

if_nalgebra_lapack! {
lapack_symmetric_eigen_impl!(f64);
}
if_nalgebra_lapack! {
lapack_symmetric_eigen_impl!(f32);
}

impl<F> RankOneUpdate<DVector<F>, F> for DMatrix<F>
where
    F: Scalar + ClosedAddAssign + ClosedMulAssign,
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
    F: Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn general_rank_one_update(
        &mut self,
        alpha: F,
        u: &DVector<F>,
        v: &DVector<F>,
    ) {
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

// Pure-Rust Cholesky (default). Generic over every `F: ComplexField`. Swapped
// out for the LAPACK impl below when the `nalgebra-lapack` feature is on; the
// two are mutually exclusive (`#[cfg]`) so coherence is never violated.
if_not_nalgebra_lapack! {
impl<F> LinearSolveSpd<DVector<F>> for DMatrix<F>
where
    F: Scalar + nalgebra::ComplexField,
{
    fn solve_spd(
        &self,
        b: &DVector<F>,
    ) -> Result<DVector<F>, LinearSolveError> {
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
        // nalgebra's `cholesky` consumes the matrix, so a clone is unavoidable
        // without a separate factorize/solve split.
        self.clone()
            .cholesky()
            .ok_or(LinearSolveError::NotPositiveDefinite)
            .map(|chol| chol.solve(b))
    }
}
}

// LAPACK-backed Cholesky (`nalgebra-lapack` feature). Behaviorally
// interchangeable with the pure-Rust impl above for the LAPACK scalar set:
// same assertions, same `NotPositiveDefinite` mapping when factorization fails.
// The bound narrows from any `F: ComplexField` to `nalgebra_lapack`'s
// `CholeskyScalar` (f32/f64 and their complex counterparts); see the
// `nalgebra-lapack` feature note in `Cargo.toml`.
if_nalgebra_lapack! {
impl<F> LinearSolveSpd<DVector<F>> for DMatrix<F>
where
    F: Scalar + nalgebra_lapack::CholeskyScalar + num_traits::Zero,
{
    fn solve_spd(
        &self,
        b: &DVector<F>,
    ) -> Result<DVector<F>, LinearSolveError> {
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
        // `Cholesky::new` consumes the matrix (clone unavoidable). `None` from
        // `new` is a failed factorization (not SPD); `None` from `solve` can
        // only follow a degenerate factor, so both map to `NotPositiveDefinite`.
        let chol = nalgebra_lapack::Cholesky::new(self.clone())
            .ok_or(LinearSolveError::NotPositiveDefinite)?;
        chol.solve(b).ok_or(LinearSolveError::NotPositiveDefinite)
    }
}
}

impl<F: Scalar + nalgebra::RealField> super::FactorizePivotedQr<DVector<F>, F>
    for DMatrix<F>
{
    type Factorization = super::QrFactorization<F>;
    fn factorize_pivoted_qr(
        &self,
        b: &DVector<F>,
    ) -> Result<Self::Factorization, super::QrSolveError> {
        let (m, n) = self.shape();
        assert_eq!(b.len(), m, "pivoted QR right-hand side length mismatch");
        let norms = super::dense_qr::column_norms(m, n, |i, j| self[(i, j)])?;
        if b.iter().any(|x| !num_traits::Float::is_finite(*x)) {
            return Err(super::QrSolveError::NonFinite);
        }
        let scaled = DMatrix::from_fn(m, n, |i, j| {
            if norms[j] > F::zero() {
                self[(i, j)] / norms[j]
            } else {
                F::zero()
            }
        });
        let qr = scaled.col_piv_qr();
        let mut rhs = b.clone();
        qr.q_tr_mul(&mut rhs);
        let mut permutation = DVector::from_iterator(n, 0..n);
        qr.p().permute_rows(&mut permutation);
        let upper = qr.unpack_r();
        let mut r = vec![F::zero(); n * n];
        for i in 0..m.min(n) {
            for j in i..n {
                r[i * n + j] = upper[(i, j)] * norms[permutation[j]];
            }
        }
        let mut qtb: Vec<F> = rhs.data.into();
        qtb.resize(n, F::zero());
        super::QrFactorization::from_parts(
            m,
            r,
            qtb,
            permutation.data.into(),
            norms,
        )
    }
}
impl<F: Scalar> super::RegularizedQrSolve<DVector<F>, F>
    for super::QrFactorization<F>
{
    fn column_norms_squared(&self) -> DVector<F> {
        DVector::from_vec(self.norms_squared())
    }
    fn solve_regularized(
        &self,
        mu: F,
        diagonal: &DVector<F>,
        tolerance: Option<F>,
    ) -> Result<DVector<F>, super::QrSolveError> {
        self.solve(mu, diagonal.as_slice(), tolerance)
            .map(DVector::from_vec)
    }
}

impl<F: Scalar, R: Dim, C: Dim, S: Storage<F, R, C>> super::MatrixShape
    for Matrix<F, R, C, S>
{
    fn matrix_rows(&self) -> usize {
        self.nrows()
    }
    fn matrix_cols(&self) -> usize {
        self.ncols()
    }
}

impl<F: Scalar, R: Dim, C: Dim, S: Storage<F, R, C>> super::MatrixFinite
    for Matrix<F, R, C, S>
{
    fn matrix_is_finite(&self) -> bool {
        self.iter().all(|x| x.is_finite())
    }
}

impl<F: Scalar, R: Dim, C: Dim, S: Storage<F, R, C>> super::MatrixIndex<F>
    for Matrix<F, R, C, S>
{
    fn matrix_entry(&self, row: usize, col: usize) -> F {
        self[(row, col)]
    }
}

impl<F: Scalar, R: Dim, C: Dim, S: StorageMut<F, R, C>>
    super::ScaleRowsInPlace<F> for Matrix<F, R, C, S>
{
    fn scale_rows_in_place(&mut self, factors: &[F]) {
        assert_eq!(self.nrows(), factors.len(), "row scale shape mismatch");
        for j in 0..self.ncols() {
            for (i, &factor) in factors.iter().enumerate() {
                self[(i, j)] = self[(i, j)] * factor;
            }
        }
    }
}

impl<F: Scalar> super::DenseBackend<F> for nalgebra::DVector<F> {
    type Matrix = nalgebra::DMatrix<F>;
}

impl<V, F> super::MatVecInto<V, F> for nalgebra::DMatrix<F>
where
    V: super::VectorLen + std::ops::Index<usize, Output = F>,
    F: super::Scalar,
{
    fn matvec_into(&self, input: &V, output: &mut [F]) {
        assert_eq!(input.vec_len(), super::MatrixShape::matrix_cols(self));
        assert_eq!(output.len(), super::MatrixShape::matrix_rows(self));
        for (i, value) in output.iter_mut().enumerate() {
            *value = (0..super::MatrixShape::matrix_cols(self))
                .map(|j| {
                    super::MatrixIndex::matrix_entry(self, i, j) * input[j]
                })
                .sum();
        }
    }
}

impl<V, F, const R: usize, const C: usize> super::MatVecInto<V, F>
    for SMatrix<F, R, C>
where
    V: super::VectorLen + std::ops::Index<usize, Output = F>,
    F: super::Scalar,
{
    fn matvec_into(&self, input: &V, output: &mut [F]) {
        assert_eq!(input.vec_len(), C);
        assert_eq!(output.len(), R);
        for (i, value) in output.iter_mut().enumerate() {
            *value = (0..C).map(|j| self[(i, j)] * input[j]).sum();
        }
    }
}
impl<V, F> super::MatTransposeVecInto<V, F> for nalgebra::DMatrix<F>
where
    V: super::VectorLen + std::ops::IndexMut<usize, Output = F>,
    F: super::Scalar,
{
    fn mat_transpose_vec_into(&self, input: &[F], output: &mut V) {
        assert_eq!(input.len(), super::MatrixShape::matrix_rows(self));
        assert_eq!(output.vec_len(), super::MatrixShape::matrix_cols(self));
        for j in 0..super::MatrixShape::matrix_cols(self) {
            output[j] = (0..super::MatrixShape::matrix_rows(self))
                .map(|i| {
                    super::MatrixIndex::matrix_entry(self, i, j) * input[i]
                })
                .sum();
        }
    }
}

impl<V, F, const R: usize, const C: usize> super::MatTransposeVecInto<V, F>
    for SMatrix<F, R, C>
where
    V: super::VectorLen + std::ops::IndexMut<usize, Output = F>,
    F: super::Scalar,
{
    fn mat_transpose_vec_into(&self, input: &[F], output: &mut V) {
        assert_eq!(input.len(), R);
        assert_eq!(output.vec_len(), C);
        for j in 0..C {
            output[j] = (0..R).map(|i| self[(i, j)] * input[i]).sum();
        }
    }
}

impl<V, F, const C: usize> super::MatVecInto<V, F> for FixedColumns<F, C>
where
    V: super::VectorLen + std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    fn matvec_into(&self, input: &V, output: &mut [F]) {
        assert_eq!(input.vec_len(), C);
        assert_eq!(output.len(), self.nrows());
        for (i, value) in output.iter_mut().enumerate() {
            *value = (0..C).map(|j| self[(i, j)] * input[j]).sum();
        }
    }
}

impl<V, F, const C: usize> super::MatTransposeVecInto<V, F>
    for FixedColumns<F, C>
where
    V: super::VectorLen + std::ops::IndexMut<usize, Output = F>,
    F: Scalar,
{
    fn mat_transpose_vec_into(&self, input: &[F], output: &mut V) {
        assert_eq!(input.len(), self.nrows());
        assert_eq!(output.vec_len(), C);
        for j in 0..C {
            output[j] =
                (0..self.nrows()).map(|i| self[(i, j)] * input[i]).sum();
        }
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
    fn fixed_spd_uses_only_the_lower_triangle() {
        let a = SMatrix::<f32, 2, 2>::new(4.0, f32::NAN, 2.0, 5.0);
        let b = SVector::<f32, 2>::new(8.0, 12.0);
        assert_eq!(a.solve_spd(&b), Ok(SVector::<f32, 2>::new(1.0, 2.0)));
    }

    #[test]
    fn fixed_spd_rejects_zero_and_negative_pivots() {
        let b = SVector::<f64, 2>::new(1.0, 1.0);
        for diagonal in [1.0, 0.0] {
            let a = SMatrix::<f64, 2, 2>::new(1.0, 0.0, 1.0, diagonal);
            assert_eq!(
                a.solve_spd(&b),
                Err(LinearSolveError::NotPositiveDefinite)
            );
        }
    }

    #[test]
    fn fixed_spd_accepts_empty_system() {
        let a = SMatrix::<f64, 0, 0>::zeros();
        let b = SVector::<f64, 0>::zeros();
        assert_eq!(a.solve_spd(&b), Ok(b));
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
    fn add_diagonal_vector_in_place_adds_per_index() {
        let mut a = DMatrix::from_row_slice(
            3,
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        a.add_diagonal_vector_in_place(&DVector::from_vec(vec![
            10.0, 100.0, 1000.0,
        ]));
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
        // Anchor to f64: under the `nalgebra-lapack` feature `try_eigh` has
        // concrete f32/f64 impls, so the matrix element type can't be inferred
        // from the literals alone.
        let c = DMatrix::<f64>::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let (b, lambda) = c.try_eigh().expect("eigendecomposition");
        // B Λ Bᵀ
        let mut lambda_diag = DMatrix::<f64>::zeros(2, 2);
        for i in 0..2 {
            lambda_diag[(i, i)] = lambda[i];
        }
        let recomposed = &b * &lambda_diag * b.transpose();
        for r in 0..2 {
            for c_idx in 0..2 {
                assert!(approx_eq(
                    recomposed[(r, c_idx)],
                    c[(r, c_idx)],
                    1e-10
                ));
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
        g.add_diagonal_vector_in_place(&DVector::from_vec(vec![1e-3, 1e-3]));
        let x = g
            .solve_spd(&DVector::from_vec(vec![1.0, 1.0]))
            .expect("damped gram must be SPD");
        assert_eq!(x.len(), 2);
    }
}
