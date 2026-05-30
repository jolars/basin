use ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

use super::Scalar;
use super::cl_scaling::{
    BoxAffineScaling, cl_scaling_pair, max_feasible_step_component,
    project_strictly_inside_component,
};
use super::sample::{SampleStandardNormal, SampleUniformBox};
use super::{
    ClampInPlace, ComponentDivAssign, ComponentMaxAssign, ComponentMulAssign, Dot,
    FloorZerosInPlace, MatDiagonal, MatTransposeVec, MatVec, MatrixFromDiagonal, MatrixIdentity,
    NegInPlace, NormInfinity, NormSquared, RankOneUpdate, ScaleInPlace, ScaledAdd, SymmetricEigen,
    SymmetricEigenError, VectorIndex, VectorLen,
};

impl<F, S, D> ScaledAdd<F> for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn scaled_add(&mut self, scalar: F, other: &Self) {
        assert_eq!(self.shape(), other.shape(), "scaled_add: shape mismatch");
        self.zip_mut_with(other, |x, y| *x = *x + scalar * *y);
    }
}

impl<F, S, D> NormSquared<F> for ArrayBase<S, D>
where
    F: Scalar,
    S: Data<Elem = F>,
    D: Dimension,
{
    fn norm_squared(&self) -> F {
        self.iter().map(|x| *x * *x).sum()
    }
}

impl<F, S, D> NormInfinity<F> for ArrayBase<S, D>
where
    F: Scalar,
    S: Data<Elem = F>,
    D: Dimension,
{
    fn norm_infinity(&self) -> F {
        self.iter().map(|x| x.abs()).fold(F::zero(), F::max)
    }
}

impl<F, S, D> Dot<F> for ArrayBase<S, D>
where
    F: Scalar,
    S: Data<Elem = F>,
    D: Dimension,
{
    fn dot(&self, other: &Self) -> F {
        assert_eq!(self.shape(), other.shape(), "dot: shape mismatch");
        self.iter().zip(other.iter()).map(|(a, b)| *a * *b).sum()
    }
}

impl<F, S, D> NegInPlace for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn neg_in_place(&mut self) {
        self.map_inplace(|x| *x = -*x);
    }
}

impl<F: Scalar + SampleUniform> SampleUniformBox for Array1<F> {
    fn sample_uniform_box<R: Rng + ?Sized>(lower: &Self, upper: &Self, rng: &mut R) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "sample_uniform_box: bounds length mismatch"
        );
        Array1::from_shape_fn(lower.len(), |i| rng.random_range(lower[i]..=upper[i]))
    }
}

impl<F: Scalar> VectorLen for Array1<F> {
    fn vec_len(&self) -> usize {
        self.len()
    }
}

// Matrix-vector ops (the only `linalg`-tier traits ndarray implements). They
// lift the backend gate on the linear-constraint solvers; the factorization
// ops stay nalgebra/faer-only, since `ndarray-linalg` needs system BLAS/LAPACK
// (breaks the wasm-default tenet).
//
// These iterate rows/columns explicitly rather than calling ndarray's inherent
// `.dot()`: with basin's own [`Dot`] trait in scope, the bare `dot` method name
// makes inference explore ndarray's recursive `Dot`/`Not` bounds and overflow.
// `assert_eq!` on the lengths matches the panic-on-shape-mismatch contract.

impl<F: Scalar> MatVec<Array1<F>> for Array2<F> {
    fn matvec(&self, x: &Array1<F>) -> Array1<F> {
        assert_eq!(
            x.len(),
            self.ncols(),
            "matvec: x has length {} but the matrix has {} columns",
            x.len(),
            self.ncols()
        );
        Array1::from_shape_fn(self.nrows(), |i| {
            self.row(i)
                .iter()
                .zip(x.iter())
                .map(|(a, xj)| *a * *xj)
                .sum()
        })
    }
}

impl<F: Scalar> MatTransposeVec<Array1<F>> for Array2<F> {
    fn mat_transpose_vec(&self, x: &Array1<F>) -> Array1<F> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec: x has length {} but the matrix has {} rows",
            x.len(),
            self.nrows()
        );
        Array1::from_shape_fn(self.ncols(), |j| {
            self.column(j)
                .iter()
                .zip(x.iter())
                .map(|(a, xi)| *a * *xi)
                .sum()
        })
    }
}

impl<F: Scalar> MatrixIdentity for Array2<F> {
    fn identity(n: usize) -> Self {
        Array2::eye(n)
    }
}

impl<F: Scalar> MatrixFromDiagonal<Array1<F>> for Array2<F> {
    fn from_diagonal(diag: &Array1<F>) -> Self {
        Array2::from_diag(diag)
    }
}

impl<F: Scalar> MatDiagonal<Array1<F>> for Array2<F> {
    fn diagonal(&self) -> Array1<F> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        self.diag().to_owned()
    }
}

impl<F: Scalar> RankOneUpdate<Array1<F>, F> for Array2<F> {
    fn rank_one_update(&mut self, alpha: F, v: &Array1<F>) {
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
        // self[i, j] ← self[i, j] + α · v[i] · v[j] — the symmetric `u == v`
        // case of `general_rank_one_update`. Explicit double-loop matches the
        // `DenseMatrix` pattern; ndarray's `outer` / broadcasting forms would
        // allocate.
        let n = self.nrows();
        for i in 0..n {
            let av = alpha * v[i];
            let mut row = self.row_mut(i);
            for j in 0..n {
                let entry = &mut row[j];
                *entry = *entry + av * v[j];
            }
        }
    }
}

impl<F: Scalar> SymmetricEigen<Array1<F>> for Array2<F> {
    fn try_eigh(&self) -> Result<(Self, Array1<F>), SymmetricEigenError> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "try_eigh: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        let n = self.nrows();
        // `jacobi_eigen` takes a row-major `&[F]`. `as_standard_layout`
        // returns a `CowArray` that is contiguous in row-major (C) order —
        // borrowing when `self` is already standard, otherwise cloning.
        let standard = self.as_standard_layout();
        let slice = standard
            .as_slice()
            .expect("as_standard_layout produces a contiguous row-major slice");
        let (eigenvalues, eigenvectors) =
            super::dense_eig::jacobi_eigen(slice, n).ok_or(SymmetricEigenError::Failed)?;
        // `jacobi_eigen` returns the eigenvectors row-major with column `k`
        // the eigenvector for `eigenvalues[k]` — exactly what
        // `Array2::from_shape_vec` with the default C-order produces.
        let b =
            Array2::from_shape_vec((n, n), eigenvectors).expect("jacobi_eigen returns n*n entries");
        Ok((b, Array1::from_vec(eigenvalues)))
    }
}

impl<F: Scalar> VectorIndex<F> for Array1<F> {
    fn get_scalar(&self, i: usize) -> F {
        self[i]
    }
    fn set_scalar(&mut self, i: usize, value: F) {
        self[i] = value;
    }
}

impl<F: Scalar> SampleStandardNormal for Array1<F>
where
    StandardNormal: Distribution<F>,
{
    fn sample_standard_normal<R: Rng + ?Sized>(template: &Self, rng: &mut R) -> Self {
        Array1::from_shape_fn(template.len(), |_| StandardNormal.sample(rng))
    }
}

impl<F, S, D> ScaleInPlace<F> for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn scale_in_place(&mut self, scalar: F) {
        self.map_inplace(|x| *x = *x * scalar);
    }
}

impl<F, S, D> ComponentMulAssign for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn component_mul_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_mul_assign: shape mismatch"
        );
        self.zip_mut_with(other, |x, y| *x = *x * *y);
    }
}

impl<F, S, D> ComponentMaxAssign for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn component_max_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_max_assign: shape mismatch"
        );
        self.zip_mut_with(other, |x, y| *x = x.max(*y));
    }
}

impl<F, S, D> FloorZerosInPlace<F> for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn floor_zeros_in_place(&mut self, value: F) {
        self.map_inplace(|x| {
            if *x <= F::zero() {
                *x = value;
            }
        });
    }
}

impl<F, S, D> ComponentDivAssign for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn component_div_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_div_assign: shape mismatch"
        );
        self.zip_mut_with(other, |x, y| *x = *x / *y);
    }
}

impl<F, S, D> ClampInPlace for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
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
        ndarray::Zip::from(self)
            .and(lower)
            .and(upper)
            // `Float` has no `clamp`; `max(lo).min(hi)` matches the
            // `f64::clamp` result on finite, ordered bounds.
            .for_each(|x, &lo, &hi| *x = (*x).max(lo).min(hi));
    }
}

impl<F, S, D> BoxAffineScaling<F> for ArrayBase<S, D>
where
    F: Scalar,
    S: DataMut<Elem = F>,
    D: Dimension,
{
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    ) {
        assert_eq!(
            self.shape(),
            gradient.shape(),
            "compute_cl_scaling: gradient shape mismatch"
        );
        assert_eq!(
            self.shape(),
            lower.shape(),
            "compute_cl_scaling: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "compute_cl_scaling: upper shape mismatch"
        );
        assert_eq!(
            self.shape(),
            d_sq.shape(),
            "compute_cl_scaling: d_sq shape mismatch"
        );
        assert_eq!(
            self.shape(),
            c_diag.shape(),
            "compute_cl_scaling: c_diag shape mismatch"
        );
        ndarray::Zip::from(d_sq)
            .and(c_diag)
            .and(self)
            .and(gradient)
            .and(lower)
            .and(upper)
            .for_each(|d, c, &x, &g, &l, &u| {
                let (d_sq_i, c_i) = cl_scaling_pair::<F>(x, g, l, u);
                *d = d_sq_i;
                *c = c_i;
            });
    }

    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> F {
        assert_eq!(
            self.shape(),
            step.shape(),
            "max_feasible_step: step shape mismatch"
        );
        assert_eq!(
            self.shape(),
            lower.shape(),
            "max_feasible_step: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "max_feasible_step: upper shape mismatch"
        );
        let mut tau = F::infinity();
        ndarray::Zip::from(self)
            .and(step)
            .and(lower)
            .and(upper)
            .for_each(|&x, &s, &l, &u| {
                let t = max_feasible_step_component::<F>(x, s, l, u);
                if t < tau {
                    tau = t;
                }
            });
        tau
    }

    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> F {
        assert_eq!(
            self.shape(),
            d_sq.shape(),
            "cl_kkt_inf_norm: shape mismatch"
        );
        let mut best = F::zero();
        ndarray::Zip::from(self).and(d_sq).for_each(|&v, &d| {
            let candidate = <F as num_traits::Float>::abs(v) / d;
            if candidate > best {
                best = candidate;
            }
        });
        best
    }

    fn weighted_norm_squared(&self, weights: &Self) -> F {
        assert_eq!(
            self.shape(),
            weights.shape(),
            "weighted_norm_squared: shape mismatch"
        );
        let mut sum = F::zero();
        ndarray::Zip::from(self)
            .and(weights)
            .for_each(|&v, &w| sum = sum + v * v * w);
        sum
    }

    fn project_strictly_inside(&mut self, lower: &Self, upper: &Self, rstep: F) {
        assert_eq!(
            self.shape(),
            lower.shape(),
            "project_strictly_inside: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "project_strictly_inside: upper shape mismatch"
        );
        ndarray::Zip::from(self)
            .and(lower)
            .and(upper)
            .for_each(|x, &l, &u| {
                *x = project_strictly_inside_component::<F>(*x, l, u, rstep);
            });
    }
}
