//! Curvature recovery must retain a usable Nielsen step in both precisions.
use basin::{
    ArctanLoss, Executor, HuberLoss, Jacobian, LevenbergMarquardt, LmDamping,
    LossFunction, PointState, Residual, Scalar, Solver, TerminationCode,
    VectorIndex,
};
use std::convert::Infallible;

struct LinearModel<V, M, F> {
    make: fn(&[F]) -> V,
    jacobian: M,
    rank_deficient: bool,
}

impl<V: VectorIndex<F>, M, F: Scalar> Residual for LinearModel<V, M, F> {
    type Param = V;
    type Output = V;
    type Error = Infallible;
    fn residual(&self, x: &V) -> Result<V, Self::Error> {
        if !self.rank_deficient {
            let mut residuals = Vec::with_capacity(12);
            for (j, target) in [1., -1., 0.5, -0.5].into_iter().enumerate() {
                let d = x.get_scalar(j) - F::from_f64(target).unwrap();
                residuals.extend([d, F::from_f64(2.).unwrap() * d, -d]);
            }
            return Ok((self.make)(&residuals));
        }
        let a = x.get_scalar(0) + x.get_scalar(1) - F::one();
        let b = x.get_scalar(2) + x.get_scalar(3) + F::one();
        let two = F::from_f64(2.).unwrap();
        Ok((self.make)(&[a, two * a, -a, b, two * b, -b, two, -two]))
    }
}

impl<V: VectorIndex<F>, M: Clone, F: Scalar> Jacobian for LinearModel<V, M, F> {
    type Jacobian = M;
    fn jacobian(&self, _: &V) -> Result<M, Self::Error> {
        Ok(self.jacobian.clone())
    }
}

fn check<V, M, F, L, S>(
    rank_deficient: bool,
    make: fn(&[F]) -> V,
    matrix: M,
    loss: L,
    reference_cost: f64,
    solver: S,
) where
    F: Scalar,
    V: VectorIndex<F> + Clone,
    M: Clone,
    L: LossFunction<F>,
    S: Solver<
            basin::RobustLeastSquares<LinearModel<V, M, F>, L, F>,
            PointState<V, F>,
            Error = Infallible,
        >,
{
    let half = F::from_f64(0.5).unwrap();
    let start = if rank_deficient {
        make(&[-half, -half, half, half])
    } else {
        make(&[-F::one(), F::one(), -half, half])
    };
    let result = Executor::new(
        basin::RobustLeastSquares::new(
            LinearModel {
                make,
                jacobian: matrix,
                rank_deficient,
            },
            loss,
        ),
        solver,
        PointState::new(start),
    )
    .max_iter(500)
    .run()
    .unwrap();
    assert!(matches!(
        result.report.code(),
        TerminationCode::SolverConverged | TerminationCode::NumericalNoProgress
    ));
    let tolerance =
        F::from_f64(if F::epsilon() > F::from_f64(1e-10).unwrap() {
            1e-3
        } else {
            1e-6
        })
        .unwrap();
    if !rank_deficient {
        for (j, target) in [1., -1., 0.5, -0.5].into_iter().enumerate() {
            let error =
                result.param().get_scalar(j) - F::from_f64(target).unwrap();
            assert!(
                F::from_f64(6.).unwrap() * error.abs() <= tolerance,
                "{error:?}"
            );
        }
    }
    for pair in 0..if rank_deficient { 2 } else { 0 } {
        let target = if pair == 0 { F::one() } else { -F::one() };
        let error = result.param().get_scalar(2 * pair)
            + result.param().get_scalar(2 * pair + 1)
            - target;
        // Both loss gradients are bounded by 6*|error| near the affine minimum.
        assert!(
            F::from_f64(6.).unwrap() * error.abs() <= tolerance,
            "{error:?}"
        );
    }
    assert!(
        (result.cost() - F::from_f64(reference_cost).unwrap()).abs()
            <= tolerance
    );
}

fn entry<F: Scalar>(row: usize, column: usize) -> F {
    if row < 6 && column / 2 == row / 3 {
        F::from_f64([1., 2., -1.][row % 3]).unwrap()
    } else {
        F::zero()
    }
}

fn full_entry<F: Scalar>(row: usize, column: usize) -> F {
    if column == row / 3 {
        F::from_f64([1., 2., -1.][row % 3]).unwrap()
    } else {
        F::zero()
    }
}

macro_rules! dense_checks {
    ($v:ty, $m:ty, $f:ty, $make:expr, $matrix:expr, $full_matrix:expr) => {{
        type V = $v;
        type M = $m;
        type F = $f;
        let make: fn(&[F]) -> V = $make;
        let matrix: M = $matrix;
        let full_matrix: M = $full_matrix;
        for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
            let solver = || {
                LevenbergMarquardt::<V, M, F>::default().with_damping(damping)
            };
            check(false, make, full_matrix.clone(), ArctanLoss, 0., solver());
            check(
                false,
                make,
                full_matrix.clone(),
                ArctanLoss,
                0.,
                solver().with_pivoted_qr(),
            );
            check(true, make, matrix.clone(), HuberLoss, 3., solver());
            check(
                true,
                make,
                matrix.clone(),
                ArctanLoss,
                4_f64.atan(),
                solver(),
            );
            check(
                true,
                make,
                matrix.clone(),
                HuberLoss,
                3.,
                solver().with_pivoted_qr(),
            );
            check(
                true,
                make,
                matrix.clone(),
                ArctanLoss,
                4_f64.atan(),
                solver().with_pivoted_qr(),
            );
        }
    }};
}

macro_rules! vec_checks {
    ($name:ident, $f:ty) => {
        #[test]
        fn $name() {
            let entries: Vec<$f> = (0..8)
                .flat_map(|i| (0..4).map(move |j| entry(i, j)))
                .collect();
            dense_checks!(
                Vec<$f>,
                basin::DenseMatrix<$f>,
                $f,
                |x| x.to_vec(),
                basin::DenseMatrix::from_row_slice(8, 4, &entries),
                basin::DenseMatrix::from_row_slice(
                    12,
                    4,
                    &(0..12)
                        .flat_map(|i| (0..4).map(move |j| full_entry(i, j)))
                        .collect::<Vec<$f>>()
                )
            );
        }
    };
}
vec_checks!(vec32, f32);
vec_checks!(vec64, f64);

#[allow(unused_macros)]
macro_rules! nalgebra_checks {
    ($module:ident, $backend:ident, $sparse_backend:ident) => {
        mod $module {
            use super::*;
            use $backend::{DMatrix, DVector};
            use $sparse_backend::{CooMatrix, CscMatrix};
            macro_rules! tests {
                ($dense:ident, $sparse:ident, $f:ty) => {
                    #[test]
                    fn $dense() {
                        dense_checks!(
                            DVector<$f>,
                            DMatrix<$f>,
                            $f,
                            DVector::from_column_slice,
                            DMatrix::from_fn(8, 4, entry),
                            DMatrix::from_fn(12, 4, full_entry)
                        );
                    }
                    #[test]
                    fn $sparse() {
                        let mut coo = CooMatrix::new(8, 4);
                        for i in 0..6 {
                            for j in 0..4 {
                                let value: $f = entry(i, j);
                                if value != 0. {
                                    coo.push(i, j, value);
                                }
                            }
                        }
                        let matrix = CscMatrix::from(&coo);
                        for damping in
                            [LmDamping::Nielsen, LmDamping::TrustRegion]
                        {
                            let solver = || {
                                LevenbergMarquardt::<
                                    DVector<$f>,
                                    CscMatrix<$f>,
                                    $f,
                                >::default()
                                .with_damping(damping)
                            };
                            check(
                                true,
                                DVector::from_column_slice,
                                matrix.clone(),
                                HuberLoss,
                                3.,
                                solver(),
                            );
                            check(
                                true,
                                DVector::from_column_slice,
                                matrix.clone(),
                                ArctanLoss,
                                4_f64.atan(),
                                solver(),
                            );
                        }
                    }
                };
            }
            tests!(dense32, sparse32, f32);
            tests!(dense64, sparse64, f64);
        }
    };
}
#[cfg(feature = "nalgebra_v0_32")]
nalgebra_checks!(nalgebra32, nalgebra_0_32, nalgebra_sparse_0_9);
#[cfg(feature = "nalgebra_v0_33")]
nalgebra_checks!(nalgebra33, nalgebra_0_33, nalgebra_sparse_0_10);
#[cfg(feature = "nalgebra_v0_34")]
nalgebra_checks!(nalgebra34, nalgebra_0_34, nalgebra_sparse_0_11);
#[cfg(feature = "nalgebra_v0_35")]
nalgebra_checks!(nalgebra35, nalgebra, nalgebra_sparse);

#[allow(unused_macros)]
macro_rules! ndarray_checks {
    ($module:ident, $backend:ident) => {
        mod $module {
            use super::*;
            use $backend::{Array1, Array2};
            macro_rules! tests {
                ($name:ident, $f:ty) => {
                    #[test]
                    fn $name() {
                        dense_checks!(
                            Array1<$f>,
                            Array2<$f>,
                            $f,
                            |x| Array1::from_vec(x.to_vec()),
                            Array2::from_shape_fn((8, 4), |(i, j)| entry(i, j)),
                            Array2::from_shape_fn(
                                (12, 4),
                                |(i, j)| full_entry(i, j)
                            )
                        );
                    }
                };
            }
            tests!(f32, f32);
            tests!(f64, f64);
        }
    };
}
#[cfg(feature = "ndarray_v0_15")]
ndarray_checks!(ndarray15, ndarray_0_15);
#[cfg(feature = "ndarray_v0_16")]
ndarray_checks!(ndarray16, ndarray_0_16);
#[cfg(feature = "ndarray_v0_17")]
ndarray_checks!(ndarray17, ndarray);

#[allow(unused_macros)]
macro_rules! faer_checks {
    ($module:ident, $backend:ident) => {
        mod $module {
            use super::*;
            use $backend::sparse::{SparseColMat, Triplet};
            use $backend::{Col, Mat};
            macro_rules! tests {
                ($dense:ident, $sparse:ident, $f:ty) => {
                    #[test]
                    fn $dense() {
                        dense_checks!(
                            Col<$f>,
                            Mat<$f>,
                            $f,
                            |x| Col::from_fn(x.len(), |i| x[i]),
                            Mat::from_fn(8, 4, entry),
                            Mat::from_fn(12, 4, full_entry)
                        );
                    }
                    #[test]
                    fn $sparse() {
                        let entries: Vec<_> = (0..6)
                            .flat_map(|i| {
                                (0..4).filter_map(move |j| {
                                    let value: $f = entry(i, j);
                                    (value != 0.)
                                        .then_some(Triplet::new(i, j, value))
                                })
                            })
                            .collect();
                        let matrix =
                            SparseColMat::try_new_from_triplets(8, 4, &entries)
                                .unwrap();
                        for damping in
                            [LmDamping::Nielsen, LmDamping::TrustRegion]
                        {
                            let solver = || {
                                LevenbergMarquardt::<
                                    Col<$f>,
                                    SparseColMat<usize, $f>,
                                    $f,
                                >::default()
                                .with_damping(damping)
                            };
                            let make =
                                |x: &[$f]| Col::from_fn(x.len(), |i| x[i]);
                            check(
                                true,
                                make,
                                matrix.clone(),
                                HuberLoss,
                                3.,
                                solver(),
                            );
                            check(
                                true,
                                make,
                                matrix.clone(),
                                ArctanLoss,
                                4_f64.atan(),
                                solver(),
                            );
                        }
                    }
                };
            }
            tests!(dense32, sparse32, f32);
            tests!(dense64, sparse64, f64);
        }
    };
}
#[cfg(feature = "faer_v0_22")]
faer_checks!(faer22, faer_0_22);
#[cfg(feature = "faer_v0_23")]
faer_checks!(faer23, faer_0_23);
#[cfg(feature = "faer_v0_24")]
faer_checks!(faer24, faer);
