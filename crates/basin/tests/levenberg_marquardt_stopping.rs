//! Stopping decisions must not depend on whether an intermediate square fits.

use basin::{
    DenseMatrix, Executor, Jacobian, LevenbergMarquardt, LevenbergMarquardtQr,
    LmDamping, Residual, TerminationReason,
};
use std::convert::Infallible;

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

macro_rules! stopping_checks {
    ($module:ident, $scalar:ty, $tiny:expr, $vector:ty, $matrix:ty, $vector_new:expr, $matrix_new:expr, $solver:expr) => {
        mod $module {
            use super::*;

            struct Affine {
                scale: [$scalar; 2],
                target: [$scalar; 2],
            }

            impl Residual for Affine {
                type Param = $vector;
                type Output = $vector;
                type Error = Infallible;

                fn residual(&self, x: &$vector) -> Result<$vector, Infallible> {
                    Ok(($vector_new)(&[
                        self.scale[0] * (x[0] - self.target[0]),
                        self.scale[1] * (x[1] - self.target[1]),
                    ]))
                }
            }

            impl Jacobian for Affine {
                type Jacobian = $matrix;

                fn jacobian(&self, _: &$vector) -> Result<$matrix, Infallible> {
                    Ok(($matrix_new)(&[self.scale[0], 0., 0., self.scale[1]]))
                }
            }

            #[test]
            fn orthogonality_normalizes_before_squaring() {
                let tiny: $scalar = $tiny;
                for scale in [tiny, 1., 1. / tiny] {
                    for tolerance in [None, Some(0.), Some(0.25), Some(2.)] {
                        for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
                            let result = Executor::from_start(
                                Affine { scale: [scale, scale], target: [-1., -1.] },
                                ($solver)
                                    .with_damping(damping)
                                    .with_initial_step_bound(10. / tiny)
                                    .with_absolute_gradient_tolerance(0.)
                                    .with_gradient_orthogonality_tolerance(tolerance),
                                ($vector_new)(&[0., 0.]),
                            )
                            .max_iter(1)
                            .run()
                            .unwrap();
                            // Each column's cosine is 1/sqrt(2), at every scale.
                            let converged = tolerance == Some(2.);
                            assert_eq!(result.reason, if converged {
                                TerminationReason::SolverConverged
                            } else {
                                TerminationReason::MaxIter
                            }, "scale={scale}, tolerance={tolerance:?}, damping={damping:?}");
                            assert_eq!(result.cost_evals(), if converged { 1 } else { 2 });
                            if !converged {
                                assert!(result.param()[0] < -0.9);
                            }
                        }
                    }
                }
            }

            #[test]
            fn relative_step_survives_square_underflow_and_overflow() {
                let tiny: $scalar = $tiny;
                for scale in [tiny, 1. / tiny] {
                    let target = 1. / scale / scale;
                    for tolerance in [None, Some(0.), Some(0.5), Some(2.)] {
                        for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
                            let result = Executor::from_start(
                                Affine { scale: [scale, scale], target: [target, target] },
                                ($solver)
                                    .with_damping(damping)
                                    .with_initial_step_bound(10. / tiny)
                                    .with_absolute_gradient_tolerance(None)
                                    .with_relative_step_tolerance(tolerance),
                                ($vector_new)(&[0., 0.]),
                            )
                            .max_iter(1)
                            .run()
                            .unwrap();
                            // Starting at zero makes the accepted step and iterate identical.
                            assert_eq!(result.reason, if tolerance == Some(2.) {
                                TerminationReason::SolverConverged
                            } else {
                                TerminationReason::MaxIter
                            }, "scale={scale}, tolerance={tolerance:?}, damping={damping:?}");
                            assert!(result.param()[0] > 0.);
                            assert_eq!(result.cost_evals(), 2);
                        }
                    }
                }
            }

            #[test]
            fn tiny_tolerance_can_accept_a_step_beside_a_large_coordinate() {
                let tiny: $scalar = $tiny;
                let large = 1. / tiny / tiny;
                // The tolerance's square underflows, but tolerance * ||x|| is two.
                for (tolerance, converged) in [(tiny * tiny * 0.5, false), (tiny * tiny * 2., true)] {
                    let result = Executor::from_start(
                        Affine { scale: [1., 1.], target: [1., large] },
                        ($solver)
                            .with_absolute_gradient_tolerance(None)
                            .with_relative_step_tolerance(tolerance),
                        ($vector_new)(&[0., large]),
                    )
                    .max_iter(1)
                    .run()
                    .unwrap();
                    assert_eq!(result.reason, if converged {
                        TerminationReason::SolverConverged
                    } else {
                        TerminationReason::MaxIter
                    });
                    assert!(result.param()[0] > 0.9);
                }
            }

            #[test]
            fn zero_residual_distinguishes_disabled_and_exact_zero_tests() {
                let tiny: $scalar = $tiny;
                for coordinate in [0., tiny * tiny, 1. / tiny / tiny] {
                    for check in 0..4 {
                        for tolerance in [None, Some(0.)] {
                            let mut solver = ($solver).with_absolute_gradient_tolerance(None);
                            solver = match check {
                                0 => solver.with_absolute_gradient_tolerance(tolerance),
                                1 => solver.with_gradient_orthogonality_tolerance(tolerance),
                                2 => solver.with_relative_model_reduction_tolerance(tolerance),
                                _ => solver.with_relative_step_tolerance(tolerance),
                            };
                            let result = Executor::from_start(
                                Affine { scale: [1., 1.], target: [coordinate, coordinate] },
                                solver,
                                ($vector_new)(&[coordinate, coordinate]),
                            )
                            .max_iter(1)
                            .run()
                            .unwrap();
                            assert_eq!(result.reason, if tolerance.is_some() {
                                TerminationReason::SolverConverged
                            } else {
                                TerminationReason::MaxIter
                            }, "coordinate={coordinate}, check={check}, tolerance={tolerance:?}");
                        }
                    }
                }
            }
        }
    };
}

mod vec_f64 {
    use super::*;

    type F = f64;
    stopping_checks!(
        cholesky,
        f64,
        1e-100,
        Vec<F>,
        DenseMatrix<F>,
        |a: &[F]| a.to_vec(),
        |a| DenseMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardt::new()
    );
    stopping_checks!(
        qr,
        f64,
        1e-100,
        Vec<F>,
        DenseMatrix<F>,
        |a: &[F]| a.to_vec(),
        |a| DenseMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

mod vec_f32 {
    use super::*;

    type F = f32;
    stopping_checks!(
        qr,
        f32,
        1e-12,
        Vec<F>,
        DenseMatrix<F>,
        |a: &[F]| a.to_vec(),
        |a| DenseMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_f64 {
    use super::*;
    use backend_aliases::nalgebra::{DMatrix, DVector};
    type F = f64;
    stopping_checks!(
        cholesky,
        f64,
        1e-100,
        DVector<F>,
        DMatrix<F>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardt::new()
    );
    stopping_checks!(
        qr,
        f64,
        1e-100,
        DVector<F>,
        DMatrix<F>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_f32 {
    use super::*;
    use backend_aliases::nalgebra::{DMatrix, DVector};
    type F = f32;
    stopping_checks!(
        qr,
        f32,
        1e-12,
        DVector<F>,
        DMatrix<F>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

#[cfg(feature = "ndarray_all")]
mod ndarray_f64 {
    use super::*;
    use backend_aliases::ndarray::{Array1, Array2};
    type F = f64;
    stopping_checks!(
        cholesky,
        f64,
        1e-100,
        Array1<F>,
        Array2<F>,
        |a: &[F]| Array1::from_vec(a.to_vec()),
        |a: &[F]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap(),
        LevenbergMarquardt::new()
    );
    stopping_checks!(
        qr,
        f64,
        1e-100,
        Array1<F>,
        Array2<F>,
        |a: &[F]| Array1::from_vec(a.to_vec()),
        |a: &[F]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap(),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

#[cfg(feature = "ndarray_all")]
mod ndarray_f32 {
    use super::*;
    use backend_aliases::ndarray::{Array1, Array2};
    type F = f32;
    stopping_checks!(
        qr,
        f32,
        1e-12,
        Array1<F>,
        Array2<F>,
        |a: &[F]| Array1::from_vec(a.to_vec()),
        |a: &[F]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap(),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

#[cfg(feature = "faer_all")]
mod faer_f64 {
    use super::*;
    use backend_aliases::faer::{Col, Mat};
    type F = f64;
    stopping_checks!(
        cholesky,
        f64,
        1e-100,
        Col<F>,
        Mat<F>,
        |a: &[F]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[F]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j]),
        LevenbergMarquardt::new()
    );
    stopping_checks!(
        qr,
        f64,
        1e-100,
        Col<F>,
        Mat<F>,
        |a: &[F]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[F]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j]),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

#[cfg(feature = "faer_all")]
mod faer_f32 {
    use super::*;
    use backend_aliases::faer::{Col, Mat};
    type F = f32;
    stopping_checks!(
        qr,
        f32,
        1e-12,
        Col<F>,
        Mat<F>,
        |a: &[F]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[F]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j]),
        LevenbergMarquardtQr::<_, _, F>::new()
    );
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_sparse {
    use super::*;
    use backend_aliases::nalgebra::DVector;
    use backend_aliases::nalgebra_sparse::{CooMatrix, CscMatrix};

    stopping_checks!(
        cholesky,
        f64,
        1e-100,
        DVector<f64>,
        CscMatrix<f64>,
        DVector::from_column_slice,
        |a: &[f64]| {
            let mut coo = CooMatrix::new(2, 2);
            coo.push(0, 0, a[0]);
            coo.push(1, 1, a[3]);
            CscMatrix::from(&coo)
        },
        LevenbergMarquardt::new()
    );
}

#[cfg(feature = "faer_all")]
mod faer_sparse {
    use super::*;
    use backend_aliases::faer::Col;
    use backend_aliases::faer::sparse::{SparseColMat, Triplet};

    stopping_checks!(
        cholesky, f64, 1e-100, Col<f64>, SparseColMat<usize, f64>,
        |a: &[f64]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f64]| SparseColMat::try_new_from_triplets(2, 2, &[
            Triplet::new(0, 0, a[0]), Triplet::new(1, 1, a[3]),
        ]).unwrap(),
        LevenbergMarquardt::new()
    );
}
