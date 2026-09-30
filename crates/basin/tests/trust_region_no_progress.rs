//! A collapsed trust-region model does not establish stationarity.

use basin::{
    CostFunction, DenseMatrix, Executor, FirstOrderState, Gradient,
    GradientState, Hessian, HessianProduct, Solver, State, Steihaug,
    TerminationReason, TrustRegion,
};
use std::convert::Infallible;

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

macro_rules! backend_checks {
    ($module:ident, $scalar:ty, $vector:ty, $matrix:ty, $vector_new:expr, $matrix_new:expr) => {
        mod $module {
            use super::*;

            struct OffsetQuadratic;

            impl CostFunction for OffsetQuadratic {
                type Param = $vector;
                type Output = $scalar;
                type Error = Infallible;

                fn cost(&self, x: &$vector) -> Result<$scalar, Infallible> {
                    // The constant hides cost reductions near x = 1 through
                    // rounding, but the analytic gradient there is still 1.
                    Ok(16. / <$scalar>::EPSILON + 0.5 * x[0] * x[0])
                }
            }

            impl Gradient for OffsetQuadratic {
                type Gradient = $vector;

                fn gradient(&self, x: &$vector) -> Result<$vector, Infallible> {
                    Ok(x.clone())
                }
            }

            impl Hessian for OffsetQuadratic {
                type Hessian = $matrix;

                fn hessian(&self, _: &$vector) -> Result<$matrix, Infallible> {
                    Ok(($matrix_new)())
                }
            }

            impl HessianProduct for OffsetQuadratic {
                fn hessian_product(
                    &self,
                    _: &$vector,
                    v: &$vector,
                ) -> Result<$vector, Infallible> {
                    Ok(v.clone())
                }
            }

            fn check_stop<S>(
                solver: S,
                start: $scalar,
                reason: TerminationReason,
            ) where
                S: Solver<
                        OffsetQuadratic,
                        FirstOrderState<$vector, $scalar>,
                        Error = Infallible,
                    >,
            {
                let result = Executor::new(
                    OffsetQuadratic,
                    solver,
                    FirstOrderState::new(($vector_new)(start)),
                )
                .max_iter(100)
                .run()
                .unwrap();

                assert_eq!(result.reason, reason);
                assert_eq!(result.state.param()[0], start);
                assert_eq!(result.state.gradient().unwrap()[0], start);
                assert_eq!(
                    result.state.cost(),
                    OffsetQuadratic.cost(&($vector_new)(start)).unwrap()
                );
            }

            #[test]
            fn repeated_rejections_do_not_establish_convergence() {
                check_stop(
                    TrustRegion::with_subproblem(Steihaug::new()),
                    1.,
                    TerminationReason::NumericalNoProgress,
                );
                check_stop(
                    TrustRegion::matrix_free_with(Steihaug::new()),
                    1.,
                    TerminationReason::NumericalNoProgress,
                );
            }

            #[test]
            fn tiny_initial_radius_does_not_establish_convergence() {
                check_stop(
                    TrustRegion::with_subproblem(Steihaug::new())
                        .with_radius(<$scalar>::MIN_POSITIVE),
                    1.,
                    TerminationReason::NumericalNoProgress,
                );
                check_stop(
                    TrustRegion::matrix_free_with(Steihaug::new())
                        .with_radius(<$scalar>::MIN_POSITIVE),
                    1.,
                    TerminationReason::NumericalNoProgress,
                );
            }

            #[test]
            fn zero_gradient_still_establishes_convergence() {
                check_stop(
                    TrustRegion::with_subproblem(Steihaug::new()),
                    0.,
                    TerminationReason::SolverConverged,
                );
                check_stop(
                    TrustRegion::matrix_free_with(Steihaug::new()),
                    0.,
                    TerminationReason::SolverConverged,
                );
            }

            #[test]
            fn configured_gradient_tolerances_take_precedence() {
                check_stop(
                    TrustRegion::with_subproblem(Steihaug::new())
                        .with_radius(<$scalar>::MIN_POSITIVE)
                        .with_absolute_gradient_tolerance(0.1),
                    0.01,
                    TerminationReason::GradientTolerance,
                );
                check_stop(
                    TrustRegion::matrix_free_with(Steihaug::new())
                        .with_radius(<$scalar>::MIN_POSITIVE)
                        .with_relative_gradient_tolerance(1.),
                    0.01,
                    TerminationReason::RelativeGradientTolerance,
                );
            }

            #[test]
            fn unmet_gradient_tolerance_does_not_establish_convergence() {
                check_stop(
                    TrustRegion::with_subproblem(Steihaug::new())
                        .with_absolute_gradient_tolerance(0.),
                    1.,
                    TerminationReason::NumericalNoProgress,
                );
                check_stop(
                    TrustRegion::matrix_free_with(Steihaug::new())
                        .with_relative_gradient_tolerance(0.),
                    1.,
                    TerminationReason::NumericalNoProgress,
                );
            }
        }
    };
}

backend_checks!(
    vec_f64,
    f64,
    Vec<f64>,
    DenseMatrix<f64>,
    |x| vec![x],
    || { DenseMatrix::from_row_slice(1, 1, &[1.]) }
);
backend_checks!(
    vec_f32,
    f32,
    Vec<f32>,
    DenseMatrix<f32>,
    |x| vec![x],
    || { DenseMatrix::from_row_slice(1, 1, &[1.]) }
);

#[cfg(feature = "nalgebra_all")]
use backend_aliases::nalgebra::{DMatrix, DVector};
#[cfg(feature = "nalgebra_all")]
backend_checks!(
    nalgebra_f64,
    f64,
    DVector<f64>,
    DMatrix<f64>,
    |x| { DVector::from_vec(vec![x]) },
    || DMatrix::identity(1, 1)
);
#[cfg(feature = "nalgebra_all")]
backend_checks!(
    nalgebra_f32,
    f32,
    DVector<f32>,
    DMatrix<f32>,
    |x| { DVector::from_vec(vec![x]) },
    || DMatrix::identity(1, 1)
);

#[cfg(feature = "ndarray_all")]
use backend_aliases::ndarray::{Array1, Array2};
#[cfg(feature = "ndarray_all")]
backend_checks!(
    ndarray_f64,
    f64,
    Array1<f64>,
    Array2<f64>,
    |x| { Array1::from_vec(vec![x]) },
    || Array2::eye(1)
);
#[cfg(feature = "ndarray_all")]
backend_checks!(
    ndarray_f32,
    f32,
    Array1<f32>,
    Array2<f32>,
    |x| { Array1::from_vec(vec![x]) },
    || Array2::eye(1)
);

#[cfg(feature = "faer_all")]
use backend_aliases::faer::{Col, Mat};
#[cfg(feature = "faer_all")]
backend_checks!(
    faer_f64,
    f64,
    Col<f64>,
    Mat<f64>,
    |x| { Col::from_fn(1, |_| x) },
    || Mat::identity(1, 1)
);
#[cfg(feature = "faer_all")]
backend_checks!(
    faer_f32,
    f32,
    Col<f32>,
    Mat<f32>,
    |x| { Col::from_fn(1, |_| x) },
    || Mat::identity(1, 1)
);
