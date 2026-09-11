//! Damping selection is independent of factorization and parameter scaling.

use basin::core::problem::Problem;
use basin::{
    DenseMatrix, Executor, Jacobian, LevenbergMarquardt, LevenbergMarquardtQr,
    LmDamping, NllsState, Residual, Solver, State, TerminationReason,
};
use std::{cell::Cell, convert::Infallible, rc::Rc};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

macro_rules! backend_checks {
    ($module:ident, $scalar:ty, $large_coordinate:expr, $vector:ty, $matrix:ty, $vector_new:expr, $matrix_new:expr, $solver:expr) => {
        mod $module {
            use super::*;

            struct Affine;
            impl Residual for Affine {
                type Param = $vector;
                type Output = $vector;
                type Error = Infallible;

                fn residual(&self, x: &$vector) -> Result<$vector, Infallible> {
                    Ok(($vector_new)(&[2. * (x[0] - 1.), 0.25 * (x[1] - 2.)]))
                }
            }
            impl Jacobian for Affine {
                type Jacobian = $matrix;

                fn jacobian(&self, _: &$vector) -> Result<$matrix, Infallible> {
                    Ok(($matrix_new)(&[2., 0., 0., 0.25]))
                }
            }

            #[test]
            fn large_radius_takes_the_undamped_gauss_newton_step() {
                let result = Executor::from_start(
                    Affine,
                    ($solver).with_damping(LmDamping::TrustRegion),
                    ($vector_new)(&[0., 0.]),
                )
                .max_iter(1)
                .run()
                .unwrap();
                let tolerance = 64. * <$scalar>::EPSILON;
                assert!((result.param()[0] - 1.).abs() < tolerance);
                assert!((result.param()[1] - 2.).abs() < tolerance);
                assert_eq!(result.cost_evals(), 2);
                assert_eq!(result.state.jacobian_evals(), 1);
            }

            #[test]
            fn small_radius_limits_the_step_in_the_marquardt_metric() {
                let starts: [[$scalar; 2]; 2] = [[0., 0.], [0.5, 1.]];
                for start in starts {
                    let factor: $scalar = 0.1;
                    let scaled_start_norm = (4. * start[0] * start[0]
                        + 0.0625 * start[1] * start[1])
                        .sqrt();
                    let radius = if scaled_start_norm == 0. {
                        factor
                    } else {
                        factor * scaled_start_norm
                    };
                    let result = Executor::from_start(
                        Affine,
                        ($solver)
                            .with_damping(LmDamping::TrustRegion)
                            .with_initial_step_bound(factor),
                        ($vector_new)(&start),
                    )
                    .max_iter(1)
                    .run()
                    .unwrap();
                    let h0 = result.param()[0] - start[0];
                    let h1 = result.param()[1] - start[1];
                    let scaled_step_norm =
                        (4. * h0 * h0 + 0.0625 * h1 * h1).sqrt();
                    // The secular solve may approximate the boundary within ten percent.
                    assert!(scaled_step_norm >= 0.89 * radius);
                    assert!(scaled_step_norm <= 1.11 * radius);
                    let tolerance = 128. * <$scalar>::EPSILON;
                    assert!(
                        (h0 / (1. - start[0]) - h1 / (2. - start[1])).abs()
                            < tolerance
                    );
                    assert_eq!(result.cost_evals(), 2);
                    assert_eq!(result.state.jacobian_evals(), 1);
                }
            }

            #[test]
            fn insensitive_columns_and_exact_rank_deficiency_remain_solvable() {
                struct Deficient(bool);
                impl Residual for Deficient {
                    type Param = $vector;
                    type Output = $vector;
                    type Error = Infallible;

                    fn residual(
                        &self,
                        x: &$vector,
                    ) -> Result<$vector, Infallible> {
                        Ok(if self.0 {
                            ($vector_new)(&[x[0] - 1., 0.])
                        } else {
                            let residual = x[0] + x[1] - 3.;
                            ($vector_new)(&[residual, 2. * residual])
                        })
                    }
                }
                impl Jacobian for Deficient {
                    type Jacobian = $matrix;

                    fn jacobian(
                        &self,
                        _: &$vector,
                    ) -> Result<$matrix, Infallible> {
                        Ok(($matrix_new)(if self.0 {
                            &[1., 0., 0., 0.]
                        } else {
                            &[1., 1., 2., 2.]
                        }))
                    }
                }
                for insensitive in [false, true] {
                    let tolerance = <$scalar>::EPSILON.sqrt();
                    let result = Executor::from_start(
                        Deficient(insensitive),
                        ($solver)
                            .with_damping(LmDamping::TrustRegion)
                            .with_absolute_gradient_tolerance(tolerance),
                        ($vector_new)(&[0., if insensitive { 7. } else { 0. }]),
                    )
                    .max_iter(100)
                    .run()
                    .unwrap();
                    assert_eq!(
                        result.reason,
                        TerminationReason::SolverConverged
                    );
                    assert!(result.cost() < tolerance * tolerance);
                    if insensitive {
                        assert!((result.param()[0] - 1.).abs() < tolerance);
                        assert_eq!(result.param()[1], 7.);
                    } else {
                        assert!(
                            (result.param()[0] + result.param()[1] - 3.).abs()
                                < tolerance
                        );
                    }
                }
            }

            #[test]
            fn large_inactive_coordinate_does_not_overflow_the_scaled_norm() {
                struct Insensitive;
                impl Residual for Insensitive {
                    type Param = $vector;
                    type Output = $vector;
                    type Error = Infallible;

                    fn residual(
                        &self,
                        x: &$vector,
                    ) -> Result<$vector, Infallible> {
                        Ok(($vector_new)(&[x[0] - 1., 0.]))
                    }
                }
                impl Jacobian for Insensitive {
                    type Jacobian = $matrix;

                    fn jacobian(
                        &self,
                        _: &$vector,
                    ) -> Result<$matrix, Infallible> {
                        Ok(($matrix_new)(&[1., 0., 0., 0.]))
                    }
                }
                // The coordinate and scaled radius fit even though their squares do not.
                let inactive: $scalar = $large_coordinate;
                let tolerance = <$scalar>::EPSILON.sqrt();
                let result = Executor::from_start(
                    Insensitive,
                    ($solver)
                        .with_damping(LmDamping::TrustRegion)
                        .with_absolute_gradient_tolerance(tolerance),
                    ($vector_new)(&[0., inactive]),
                )
                .max_iter(100)
                .run()
                .unwrap();
                assert_eq!(result.reason, TerminationReason::SolverConverged);
                assert!((result.param()[0] - 1.).abs() < tolerance);
                assert_eq!(result.param()[1], inactive);
                assert!(result.cost() < tolerance * tolerance);
            }

            #[test]
            fn zero_residual_with_disabled_stopping_keeps_finite_state() {
                let result = Executor::from_start(
                    Affine,
                    ($solver)
                        .with_damping(LmDamping::TrustRegion)
                        .with_absolute_gradient_tolerance(None),
                    ($vector_new)(&[1., 2.]),
                )
                .max_iter(4)
                .run()
                .unwrap();
                assert_eq!(result.reason, TerminationReason::MaxIter);
                assert_eq!(result.cost(), 0.);
                assert_eq!(result.param()[0], 1.);
                assert_eq!(result.param()[1], 2.);
            }
        }
    };
}

backend_checks!(
    vec_cholesky,
    f64,
    1e200,
    Vec<f64>,
    DenseMatrix,
    |a: &[f64]| a.to_vec(),
    |a| DenseMatrix::from_row_slice(2, 2, a),
    LevenbergMarquardt::new()
);
backend_checks!(
    vec_qr,
    f64,
    1e200,
    Vec<f64>,
    DenseMatrix,
    |a: &[f64]| a.to_vec(),
    |a| DenseMatrix::from_row_slice(2, 2, a),
    LevenbergMarquardtQr::new()
);
backend_checks!(
    vec_f32_qr,
    f32,
    1e20,
    Vec<f32>,
    DenseMatrix<f32>,
    |a: &[f32]| a.to_vec(),
    |a| DenseMatrix::from_row_slice(2, 2, a),
    LevenbergMarquardtQr::<_, _, f32>::new()
);

#[cfg(feature = "nalgebra_all")]
mod nalgebra {
    use super::*;
    use backend_aliases::nalgebra::{DMatrix, DVector};

    backend_checks!(
        cholesky,
        f64,
        1e200,
        DVector<f64>,
        DMatrix<f64>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardt::new()
    );
    backend_checks!(
        qr,
        f64,
        1e200,
        DVector<f64>,
        DMatrix<f64>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardtQr::new()
    );
    backend_checks!(
        f32_qr,
        f32,
        1e20,
        DVector<f32>,
        DMatrix<f32>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a),
        LevenbergMarquardtQr::<_, _, f32>::new()
    );
}

#[cfg(feature = "ndarray_all")]
mod ndarray {
    use super::*;
    use backend_aliases::ndarray::{Array1, Array2};

    backend_checks!(
        cholesky,
        f64,
        1e200,
        Array1<f64>,
        Array2<f64>,
        |a: &[f64]| Array1::from_vec(a.to_vec()),
        |a: &[f64]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap(),
        LevenbergMarquardt::new()
    );
    backend_checks!(
        qr,
        f64,
        1e200,
        Array1<f64>,
        Array2<f64>,
        |a: &[f64]| Array1::from_vec(a.to_vec()),
        |a: &[f64]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap(),
        LevenbergMarquardtQr::new()
    );
    backend_checks!(
        f32_qr,
        f32,
        1e20,
        Array1<f32>,
        Array2<f32>,
        |a: &[f32]| Array1::from_vec(a.to_vec()),
        |a: &[f32]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap(),
        LevenbergMarquardtQr::<_, _, f32>::new()
    );
}

#[cfg(feature = "faer_all")]
mod faer {
    use super::*;
    use backend_aliases::faer::{Col, Mat};

    backend_checks!(
        cholesky,
        f64,
        1e200,
        Col<f64>,
        Mat<f64>,
        |a: &[f64]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f64]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j]),
        LevenbergMarquardt::new()
    );
    backend_checks!(
        qr,
        f64,
        1e200,
        Col<f64>,
        Mat<f64>,
        |a: &[f64]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f64]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j]),
        LevenbergMarquardtQr::new()
    );
    backend_checks!(
        f32_qr,
        f32,
        1e20,
        Col<f32>,
        Mat<f32>,
        |a: &[f32]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f32]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j]),
        LevenbergMarquardtQr::<_, _, f32>::new()
    );
}

#[derive(Clone, Default)]
struct Nonlinear {
    residuals: Rc<Cell<usize>>,
    jacobians: Rc<Cell<usize>>,
    error_at_residual: usize,
    error_at_jacobian: usize,
}

#[derive(Debug, PartialEq)]
struct CallbackError;

impl Residual for Nonlinear {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = CallbackError;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, CallbackError> {
        self.residuals.set(self.residuals.get() + 1);
        if self.residuals.get() == self.error_at_residual {
            return Err(CallbackError);
        }
        Ok(vec![x[0] * x[0] - 1.])
    }
}

impl Jacobian for Nonlinear {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, CallbackError> {
        self.jacobians.set(self.jacobians.get() + 1);
        if self.jacobians.get() == self.error_at_jacobian {
            return Err(CallbackError);
        }
        Ok(DenseMatrix::from_row_slice(1, 1, &[2. * x[0]]))
    }
}

#[test]
fn explicit_nielsen_preserves_default_trajectories() {
    macro_rules! check {
        ($solver:expr) => {
            for iterations in [1, 2, 5, 20] {
                let run = |solver| {
                    Executor::from_start(
                        Nonlinear::default(),
                        solver,
                        vec![0.1],
                    )
                    .max_iter(iterations)
                    .run()
                    .unwrap()
                };
                let default = run($solver);
                let explicit = run(($solver).with_damping(LmDamping::Nielsen));
                assert_eq!(default.param(), explicit.param());
                assert_eq!(default.cost(), explicit.cost());
                assert_eq!(default.reason, explicit.reason);
                assert_eq!(default.cost_evals(), explicit.cost_evals());
                assert_eq!(
                    default.state.jacobian_evals(),
                    explicit.state.jacobian_evals()
                );
            }
        };
    }
    check!(LevenbergMarquardt::new());
    check!(LevenbergMarquardtQr::new());
}

#[test]
fn damping_configuration_survives_qr_conversion_in_either_order() {
    let run = |solver| {
        Executor::from_start(Nonlinear::default(), solver, vec![0.1])
            .max_iter(1)
            .run()
            .unwrap()
    };
    let before = run(LevenbergMarquardt::new()
        .with_damping(LmDamping::TrustRegion)
        .with_initial_step_bound(0.1)
        .with_pivoted_qr());
    let after = run(LevenbergMarquardt::new()
        .with_pivoted_qr()
        .with_damping(LmDamping::TrustRegion)
        .with_initial_step_bound(0.1));
    let direct = run(LevenbergMarquardtQr::new()
        .with_damping(LmDamping::TrustRegion)
        .with_initial_step_bound(0.1));
    assert_eq!(before.param(), after.param());
    assert_eq!(before.param(), direct.param());
    assert!(before.param()[0] > 0.108 && before.param()[0] < 0.112);
}

#[test]
fn trust_qr_recovers_weak_linear_direction_with_relative_stopping() {
    struct NearlyCollinear;
    impl Residual for NearlyCollinear {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = Infallible;

        fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            // The fixed residual exposes stopping before the weak direction is fitted.
            Ok(vec![x[0] + x[1] - 3., 1e-6 * (x[1] - 2.), 1.])
        }
    }
    impl Jacobian for NearlyCollinear {
        type Jacobian = DenseMatrix;

        fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
            Ok(DenseMatrix::from_row_slice(
                3,
                2,
                &[1., 1., 0., 1e-6, 0., 0.],
            ))
        }
    }
    let result = Executor::from_start(
        NearlyCollinear,
        LevenbergMarquardtQr::new()
            .with_damping(LmDamping::TrustRegion)
            .with_absolute_gradient_tolerance(None)
            .with_gradient_orthogonality_tolerance(1e-12)
            .with_relative_model_reduction_tolerance(1e-12)
            .with_relative_step_tolerance(1e-12),
        vec![0., 0.],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param()[0] - 1.).abs() < 1e-8);
    assert!((result.param()[1] - 2.).abs() < 1e-8);
    assert_eq!(result.cost(), 0.5);
}

fn check_rejection_caching_and_reset<S>(mut solver: S)
where
    S: Solver<Nonlinear, NllsState<Vec<f64>>, Error = CallbackError>,
{
    let counts = Nonlinear::default();
    let mut problem = Problem::new(counts.clone());
    let initial = solver
        .init(&mut problem, NllsState::new(vec![0.1]))
        .unwrap();
    let (mut state, reason) = solver.next_iter(&mut problem, initial).unwrap();
    assert!(reason.is_none());
    assert_eq!(state.param(), &vec![0.1]);
    assert_eq!(counts.residuals.get(), 2);
    assert_eq!(counts.jacobians.get(), 1);
    for _ in 0..20 {
        if state.param()[0] != 0.1 {
            break;
        }
        (state, _) = solver.next_iter(&mut problem, state).unwrap();
        assert_eq!(counts.jacobians.get(), 1);
    }
    assert_ne!(state.param()[0], 0.1);
    let reset = solver
        .init(&mut problem, NllsState::new(vec![0.1]))
        .unwrap();
    let (reset, _) = solver.next_iter(&mut problem, reset).unwrap();
    // A fresh solve must again reject its first, undamped Newton step.
    assert_eq!(reset.param(), &vec![0.1]);
    assert_eq!(counts.jacobians.get(), 2);
}

#[test]
fn trust_rejections_reuse_jacobians_and_init_resets_radius() {
    check_rejection_caching_and_reset(
        LevenbergMarquardt::new().with_damping(LmDamping::TrustRegion),
    );
    check_rejection_caching_and_reset(
        LevenbergMarquardtQr::new().with_damping(LmDamping::TrustRegion),
    );
}

#[test]
fn trust_callback_errors_propagate_unchanged() {
    macro_rules! check {
        ($solver:expr) => {
            for (error_at_residual, error_at_jacobian) in
                [(1, 0), (2, 0), (0, 1), (0, 2)]
            {
                let result = Executor::from_start(
                    Nonlinear {
                        error_at_residual,
                        error_at_jacobian,
                        ..Default::default()
                    },
                    ($solver).with_damping(LmDamping::TrustRegion),
                    vec![2.],
                )
                .max_iter(50)
                .run();
                assert!(matches!(result, Err(CallbackError)));
            }
        };
    }
    check!(LevenbergMarquardt::new());
    check!(LevenbergMarquardtQr::new());
}

#[test]
fn initial_step_bound_requires_finite_positive_values() {
    for value in [0., -1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(|| {
                LevenbergMarquardt::<Vec<f64>, DenseMatrix>::new()
                    .with_initial_step_bound(value)
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new()
                    .with_initial_step_bound(value)
            })
            .is_err()
        );
    }
}
