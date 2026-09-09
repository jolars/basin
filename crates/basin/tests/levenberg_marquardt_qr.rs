use basin::{
    DenseMatrix, Executor, Jacobian, LevenbergMarquardt, LevenbergMarquardtQr,
    NllsState, Residual, TerminationReason,
};

use basin::core::problem::Problem;
use basin::{
    FactorizePivotedQr, MatTransposeVec, NormInfinity, NormSquared,
    QrFactorization, QrSolveError, RegularizedQrSolve, Solver, State,
};
use std::{cell::Cell, rc::Rc};

// This downstream-style matrix deliberately implements neither Gram nor SPD solve.
struct QrOnlyMatrix {
    matrix: DenseMatrix,
    factors: Rc<Cell<usize>>,
    solves: Rc<Cell<usize>>,
    failures: Rc<Cell<usize>>,
    nonfinite_gradient: Option<f64>,
}
struct CountedFactor {
    factor: QrFactorization,
    solves: Rc<Cell<usize>>,
    failures: Rc<Cell<usize>>,
}
impl MatTransposeVec<Vec<f64>> for QrOnlyMatrix {
    fn mat_transpose_vec(&self, v: &Vec<f64>) -> Vec<f64> {
        match self.nonfinite_gradient {
            Some(value) => vec![value],
            None => self.matrix.mat_transpose_vec(v),
        }
    }
}
impl FactorizePivotedQr<Vec<f64>> for QrOnlyMatrix {
    type Factorization = CountedFactor;
    fn factorize_pivoted_qr(
        &self,
        b: &Vec<f64>,
    ) -> Result<CountedFactor, QrSolveError> {
        self.factors.set(self.factors.get() + 1);
        Ok(CountedFactor {
            factor: self.matrix.factorize_pivoted_qr(b)?,
            solves: self.solves.clone(),
            failures: self.failures.clone(),
        })
    }
}
impl RegularizedQrSolve<Vec<f64>> for CountedFactor {
    fn column_norms_squared(&self) -> Vec<f64> {
        self.factor.column_norms_squared()
    }
    fn solve_regularized(
        &self,
        mu: f64,
        d: &Vec<f64>,
        tolerance: Option<f64>,
    ) -> Result<Vec<f64>, QrSolveError> {
        self.solves.set(self.solves.get() + 1);
        if self.failures.get() > 0 {
            self.failures.set(self.failures.get() - 1);
            return Err(QrSolveError::RankDeficient);
        }
        self.factor.solve_regularized(mu, d, tolerance)
    }
}
#[derive(Default, Clone)]
struct CountedProblem {
    residuals: Rc<Cell<usize>>,
    jacobians: Rc<Cell<usize>>,
    factors: Rc<Cell<usize>>,
    solves: Rc<Cell<usize>>,
    failures: Rc<Cell<usize>>,
    error_at_residual: usize,
    error_at_jacobian: usize,
    nonfinite_jacobian: bool,
    nonfinite_gradient: Option<f64>,
}
#[derive(Debug, PartialEq)]
struct CallbackError;
impl Residual for CountedProblem {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = CallbackError;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.residuals.set(self.residuals.get() + 1);
        if self.residuals.get() == self.error_at_residual {
            return Err(CallbackError);
        }
        Ok(vec![x[0] * x[0] - 1.])
    }
}
impl Jacobian for CountedProblem {
    type Jacobian = QrOnlyMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<Self::Jacobian, Self::Error> {
        self.jacobians.set(self.jacobians.get() + 1);
        if self.jacobians.get() == self.error_at_jacobian {
            return Err(CallbackError);
        }
        Ok(QrOnlyMatrix {
            matrix: DenseMatrix::from_row_slice(
                1,
                1,
                &[if self.nonfinite_jacobian {
                    f64::NAN
                } else {
                    2. * x[0]
                }],
            ),
            factors: self.factors.clone(),
            solves: self.solves.clone(),
            failures: self.failures.clone(),
            nonfinite_gradient: self.nonfinite_gradient,
        })
    }
}

#[test]
fn rejections_and_damping_retries_reuse_factorization() {
    let counts = CountedProblem::default();
    counts.failures.set(2);
    let mut problem = Problem::new(counts.clone());
    let mut solver = LevenbergMarquardt::new()
        .with_pivoted_qr()
        .with_tol_grad(0.);
    let state = solver
        .init(&mut problem, NllsState::new(vec![0.1]))
        .unwrap();
    let (mut state, reason) = solver.next_iter(&mut problem, state).unwrap();
    assert!(reason.is_none());
    assert_eq!(state.param(), &vec![0.1]);
    assert_eq!(counts.factors.get(), 1);
    assert_eq!(counts.solves.get(), 3);
    assert_eq!(counts.residuals.get(), 2);
    while state.param()[0] == 0.1 {
        (state, _) = solver.next_iter(&mut problem, state).unwrap();
        assert_eq!(counts.factors.get(), 1);
        assert!(counts.residuals.get() < 20);
    }
    let before = counts.residuals.get();
    let _ = solver.next_iter(&mut problem, state).unwrap();
    assert_eq!(counts.factors.get(), 2);
    assert_eq!(counts.jacobians.get(), 2);
    assert_eq!(counts.residuals.get(), before + 1);
    // Reusing the solver for a new run must discard the previous factor.
    solver.init(&mut problem, NllsState::new(vec![2.])).unwrap();
    assert_eq!(counts.factors.get(), 3);
}

#[test]
fn failed_solve_does_not_evaluate_trial_residual() {
    for nonfinite in [false, true] {
        let problem = CountedProblem {
            nonfinite_jacobian: nonfinite,
            ..Default::default()
        };
        problem.failures.set(10);
        let out = Executor::from_start(
            problem.clone(),
            LevenbergMarquardt::new()
                .with_pivoted_qr()
                .with_max_inner_attempts(3),
            vec![0.1],
        )
        .max_iter(20)
        .run()
        .unwrap();
        assert_eq!(out.reason, TerminationReason::SolverFailed);
        assert_eq!(out.param(), &vec![0.1]);
        assert_eq!(problem.residuals.get(), 1);
        assert_eq!(problem.factors.get(), 1);
        assert_eq!(problem.solves.get(), if nonfinite { 0 } else { 3 });
    }
}

#[test]
fn nonfinite_gradient_fails_before_solving() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let problem = CountedProblem {
            nonfinite_gradient: Some(value),
            ..Default::default()
        };
        let out = Executor::from_start(
            problem.clone(),
            LevenbergMarquardtQr::new(),
            vec![0.1],
        )
        .max_iter(20)
        .run()
        .unwrap();
        assert_eq!(out.reason, TerminationReason::SolverFailed);
        assert_eq!(out.param(), &vec![0.1]);
        assert_eq!(problem.residuals.get(), 1);
        assert_eq!(problem.solves.get(), 0);
    }
}

#[test]
fn actual_augmented_rank_loss_recovers_with_more_damping() {
    struct Redundant;
    impl Residual for Redundant {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![x[0] + x[1] - 1.])
        }
    }
    impl Jacobian for Redundant {
        type Jacobian = DenseMatrix;
        fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
            Ok(DenseMatrix::from_row_slice(1, 2, &[1., 1.]))
        }
    }
    for attempts in [1, 50] {
        let solver = LevenbergMarquardt::new()
            .with_tau(1e-30)
            .with_max_inner_attempts(attempts)
            .with_pivoted_qr()
            .with_rank_tolerance(1e-6);
        let result = Executor::from_start(Redundant, solver, vec![0., 0.])
            .max_iter(50)
            .run()
            .unwrap();
        if attempts == 1 {
            assert_eq!(result.reason, TerminationReason::SolverFailed);
            assert_eq!(result.cost_evals(), 1);
        } else {
            assert_eq!(result.reason, TerminationReason::SolverConverged);
            assert!(result.cost() < 1e-16);
            assert!((result.param()[0] - 0.5).abs() < 1e-8);
            assert!((result.param()[1] - 0.5).abs() < 1e-8);
            assert_eq!(result.cost_evals(), 2);
        }
    }
}

#[test]
fn callback_errors_propagate_unchanged() {
    for (error_at_residual, error_at_jacobian) in
        [(1, 0), (2, 0), (0, 1), (0, 2)]
    {
        let problem = CountedProblem {
            error_at_residual,
            error_at_jacobian,
            ..Default::default()
        };
        let result = Executor::from_start(
            problem,
            LevenbergMarquardt::new().with_pivoted_qr(),
            vec![2.],
        )
        .max_iter(20)
        .run();
        assert!(matches!(result, Err(CallbackError)));
    }
}

struct Linear;
impl Residual for Linear {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![x[0] + x[1] - 3., 2. * x[0] - x[1]])
    }
}
impl Jacobian for Linear {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(DenseMatrix::from_row_slice(2, 2, &[1., 1., 2., -1.]))
    }
}

#[test]
fn builder_and_direct_constructor_agree() {
    for solver in [
        LevenbergMarquardt::new()
            .with_tol_grad(1e-12)
            .with_pivoted_qr(),
        LevenbergMarquardtQr::new().with_tol_grad(1e-12),
    ] {
        let result =
            Executor::new(Linear, solver, NllsState::new(vec![0., 0.]))
                .max_iter(50)
                .run()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert!((result.param()[0] - 1.).abs() < 1e-10);
        assert!((result.param()[1] - 2.).abs() < 1e-10);
    }
}

#[cfg(feature = "problems")]
#[test]
fn nonlinear_and_deficient_problems() {
    use basin::problems::{PowellSingular, RosenbrockResiduals};
    let result = Executor::new(
        RosenbrockResiduals::<Vec<f64>>::new(),
        LevenbergMarquardt::new().with_pivoted_qr(),
        NllsState::new(vec![-1.2, 1.]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-15);
    let result = Executor::new(
        PowellSingular::<Vec<f64>>::new(),
        LevenbergMarquardt::new().with_pivoted_qr(),
        NllsState::new(vec![3., -1., 0., 1.]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-10);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

macro_rules! scaled_gradient_checks {
    ($solve:ident, $norm:ident, $scalar:ty, $scale:expr, $vector:ty, $matrix:ty, $make_vector:expr, $make_matrix:expr) => {
        #[test]
        fn $solve() {
            struct Fit;
            impl Residual for Fit {
                type Param = $vector;
                type Output = $vector;
                type Error = std::convert::Infallible;
                fn residual(
                    &self,
                    x: &$vector,
                ) -> Result<$vector, Self::Error> {
                    Ok(($make_vector)(&[$scale * (x[0] - 1.), 0.]))
                }
            }
            impl Jacobian for Fit {
                type Jacobian = $matrix;
                fn jacobian(
                    &self,
                    _: &$vector,
                ) -> Result<$matrix, Self::Error> {
                    Ok(($make_matrix)(&[$scale, 0., 0., 0.]))
                }
            }
            let start = ($make_vector)(&[0., 0.]);
            let residual = Fit.residual(&start).unwrap();
            let gradient =
                Fit.jacobian(&start).unwrap().mat_transpose_vec(&residual);
            assert!(residual.norm_squared().is_finite());
            assert!(gradient.norm_infinity().is_finite());
            assert!(gradient.norm_squared().is_infinite());

            let tau: $scalar = 1e-3;
            let tol = 10. * <$scalar>::EPSILON;
            for max_iter in [1, 20] {
                let solver =
                    LevenbergMarquardtQr::<_, _, $scalar>::new().with_tau(tau);
                let out = Executor::from_start(Fit, solver, start.clone())
                    .max_iter(max_iter)
                    .run()
                    .unwrap();
                if max_iter == 1 {
                    // The damped step is independent of residual scale.
                    assert!((out.param()[0] - 1. / (1. + tau)).abs() < tol);
                    assert!(out.cost().is_finite());
                    assert!(out.cost() < 0.5 * residual.norm_squared());
                } else {
                    assert_eq!(out.reason, TerminationReason::SolverConverged);
                    assert!((out.param()[0] - 1.).abs() < tol);
                }
                // An insensitive parameter retains its initial value.
                assert_eq!(out.param()[1], 0.);
            }
        }

        #[test]
        fn $norm() {
            let make_vector: fn(&[$scalar]) -> $vector = $make_vector;
            for values in [
                [<$scalar>::NAN, 1.],
                [1., <$scalar>::NAN],
                [<$scalar>::NAN, <$scalar>::INFINITY],
                [<$scalar>::INFINITY, <$scalar>::NAN],
                [<$scalar>::NAN, <$scalar>::NAN],
            ] {
                assert!(make_vector(&values).norm_infinity().is_nan());
            }
            for value in [<$scalar>::INFINITY, <$scalar>::NEG_INFINITY] {
                assert_eq!(
                    make_vector(&[1., value]).norm_infinity(),
                    <$scalar>::INFINITY
                );
            }
            assert_eq!(make_vector(&[0., 0.]).norm_infinity(), 0.);
            assert_eq!(make_vector(&[]).norm_infinity(), 0.);
            assert_eq!(make_vector(&[-$scale, 1.]).norm_infinity(), $scale);
        }
    };
}

scaled_gradient_checks!(
    f32_large_gradient,
    f32_gradient_norm,
    f32,
    1e10_f32,
    Vec<f32>,
    DenseMatrix<f32>,
    |a: &[f32]| a.to_vec(),
    |a| DenseMatrix::from_row_slice(2, 2, a)
);
scaled_gradient_checks!(
    f64_large_gradient,
    f64_gradient_norm,
    f64,
    1e100_f64,
    Vec<f64>,
    DenseMatrix<f64>,
    |a: &[f64]| a.to_vec(),
    |a| DenseMatrix::from_row_slice(2, 2, a)
);

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
macro_rules! backend_solve {
    ($name:ident, $scalar:ty, $vector:ty, $matrix:ty, $make_vector:expr, $make_matrix:expr) => {
        #[test]
        fn $name() {
            struct Fit;
            impl Residual for Fit {
                type Param = $vector;
                type Output = $vector;
                type Error = std::convert::Infallible;
                fn residual(
                    &self,
                    x: &$vector,
                ) -> Result<$vector, Self::Error> {
                    Ok(($make_vector)(&[x[0] + x[1] - 3., 2. * x[0] - x[1]]))
                }
            }
            impl Jacobian for Fit {
                type Jacobian = $matrix;
                fn jacobian(
                    &self,
                    _: &$vector,
                ) -> Result<$matrix, Self::Error> {
                    Ok(($make_matrix)(&[1., 1., 2., -1.]))
                }
            }
            let tol = <$scalar>::EPSILON.sqrt();
            let solver =
                LevenbergMarquardtQr::<_, _, $scalar>::new().with_tol_grad(tol);
            let out =
                Executor::from_start(Fit, solver, ($make_vector)(&[0., 0.]))
                    .max_iter(50)
                    .run()
                    .unwrap();
            assert_eq!(out.reason, TerminationReason::SolverConverged);
            assert!((out.param()[0] - 1.).abs() < tol);
            assert!((out.param()[1] - 2.).abs() < tol);
        }
    };
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra {
    use super::*;
    use backend_aliases::nalgebra::{DMatrix, DVector};
    scaled_gradient_checks!(
        f32_large_gradient,
        f32_gradient_norm,
        f32,
        1e10_f32,
        DVector<f32>,
        DMatrix<f32>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a)
    );
    scaled_gradient_checks!(
        f64_large_gradient,
        f64_gradient_norm,
        f64,
        1e100_f64,
        DVector<f64>,
        DMatrix<f64>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a)
    );
    backend_solve!(
        f64_solve,
        f64,
        DVector<f64>,
        DMatrix<f64>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a)
    );
    backend_solve!(
        f32_solve,
        f32,
        DVector<f32>,
        DMatrix<f32>,
        DVector::from_column_slice,
        |a| DMatrix::from_row_slice(2, 2, a)
    );
}
#[cfg(feature = "ndarray_all")]
mod ndarray {
    use super::*;
    use backend_aliases::ndarray::{Array1, Array2};
    scaled_gradient_checks!(
        f32_large_gradient,
        f32_gradient_norm,
        f32,
        1e10_f32,
        Array1<f32>,
        Array2<f32>,
        |a: &[f32]| Array1::from_vec(a.to_vec()),
        |a: &[f32]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap()
    );
    scaled_gradient_checks!(
        f64_large_gradient,
        f64_gradient_norm,
        f64,
        1e100_f64,
        Array1<f64>,
        Array2<f64>,
        |a: &[f64]| Array1::from_vec(a.to_vec()),
        |a: &[f64]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap()
    );
    backend_solve!(
        f64_solve,
        f64,
        Array1<f64>,
        Array2<f64>,
        |a: &[f64]| Array1::from_vec(a.to_vec()),
        |a: &[f64]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap()
    );
    backend_solve!(
        f32_solve,
        f32,
        Array1<f32>,
        Array2<f32>,
        |a: &[f32]| Array1::from_vec(a.to_vec()),
        |a: &[f32]| Array2::from_shape_vec((2, 2), a.to_vec()).unwrap()
    );
}
#[cfg(feature = "faer_all")]
mod faer {
    use super::*;
    use backend_aliases::faer::{Col, Mat};
    scaled_gradient_checks!(
        f32_large_gradient,
        f32_gradient_norm,
        f32,
        1e10_f32,
        Col<f32>,
        Mat<f32>,
        |a: &[f32]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f32]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j])
    );
    scaled_gradient_checks!(
        f64_large_gradient,
        f64_gradient_norm,
        f64,
        1e100_f64,
        Col<f64>,
        Mat<f64>,
        |a: &[f64]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f64]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j])
    );
    backend_solve!(
        f64_solve,
        f64,
        Col<f64>,
        Mat<f64>,
        |a: &[f64]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f64]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j])
    );
    backend_solve!(
        f32_solve,
        f32,
        Col<f32>,
        Mat<f32>,
        |a: &[f32]| Col::from_fn(a.len(), |i| a[i]),
        |a: &[f32]| Mat::from_fn(2, 2, |i, j| a[i * 2 + j])
    );
}
