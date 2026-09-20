use basin::{
    BoxConstraints, CostFunction, DenseMatrix, Executor, GaussNewton, Gradient,
    HuberLoss, Jacobian, LevenbergMarquardt, LevenbergMarquardtQr, NllsState,
    Residual, RobustLeastSquares, Scalar, Solver, SquaredLoss,
    TerminationReason, Trf, TrustRegionReflective,
};
use std::{cell::Cell, convert::Infallible, rc::Rc};

struct Fit {
    outlier: bool,
    invalid_jacobian: bool,
    fused_calls: Rc<Cell<usize>>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Fit {
    fn new(outlier: bool) -> Self {
        Self {
            outlier,
            invalid_jacobian: false,
            fused_calls: Rc::default(),
            lower: vec![-1.0; 2],
            upper: vec![1.0; 2],
        }
    }
}
impl CostFunction for Fit {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Infallible> {
        panic!("the raw cost must not be called")
    }
}
impl Residual for Fit {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        let sum = x[0] + x[1];
        Ok(vec![
            sum,
            sum,
            sum,
            sum - if self.outlier { 10.0 } else { 0.0 },
        ])
    }
}
impl Jacobian for Fit {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(
            4,
            2,
            &[if self.invalid_jacobian { f64::NAN } else { 1.0 }; 8],
        ))
    }
    fn residual_and_jacobian(
        &self,
        x: &Vec<f64>,
    ) -> Result<(Vec<f64>, DenseMatrix), Infallible> {
        self.fused_calls.set(self.fused_calls.get() + 1);
        Ok((self.residual(x)?, self.jacobian(x)?))
    }
}
impl BoxConstraints for Fit {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn check_initial<S>(solver: impl Fn() -> S)
where
    S: Solver<
            RobustLeastSquares<Fit, HuberLoss>,
            NllsState<Vec<f64>>,
            Error = Infallible,
        >,
{
    for invalid in [false, true] {
        let mut fit = Fit::new(false);
        fit.invalid_jacobian = invalid;
        let calls = Rc::clone(&fit.fused_calls);
        let result = Executor::new(
            RobustLeastSquares::new(fit, HuberLoss),
            solver(),
            NllsState::new(vec![0.0; 2]),
        )
        .max_iter(10)
        .run_with_solver()
        .unwrap();
        assert_eq!(
            result.reason,
            if invalid {
                TerminationReason::SolverFailed
            } else {
                TerminationReason::SolverConverged
            }
        );
        assert_eq!(result.cost(), 0.0);
        assert_eq!(result.counts.residual_evals, 1);
        assert_eq!(result.counts.jacobian_evals, 1);
        assert_eq!(result.counts.cost_evals, 0);
        assert_eq!(calls.get(), 1);
    }
}

#[test]
fn zero_residuals_and_invalid_derivatives_use_one_fused_evaluation() {
    check_initial(GaussNewton::new);
    check_initial(LevenbergMarquardt::new);
    check_initial(LevenbergMarquardtQr::new);
    check_initial(Trf::new);
    check_initial(TrustRegionReflective::new);
}

fn check_rank_deficient<S>(solver: S)
where
    S: Solver<
            RobustLeastSquares<Fit, HuberLoss>,
            NllsState<Vec<f64>>,
            Error = Infallible,
        >,
{
    let result = Executor::new(
        RobustLeastSquares::new(Fit::new(true), HuberLoss),
        solver,
        NllsState::new(vec![0.0; 2]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param().iter().sum::<f64>() - 1.0 / 3.0).abs() < 1e-7);
    assert!((result.cost() - 28.0 / 3.0).abs() < 1e-12);
}

#[test]
fn damping_and_dense_svd_handle_a_rank_deficient_robust_model() {
    check_rank_deficient(LevenbergMarquardt::new());
    check_rank_deficient(LevenbergMarquardtQr::new());
    check_rank_deficient(Trf::new());
    check_rank_deficient(TrustRegionReflective::new());
}

struct IdentityResidual<F>(std::marker::PhantomData<F>);

impl<F: Scalar> Residual for IdentityResidual<F> {
    type Param = Vec<F>;
    type Output = Vec<F>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<F>) -> Result<Vec<F>, Infallible> {
        Ok(x.clone())
    }
}

impl<F: Scalar> Jacobian for IdentityResidual<F> {
    type Jacobian = DenseMatrix<F>;

    fn jacobian(&self, _: &Vec<F>) -> Result<DenseMatrix<F>, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[F::one()]))
    }
}

fn check_extreme_scales<F: Scalar>(large: F, small: F) {
    let one = F::one();
    let half = F::from_f64(0.5).unwrap();
    let tolerance = F::epsilon().sqrt();
    for scale in [large, small, F::max_value(), F::min_positive_value()] {
        let objective = RobustLeastSquares::new(
            IdentityResidual::<F>(std::marker::PhantomData),
            SquaredLoss,
        )
        .with_scale(scale);
        assert_eq!(
            objective.cost(&vec![one]).unwrap(),
            half,
            "scale={scale:?}"
        );
        assert_eq!(objective.gradient(&vec![one]).unwrap(), vec![one]);
        let result = Executor::from_start(
            objective,
            LevenbergMarquardt::<Vec<F>, DenseMatrix<F>, F>::default()
                .with_absolute_gradient_tolerance(tolerance),
            vec![one],
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(
            result.reason,
            TerminationReason::SolverConverged,
            "scale={scale:?}"
        );
        assert!(result.param()[0].abs() <= tolerance, "scale={scale:?}");
        assert!(
            result.cost() <= half * tolerance * tolerance,
            "scale={scale:?}"
        );
    }
}

#[test]
fn squared_loss_scale_cancels_in_cost_gradient_and_lm() {
    check_extreme_scales::<f32>(1e23, 1e-23);
    check_extreme_scales::<f64>(1e200, 1e-200);
}
