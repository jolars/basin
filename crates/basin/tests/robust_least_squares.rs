use basin::{
    ArctanLoss, CauchyLoss, CostFunction, DenseMatrix, Executor, GaussNewton,
    Gradient, HuberLoss, Jacobian, LevenbergMarquardt, LevenbergMarquardtQr,
    LossFunction, NllsState, Residual, RobustLeastSquares, SoftL1Loss,
    SquaredLoss, TerminationReason,
};

#[derive(Clone)]
struct Location;

impl Residual for Location {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![x[0], x[0], x[0], x[0] - 10.0])
    }
}

impl Jacobian for Location {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(DenseMatrix::from_row_slice(4, 1, &[1.0; 4]))
    }
}

#[test]
fn losses_have_analytic_values_and_derivatives() {
    for loss in [
        &SquaredLoss as &dyn LossFunction,
        &HuberLoss,
        &SoftL1Loss,
        &CauchyLoss,
        &ArctanLoss,
    ] {
        let zero = loss.evaluate(0.0);
        assert_eq!(zero.value, 0.0);
        assert_eq!(zero.first_derivative, 1.0);
        for z in [0.1, 0.8, 1.3, 4.0] {
            let h = 1e-5;
            let value = loss.evaluate(z);
            let plus = loss.evaluate(z + h);
            let minus = loss.evaluate(z - h);
            assert!(
                (value.first_derivative
                    - (plus.value - minus.value) / (2.0 * h))
                    .abs()
                    < 1e-8
            );
            assert!(
                (value.second_derivative
                    - (plus.first_derivative - minus.first_derivative)
                        / (2.0 * h))
                    .abs()
                    < 1e-8
            );
        }
    }
}

#[test]
fn adapter_exposes_the_robust_objective_and_gradient() {
    let problem = RobustLeastSquares::new(Location, HuberLoss);
    let (cost, gradient) = problem.cost_and_gradient(&vec![1.0 / 3.0]).unwrap();
    assert!((cost - 28.0 / 3.0).abs() < 1e-12);
    assert!(gradient[0].abs() < 1e-12);
    assert_eq!(cost, problem.cost(&vec![1.0 / 3.0]).unwrap());
}

macro_rules! location_solver {
    ($name:ident, $solver:expr) => {
        #[test]
        fn $name() {
            let result = Executor::new(
                RobustLeastSquares::new(Location, HuberLoss),
                $solver,
                NllsState::new(vec![0.0]),
            )
            .max_iter(100)
            .run()
            .unwrap();
            assert_eq!(result.reason, TerminationReason::SolverConverged);
            assert!((result.param()[0] - 1.0 / 3.0).abs() < 1e-7);
            assert!((result.cost() - 28.0 / 3.0).abs() < 1e-12);
        }
    };
}

location_solver!(normal_equations, LevenbergMarquardt::new());
location_solver!(pivoted_qr, LevenbergMarquardtQr::new());
location_solver!(gauss_newton, GaussNewton::new());
location_solver!(
    relative_gradient,
    LevenbergMarquardt::new()
        .with_absolute_gradient_tolerance(None)
        .with_gradient_orthogonality_tolerance(1e-8)
);

#[path = "support/robust_backend.rs"]
#[allow(dead_code)]
mod robust_backend;

fn bounded(
    lower: f64,
    upper: f64,
) -> robust_backend::Location<Vec<f64>, DenseMatrix, f64> {
    robust_backend::Location {
        make: |x| x.to_vec(),
        jacobian: DenseMatrix::from_row_slice(4, 1, &[1.0; 4]),
        lower: vec![lower],
        upper: vec![upper],
    }
}

#[test]
fn bounds_and_fixed_coordinates_use_the_robust_objective() {
    for fixed in [false, true] {
        let problem = RobustLeastSquares::new(
            bounded(if fixed { 0.2 } else { -1.0 }, 0.2),
            HuberLoss,
        );
        let result = Executor::from_start(
            problem.clone(),
            basin::TrustRegionReflective::new(),
            vec![0.0],
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert!((result.param()[0] - 0.2).abs() < 1e-6);
        assert!(
            (result.cost() - problem.cost(result.param()).unwrap()).abs()
                < 1e-12
        );
        if fixed {
            assert_eq!(result.state.jacobian_evals(), 0);
        }
    }
    let problem = RobustLeastSquares::new(bounded(-1.0, 0.2), HuberLoss);
    let result = Executor::from_start(problem, basin::Trf::new(), vec![0.0])
        .max_iter(100)
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param()[0] - 0.2).abs() < 1e-6);
}

#[test]
fn general_optimizer_and_finite_differences_use_the_same_loss() {
    let objective = RobustLeastSquares::new(bounded(-10.0, 10.0), HuberLoss);
    let result =
        Executor::from_start(objective, basin::Lbfgsb::new(), vec![0.0])
            .max_iter(100)
            .run()
            .unwrap();
    assert!((result.param()[0] - 1.0 / 3.0).abs() < 1e-7);
    let numerical =
        RobustLeastSquares::new(basin::FiniteDiff::new(Location), HuberLoss);
    let result =
        Executor::from_start(numerical, LevenbergMarquardt::new(), vec![0.0])
            .max_iter(100)
            .run()
            .unwrap();
    assert!((result.param()[0] - 1.0 / 3.0).abs() < 1e-7);
    let numerical = RobustLeastSquares::new(
        basin::BoundedFiniteDiff::new(
            bounded(-1.0, 0.2),
            vec![-1.0],
            vec![0.2],
        ),
        HuberLoss,
    );
    let result = Executor::from_start(
        numerical,
        basin::TrustRegionReflective::new(),
        vec![0.0],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!((result.param()[0] - 0.2).abs() < 1e-6);
}

#[test]
fn checkpoint_and_fresh_reuse_preserve_the_objective_and_counts() {
    fn check<S>(solver: impl Fn() -> S)
    where
        S: basin::Solver<
                RobustLeastSquares<
                    robust_backend::Location<Vec<f64>, DenseMatrix, f64>,
                    HuberLoss,
                >,
                NllsState<Vec<f64>>,
                Error = std::convert::Infallible,
            >,
    {
        let objective =
            || RobustLeastSquares::new(bounded(-10.0, 10.0), HuberLoss);
        let reference =
            Executor::new(objective(), solver(), NllsState::new(vec![0.0]))
                .max_iter(100)
                .run_with_solver()
                .unwrap();
        let partial =
            Executor::new(objective(), solver(), NllsState::new(vec![0.0]))
                .max_iter(1)
                .run_with_solver()
                .unwrap();
        let resumed = Executor::resume_from_checkpoint(
            objective(),
            partial.into_checkpoint(),
        )
        .max_iter(100)
        .run_with_solver()
        .unwrap();
        assert_eq!(reference.param(), resumed.param());
        assert_eq!(reference.cost(), resumed.cost());
        assert_eq!(reference.counts, resumed.counts);
        let fresh = Executor::new(
            objective(),
            resumed.solver,
            NllsState::new(vec![0.0]),
        )
        .max_iter(100)
        .run_with_solver()
        .unwrap();
        assert_eq!(reference.param(), fresh.param());
        assert_eq!(reference.cost(), fresh.cost());
        assert_eq!(reference.counts, fresh.counts);
    }
    check(GaussNewton::new);
    check(LevenbergMarquardt::new);
    check(LevenbergMarquardtQr::new);
    check(basin::Trf::new);
    check(basin::TrustRegionReflective::new);
}

#[derive(Clone)]
struct Domain {
    initial_invalid: bool,
    hard_error: bool,
}
impl Residual for Domain {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        if self.hard_error {
            return Err("residual error");
        }
        Ok(vec![if self.initial_invalid || x[0] > 0.6 {
            f64::NAN
        } else {
            x[0] - 1.0
        }])
    }
}
impl Jacobian for Domain {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[1.0]))
    }
}

#[test]
fn invalid_evaluations_cannot_be_hidden_by_a_bounded_loss() {
    let objective = RobustLeastSquares::new(
        Domain {
            initial_invalid: true,
            hard_error: false,
        },
        ArctanLoss,
    );
    assert_eq!(objective.cost(&vec![0.0]).unwrap(), f64::INFINITY);
    let result =
        Executor::from_start(objective, LevenbergMarquardt::new(), vec![0.0])
            .max_iter(10)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    let result = Executor::from_start(
        RobustLeastSquares::new(
            Domain {
                initial_invalid: false,
                hard_error: true,
            },
            HuberLoss,
        ),
        LevenbergMarquardt::new(),
        vec![0.0],
    )
    .run();
    assert!(matches!(result, Err("residual error")));
}

#[test]
fn rejected_trials_reuse_the_model_and_gauss_newton_fails_cleanly() {
    let objective = || {
        RobustLeastSquares::new(
            Domain {
                initial_invalid: false,
                hard_error: false,
            },
            HuberLoss,
        )
    };
    let result =
        Executor::from_start(objective(), LevenbergMarquardt::new(), vec![0.0])
            .max_iter(2)
            .run_with_solver()
            .unwrap();
    assert_eq!(result.param(), &vec![0.0]);
    assert_eq!(result.cost(), 0.5);
    assert_eq!(result.counts.residual_evals, 3);
    assert_eq!(result.counts.jacobian_evals, 1);
    let result =
        Executor::from_start(objective(), GaussNewton::new(), vec![0.0])
            .max_iter(2)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.cost(), f64::INFINITY);
}

struct CustomLoss;
impl LossFunction for CustomLoss {
    fn evaluate(&self, z: f64) -> basin::LossEvaluation {
        basin::LossEvaluation {
            value: 3.0 * z,
            first_derivative: 3.0,
            second_derivative: 0.0,
        }
    }
}

#[test]
fn custom_losses_and_scale_have_consistent_derivatives() {
    let custom = RobustLeastSquares::new(Location, CustomLoss).with_scale(0.25);
    let quadratic =
        RobustLeastSquares::new(Location, SquaredLoss).with_scale(7.0);
    assert!(
        (custom.cost(&vec![2.0]).unwrap()
            - 3.0 * quadratic.cost(&vec![2.0]).unwrap())
        .abs()
            < 1e-12
    );
    assert_eq!(custom.gradient(&vec![2.0]).unwrap(), vec![-6.0]);
    for scale in [0.25, 1.0, 2.0] {
        let robust =
            RobustLeastSquares::new(Location, HuberLoss).with_scale(scale);
        let x = scale / 3.0;
        assert!(robust.gradient(&vec![x]).unwrap()[0].abs() < 1e-12);
        let result =
            Executor::from_start(robust, LevenbergMarquardt::new(), vec![0.0])
                .max_iter(100)
                .run()
                .unwrap();
        assert!((result.param()[0] - x).abs() < 1e-7);
    }
}

#[test]
fn squared_loss_preserves_quadratic_iterations_and_counts() {
    let ordinary =
        Executor::from_start(Location, LevenbergMarquardt::new(), vec![0.0])
            .max_iter(100)
            .run_with_solver()
            .unwrap();
    let robust = Executor::from_start(
        RobustLeastSquares::new(Location, SquaredLoss),
        LevenbergMarquardt::new(),
        vec![0.0],
    )
    .max_iter(100)
    .run_with_solver()
    .unwrap();
    assert_eq!(ordinary.param(), robust.param());
    assert_eq!(ordinary.cost(), robust.cost());
    assert_eq!(ordinary.counts, robust.counts);
}
