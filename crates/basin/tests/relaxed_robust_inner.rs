//! Box relaxation retains robust least-squares evaluation and accounting.

use basin::{
    BoundedCmaEs, BoundedCmaInject, BoxBounds, Constrained,
    ConstraintSolveError, Constraints, De, DeInject, DenseMatrix, Executor,
    GaussNewton, HuberLoss, Jacobian, LevenbergMarquardt, LevenbergMarquardtQr,
    NoConstraints, PointState, PopulationProgress, Problem, ProblemConstraints,
    RelaxedInner, Residual, RobustLeastSquares, Solver,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

#[derive(Clone, Default)]
struct Location {
    residual_calls: Arc<AtomicU64>,
    jacobian_calls: Arc<AtomicU64>,
    fail: Arc<AtomicBool>,
}
impl ProblemConstraints for Location {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        Constraints::new()
    }
}
impl Residual for Location {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.residual_calls.fetch_add(1, Ordering::Relaxed);
        if self.fail.load(Ordering::Relaxed) {
            return Err("residual failed");
        }
        Ok(vec![x[0], x[0], x[0], x[0] - 10.0])
    }
}
impl Jacobian for Location {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        self.jacobian_calls.fetch_add(1, Ordering::Relaxed);
        Ok(DenseMatrix::from_row_slice(4, 1, &[1.0; 4]))
    }
}

type Objective = RobustLeastSquares<
    Constrained<Location, Constraints<BoxBounds<Vec<f64>>>>,
    HuberLoss,
>;

fn objective(location: Location, upper: f64) -> Objective {
    RobustLeastSquares::new(
        Constrained::new(
            location,
            Constraints::new().with_bounds(BoxBounds::new(
                Some(vec![-1.0]),
                Some(vec![upper]),
            )),
        ),
        HuberLoss,
    )
    .with_scale(0.6)
}

macro_rules! relaxed_solver {
    ($name:ident, $solver:expr) => {
        #[test]
        fn $name() {
            let reference = Executor::from_start(
                RobustLeastSquares::new(Location::default(), HuberLoss)
                    .with_scale(0.6),
                $solver,
                vec![0.0],
            )
            .max_iter(100)
            .run()
            .unwrap();
            let location = Location::default();
            let result = Executor::from_start(
                objective(location.clone(), 0.1),
                RelaxedInner::new($solver),
                vec![0.0],
            )
            .max_iter(100)
            .run()
            .unwrap();
            assert!((result.param()[0] - 0.2).abs() < 1e-7);
            assert!((result.cost() - 5.76).abs() < 1e-12);
            assert_eq!(result.param(), reference.param());
            assert_eq!(result.counts, reference.counts);
            assert_eq!(result.report.code(), reference.report.code());
            assert_eq!(
                result.counts.residual_evals,
                location.residual_calls.load(Ordering::Relaxed)
            );
            assert_eq!(
                result.counts.jacobian_evals,
                location.jacobian_calls.load(Ordering::Relaxed)
            );
            assert_eq!(result.counts.cost_evals, 0);
        }
    };
}
relaxed_solver!(relaxed_lm_retains_loss_and_scale, LevenbergMarquardt::new());
relaxed_solver!(
    relaxed_qr_retains_loss_and_scale,
    LevenbergMarquardtQr::new()
);
relaxed_solver!(
    relaxed_gauss_newton_retains_loss_and_scale,
    GaussNewton::new()
);

macro_rules! injected_solver {
    ($name:ident, $solver:expr) => {
        #[test]
        fn $name() {
            let location = Location::default();
            let problem = objective(location.clone(), 1.0);
            let result = Executor::new(
                problem,
                $solver,
                PopulationProgress::from_point(vec![0.8]),
            )
            .max_iter(100)
            .run()
            .unwrap();
            assert!((result.param()[0] - 0.2).abs() < 1e-4);
            assert!((result.cost() - 5.76).abs() < 1e-8);
            assert!(result.counts.jacobian_evals > 0);
            assert!(result.counts.cost_evals > 0);
            assert_eq!(
                result.counts.residual_evals + result.counts.cost_evals,
                location.residual_calls.load(Ordering::Relaxed)
            );
            assert_eq!(
                result.counts.jacobian_evals,
                location.jacobian_calls.load(Ordering::Relaxed)
            );
        }
    };
}
injected_solver!(
    bounded_cma_inject_accepts_robust_lm,
    BoundedCmaInject::with_inner_solver(
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42, 0.3),
        RelaxedInner::new(LevenbergMarquardt::new()),
    )
);
injected_solver!(
    de_inject_accepts_robust_lm,
    DeInject::with_inner_solver(
        De::new(42),
        RelaxedInner::new(LevenbergMarquardt::new()),
    )
);

#[test]
fn relaxed_robust_callback_errors_retain_counts() {
    let location = Location::default();
    let mut problem = Problem::new(objective(location.clone(), 0.1));
    let mut solver = RelaxedInner::new(LevenbergMarquardt::new());
    problem.counts_mut().cost_evals = 7;
    problem.counts_mut().hessian_evals = 3;
    let state = solver
        .init(&mut problem, PointState::new(vec![0.0]))
        .unwrap();
    assert_eq!(problem.counts().residual_evals, 1);
    assert_eq!(problem.counts().jacobian_evals, 1);
    location.fail.store(true, Ordering::Relaxed);
    let error = solver.next_iter(&mut problem, state).unwrap_err();
    assert_eq!(error, ConstraintSolveError::Evaluation("residual failed"));
    assert_eq!(problem.counts().cost_evals, 7);
    assert_eq!(problem.counts().hessian_evals, 3);
    assert_eq!(problem.counts().residual_evals, 2);
    assert_eq!(problem.counts().jacobian_evals, 1);
    assert_eq!(location.residual_calls.load(Ordering::Relaxed), 2);
    assert_eq!(location.jacobian_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn relaxed_robust_initial_error_retains_attempted_combined_counts() {
    let location = Location::default();
    location.fail.store(true, Ordering::Relaxed);
    let mut problem = Problem::new(objective(location.clone(), 0.1));
    let error = RelaxedInner::new(LevenbergMarquardt::new())
        .init(&mut problem, PointState::new(vec![0.0]))
        .unwrap_err();
    assert_eq!(error, ConstraintSolveError::Evaluation("residual failed"));
    assert_eq!(problem.counts().residual_evals, 1);
    assert_eq!(problem.counts().jacobian_evals, 1);
    assert_eq!(location.residual_calls.load(Ordering::Relaxed), 1);
    assert_eq!(location.jacobian_calls.load(Ordering::Relaxed), 0);
}
