//! First-order progress stays coherent across fresh runs, rejection, and continuation.

use basin::{
    CostFunction, DenseMatrix, Executor, FirstOrderState, Gradient,
    GradientDescent, GradientState, Hessian, HessianProduct,
    ProjectedGradientDescent, State, TrustRegion,
};
use std::convert::Infallible;

struct Quadratic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Quadratic {
    fn new(n: usize) -> Self {
        Self {
            lower: vec![-5.0; n],
            upper: vec![5.0; n],
        }
    }
}

impl basin::ProblemConstraints for Quadratic {
    type Constraints = basin::NoConstraints;
    fn constraint_set(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}
fn bounded(
    p: Quadratic,
) -> basin::Constrained<Quadratic, basin::Constraints<basin::BoxBounds<Vec<f64>>>>
{
    let bounds =
        basin::BoxBounds::new(Some(p.lower.clone()), Some(p.upper.clone()));
    basin::Constrained::new(p, basin::Constraints::new().with_bounds(bounds))
}
impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, x)| (i + 1) as f64 * x * x)
            .sum())
    }
}
impl Gradient for Quadratic {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, x)| 2.0 * (i + 1) as f64 * x)
            .collect())
    }
}
impl Hessian for Quadratic {
    type Hessian = DenseMatrix;
    fn hessian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        use basin::MatrixFromDiagonal;
        Ok(DenseMatrix::from_diagonal(
            &(0..x.len()).map(|i| 2.0 * (i + 1) as f64).collect(),
        ))
    }
}
impl HessianProduct for Quadratic {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Infallible> {
        self.gradient(v)
    }
}

fn consistent(state: &FirstOrderState<Vec<f64>>) {
    let (x, cost, gradient) = state.current().unwrap();
    let problem = Quadratic::new(x.len());
    assert_eq!(cost, problem.cost(x).unwrap());
    assert_eq!(gradient, &problem.gradient(x).unwrap());
    assert_eq!(state.cost_evals(), state.counts().cost_evals);
    assert_eq!(state.gradient_evals(), state.counts().gradient_evals);
}

macro_rules! lifecycle {
    ($name:ident, $solver:expr, $prepare:expr) => {
        #[test]
        fn $name() {
            let make = || {
                Executor::new(
                    ($prepare)(Quadratic::new(2)),
                    $solver,
                    FirstOrderState::new(vec![3.0, 2.0]),
                )
                .require_evaluated_state()
            };
            let expected = make().max_iter(5).run_with_solver().unwrap();
            consistent(&expected.state);
            for split in [0, 1, 3] {
                let initial = make().max_iter(split).run_with_solver().unwrap();
                let resumed = Executor::resume_from_checkpoint(
                    ($prepare)(Quadratic::new(2)),
                    initial.into_checkpoint(),
                )
                .require_evaluated_state()
                .max_iter(5)
                .run_with_solver()
                .unwrap();
                assert_eq!(resumed.state, expected.state);
                assert_eq!(resumed.counts, expected.counts);
            }
            let mut state = expected.state;
            state
                .replace(vec![3.0, 2.0, 1.0], 0.0, vec![0.0; 3])
                .unwrap();
            let initialized = Executor::new(
                ($prepare)(Quadratic::new(3)),
                expected.solver,
                state,
            )
            .require_evaluated_state()
            .max_iter(0)
            .run_with_solver()
            .unwrap();
            consistent(&initialized.state);
            assert_eq!(initialized.state.iter(), 0);
            assert_eq!(initialized.state.best_iter(), 0);
            assert_eq!(initialized.counts.cost_evals, 1);
            assert_eq!(initialized.counts.gradient_evals, 1);
            assert_eq!(initialized.counts.total_work(), 2);
            let resumed = Executor::resume_from_checkpoint(
                ($prepare)(Quadratic::new(3)),
                initialized.into_checkpoint(),
            )
            .max_iter(3)
            .run_with_solver()
            .unwrap();
            let rebuilt = Executor::from_start(
                ($prepare)(Quadratic::new(3)),
                $solver,
                vec![3.0, 2.0, 1.0],
            )
            .max_iter(3)
            .run_with_solver()
            .unwrap();
            assert_eq!(resumed.state, rebuilt.state);
            assert_eq!(resumed.counts, rebuilt.counts);
        }
    };
}

lifecycle!(
    descent_momentum,
    GradientDescent::new(0.05).with_momentum(0.8),
    |p| p
);
lifecycle!(
    projected_descent,
    ProjectedGradientDescent::new(0.05),
    bounded
);
lifecycle!(
    trust_region_exact,
    TrustRegion::new().with_radius(0.2),
    |p| p
);
lifecycle!(
    trust_region_matrix_free,
    TrustRegion::matrix_free().with_radius(0.2),
    |p| p
);

#[test]
fn curvature_work_is_separate_from_gradients() {
    let exact = Executor::from_start(
        Quadratic::new(2),
        TrustRegion::new(),
        vec![3.0, 2.0],
    )
    .require_evaluated_state()
    .max_iter(1)
    .run()
    .unwrap();
    consistent(&exact.state);
    assert_eq!(exact.state.counts().hessian_evals, 1);
    assert_eq!(exact.state.gradient_evals(), 2);
    let matrix_free = Executor::from_start(
        Quadratic::new(2),
        TrustRegion::matrix_free(),
        vec![3.0, 2.0],
    )
    .require_evaluated_state()
    .max_iter(1)
    .run()
    .unwrap();
    consistent(&matrix_free.state);
    assert!(matrix_free.state.counts().hessian_product_evals > 0);
    assert_eq!(matrix_free.state.gradient_evals(), 2);
}

// A non-Clone component whose evolving step scale must reset only on fresh runs.
struct StatefulSearch {
    calls: u64,
}
impl<P: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>>
    basin::LineSearch<P, Vec<f64>> for StatefulSearch
{
    type Error = Infallible;
    fn reset(&mut self) {
        self.calls = 0;
    }
    fn next(
        &mut self,
        _: &mut basin::Problem<P>,
        _: &Vec<f64>,
        _: f64,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<f64, Infallible> {
        self.calls += 1;
        Ok(0.02 / self.calls as f64)
    }
}

lifecycle!(
    descent_stateful_search,
    GradientDescent::with_line_search(StatefulSearch { calls: 99 })
        .with_momentum(0.8),
    |p| p
);
lifecycle!(
    projected_stateful_search,
    ProjectedGradientDescent::with_line_search(StatefulSearch { calls: 99 }),
    bounded
);
lifecycle!(
    nonlinear_cg_stateful_search,
    basin::NonlinearCg::with_line_search(StatefulSearch { calls: 99 }),
    |p| p
);

#[cfg(feature = "serde")]
fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
) -> T {
    postcard::from_bytes(&postcard::to_allocvec(value).unwrap()).unwrap()
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoints_preserve_solver_history() {
    macro_rules! check {
        ($solver:expr, $prepare:expr) => {{
            let make = || {
                Executor::from_start(
                    ($prepare)(Quadratic::new(3)),
                    $solver,
                    vec![3.0; 3],
                )
            };
            let expected = make().max_iter(5).run_with_solver().unwrap();
            let checkpoint = round_trip(
                &make()
                    .max_iter(2)
                    .run_with_solver()
                    .unwrap()
                    .into_checkpoint(),
            );
            let resumed = Executor::resume_from_checkpoint(
                ($prepare)(Quadratic::new(3)),
                checkpoint,
            )
            .require_evaluated_state()
            .max_iter(5)
            .run_with_solver()
            .unwrap();
            assert_eq!(resumed.state, expected.state);
            assert_eq!(resumed.counts, expected.counts);
            assert_eq!(
                postcard::to_allocvec(&resumed.solver).unwrap(),
                postcard::to_allocvec(&expected.solver).unwrap()
            );
        }};
    }
    check!(GradientDescent::new(0.05).with_momentum(0.8), |p| p);
    check!(ProjectedGradientDescent::new(0.05), bounded);
    check!(basin::NonlinearCg::new(), |p| p);
    check!(TrustRegion::new().with_radius(0.2), |p| p);
    check!(TrustRegion::matrix_free().with_radius(0.2), |p| p);
    check!(
        TrustRegion::with_subproblem(basin::Dogleg).with_radius(0.2),
        |p| p
    );
    check!(
        TrustRegion::with_subproblem(basin::MoreSorensen::new())
            .with_radius(0.2),
        |p| p
    );
    check!(
        TrustRegion::with_subproblem(basin::CauchyPoint).with_radius(0.2),
        |p| p
    );
    check!(
        TrustRegion::with_subproblem(
            basin::Steihaug::new().with_forcing_parameters(0.1, 0.5)
        )
        .with_radius(0.2),
        |p| p
    );
}
