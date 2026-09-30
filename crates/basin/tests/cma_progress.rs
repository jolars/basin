//! Distribution ownership, coherent CMA progress, and restart contracts.
use basin::{
    BoundedCmaEs, BoxConstraints, CmaEs, CostFunction, CountsMirror,
    DenseMatrix, EvalCounts, Executor, PopulationProgress, State,
};
use std::convert::Infallible;
struct Sphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Sphere {
    fn new(n: usize) -> Self {
        Self {
            lower: vec![-0.5; n],
            upper: vec![0.5; n],
        }
    }
}
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        assert_eq!(x.len(), self.lower.len());
        Ok(x.iter().map(|v| v * v).sum())
    }
}
impl BoxConstraints for Sphere {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}
#[test]
fn fresh_run_reevaluates_and_resets_distribution() {
    let prior = Executor::new(
        Sphere::new(2),
        CmaEs::<_, DenseMatrix>::new(91, 0.7).with_lambda(6),
        PopulationProgress::from_point(vec![0.4; 2]),
    )
    .max_iter(4)
    .run_with_solver()
    .unwrap();
    let fresh = Executor::new(Sphere::new(2), prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 7);
}
#[test]
fn categories_are_raw() {
    let mut state = PopulationProgress::<Vec<f64>>::from_point(vec![0.4; 2]);
    let counts = EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 4,
        jacobian_evals: 5,
        hessian_evals: 6,
        hessian_product_evals: 7,
    };
    state.mirror(&counts);
    assert_eq!(state.cost_evals(), 2);
}
#[test]
fn bounded_members_are_evaluated_points_with_raw_costs() {
    let result = Executor::new(
        Sphere::new(2),
        BoundedCmaEs::<_, DenseMatrix>::new(91, 2.0).with_lambda(6),
        PopulationProgress::from_point(vec![0.4; 2]),
    )
    .max_iter(3)
    .run()
    .unwrap();
    for (x, &f) in result.state.candidates().iter().zip(result.state.costs()) {
        assert!(x.iter().all(|v| (-0.5..=0.5).contains(v)));
        assert_eq!(f, x.iter().map(|v| v * v).sum::<f64>());
    }
}

fn coherent(state: &PopulationProgress<Vec<f64>>, bounded: bool) {
    assert_eq!(state.candidates().len(), state.costs().len());
    for (x, f) in state
        .candidates()
        .iter()
        .zip(state.costs().iter().copied())
        .chain(state.current())
        .chain(state.best())
    {
        if bounded {
            assert!(x.iter().all(|v| (-0.5..=0.5).contains(v)));
        }
        assert_eq!(f, x.iter().map(|v| v * v).sum::<f64>());
    }
}

macro_rules! continuation {
    ($name:ident, $solver:expr, $bounded:expr) => {
        #[test]
        fn $name() {
            let make = || {
                Executor::new(
                    Sphere::new(2),
                    $solver,
                    PopulationProgress::from_point(vec![0.4; 2]),
                )
            };
            let expected = make().max_iter(18).run_with_solver().unwrap();
            coherent(&expected.state, $bounded);
            for split in [0, 1, 5, 17] {
                let checkpoint = make()
                    .max_iter(split)
                    .run_with_solver()
                    .unwrap()
                    .into_checkpoint();
                let resumed = Executor::resume_from_checkpoint(
                    Sphere::new(2),
                    checkpoint,
                )
                .require_evaluated_state()
                .max_iter(18)
                .run_with_solver()
                .unwrap();
                coherent(&resumed.state, $bounded);
                assert_eq!(resumed.state, expected.state);
                assert_eq!(resumed.counts, expected.counts);
                #[cfg(feature = "serde")]
                assert_eq!(
                    postcard::to_allocvec(&resumed.solver).unwrap(),
                    postcard::to_allocvec(&expected.solver).unwrap()
                );
            }
            #[cfg(feature = "serde")]
            {
                // Let inference recover the full inner solver type without erasing it.
                fn restore<
                    T: serde::Serialize + serde::de::DeserializeOwned,
                >(
                    value: T,
                ) -> T {
                    postcard::from_bytes(
                        &postcard::to_allocvec(&value).unwrap(),
                    )
                    .unwrap()
                }
                let checkpoint = restore(
                    make()
                        .max_iter(7)
                        .run_with_solver()
                        .unwrap()
                        .into_checkpoint(),
                );
                let resumed = Executor::resume_from_checkpoint(
                    Sphere::new(2),
                    checkpoint,
                )
                .require_evaluated_state()
                .max_iter(18)
                .run_with_solver()
                .unwrap();
                assert_eq!(resumed.state, expected.state);
                assert_eq!(resumed.counts, expected.counts);
                assert_eq!(
                    postcard::to_allocvec(&resumed.solver).unwrap(),
                    postcard::to_allocvec(&expected.solver).unwrap()
                );
            }
        }
    };
}
continuation!(
    unbounded_owned_and_serialized_continuation,
    CmaEs::<_, DenseMatrix>::new(91, 0.7)
        .with_lambda(6)
        .with_stds(vec![1.0, 0.3]),
    false
);
continuation!(
    bounded_owned_and_serialized_continuation,
    BoundedCmaEs::<_, DenseMatrix>::new(91, 0.7)
        .with_lambda(6)
        .with_stds(vec![1.0, 0.3]),
    true
);
continuation!(
    injected_owned_and_serialized_continuation,
    basin::CmaInject::with_inner_solver(
        CmaEs::<_, DenseMatrix>::new(91, 0.7).with_lambda(6),
        basin::NelderMead::adaptive()
    )
    .with_k(2)
    .with_inner_max_iter(3),
    false
);
continuation!(
    bounded_injected_owned_and_serialized_continuation,
    basin::BoundedCmaInject::with_inner_solver(
        BoundedCmaEs::<_, DenseMatrix>::new(91, 0.7).with_lambda(6),
        basin::NelderMead::adaptive()
    )
    .with_k(2)
    .with_inner_max_iter(3),
    true
);

macro_rules! fresh {
    ($name:ident, $kind:ident) => {
        #[test]
        fn $name() {
            let make = || $kind::<_, DenseMatrix>::new(91, 0.7).with_lambda(6);
            let prior =
                Executor::from_start(Sphere::new(2), make(), vec![0.4; 2])
                    .max_iter(9)
                    .run_with_solver()
                    .unwrap();
            let warm = prior.state.param().clone();
            let actual =
                Executor::new(Sphere::new(2), prior.solver, prior.state)
                    .max_iter(0)
                    .run_with_solver()
                    .unwrap();
            let expected = Executor::from_start(Sphere::new(2), make(), warm)
                .max_iter(0)
                .run_with_solver()
                .unwrap();
            assert_eq!(actual.state, expected.state);
            assert_eq!(actual.solver.sigma(), Some(0.7));
            assert_eq!(actual.counts.cost_evals, 7);
            #[cfg(feature = "serde")]
            assert_eq!(
                postcard::to_allocvec(&actual.solver).unwrap(),
                postcard::to_allocvec(&expected.solver).unwrap()
            );
            let resized = Executor::from_start(
                Sphere::new(3),
                actual.solver,
                vec![0.2; 3],
            )
            .max_iter(0)
            .run_with_solver()
            .unwrap();
            let expected =
                Executor::from_start(Sphere::new(3), make(), vec![0.2; 3])
                    .max_iter(0)
                    .run_with_solver()
                    .unwrap();
            assert_eq!(resized.state, expected.state);
            #[cfg(feature = "serde")]
            assert_eq!(
                postcard::to_allocvec(&resized.solver).unwrap(),
                postcard::to_allocvec(&expected.solver).unwrap()
            );
        }
    };
}
fresh!(unbounded_fresh_model_and_dimension_reset, CmaEs);
fresh!(bounded_fresh_model_and_dimension_reset, BoundedCmaEs);

#[test]
fn unevaluated_mean_survives_reset_without_publishing_a_cost() {
    let mut state = PopulationProgress::<Vec<f64>>::from_point(vec![0.4; 2]);
    assert_eq!(state.current(), None);
    assert_eq!(state.best(), None);
    assert_eq!(state.evaluated_members(), None);
    state.reset();
    assert_eq!(state.param(), &vec![0.4; 2]);
    let mut state = Executor::new(
        Sphere::new(2),
        CmaEs::<_, DenseMatrix>::new(91, 0.7),
        state,
    )
    .max_iter(3)
    .run()
    .unwrap()
    .state;
    let mean = state.param().clone();
    let members = state.candidates().to_vec();
    state.reset();
    assert_eq!(state.param(), &mean);
    assert_eq!(state.candidates(), members);
    assert_eq!(state.current(), None);
    assert_eq!(state.best(), None);
    assert_eq!(state.counts(), &EvalCounts::default());
}

#[test]
fn nonfinite_population_has_no_objective_incumbent() {
    struct Nonfinite(f64);
    impl CostFunction for Nonfinite {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(self.0)
        }
    }
    for cost in [f64::NAN, f64::INFINITY] {
        let state = Executor::from_start(
            Nonfinite(cost),
            CmaEs::<_, DenseMatrix>::new(91, 0.7),
            vec![0.4; 2],
        )
        .max_iter(1)
        .run()
        .unwrap()
        .state;
        assert!(state.current().is_some());
        assert_eq!(state.best(), None);
        assert_eq!(state.best_counts(), None);
    }
}

#[test]
fn initial_scale_must_be_finite_and_positive() {
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| CmaEs::<Vec<f64>, DenseMatrix>::new(
                1, sigma
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(1, sigma)
            })
            .is_err()
        );
    }
}
