//! Distribution ownership, coherent CMA progress, and restart contracts.
use basin::{
    BoundedCmaEs, CmaEs, CostFunction, CountsMirror, DenseMatrix, EvalCounts,
    Executor, PopulationProgress, State,
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

impl basin::Constraints for Sphere {
    type Set = basin::ConstraintSet<basin::BoxBounds<Vec<f64>>>;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
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

#[test]
fn fresh_run_reevaluates_and_resets_distribution() {
    let prior = Executor::new(
        basin::SamplingDomain::new(Sphere::new(2)),
        CmaEs::<_, DenseMatrix>::new(91, 0.7).with_lambda(6),
        PopulationProgress::from_point(vec![0.4; 2]),
    )
    .max_iter(4)
    .run_with_solver()
    .unwrap();
    let fresh = Executor::new(
        basin::SamplingDomain::new(Sphere::new(2)),
        prior.solver,
        prior.state,
    )
    .max_iter(0)
    .run_with_solver()
    .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 7);
}

#[test]
fn periodic_axes_refresh_on_schedule() {
    use basin::SymmetricEigen;
    let mut stepper = Executor::new(
        basin::SamplingDomain::new(Sphere::new(4)),
        CmaEs::<_, DenseMatrix>::new(91, 0.7)
            .with_lambda(8)
            .with_eigen_update_period(3),
        PopulationProgress::from_point(vec![0.4; 4]),
    )
    .into_stepper()
    .unwrap();
    for generation in 1..=9 {
        let before = stepper.solver().distribution().unwrap();
        let axes = before.principal_axis_stds.clone();
        let covariance = before.covariance.clone();
        assert!(matches!(
            stepper.step().unwrap(),
            basin::StepOutcome::Continue
        ));
        let after = stepper.solver().distribution().unwrap();
        assert_eq!(after.generation, generation);
        assert_ne!(*after.covariance, covariance);
        if generation % 3 == 0 {
            let (_, values) = after.covariance.try_eigh().unwrap();
            let expected: Vec<_> =
                values.iter().map(|v| v.max(1e-30).sqrt()).collect();
            assert_eq!(*after.principal_axis_stds, expected);
            assert_ne!(*after.principal_axis_stds, axes);
        } else {
            assert_eq!(*after.principal_axis_stds, axes);
        }
        assert_eq!(stepper.counts().cost_evals, (generation + 1) * 9);
    }
}

#[test]
fn explicit_every_generation_matches_default() {
    let run = |solver| {
        Executor::new(
            basin::SamplingDomain::new(Sphere::new(4)),
            solver,
            PopulationProgress::from_point(vec![0.4; 4]),
        )
        .max_iter(12)
        .run_with_solver()
        .unwrap()
    };
    let a = run(CmaEs::<_, DenseMatrix>::new(91, 0.7));
    let b =
        run(CmaEs::<_, DenseMatrix>::new(91, 0.7).with_eigen_update_period(1));
    assert_eq!(a.state, b.state);
    assert_eq!(a.counts, b.counts);
}

#[cfg(feature = "serde")]
#[test]
fn refresh_period_accepts_older_maps_and_rejects_zero() {
    type Cma = CmaEs<Vec<f64>, DenseMatrix>;
    let mut value = serde_json::to_value(Cma::new(91, 0.7)).unwrap();
    value.as_object_mut().unwrap().remove("eigen_update_period");
    let restored: Cma = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        serde_json::to_value(restored).unwrap()["eigen_update_period"],
        1
    );
    value["eigen_update_period"] = 0.into();
    assert!(serde_json::from_value::<Cma>(value).is_err());
}
#[test]
fn categories_are_raw() {
    let mut state = PopulationProgress::<Vec<f64>>::from_point(vec![0.4; 2]);
    let mut counts = EvalCounts::default();
    counts.cost_evals = 2;
    counts.gradient_evals = 3;
    counts.residual_evals = 4;
    counts.jacobian_evals = 5;
    counts.hessian_evals = 6;
    counts.hessian_product_evals = 7;
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
    ($name:ident, $solver:expr, $bounded:expr, $prepare:expr) => {
        #[test]
        fn $name() {
            let make = || {
                Executor::new(
                    ($prepare)(Sphere::new(2)),
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
                    ($prepare)(Sphere::new(2)),
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
                    ($prepare)(Sphere::new(2)),
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
    false,
    basin::SamplingDomain::new
);
continuation!(
    periodic_owned_and_serialized_continuation,
    CmaEs::<_, DenseMatrix>::new(91, 0.7)
        .with_lambda(6)
        .with_stds(vec![1.0, 0.3])
        .with_eigen_update_period(3),
    false,
    basin::SamplingDomain::new
);
continuation!(
    bounded_owned_and_serialized_continuation,
    BoundedCmaEs::<_, DenseMatrix>::new(91, 0.7)
        .with_lambda(6)
        .with_stds(vec![1.0, 0.3]),
    true,
    |p| p
);
continuation!(
    injected_owned_and_serialized_continuation,
    basin::CmaInject::with_inner_solver(
        CmaEs::<_, DenseMatrix>::new(91, 0.7).with_lambda(6),
        basin::NelderMead::adaptive()
    )
    .with_k(2)
    .with_inner_max_iter(3),
    false,
    basin::SamplingDomain::new
);
continuation!(
    bounded_injected_owned_and_serialized_continuation,
    basin::BoundedCmaInject::with_inner_solver(
        BoundedCmaEs::<_, DenseMatrix>::new(91, 0.7).with_lambda(6),
        basin::RelaxedInner::new(basin::NelderMead::adaptive())
    )
    .with_k(2)
    .with_inner_max_iter(3),
    true,
    |p| p
);

macro_rules! fresh {
    ($name:ident, $kind:ident, $prepare:expr) => {
        #[test]
        fn $name() {
            let make = || $kind::<_, DenseMatrix>::new(91, 0.7).with_lambda(6);
            let prior = Executor::from_start(
                ($prepare)(Sphere::new(2)),
                make(),
                vec![0.4; 2],
            )
            .max_iter(9)
            .run_with_solver()
            .unwrap();
            let warm = prior.state.param().clone();
            let actual = Executor::new(
                ($prepare)(Sphere::new(2)),
                prior.solver,
                prior.state,
            )
            .max_iter(0)
            .run_with_solver()
            .unwrap();
            let expected =
                Executor::from_start(($prepare)(Sphere::new(2)), make(), warm)
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
                ($prepare)(Sphere::new(3)),
                actual.solver,
                vec![0.2; 3],
            )
            .max_iter(0)
            .run_with_solver()
            .unwrap();
            let expected = Executor::from_start(
                ($prepare)(Sphere::new(3)),
                make(),
                vec![0.2; 3],
            )
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
fresh!(
    unbounded_fresh_model_and_dimension_reset,
    CmaEs,
    basin::SamplingDomain::new
);
fresh!(bounded_fresh_model_and_dimension_reset, BoundedCmaEs, |p| p);

#[test]
fn unevaluated_mean_survives_reset_without_publishing_a_cost() {
    let mut state = PopulationProgress::<Vec<f64>>::from_point(vec![0.4; 2]);
    assert_eq!(state.current(), None);
    assert_eq!(state.best(), None);
    assert_eq!(state.evaluated_members(), None);
    state.reset();
    assert_eq!(state.param(), &vec![0.4; 2]);
    let mut state = Executor::new(
        basin::SamplingDomain::new(Sphere::new(2)),
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

    impl basin::Constraints for Nonfinite {
        type Set = basin::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new()
        }
    }
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
