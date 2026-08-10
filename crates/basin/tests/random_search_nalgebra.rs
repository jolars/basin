#![cfg(feature = "nalgebra")]

use basin::problems::BoothBoxed;
use basin::{
    BasicPopulationState, Executor, PopulationState, RandomSearch, State,
    StepOutcome,
};
use nalgebra::DVector;

/// Same seed → same trajectory, on the nalgebra backend. Sample
/// reproducibility is platform- and backend-independent.
#[test]
fn same_seed_yields_identical_trajectory() {
    let lower = DVector::from_vec(vec![-1.0, -1.0]);
    let upper = DVector::from_vec(vec![1.0, 1.0]);

    let result_a = Executor::new(
        BoothBoxed::<DVector<f64>>::new(lower.clone(), upper.clone()),
        RandomSearch::new(16, 42),
        BasicPopulationState::<DVector<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<DVector<f64>>::new(lower, upper),
        RandomSearch::new(16, 42),
        BasicPopulationState::<DVector<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Convergence on `BoothBoxed` with the tight `[-1, 1]²` box: the
/// constrained optimum sits at the corner `(1, 1)`; see the
/// equivalent Vec-backend test for the convergence-rate reasoning.
#[test]
fn converges_to_box_corner_on_tight_booth() {
    let lower = DVector::from_vec(vec![-1.0, -1.0]);
    let upper = DVector::from_vec(vec![1.0, 1.0]);

    let result = Executor::new(
        BoothBoxed::<DVector<f64>>::new(lower, upper),
        RandomSearch::new(64, 7),
        BasicPopulationState::<DVector<f64>>::with_size(64),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 0.05,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 0.05,
        "x[1] = {}",
        result.param()[1]
    );
}

/// Elitism keeps `state.cost()` non-increasing across `next_iter`.
#[test]
fn elite_keeps_cost_monotone_across_iterations() {
    let lower = DVector::from_vec(vec![-3.0, -3.0]);
    let upper = DVector::from_vec(vec![3.0, 3.0]);

    let mut stepper = Executor::new(
        BoothBoxed::<DVector<f64>>::new(lower, upper),
        RandomSearch::new(8, 99),
        BasicPopulationState::<DVector<f64>>::with_size(8),
    )
    .max_iter(50)
    .into_stepper()
    .unwrap();

    let mut prev = stepper.state().cost();
    while let StepOutcome::Continue = stepper.step().unwrap() {
        let current = stepper.state().cost();
        assert!(current <= prev, "cost increased: {prev} → {current}");
        prev = current;
    }
}

/// Sort + length invariants survive iteration on the nalgebra
/// backend.
#[test]
fn population_invariants_hold_after_iteration() {
    let lower = DVector::from_vec(vec![-2.0, -2.0]);
    let upper = DVector::from_vec(vec![2.0, 2.0]);
    let lambda = 12;

    let mut stepper = Executor::new(
        BoothBoxed::<DVector<f64>>::new(lower, upper),
        RandomSearch::new(lambda, 1234),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(10)
    .into_stepper()
    .unwrap();

    for _ in 0..10 {
        let StepOutcome::Continue = stepper.step().unwrap() else {
            break;
        };
        let state = stepper.state();
        assert_eq!(state.candidates().len(), lambda);
        assert_eq!(state.costs().len(), lambda);
        for window in state.costs().windows(2) {
            assert!(window[0] <= window[1]);
        }
    }
}
