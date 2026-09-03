#![cfg(feature = "ndarray_all")]

use crate::backend_aliases::ndarray::Array1;
use basin::problems::BoothBoxed;
use basin::{
    BasicPopulationState, Executor, PopulationState, RandomSearch, State,
    StepOutcome,
};

#[test]
fn same_seed_yields_identical_trajectory() {
    let lower = Array1::from(vec![-1.0, -1.0]);
    let upper = Array1::from(vec![1.0, 1.0]);

    let result_a = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower.clone(), upper.clone()),
        RandomSearch::new(16, 42),
        BasicPopulationState::<Array1<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        RandomSearch::new(16, 42),
        BasicPopulationState::<Array1<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

#[test]
fn converges_to_box_corner_on_tight_booth() {
    let lower = Array1::from(vec![-1.0, -1.0]);
    let upper = Array1::from(vec![1.0, 1.0]);

    let result = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        RandomSearch::new(64, 7),
        BasicPopulationState::<Array1<f64>>::with_size(64),
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

#[test]
fn elite_keeps_cost_monotone_across_iterations() {
    let lower = Array1::from(vec![-3.0, -3.0]);
    let upper = Array1::from(vec![3.0, 3.0]);

    let mut stepper = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        RandomSearch::new(8, 99),
        BasicPopulationState::<Array1<f64>>::with_size(8),
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

#[test]
fn population_invariants_hold_after_iteration() {
    let lower = Array1::from(vec![-2.0, -2.0]);
    let upper = Array1::from(vec![2.0, 2.0]);
    let lambda = 12;

    let mut stepper = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        RandomSearch::new(lambda, 1234),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
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

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
