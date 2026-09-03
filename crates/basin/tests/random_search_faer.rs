#![cfg(feature = "faer_all")]

use crate::backend_aliases::faer::Col;
use basin::problems::BoothBoxed;
use basin::{
    BasicPopulationState, Executor, PopulationState, RandomSearch, State,
    StepOutcome,
};

fn col2(a: f64, b: f64) -> Col<f64> {
    Col::<f64>::from_fn(2, |i| if i == 0 { a } else { b })
}

#[test]
fn same_seed_yields_identical_trajectory() {
    let result_a = Executor::new(
        BoothBoxed::<Col<f64>>::new(col2(-1.0, -1.0), col2(1.0, 1.0)),
        RandomSearch::new(16, 42),
        BasicPopulationState::<Col<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<Col<f64>>::new(col2(-1.0, -1.0), col2(1.0, 1.0)),
        RandomSearch::new(16, 42),
        BasicPopulationState::<Col<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    let a = result_a.param();
    let b = result_b.param();
    assert_eq!(a.nrows(), b.nrows());
    for i in 0..a.nrows() {
        assert_eq!(a[i], b[i]);
    }
}

#[test]
fn converges_to_box_corner_on_tight_booth() {
    let result = Executor::new(
        BoothBoxed::<Col<f64>>::new(col2(-1.0, -1.0), col2(1.0, 1.0)),
        RandomSearch::new(64, 7),
        BasicPopulationState::<Col<f64>>::with_size(64),
    )
    .max_iter(200)
    .run()
    .unwrap();

    let p = result.param();
    assert!((p[0] - 1.0).abs() < 0.05, "x[0] = {}", p[0]);
    assert!((p[1] - 1.0).abs() < 0.05, "x[1] = {}", p[1]);
}

#[test]
fn elite_keeps_cost_monotone_across_iterations() {
    let mut stepper = Executor::new(
        BoothBoxed::<Col<f64>>::new(col2(-3.0, -3.0), col2(3.0, 3.0)),
        RandomSearch::new(8, 99),
        BasicPopulationState::<Col<f64>>::with_size(8),
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
    let lambda = 12;

    let mut stepper = Executor::new(
        BoothBoxed::<Col<f64>>::new(col2(-2.0, -2.0), col2(2.0, 2.0)),
        RandomSearch::new(lambda, 1234),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
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
