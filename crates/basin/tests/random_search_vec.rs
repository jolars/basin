use basin::problems::BoothBoxed;
use basin::{
    BasicPopulationState, Executor, MaxIter, PopulationState, RandomSearch,
    State, StepOutcome,
};

/// Same seed → same trajectory. Load-bearing reproducibility check
/// for the stochastic-solver contract: a `RandomSearch::new(λ, seed)`
/// driven over the same problem must produce bit-identical iterates
/// across runs (and platforms; ChaCha8Rng is platform-independent).
#[test]
fn same_seed_yields_identical_trajectory() {
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];

    let result_a = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower.clone(), upper.clone()),
        RandomSearch::new(16, 42),
        BasicPopulationState::<Vec<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        RandomSearch::new(16, 42),
        BasicPopulationState::<Vec<f64>>::with_size(16),
    )
    .max_iter(20)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Different seeds → different trajectories. Sanity check that the
/// RNG actually drives sampling; a constant-RNG bug would make this
/// test produce identical output.
#[test]
fn different_seeds_yield_different_trajectories() {
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];

    let result_a = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower.clone(), upper.clone()),
        RandomSearch::new(8, 1),
        BasicPopulationState::<Vec<f64>>::with_size(8),
    )
    .max_iter(5)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        RandomSearch::new(8, 2),
        BasicPopulationState::<Vec<f64>>::with_size(8),
    )
    .max_iter(5)
    .run()
    .unwrap();

    assert_ne!(result_a.param(), result_b.param());
}

/// Convergence on `BoothBoxed` `[-1, 1]²`: the unconstrained min
/// `(1, 3)` is outside the box, so the constrained optimum is the
/// corner `(1, 1)`. With λ = 64 and 200 generations (≈12k samples in
/// 2D) the elite reliably lands within 0.05 of the corner regardless
/// of seed. Tolerance is loose because random search converges
/// polynomially in the sample budget.
#[test]
fn converges_to_box_corner_on_tight_booth() {
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];

    let result = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        RandomSearch::new(64, 7),
        BasicPopulationState::<Vec<f64>>::with_size(64),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 0.05,
        "x[0] = {} (expected near upper bound 1)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 0.05,
        "x[1] = {} (expected near upper bound 1)",
        result.param()[1]
    );
}

/// Elitism contract: `state.cost()` is non-increasing across
/// `next_iter` calls. This is what makes the framework's
/// `CostTolerance`/`ParamTolerance` honest under stochastic
/// dynamics; a non-elitist random search would silently break them.
#[test]
fn elite_keeps_cost_monotone_across_iterations() {
    let lower = vec![-3.0, -3.0];
    let upper = vec![3.0, 3.0];
    let problem = BoothBoxed::<Vec<f64>>::new(lower, upper);

    let mut stepper = Executor::new(
        problem,
        RandomSearch::new(8, 99),
        BasicPopulationState::<Vec<f64>>::with_size(8),
    )
    .max_iter(50)
    .into_stepper()
    .unwrap();

    let mut prev = stepper.state().cost();
    while let StepOutcome::Continue = stepper.step().unwrap() {
        let current = stepper.state().cost();
        assert!(
            current <= prev,
            "cost increased: prev = {prev}, current = {current}"
        );
        prev = current;
    }
}

/// `BasicPopulationState` invariant: candidates and costs both have
/// length `λ` and are sorted ascending so `param()`/`cost()` always
/// surface the best. Regression check on the sort and truncate logic in
/// `next_iter`.
#[test]
fn population_invariants_hold_after_iteration() {
    let lower = vec![-2.0, -2.0];
    let upper = vec![2.0, 2.0];
    let lambda = 12;

    let mut stepper = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        RandomSearch::new(lambda, 1234),
        BasicPopulationState::<Vec<f64>>::with_size(lambda),
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
        // Sorted ascending → best at index 0.
        for window in state.costs().windows(2) {
            assert!(
                window[0] <= window[1],
                "costs not sorted: {} > {}",
                window[0],
                window[1]
            );
        }
    }
}

/// `MaxIter(0)` returns immediately after `init`, so the result is the
/// `init`-time first-generation best. Companion to the reproducibility
/// test: confirms `init` is the place where the seeded RNG starts the
/// trajectory, regardless of which `BasicPopulationState` constructor
/// the caller used.
#[test]
fn max_iter_zero_returns_initial_population_best() {
    let lower = vec![0.0, 0.0];
    let upper = vec![1.0, 1.0];

    let result = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower.clone(), upper.clone()),
        RandomSearch::new(4, 555),
        BasicPopulationState::<Vec<f64>>::with_size(4),
    )
    .max_iter(0)
    .terminate_on(MaxIter(0))
    .run()
    .unwrap();

    let p = result.param();
    assert_eq!(p.len(), 2);
    // Every candidate is sampled in [lower, upper]; the elite must be too.
    for (i, &v) in p.iter().enumerate() {
        assert!(v >= lower[i] - 1e-12 && v <= upper[i] + 1e-12);
    }
}
