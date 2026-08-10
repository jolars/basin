use basin::problems::RastriginBoxed;
use basin::{
    BasicPopulationState, De, Executor, MaxCostEvals, PopulationState, State,
    StepOutcome,
};

/// Convergence on Rastrigin(D=5) within the [-5.12, 5.12] standard box.
/// DE/rand/1/bin is a global optimizer and Rastrigin is its canonical
/// multimodal stress test; with NP=30, F=0.8, CR=0.9 and 12_000 cost
/// evals (~400 generations) the elite reliably drops below 5; most
/// seeds end well under 1, with seed 42 hitting ~1e-3. The threshold
/// is the deterministic-CI floor across the seeds sampled, not the
/// typical performance. DE evaluates NP trials per generation, so it
/// needs a larger eval budget than steady-state SSGA to reach the same
/// threshold.
#[test]
fn converges_on_low_dim_rastrigin() {
    let problem = RastriginBoxed::<Vec<f64>>::with_standard_bounds(5);
    let solver = De::new(42).with_pop_size(30);
    let result = Executor::new(
        problem,
        solver,
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(u64::MAX)
    .terminate_on(MaxCostEvals(12_000))
    .run()
    .unwrap();

    assert!(
        result.cost() < 5.0,
        "Rastrigin(D=5) cost {} (expected < 5.0)",
        result.cost()
    );
}

/// Same seed → same trajectory. The reproducibility contract for every
/// stochastic solver in basin: a fixed `De::new(seed)` driven over the
/// same problem must produce bit-identical iterates across runs.
#[test]
fn same_seed_yields_identical_trajectory() {
    let problem_a = RastriginBoxed::<Vec<f64>>::with_standard_bounds(3);
    let problem_b = RastriginBoxed::<Vec<f64>>::with_standard_bounds(3);
    let result_a = Executor::new(
        problem_a,
        De::new(7).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(30)
    .run()
    .unwrap();
    let result_b = Executor::new(
        problem_b,
        De::new(7).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(30)
    .run()
    .unwrap();
    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Different seeds → different trajectories. Sanity check that the RNG
/// actually drives sampling.
#[test]
fn different_seeds_yield_different_trajectories() {
    let result_a = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(3),
        De::new(1).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(20)
    .run()
    .unwrap();
    let result_b = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(3),
        De::new(2).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(20)
    .run()
    .unwrap();
    assert_ne!(result_a.param(), result_b.param());
}

/// Greedy selection ensures `state.cost()` is non-increasing across
/// generations: index 0 is the sorted best, and trials only replace
/// targets when their cost is `≤`. Same monotonicity contract as
/// `RandomSearch`/`Ssga` so framework `CostTolerance`/`ParamTolerance`
/// are honest under DE dynamics.
#[test]
fn elite_keeps_cost_monotone_across_iterations() {
    let mut stepper = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(4),
        De::new(99).with_pop_size(20),
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(40)
    .into_stepper()
    .unwrap();

    let mut prev = stepper.state().cost();
    while let StepOutcome::Continue = stepper.step().unwrap() {
        let curr = stepper.state().cost();
        assert!(
            curr <= prev,
            "cost increased: prev = {prev}, current = {curr}"
        );
        prev = curr;
    }
}

/// Population invariants: candidates and costs are length `pop_size`
/// and sorted ascending so `param()`/`cost()` always surface the best.
#[test]
fn population_invariants_hold_after_iteration() {
    let pop_size = 12;
    let mut stepper = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(3),
        De::new(1234).with_pop_size(pop_size),
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(20)
    .into_stepper()
    .unwrap();

    for _ in 0..20 {
        let StepOutcome::Continue = stepper.step().unwrap() else {
            break;
        };
        let s = stepper.state();
        assert_eq!(s.candidates().len(), pop_size);
        assert_eq!(s.costs().len(), pop_size);
        for w in s.costs().windows(2) {
            assert!(w[0] <= w[1], "costs not sorted: {} > {}", w[0], w[1]);
        }
    }
}

/// Reinit-per-coord bound repair keeps every candidate inside the box
/// across iterations, even when F = 1.5 pushes donor vectors well
/// outside [-5.12, 5.12].
#[test]
fn population_stays_feasible_even_with_aggressive_f() {
    let n = 4;
    let lo = -5.12;
    let hi = 5.12;
    let mut stepper = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(n),
        // F > 1 routinely sends donors out of the box, exercising the
        // reinit-per-coord repair on every iteration.
        De::new(2024).with_pop_size(15).with_f(1.5),
        BasicPopulationState::<Vec<f64>>::with_size(1),
    )
    .max_iter(30)
    .into_stepper()
    .unwrap();

    for _ in 0..30 {
        let StepOutcome::Continue = stepper.step().unwrap() else {
            break;
        };
        for (i, x) in stepper.state().candidates().iter().enumerate() {
            for (j, &v) in x.iter().enumerate() {
                assert!(
                    v >= lo - 1e-12 && v <= hi + 1e-12,
                    "candidate {} component {} = {} outside [{}, {}]",
                    i,
                    j,
                    v,
                    lo,
                    hi
                );
            }
        }
    }
}
