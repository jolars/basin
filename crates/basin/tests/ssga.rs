use basin::problems::RastriginBoxed;
use basin::{
    BasicPopulationState, Executor, MaxCostEvals, PopulationState, Ssga, State, StepOutcome,
};

/// Convergence on Rastrigin(D=5) within the [-5.12, 5.12] standard box.
/// SSGA is a global optimizer and Rastrigin is its canonical multimodal
/// stress test; with pop=30 and 4000 cost evals the elite reliably
/// drops below 5. (Tightness depends on seed luck; this loose bound is
/// the deterministic-CI threshold.)
#[test]
fn converges_on_low_dim_rastrigin() {
    let problem = RastriginBoxed::<Vec<f64>>::with_standard_bounds(5);
    let solver = Ssga::new(42).with_pop_size(30);
    let result = Executor::new(
        problem,
        solver,
        BasicPopulationState::<Vec<f64>>::with_size(30),
    )
    .max_iter(u64::MAX)
    .terminate_on(MaxCostEvals(4000))
    .run()
    .unwrap();

    assert!(
        result.cost() < 5.0,
        "Rastrigin(D=5) cost {} (expected < 5.0)",
        result.cost()
    );
}

/// Same seed → same trajectory. The reproducibility contract for every
/// stochastic solver in basin: a fixed `Ssga::new(seed)` driven over
/// the same problem must produce bit-identical iterates across runs.
#[test]
fn same_seed_yields_identical_trajectory() {
    let problem_a = RastriginBoxed::<Vec<f64>>::with_standard_bounds(3);
    let problem_b = RastriginBoxed::<Vec<f64>>::with_standard_bounds(3);
    let result_a = Executor::new(
        problem_a,
        Ssga::new(7).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(10),
    )
    .max_iter(30)
    .run()
    .unwrap();
    let result_b = Executor::new(
        problem_b,
        Ssga::new(7).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(10),
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
        Ssga::new(1).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(10),
    )
    .max_iter(20)
    .run()
    .unwrap();
    let result_b = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(3),
        Ssga::new(2).with_pop_size(10),
        BasicPopulationState::<Vec<f64>>::with_size(10),
    )
    .max_iter(20)
    .run()
    .unwrap();
    assert_ne!(result_a.param(), result_b.param());
}

/// Replace-worst means `state.cost()` is non-increasing across
/// generations; the elite at position 0 can only be replaced by a
/// strictly better individual. Same monotonicity contract as
/// `RandomSearch` so framework `CostTolerance`/`ParamTolerance` are
/// honest under SSGA dynamics.
#[test]
fn elite_keeps_cost_monotone_across_iterations() {
    let mut stepper = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(4),
        Ssga::new(99).with_pop_size(20),
        BasicPopulationState::<Vec<f64>>::with_size(20),
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

/// Population invariants: candidates and costs are length pop_size and
/// sorted ascending so `param()`/`cost()` always surface the best.
#[test]
fn population_invariants_hold_after_iteration() {
    let pop_size = 12;
    let mut stepper = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(3),
        Ssga::new(1234).with_pop_size(pop_size),
        BasicPopulationState::<Vec<f64>>::with_size(pop_size),
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

/// Every individual after init and every offspring is clipped to the
/// problem box. Drive a small budget and check no candidate has drifted
/// outside the [-5.12, 5.12]^n standard Rastrigin box.
#[test]
fn population_stays_feasible() {
    let n = 4;
    let lo = -5.12;
    let hi = 5.12;
    let mut stepper = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(n),
        Ssga::new(2024).with_pop_size(15),
        BasicPopulationState::<Vec<f64>>::with_size(15),
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
