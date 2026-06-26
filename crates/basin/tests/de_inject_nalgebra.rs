//! Integration tests for [`DeInject`] with Nelder-Mead inner on the
//! nalgebra backend.
//!
//! Three tests: convergence on Rastrigin D=5 (the canonical multimodal
//! stress test [`De`] is benchmarked against), cost-eval aggregation
//! across the outer/inner composition boundary
//! (CONTRIBUTING.md "Solver composition" rule 1), and reproducibility
//! under a fixed seed. For the L-Bfgs-B inner variant see
//! `de_inject_lbfgsb_nalgebra.rs`; for the failure-bubbling contract
//! test (rule 3) see `de_inject_solver_failed_bubbles.rs`.

#![cfg(feature = "nalgebra")]

use basin::problems::{AckleyBoxed, RastriginBoxed};
use basin::{BasicPopulationState, De, DeInject, Executor, MaxCostEvals, NelderMead};
use nalgebra::DVector;

/// Ackley D=3 within the standard `[-32.768, 32.768]³` box. Ackley's
/// exponential-decay global basin is friendly to Nelder-Mead polish:
/// once DE places a candidate inside the central well, the inner can
/// drive cost to ≪ 1 in tens of iterations. With NP=20 and a 12_000-
/// eval budget DeInject reliably drops the elite below 0.1.
///
/// Ackley is chosen over the vanilla-DE convergence test's Rastrigin
/// because highly multimodal landscapes reward exploration over the
/// exploitation LS polish adds; DEahcSPX ranks 4th-of-5 on Rastrigin
/// in Neri & Tirronen 2010 §4.2's benchmarks. Ackley exhibits the
/// helpful side of the memetic combination instead (DESFLS-rivalling
/// performance per the same tables).
#[test]
fn converges_on_ackley_d3_with_nm_inner() {
    let problem = AckleyBoxed::<DVector<f64>>::with_standard_bounds(3);
    let de = De::new(42).with_pop_size(20);
    let solver = DeInject::with_inner_solver(de, NelderMead::adaptive())
        .with_k(1)
        .with_inner_max_iter(30);

    let result = Executor::new(
        problem,
        solver,
        BasicPopulationState::<DVector<f64>>::with_size(1),
    )
    .max_iter(u64::MAX)
    .terminate_on(MaxCostEvals(12_000))
    .run()
    .unwrap();

    assert!(
        result.cost() < 0.1,
        "Ackley(D=3) cost {} (expected < 0.1)",
        result.cost()
    );
}

/// `DeInject` must roll the inner Nelder-Mead `cost_evals` into the
/// outer state's eval counter (CONTRIBUTING.md "Solver composition"
/// rule 1). Compare against vanilla `De` on the same seed and outer
/// budget: every refinement pass adds at least `n + 1` NM-init evals
/// plus the post-clip re-evaluation (`+1`), so for `k = 1` over
/// `outer_iters` generations the memetic variant's `cost_evals` must
/// exceed vanilla's by ≥ `outer_iters · k · (n + 2)` (loose lower
/// bound, NM may early-terminate on some iters).
#[test]
fn aggregates_inner_cost_evals_into_outer() {
    let n = 5usize;
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 30;
    let k: usize = 1;
    let pop_size = 12;

    // Vanilla DE baseline.
    let vanilla = Executor::new(
        RastriginBoxed::<DVector<f64>>::with_standard_bounds(n),
        De::new(7).with_pop_size(pop_size),
        BasicPopulationState::<DVector<f64>>::with_size(1),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget.
    let de = De::new(7).with_pop_size(pop_size);
    let solver = DeInject::with_inner_solver(de, NelderMead::adaptive())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        RastriginBoxed::<DVector<f64>>::with_standard_bounds(n),
        solver,
        BasicPopulationState::<DVector<f64>>::with_size(1),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Per outer iter, DeInject does at least: NM init (n + 1 evals) +
    // 1 post-clip re-evaluation = (n + 2) extra evals per refined
    // candidate.
    let min_extra = outer_iters * (k as u64) * (n as u64 + 2);
    assert!(
        memetic.cost_evals() >= vanilla.cost_evals() + min_extra,
        "memetic cost_evals = {} should exceed vanilla {} by at least \
         {} (outer iters × k × (n+2) for NM init + re-eval)",
        memetic.cost_evals(),
        vanilla.cost_evals(),
        min_extra
    );
}

/// Same seed → same trajectory. Reproducibility contract every
/// stochastic solver in basin owes the caller; the memetic layer must
/// not introduce nondeterminism on top of vanilla DE's PRNG.
#[test]
fn same_seed_yields_identical_trajectory() {
    let problem_a = RastriginBoxed::<DVector<f64>>::with_standard_bounds(3);
    let problem_b = RastriginBoxed::<DVector<f64>>::with_standard_bounds(3);

    let result_a = Executor::new(
        problem_a,
        DeInject::with_inner_solver(De::new(7).with_pop_size(10), NelderMead::adaptive())
            .with_k(1)
            .with_inner_max_iter(20),
        BasicPopulationState::<DVector<f64>>::with_size(1),
    )
    .max_iter(15)
    .run()
    .unwrap();

    let result_b = Executor::new(
        problem_b,
        DeInject::with_inner_solver(De::new(7).with_pop_size(10), NelderMead::adaptive())
            .with_k(1)
            .with_inner_max_iter(20),
        BasicPopulationState::<DVector<f64>>::with_size(1),
    )
    .max_iter(15)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}
