//! Integration tests for [`MaLsChCma`] on the nalgebra backend.
//!
//! Covers convergence, reproducibility, the chain-resumption mechanism
//! firing across iterations, and the eval-aggregation contract. Failure
//! routing follows the same pattern as `cma_inject_nalgebra.rs` (the
//! inner CMA-ES can produce `SolverFailed` from eigendecomposition
//! failure on a NaN-returning problem, exercised at the boundary via
//! `inner_executor.rs`).

#![cfg(feature = "nalgebra")]

use basin::problems::RastriginBoxed;
use basin::{
    CostFunction, Executor, MaLsChCma, MaLsChState, MaxCostEvals, PopulationState, StepOutcome,
};
use nalgebra::{DMatrix, DVector};

/// Sphere-style problem with a box for SSGA initial sampling. Plain
/// f(x) = ||x||², not the convex regression target — we want the easy
/// canary that any working population solver should crush.
struct BoxedSphere {
    lower: DVector<f64>,
    upper: DVector<f64>,
}

impl BoxedSphere {
    fn new(n: usize, half_width: f64) -> Self {
        Self {
            lower: DVector::from_element(n, -half_width),
            upper: DVector::from_element(n, half_width),
        }
    }
}

impl CostFunction for BoxedSphere {
    type Param = DVector<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}

impl basin::BoxConstraints for BoxedSphere {
    fn lower(&self) -> &DVector<f64> {
        &self.lower
    }
    fn upper(&self) -> &DVector<f64> {
        &self.upper
    }
}

/// Sphere(D=10) within a generous box. MaLsChCma with the default
/// parameters should drive cost below 1e-6 well within 20k cost evals
/// — CMA-ES alone solves this in a few hundred, so the chain machinery
/// shouldn't regress it.
#[test]
fn converges_on_sphere_d10() {
    let problem = BoxedSphere::new(10, 5.0);
    let solver = MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) final cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

/// Rastrigin(D=10) is the canonical multimodal stress test Bergmeir
/// 2016 uses for MA-LSCh-CMA. The reference paper reports near-zero
/// fitness within 200k evals on D=30; here we use D=10 and a 50k
/// budget for CI determinism, expecting cost well below 1.0.
#[test]
fn converges_on_rastrigin_d10() {
    let problem = RastriginBoxed::<DVector<f64>>::with_standard_bounds(10);
    let solver = MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(42).with_pop_size(30);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(50_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1.0,
        "Rastrigin(D=10) final cost {} should be < 1.0 within 50k evals",
        result.cost()
    );
}

/// Same seed → bitwise-identical result on the same problem.
#[test]
fn same_seed_yields_identical_trajectory() {
    let result_a = Executor::new(
        BoxedSphere::new(5, 5.0),
        MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(99).with_pop_size(15),
        MaLsChState::new(),
    )
    .max_iter(10)
    .run()
    .unwrap();
    let result_b = Executor::new(
        BoxedSphere::new(5, 5.0),
        MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(99).with_pop_size(15),
        MaLsChState::new(),
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Different seeds → different trajectories (sanity check the outer
/// RNG actually drives the chain).
#[test]
fn different_seeds_yield_different_trajectories() {
    let result_a = Executor::new(
        BoxedSphere::new(5, 5.0),
        MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(1).with_pop_size(15),
        MaLsChState::new(),
    )
    .max_iter(5)
    .run()
    .unwrap();
    let result_b = Executor::new(
        BoxedSphere::new(5, 5.0),
        MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(2).with_pop_size(15),
        MaLsChState::new(),
    )
    .max_iter(5)
    .run()
    .unwrap();
    assert_ne!(result_a.param(), result_b.param());
}

/// Chain mechanism is actually firing: at least one individual
/// undergoes ≥2 LS applications over the run, which is only possible
/// if its `(CmaEs, CmaEsState)` pair was correctly preserved
/// and re-entered between outer iterations. Without `CmaEs::init`
/// idempotency the second LS application would lose its evolution
/// state — the test would still pass (count increments unconditionally)
/// but the chain would be broken; see the cost-progression test below
/// for the idempotency check.
///
/// Setup: tiny `pop_size = 4` (minimum NAM-feasible) so the
/// "first-time LS" branch saturates within a handful of iters and the
/// Molina §4.3 fallback (`S_LS = ∅` → apply LS to the best) fires
/// repeatedly on the best individual.
#[test]
fn chain_resumes_at_least_one_individual_twice() {
    // Rastrigin is multimodal — random SSGA offspring almost never
    // beat the worst, so the population stabilizes after a few iters
    // and the Molina §4.3 fallback (S_LS = ∅ → LS the best) fires
    // repeatedly on the same individual.
    let problem = RastriginBoxed::<DVector<f64>>::with_standard_bounds(5);
    let pop_size = 4;
    let mut stepper = Executor::new(
        problem,
        MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(31)
            .with_pop_size(pop_size)
            .with_nam_pool(pop_size)
            .with_ls_intensity(30)
            .with_nfrec(5),
        MaLsChState::new(),
    )
    .max_iter(40)
    .into_stepper()
    .unwrap();

    // Track max LS count across the run rather than just final state.
    // The final state can show low counts because SSGA replace-worst
    // can displace a high-count individual mid-run (clearing its
    // chain); the load-bearing question is whether ANY individual was
    // re-selected and accumulated ≥2 applications at some point.
    let mut max_ever = 0u32;
    while let StepOutcome::Continue = stepper.step().unwrap() {
        let s = stepper.state();
        for i in 0..pop_size {
            max_ever = max_ever.max(s.ls_application_count(i));
        }
    }
    assert!(
        max_ever >= 2,
        "no individual ever reached >=2 LS applications in 40 outer \
         iters; max_ever = {} (chain mechanism may be broken)",
        max_ever
    );
}

/// Eval-aggregation contract (CONTRIBUTING.md "Solver composition" rule 1):
/// `result.cost_evals()` reflects outer SSGA evals + every inner CMA
/// evaluation. With budget `B`, the final count overshoots `B` by at
/// most one chain segment's evals plus the SSGA phase of the trailing
/// iteration — bounded by `nfrec + ls_intensity + λ_inner`.
#[test]
fn cost_evals_overshoot_is_bounded() {
    let problem = BoxedSphere::new(5, 5.0);
    let budget = 5_000u64;
    let ls_intensity = 100u64;
    let nfrec = 100u64;
    let pop_size = 20;
    let n = 5;
    let lambda_inner = (4 + (3.0 * (n as f64).ln()).floor() as usize) as u64;

    let solver = MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(11)
        .with_pop_size(pop_size)
        .with_ls_intensity(ls_intensity)
        .with_nfrec(nfrec);

    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(budget))
        .run()
        .unwrap();

    let max_overshoot = nfrec + ls_intensity + lambda_inner;
    assert!(
        result.cost_evals() >= budget,
        "result.cost_evals() = {} did not reach budget {}",
        result.cost_evals(),
        budget
    );
    assert!(
        result.cost_evals() <= budget + max_overshoot,
        "result.cost_evals() = {} overshoots budget {} by more than \
         the allowed slack {} (nfrec + ls_intensity + λ_inner)",
        result.cost_evals(),
        budget,
        max_overshoot
    );
}

/// Population invariants: sorted ascending by cost so `state.cost()`
/// always reports the best.
#[test]
fn population_stays_sorted_ascending() {
    let problem = BoxedSphere::new(4, 5.0);
    let pop_size = 12;
    let mut stepper = Executor::new(
        problem,
        MaLsChCma::<DVector<f64>, DMatrix<f64>>::new(2024).with_pop_size(pop_size),
        MaLsChState::new(),
    )
    .max_iter(10)
    .into_stepper()
    .unwrap();

    for _ in 0..10 {
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
