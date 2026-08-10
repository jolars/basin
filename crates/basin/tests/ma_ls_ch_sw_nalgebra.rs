#![cfg(feature = "nalgebra")]

//! nalgebra-backend tests for [`MaLsChSw`]: the deep mirror. Covers
//! convergence, reproducibility, the chain-resume mechanism, and the
//! sorted-population invariant; the other backends run smoke mirrors.

use basin::problems::{RastriginBoxed, SphereBoxed};
use basin::{
    Executor, MaLsChSw, MaLsChSwState, MaxCostEvals, PopulationState,
    StepOutcome,
};
use nalgebra::DVector;

fn boxed_sphere(n: usize) -> SphereBoxed<DVector<f64>> {
    SphereBoxed::new(
        DVector::from_element(n, -5.0),
        DVector::from_element(n, 5.0),
    )
}

#[test]
fn converges_on_sphere_d10() {
    let problem = boxed_sphere(10);
    let solver = MaLsChSw::<DVector<f64>>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChSwState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) nalgebra cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

#[test]
fn same_seed_yields_identical_trajectory() {
    let run = || {
        Executor::new(
            boxed_sphere(5),
            MaLsChSw::<DVector<f64>>::new(99).with_pop_size(15),
            MaLsChSwState::new(),
        )
        .max_iter(20)
        .run()
        .unwrap()
    };
    let result_a = run();
    let result_b = run();
    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

#[test]
fn different_seeds_yield_different_trajectories() {
    let run = |seed| {
        Executor::new(
            boxed_sphere(5),
            MaLsChSw::<DVector<f64>>::new(seed).with_pop_size(15),
            MaLsChSwState::new(),
        )
        .max_iter(10)
        .run()
        .unwrap()
    };
    assert_ne!(run(1).param(), run(2).param());
}

/// Chain mechanism is actually firing: at least one individual
/// undergoes ≥2 LS applications over the run, which requires its
/// `(SolisWets, SolisWetsState)` pair to have been preserved and
/// re-entered between outer iterations (`ResumableInner::prepare_resume`
/// plus the resume-idempotent `SolisWets::init`). `δ_LS_min = 0` makes
/// the chain store-back unconditional, so a count of 2 cannot be
/// reached by two independent fresh seeds (a displaced individual
/// resets to 0): the second application *must* have resumed the stored
/// pair. Mirrors the CMA variant's chain test; see there for why
/// max-over-the-run is tracked instead of the final state.
#[test]
fn chain_resumes_at_least_one_individual_twice() {
    let problem = RastriginBoxed::<DVector<f64>>::with_standard_bounds(5);
    let pop_size = 4;
    let mut stepper = Executor::new(
        problem,
        MaLsChSw::<DVector<f64>>::new(31)
            .with_pop_size(pop_size)
            .with_nam_pool(pop_size)
            .with_ls_intensity(30)
            .with_nfrec(5)
            .with_ls_improvement_threshold(0.0),
        MaLsChSwState::new(),
    )
    .max_iter(40)
    .into_stepper()
    .unwrap();

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

/// The population (and its parallel arrays) stays sorted ascending at
/// every outer iteration boundary.
#[test]
fn population_stays_sorted_ascending() {
    let pop_size = 10;
    let mut stepper = Executor::new(
        boxed_sphere(5),
        MaLsChSw::<DVector<f64>>::new(2024).with_pop_size(pop_size),
        MaLsChSwState::new(),
    )
    .max_iter(10)
    .into_stepper()
    .unwrap();

    while let StepOutcome::Continue = stepper.step().unwrap() {
        let s = stepper.state();
        assert_eq!(s.candidates().len(), pop_size);
        assert_eq!(s.costs().len(), pop_size);
        for w in s.costs().windows(2) {
            assert!(w[0] <= w[1], "costs not sorted: {} > {}", w[0], w[1]);
        }
    }
}
