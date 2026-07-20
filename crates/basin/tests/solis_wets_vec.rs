//! `Vec<f64>`-backend tests for [`SolisWets`] (no feature gate): the
//! full suite. The backend mirrors (`tests/solis_wets_{nalgebra,
//! ndarray,faer}.rs`) run reduced versions of the same checks to confirm
//! the per-backend `SampleStandardNormal`/`ScaledAdd` wiring.

use basin::problems::{Rosenbrock, Sphere};
use basin::{
    CmaEs, CmaEsState, CmaInject, DenseMatrix, Executor, RhoTolerance, SolisWets, SolisWetsState,
    State, StepOutcome, TerminationReason,
};

/// Same seed → same trajectory. Load-bearing reproducibility check for
/// the stochastic-solver contract (ChaCha8Rng is platform-independent).
#[test]
fn same_seed_yields_identical_trajectory() {
    let run = || {
        Executor::from_start(
            Sphere::<Vec<f64>>::new(),
            SolisWets::new(42),
            vec![2.0, -1.5],
        )
        .max_iter(200)
        .run()
        .unwrap()
    };
    let result_a = run();
    let result_b = run();
    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Different seeds → different trajectories: the RNG actually drives
/// the sampling.
#[test]
fn different_seeds_yield_different_trajectories() {
    let run = |seed| {
        Executor::from_start(
            Sphere::<Vec<f64>>::new(),
            SolisWets::new(seed),
            vec![2.0, -1.5],
        )
        .max_iter(50)
        .run()
        .unwrap()
    };
    assert_ne!(run(1).param(), run(2).param());
}

/// Sphere 5-D to tight accuracy with `RhoTolerance` as the convergence
/// test: ρ contracts to the floor near the minimum and the criterion
/// (not the iteration budget) must be what fires.
#[test]
fn converges_on_sphere_5d_via_rho_tolerance() {
    let result = Executor::from_start(
        Sphere::<Vec<f64>>::new(),
        SolisWets::new(7),
        vec![2.0, -1.0, 1.5, 0.5, -2.0],
    )
    .terminate_on(RhoTolerance::new(1e-8))
    .max_iter(100_000)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RhoTolerance);
    assert!(
        result.cost() < 1e-6,
        "sphere 5-D cost = {} (expected < 1e-6)",
        result.cost()
    );
}

/// Rosenbrock 2-D from the classic `(-1.2, 1)` start. Solis-Wets's
/// isotropic mutations are slow in the curved valley, so the tolerance
/// is deliberately loose; Sphere above is the tight-accuracy canary.
#[test]
fn makes_progress_on_rosenbrock_2d() {
    let result = Executor::from_start(
        Rosenbrock::<Vec<f64>>::new(),
        SolisWets::new(3),
        vec![-1.2, 1.0],
    )
    .terminate_on(RhoTolerance::new(1e-10))
    .max_iter(50_000)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-2,
        "rosenbrock 2-D cost = {} (expected < 1e-2)",
        result.cost()
    );
}

/// Monotonicity contract: Solis-Wets only ever accepts strict
/// improvements, so `state.cost()` is non-increasing across iterations.
#[test]
fn cost_is_monotone_nonincreasing() {
    let mut stepper = Executor::from_start(
        Sphere::<Vec<f64>>::new(),
        SolisWets::new(99),
        vec![3.0, -2.0, 1.0],
    )
    .max_iter(500)
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

/// `Executor::from_start` (via `InitialState::seed`, default ρ = 1) is
/// identical to handing in the equivalent explicit state.
#[test]
fn from_start_matches_explicit_state() {
    let via_seed =
        Executor::from_start(Sphere::<Vec<f64>>::new(), SolisWets::new(5), vec![1.0, 2.0])
            .max_iter(100)
            .run()
            .unwrap();

    let via_state = Executor::new(
        Sphere::<Vec<f64>>::new(),
        SolisWets::new(5),
        SolisWetsState::new(vec![1.0, 2.0], 1.0),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert_eq!(via_seed.cost(), via_state.cost());
    assert_eq!(via_seed.param(), via_state.param());
}

/// Smoke test for the `MemeticInner` impl: Solis-Wets as the injected
/// refinement inside `CmaInject` must run and converge (the σ-scaled
/// seed keeps the walk on the outer distribution's scale).
#[test]
fn works_as_cma_inject_inner() {
    let cma = CmaEs::<Vec<f64>, DenseMatrix>::new(17);
    let solver = CmaInject::with_inner_solver(cma, SolisWets::new(23))
        .with_k(1)
        .with_inner_max_iter(30);

    let result = Executor::new(
        Sphere::<Vec<f64>>::new(),
        solver,
        CmaEsState::<Vec<f64>, DenseMatrix>::new(vec![2.0, -1.5], 0.5),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-6,
        "cma-inject(solis-wets) sphere cost = {} (expected < 1e-6)",
        result.cost()
    );
}
