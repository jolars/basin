#![cfg(all(feature = "ndarray_all", feature = "problems"))]

use crate::backend_aliases::ndarray::{Array1, Array2};
use basin::problems::{Rosenbrock, Sphere};
use basin::{
    CmaEs, CostFunction, Executor, PopulationProgress, StepOutcome,
    TerminationCode,
};

/// Same seed → same trajectory, on the ndarray backend. Reproducibility
/// is the load-bearing contract for the seeded RNG (per
/// [`crate::core::rng`] docs); regression-guards a constant-RNG bug.
#[test]
fn same_seed_yields_identical_trajectory() {
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);

    let result_a = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(42, 0.3),
        PopulationProgress::<Array1<f64>>::from_point(m0.clone()),
    )
    .max_iter(30)
    .run()
    .unwrap();

    let result_b = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(42, 0.3),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(30)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Different seeds → different trajectories. Regression-guards an
/// always-zero or always-constant RNG bug.
#[test]
fn different_seeds_yield_different_trajectories() {
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5]);

    let result_a = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(1, 0.3),
        PopulationProgress::<Array1<f64>>::from_point(m0.clone()),
    )
    .max_iter(5)
    .run()
    .unwrap();

    let result_b = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(2, 0.3),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(5)
    .run()
    .unwrap();

    assert_ne!(result_a.param(), result_b.param());
}

/// Sphere 5-D: the canonical "every solver should solve it cleanly"
/// canary. CMA-ES drives `σ` and `‖x‖` to zero geometrically;
/// 80 iterations × λ ≈ 8 evals/iter is far more than needed for
/// 1e-6 accuracy.
#[test]
fn converges_on_sphere_5d() {
    let m0 = Array1::from_elem(5, 1.0);

    let result = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(7, 0.5),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(80)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-6,
        "sphere 5-D cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// Rosenbrock 2-D from `(-1, 1)`. The non-convex banana valley is the
/// canonical CMA-ES test case; the algorithm is a famous good fit for
/// this function. λ_default = 6 at n = 2; 800 iterations ≈ 4800 evals
/// is well within Hansen's ballpark for 2-D.
#[test]
fn converges_on_rosenbrock_2d() {
    let m0 = Array1::from_vec(vec![-1.0, 1.0]);

    let result = Executor::new(
        Rosenbrock::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(17, 0.3),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(800)
    .run()
    .unwrap();

    let p = result.param();
    assert!(
        (p[0] - 1.0).abs() < 1e-3 && (p[1] - 1.0).abs() < 1e-3,
        "rosenbrock 2-D iterate = ({}, {}), expected ≈ (1, 1)",
        p[0],
        p[1]
    );
}

/// TolX termination via the `CmaEsTolerance` criterion: CMA-ES emits
/// `CmaEsTolerance` on Sphere as `σ · max d_i` collapses below
/// `1e−12 · initial_sigma`.
#[test]
fn sphere_terminates_solver_converged_on_tol_x() {
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5]);

    let result = Executor::new(
        Sphere::<Array1<f64>>::new(),
        (CmaEs::<Array1<f64>, Array2<f64>>::new(11, 0.3))
            .with_absolute_distribution_size_tolerance(1e-12 * 0.3),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(2000)
    .run()
    .unwrap();

    assert_eq!(result.report.code(), TerminationCode::CmaEsTolerance);
}

/// `with_stds(ones)` must reproduce the isotropic `C = I` default
/// bit-for-bit: the only difference is the first-generation sampling path
/// (`m + σ B (D ⊙ z)` vs `m + σ z`), which is exactly identical when
/// `B = I` and `D = 1`. Guards the bit-identity claim that lets existing
/// users keep today's behavior unchanged.
#[test]
fn with_stds_ones_matches_default() {
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    let ones = Array1::from_elem(5, 1.0);

    let default = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(42, 0.3),
        PopulationProgress::<Array1<f64>>::from_point(m0.clone()),
    )
    .max_iter(40)
    .run()
    .unwrap();

    let with_ones = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(42, 0.3).with_stds(ones),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(40)
    .run()
    .unwrap();

    assert_eq!(default.cost(), with_ones.cost());
    assert_eq!(default.param(), with_ones.param());
    assert_eq!(default.report.code(), with_ones.report.code());
}

/// Anisotropic stds on the (well-conditioned) Sphere must still converge:
/// per-coordinate scaling rescales the initial distribution but does not
/// break the adaptation. Correctness check, not a speedup claim.
#[test]
fn with_stds_anisotropic_converges_on_sphere() {
    let m0 = Array1::from_elem(5, 1.0);
    let stds = Array1::from_vec(vec![1.0, 0.1, 10.0, 0.5, 2.0]);

    let result = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(7, 0.5).with_stds(stds),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(120)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-6,
        "anisotropic sphere 5-D cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// The motivating case: an ill-scaled quadratic `f(x) = x₀² + 1e6 · x₁²`.
/// Preconditioning with `with_stds([1, 1e-3])` rescales the search to ~unit
/// conditioning, so CMA-ES reaches a low cost within a modest budget
/// instead of spending generations learning the 1e6 scale ratio.
#[test]
fn with_stds_preconditions_ill_scaled_quadratic() {
    struct IllScaledQuadratic;
    impl CostFunction for IllScaledQuadratic {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(x[0] * x[0] + 1e6 * x[1] * x[1])
        }
    }

    let m0 = Array1::from_vec(vec![2.0, 2.0]);
    let stds = Array1::from_vec(vec![1.0, 1e-3]);

    let result = Executor::new(
        IllScaledQuadratic,
        CmaEs::<Array1<f64>, Array2<f64>>::new(7, 0.5).with_stds(stds),
        PopulationProgress::<Array1<f64>>::from_point(m0),
    )
    .max_iter(300)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-6,
        "preconditioned ill-scaled quadratic cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// `with_stds` panics when the std vector length doesn't match the mean.
#[test]
#[should_panic(expected = "stds.len() == mean.len()")]
fn with_stds_panics_on_length_mismatch() {
    let _ = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(7, 0.3)
            .with_stds(Array1::from_vec(vec![1.0; 2])),
        PopulationProgress::from_point(Array1::from_vec(vec![0.0; 3])),
    )
    .max_iter(0)
    .run()
    .unwrap();
}

/// `with_stds` panics on a non-positive std (would make `1/std` non-finite
/// in the `C^{−1/2}` factor).
#[test]
#[should_panic(expected = "every std > 0")]
fn with_stds_panics_on_nonpositive() {
    let _ = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(7, 0.3)
            .with_stds(Array1::from_vec(vec![1.0, 0.0])),
        PopulationProgress::from_point(Array1::from_vec(vec![0.0; 2])),
    )
    .max_iter(0)
    .run()
    .unwrap();
}

/// Sort + length invariants survive iteration. 's
/// contract requires `candidates`/`costs` to remain in parallel,
/// length-`λ`, sorted-ascending.
#[test]
fn population_invariants_hold_after_iteration() {
    let m0 = Array1::from_vec(vec![0.3, 0.4]);
    let lambda = 12;

    let mut stepper = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(1234, 0.5).with_lambda(lambda),
        PopulationProgress::<Array1<f64>>::from_point(m0),
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
