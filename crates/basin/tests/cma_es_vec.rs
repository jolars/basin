//! CMA-ES convergence over the default `Vec<f64>` backend (no feature gate).
//!
//! Mirrors the nalgebra tests in `tests/cma_es_nalgebra.rs` to confirm the
//! generic `Solver` impl runs on the hand-rolled `DenseMatrix` covariance,
//! whose `SymmetricEigen` is the pure-Rust cyclic Jacobi eigensolver. Asserts
//! convergence + same-seed reproducibility on this backend; it does *not*
//! assert bit-identity against nalgebra/faer, whose eigensolvers round
//! differently.

use basin::problems::{Rosenbrock, Sphere};
use basin::{
    CmaEs, CmaEsState, CmaEsTolerance, CostFunction, DenseMatrix, Executor,
    PopulationState, StepOutcome, TerminationReason,
};

/// Same seed → same trajectory on the `Vec<f64>` backend. Reproducibility is
/// the load-bearing contract for the seeded RNG; guards a constant-RNG bug.
#[test]
fn same_seed_yields_identical_trajectory() {
    let m0 = vec![0.5, 0.5, 0.5, 0.5, 0.5];

    let result_a = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0.clone(), 0.3),
    )
    .max_iter(30)
    .run()
    .unwrap();

    let result_b = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.3),
    )
    .max_iter(30)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Sphere 5-D: the canonical "every solver should solve it cleanly" canary.
#[test]
fn converges_on_sphere_5d() {
    let m0 = vec![1.0; 5];

    let result = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(7),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5),
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

/// Rosenbrock 2-D from `(-1, 1)`: the canonical non-convex CMA-ES test case.
/// Exercises the eigensolver on a genuinely anisotropic learned covariance.
#[test]
fn converges_on_rosenbrock_2d() {
    let m0 = vec![-1.0, 1.0];

    let result = Executor::new(
        Rosenbrock::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(17),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.3),
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
/// `CmaEsTolerance` on Sphere as `σ · max dᵢ` collapses below the tolerance.
#[test]
fn sphere_terminates_solver_converged_on_tol_x() {
    let m0 = vec![0.5, 0.5, 0.5];

    let result = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(11),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.3),
    )
    .terminate_on(CmaEsTolerance::new(1e-12 * 0.3))
    .max_iter(2000)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::CmaEsTolerance);
}

/// `with_stds(ones)` reproduces the isotropic `C = I` default bit-for-bit on
/// the `Vec<f64>` backend; the `MatrixFromDiagonal` seeding of `diag(1²)`
/// must equal the `MatrixIdentity` default.
#[test]
fn with_stds_ones_matches_default() {
    let m0 = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let ones = vec![1.0; 5];

    let default = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0.clone(), 0.3),
    )
    .max_iter(40)
    .run()
    .unwrap();

    let with_ones = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.3).with_stds(ones),
    )
    .max_iter(40)
    .run()
    .unwrap();

    assert_eq!(default.cost(), with_ones.cost());
    assert_eq!(default.param(), with_ones.param());
    assert_eq!(default.reason, with_ones.reason);
}

/// Anisotropic stds on the well-conditioned Sphere must still converge: the
/// `MatrixFromDiagonal` seed rescales the initial distribution without
/// breaking adaptation.
#[test]
fn with_stds_anisotropic_converges_on_sphere() {
    let m0 = vec![1.0; 5];
    let stds = vec![1.0, 0.1, 10.0, 0.5, 2.0];

    let result = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(7),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5).with_stds(stds),
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

/// The motivating preconditioning case: an ill-scaled quadratic
/// `f(x) = x₀² + 1e6 · x₁²`. `with_stds([1, 1e-3])` rescales to ~unit
/// conditioning so CMA-ES reaches a low cost within a modest budget.
#[test]
fn with_stds_preconditions_ill_scaled_quadratic() {
    struct IllScaledQuadratic;
    impl CostFunction for IllScaledQuadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x[0] * x[0] + 1e6 * x[1] * x[1])
        }
    }

    let m0 = vec![2.0, 2.0];
    let stds = vec![1.0, 1e-3];

    let result = Executor::new(
        IllScaledQuadratic,
        CmaEs::<Vec<f64>, DenseMatrix>::new(7),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5).with_stds(stds),
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

/// `PopulationState` invariants survive iteration on the `Vec<f64>` backend:
/// `candidates`/`costs` stay parallel, length-λ, and sorted-ascending.
#[test]
fn population_invariants_hold_after_iteration() {
    let m0 = vec![0.3, 0.4];
    let lambda = 12;

    let mut stepper = Executor::new(
        Sphere::<Vec<f64>>::new(),
        CmaEs::<Vec<f64>, DenseMatrix>::new(1234).with_lambda(lambda),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5),
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
