//! Public-API integration tests for the MADS (OrthoMADS) solver.
//!
//! Exercises [`Mads`] through the framework — [`Executor`] over a [`MadsState`],
//! with framework termination ([`MaxCostEvals`], [`MeshTolerance`]). The
//! direction machinery is validated against the OrthoMADS paper by the in-crate
//! `solver::mads` golden tests; these confirm the public wiring (init/next_iter,
//! the V↔Vec bridge, count mirroring, convergence / budget / early-stop paths)
//! and convergence on smooth problems across backends.

use basin::{
    CostFunction, Executor, Mads, MadsState, MaxCostEvals, MeshTolerance, TerminationReason,
};

/// Chained Rosenbrock (basin coefficient form), minimum 0 at the all-ones point.
struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((0..x.len() - 1)
            .map(|i| (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i] * x[i]).powi(2))
            .sum())
    }
}

/// Sphere, minimum 0 at the origin.
struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}

#[test]
fn converges_on_sphere() {
    let result = Executor::new(
        Sphere,
        Mads::new()
            .with_initial_poll_size(1.0)
            .with_min_poll_size(1e-8),
        MadsState::new(vec![2.0, -3.0, 1.5]),
    )
    .terminate_on(MaxCostEvals(10_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.best_cost() < 1e-10,
        "best_cost = {}",
        result.best_cost()
    );
    // The reported iterate is the best (monotone), so the two coincide.
    assert_eq!(result.cost(), result.best_cost());
}

#[test]
fn converges_on_rosenbrock_2d() {
    let result = Executor::new(
        Rosenbrock,
        Mads::new()
            .with_initial_poll_size(0.5)
            .with_min_poll_size(1e-7),
        MadsState::new(vec![-1.2, 1.0]),
    )
    // MADS is a poll-only direct search, so it needs many iterations on the
    // Rosenbrock valley; let convergence / the eval budget govern, not max_iter.
    .max_iter(100_000)
    .terminate_on(MaxCostEvals(20_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.best_cost() < 1e-4,
        "best_cost = {}",
        result.best_cost()
    );
    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-2 && (x[1] - 1.0).abs() < 1e-2,
        "x = {x:?}"
    );
}

#[test]
fn mesh_tolerance_stops_early() {
    let result = Executor::new(
        Rosenbrock,
        // Configured to drive the poll size to 1e-12, but MeshTolerance cuts it
        // off at a coarse poll size first.
        Mads::new()
            .with_initial_poll_size(0.5)
            .with_min_poll_size(1e-12),
        MadsState::new(vec![-1.2, 1.0]),
    )
    .terminate_on(MeshTolerance::new(1e-3))
    .terminate_on(MaxCostEvals(50_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MeshTolerance);
    assert!(
        result.state.poll_size() <= 1e-3,
        "poll_size = {}",
        result.state.poll_size()
    );
}

#[test]
fn respects_cost_eval_budget() {
    let result = Executor::new(
        Rosenbrock,
        Mads::new()
            .with_initial_poll_size(0.5)
            .with_min_poll_size(1e-12),
        MadsState::new(vec![-1.2, 1.0]),
    )
    .terminate_on(MaxCostEvals(50))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert!(
        result.cost_evals() >= 50,
        "cost_evals = {}",
        result.cost_evals()
    );
}

/// The motivating case: on a **discontinuous** objective, MADS converges where
/// the Nelder-Mead simplex stalls. On the Step staircase, NM's initial simplex
/// (5% offsets) lands entirely within one plateau — all vertices tie, so the
/// simplex degenerates and never escapes the start cell. MADS steps down the
/// staircase on its mesh and reaches the minimum. Both run from the same start
/// with the same budget.
#[test]
fn outperforms_nelder_mead_on_discontinuous_step() {
    use basin::problems::Step;
    use basin::{BasicSimplexState, NelderMead, SimplexTolerance};

    let start = vec![4.2, -3.2];
    let budget = 5_000;

    let mads = Executor::new(
        Step::<Vec<f64>>::default(),
        Mads::new()
            .with_initial_poll_size(1.0)
            .with_min_poll_size(1e-6),
        MadsState::new(start.clone()),
    )
    .max_iter(100_000)
    .terminate_on(MaxCostEvals(budget))
    .run()
    .unwrap();

    let nm = Executor::new(
        Step::<Vec<f64>>::default(),
        NelderMead::new(),
        BasicSimplexState::new(start.clone()),
    )
    .max_iter(100_000)
    .terminate_on(MaxCostEvals(budget))
    .terminate_on(SimplexTolerance::new(1e-12, 1e-12))
    .run()
    .unwrap();

    // MADS reaches the global-minimum plateau (value 0).
    assert_eq!(
        mads.best_cost(),
        0.0,
        "MADS best_cost = {}",
        mads.best_cost()
    );
    // Nelder-Mead degenerates on the start plateau and stalls above it.
    assert!(
        nm.best_cost() > mads.best_cost(),
        "NM {} should stall above MADS {}",
        nm.best_cost(),
        mads.best_cost()
    );
}

/// Backend-generic over the parameter vector: drive MADS on `nalgebra`'s
/// `DVector` to prove the `V`-generic solver/state work outside `Vec<f64>`.
#[cfg(feature = "nalgebra")]
#[test]
fn backend_generic_nalgebra() {
    use nalgebra::DVector;

    struct SphereN;
    impl CostFunction for SphereN {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x.iter().map(|xi| xi * xi).sum())
        }
    }

    let result = Executor::new(
        SphereN,
        Mads::new().with_min_poll_size(1e-8),
        MadsState::new(DVector::from_vec(vec![2.0, -3.0, 1.5])),
    )
    .terminate_on(MaxCostEvals(10_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.best_cost() < 1e-10,
        "best_cost = {}",
        result.best_cost()
    );
}

/// Backend-generic over `ndarray`'s `Array1`.
#[cfg(feature = "ndarray")]
#[test]
fn backend_generic_ndarray() {
    use ndarray::Array1;

    struct SphereA;
    impl CostFunction for SphereA {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x.iter().map(|xi| xi * xi).sum())
        }
    }

    let result = Executor::new(
        SphereA,
        Mads::new().with_min_poll_size(1e-8),
        MadsState::new(Array1::from_vec(vec![2.0, -3.0, 1.5])),
    )
    .terminate_on(MaxCostEvals(10_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.best_cost() < 1e-10,
        "best_cost = {}",
        result.best_cost()
    );
}

/// Backend-generic over `faer`'s `Col` vector.
#[cfg(feature = "faer")]
#[test]
fn backend_generic_faer() {
    use faer::Col;

    struct SphereF;
    impl CostFunction for SphereF {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((0..x.nrows()).map(|i| x[i] * x[i]).sum())
        }
    }

    let result = Executor::new(
        SphereF,
        Mads::new().with_min_poll_size(1e-8),
        MadsState::new(Col::<f64>::from_fn(3, |i| [2.0, -3.0, 1.5][i])),
    )
    .terminate_on(MaxCostEvals(10_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.best_cost() < 1e-10,
        "best_cost = {}",
        result.best_cost()
    );
}
