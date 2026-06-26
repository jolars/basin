//! Public-API integration tests for the nonlinearly-constrained MADS variant
//! (`Mads<Constrained>`, the progressive barrier).
//!
//! Exercises [`Mads::constrained`] through the framework: [`Executor`] over a
//! [`ConstrainedMadsState`], with a problem carrying nonlinear inequality
//! constraints via [`NonlinearInequalityConstraints`]. These confirm the public
//! wiring: init/next_iter, the `c(x)` evaluation folded into the violation
//! `h(x) = Σⱼ max(cⱼ, 0)²`, the V↔Vec bridge, count mirroring, feasibility of
//! the returned point, and (the capability the extreme barrier lacks) that an
//! **infeasible start** relaxes its way to a feasible optimum.
//!
//! The constraint trait is *function-valued* (no matrix carrier), so the
//! backend-generic tests need only the parameter vector to be the backend type,
//! guarding the support-matrix ✓ for nalgebra/ndarray/faer (same shape as
//! `cobyla_public.rs`). MADS's poll geometry is deterministic and
//! backend-independent, so every backend traces the identical run.

use basin::{
    ConstrainedMadsState, CostFunction, Executor, Mads, MaxCostEvals,
    NonlinearInequalityConstraints, TerminationReason,
};

/// `min x0·x1` s.t. `x0² + x1² ≤ 1` on `Vec<f64>` (default features). The
/// constrained optimum is `F* = −1/2` on the unit circle (at the 45° points).
struct Disk;

impl CostFunction for Disk {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x[0] * x[1])
    }
}

impl NonlinearInequalityConstraints for Disk {
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0])
    }
    fn num_constraints(&self) -> usize {
        1
    }
}

#[test]
fn converges_to_disk_optimum() {
    let result = Executor::new(
        Disk,
        Mads::new()
            .with_initial_poll_size(1.0)
            .with_min_poll_size(1e-7)
            .constrained(),
        ConstrainedMadsState::new(vec![0.0, 0.0]),
    )
    .terminate_on(MaxCostEvals(20_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-2,
        "f = {}",
        result.best_cost()
    );
    // The returned point is feasible and the reported violation is ~0.
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
    assert!(
        result.state.constraint_violation() <= 1e-12,
        "violation = {}",
        result.state.constraint_violation()
    );
}

/// The progressive barrier's defining capability: starting **outside** the
/// feasible region, it relaxes its way to a feasible optimum (the extreme
/// barrier would have nothing to descend from). Start far outside the disk.
#[test]
fn infeasible_start_reaches_feasible_optimum() {
    let start = vec![2.0, -2.0]; // x0² + x1² = 8 > 1: strongly infeasible
    let result = Executor::new(
        Disk,
        Mads::new()
            .with_initial_poll_size(1.0)
            .with_min_poll_size(1e-7)
            .constrained(),
        ConstrainedMadsState::new(start),
    )
    .terminate_on(MaxCostEvals(20_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.best_param();
    assert!(
        x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6,
        "did not reach feasibility: {x:?}"
    );
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-2,
        "f = {}",
        result.best_cost()
    );
}

/// The evaluation budget is honored: a tiny cap stops the run with
/// [`TerminationReason::MaxCostEvals`].
#[test]
fn respects_eval_budget() {
    let result = Executor::new(
        Disk,
        Mads::new().constrained(),
        ConstrainedMadsState::new(vec![0.0, 0.0]),
    )
    .terminate_on(MaxCostEvals(50))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
}

/// Backend-generic: drive constrained MADS on nalgebra `DVector`. Guards the
/// support-matrix ✓ for nalgebra (the param vector needs
/// `VectorLen + Index + IndexMut`).
#[cfg(feature = "nalgebra")]
#[test]
fn backend_generic_nalgebra() {
    use nalgebra::DVector;

    struct Disk;
    impl CostFunction for Disk {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x[0] * x[1])
        }
    }
    impl NonlinearInequalityConstraints for Disk {
        fn constraints(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok(DVector::from_vec(vec![x[0] * x[0] + x[1] * x[1] - 1.0]))
        }
        fn num_constraints(&self) -> usize {
            1
        }
    }

    let result = Executor::new(
        Disk,
        Mads::new().with_min_poll_size(1e-7).constrained(),
        ConstrainedMadsState::new(DVector::from_vec(vec![0.0, 0.0])),
    )
    .terminate_on(MaxCostEvals(20_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-2,
        "f = {}",
        result.best_cost()
    );
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}

/// Backend-generic: drive constrained MADS on ndarray `Array1`.
#[cfg(feature = "ndarray")]
#[test]
fn backend_generic_ndarray() {
    use ndarray::Array1;

    struct Disk;
    impl CostFunction for Disk {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x[0] * x[1])
        }
    }
    impl NonlinearInequalityConstraints for Disk {
        fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            Ok(Array1::from_vec(vec![x[0] * x[0] + x[1] * x[1] - 1.0]))
        }
        fn num_constraints(&self) -> usize {
            1
        }
    }

    let result = Executor::new(
        Disk,
        Mads::new().with_min_poll_size(1e-7).constrained(),
        ConstrainedMadsState::new(Array1::from_vec(vec![0.0, 0.0])),
    )
    .terminate_on(MaxCostEvals(20_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-2,
        "f = {}",
        result.best_cost()
    );
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}

/// Backend-generic: drive constrained MADS on faer `Col`.
#[cfg(feature = "faer")]
#[test]
fn backend_generic_faer() {
    use faer::Col;

    struct Disk;
    impl CostFunction for Disk {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x[0] * x[1])
        }
    }
    impl NonlinearInequalityConstraints for Disk {
        fn constraints(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(Col::from_fn(1, |_| x[0] * x[0] + x[1] * x[1] - 1.0))
        }
        fn num_constraints(&self) -> usize {
            1
        }
    }

    let result = Executor::new(
        Disk,
        Mads::new().with_min_poll_size(1e-7).constrained(),
        ConstrainedMadsState::new(Col::from_fn(2, |_| 0.0)),
    )
    .terminate_on(MaxCostEvals(20_000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-2,
        "f = {}",
        result.best_cost()
    );
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}
