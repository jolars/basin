//! Public-API integration tests for the COBYLA solver.
//!
//! Exercises [`Cobyla`] through the framework: [`Executor`] over a
//! [`CobylaState`], with framework termination ([`MaxCostEvals`],
//! [`RhoTolerance`]) and a problem carrying nonlinear inequality constraints via
//! [`NonlinearInequalityConstraints`]. These confirm the public wiring:
//! init/next_iter, the constraint evaluation + folding into the merit, the V↔Vec
//! bridge, count mirroring, feasibility of the returned point, and the
//! convergence/budget/early-stop termination paths.
//!
//! The constraint trait is *function-valued* (no matrix carrier), so the
//! backend-generic tests need only the parameter vector to be the backend type;
//! they guard the support-matrix ✓ for nalgebra/ndarray/faer the same way
//! `lincoa_public.rs` does for the linear-constrained family.

use basin::{
    Cobyla, CobylaState, CostFunction, Executor, MaxCostEvals,
    NonlinearInequalityConstraints, RhoTolerance, TerminationReason,
};

/// `min x0·x1` s.t. `x0² + x1² ≤ 1` on `Vec<f64>` (default features). The
/// constrained optimum is `F* = −1/2` on the unit circle.
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
    fn constraints(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
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
        Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-6),
        CobylaState::new(vec![1.0, 1.0]),
    )
    .terminate_on(MaxCostEvals(2000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-3,
        "f = {}",
        result.best_cost()
    );
    // The returned point is feasible: x0² + x1² ≤ 1.
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
    assert!(
        result.cost_evals() < 2000,
        "cost_evals = {}",
        result.cost_evals()
    );
}

#[test]
fn respects_cost_eval_budget() {
    let result = Executor::new(
        Disk,
        Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-12),
        CobylaState::new(vec![1.0, 1.0]),
    )
    .terminate_on(MaxCostEvals(15))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert!(
        result.cost_evals() >= 15,
        "cost_evals = {}",
        result.cost_evals()
    );
}

#[test]
fn rho_tolerance_stops_early() {
    let result = Executor::new(
        Disk,
        Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-12),
        CobylaState::new(vec![1.0, 1.0]),
    )
    .terminate_on(RhoTolerance::new(1e-3))
    .terminate_on(MaxCostEvals(5000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RhoTolerance);
    assert!(result.state.rho() <= 1e-3, "rho = {}", result.state.rho());
}

/// Backend-generic: drive COBYLA on nalgebra `DVector`. Guards the
/// support-matrix ✓ for nalgebra; the param vector must satisfy
/// `VectorLen + Index + IndexMut`.
#[cfg(feature = "nalgebra_all")]
#[test]
fn backend_generic_nalgebra() {
    use crate::backend_aliases::nalgebra::DVector;

    struct Disk;
    impl CostFunction for Disk {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(x[0] * x[1])
        }
    }
    impl NonlinearInequalityConstraints for Disk {
        fn constraints(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok(DVector::from_vec(vec![x[0] * x[0] + x[1] * x[1] - 1.0]))
        }
        fn num_constraints(&self) -> usize {
            1
        }
    }

    let result = Executor::new(
        Disk,
        Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-6),
        CobylaState::new(DVector::from_vec(vec![1.0, 1.0])),
    )
    .terminate_on(MaxCostEvals(2000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-3,
        "f = {}",
        result.best_cost()
    );
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}

/// Backend-generic: drive COBYLA on ndarray `Array1`. Guards the
/// support-matrix ✓ for ndarray.
#[cfg(feature = "ndarray_all")]
#[test]
fn backend_generic_ndarray() {
    use crate::backend_aliases::ndarray::Array1;

    struct Disk;
    impl CostFunction for Disk {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(x[0] * x[1])
        }
    }
    impl NonlinearInequalityConstraints for Disk {
        fn constraints(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            Ok(Array1::from_vec(vec![x[0] * x[0] + x[1] * x[1] - 1.0]))
        }
        fn num_constraints(&self) -> usize {
            1
        }
    }

    let result = Executor::new(
        Disk,
        Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-6),
        CobylaState::new(Array1::from_vec(vec![1.0, 1.0])),
    )
    .terminate_on(MaxCostEvals(2000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-3,
        "f = {}",
        result.best_cost()
    );
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}

/// Backend-generic: drive COBYLA on faer `Col`. Guards the support-matrix ✓ for
/// faer.
#[cfg(feature = "faer_all")]
#[test]
fn backend_generic_faer() {
    use crate::backend_aliases::faer::Col;

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
        fn constraints(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let c = x[0] * x[0] + x[1] * x[1] - 1.0;
            Ok(Col::from_fn(1, |_| c))
        }
        fn num_constraints(&self) -> usize {
            1
        }
    }

    let result = Executor::new(
        Disk,
        Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-6),
        CobylaState::new(Col::from_fn(2, |_| 1.0)),
    )
    .terminate_on(MaxCostEvals(2000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-3,
        "f = {}",
        result.best_cost()
    );
    let x = result.best_param();
    assert!(x[0] * x[0] + x[1] * x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
