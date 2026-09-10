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
fn callback_errors_abort_immediately_during_initialization_and_steps() {
    use std::cell::Cell;

    #[derive(Debug, PartialEq)]
    struct CallbackError(bool, usize);
    struct FallibleDisk<'a> {
        costs: &'a Cell<usize>,
        constraints: &'a Cell<usize>,
        fail_constraint: bool,
        fail_at: usize,
    }
    impl CostFunction for FallibleDisk<'_> {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = CallbackError;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            let call = self.costs.get();
            self.costs.set(call + 1);
            if !self.fail_constraint && call == self.fail_at {
                Err(CallbackError(false, call))
            } else {
                Ok(x[0] * x[1])
            }
        }
    }
    impl NonlinearInequalityConstraints for FallibleDisk<'_> {
        fn num_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            let call = self.constraints.get();
            self.constraints.set(call + 1);
            if self.fail_constraint && call == self.fail_at {
                Err(CallbackError(true, call))
            } else {
                Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0])
            }
        }
    }
    for fail_constraint in [false, true] {
        for fail_at in 0..16 {
            let costs = Cell::new(0);
            let constraints = Cell::new(0);
            let result = Executor::from_start(
                FallibleDisk {
                    costs: &costs,
                    constraints: &constraints,
                    fail_constraint,
                    fail_at,
                },
                Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-12),
                vec![1.0, 1.0],
            )
            .max_iter(1000)
            .run();
            assert!(
                matches!(result, Err(e) if e == CallbackError(fail_constraint, fail_at))
            );
            assert_eq!(costs.get(), fail_at + 1);
            assert_eq!(
                constraints.get(),
                fail_at + usize::from(fail_constraint)
            );
        }
    }
}

#[test]
fn projected_callbacks_retain_their_hard_budget_and_box_feasibility() {
    use std::{cell::Cell, convert::Infallible};

    struct GuardedBox<'a> {
        calls: &'a Cell<u64>,
        projected: &'a Cell<u64>,
        budget: u64,
    }
    impl CostFunction for GuardedBox<'_> {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            if self.calls.get() == self.budget {
                return Ok(f64::INFINITY);
            }
            self.calls.set(self.calls.get() + 1);
            let projected = x[0].clamp(-1.0, 1.0);
            if x[0] != projected {
                self.projected.set(self.projected.get() + 1);
            }
            Ok((projected - 2.0).powi(2))
        }
    }
    impl NonlinearInequalityConstraints for GuardedBox<'_> {
        fn num_constraints(&self) -> usize {
            2
        }
        fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            Ok(vec![-1.0 - x[0], x[0] - 1.0])
        }
    }
    for budget in [1, 2, 3, 5, 8] {
        let calls = Cell::new(0);
        let projected = Cell::new(0);
        let result = Executor::from_start(
            GuardedBox {
                calls: &calls,
                projected: &projected,
                budget,
            },
            Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-12),
            vec![1.0],
        )
        .max_iter(1000)
        .terminate_on(MaxCostEvals(budget))
        .run()
        .unwrap();
        assert_eq!(calls.get(), budget);
        assert_eq!(result.reason, TerminationReason::MaxCostEvals);
        assert!(result.best_param()[0].abs() <= 1.0 + f64::EPSILON.sqrt());
        assert!((result.best_cost() - 1.0).abs() < 1e-12);
        if budget >= 2 {
            assert!(projected.get() > 0);
        }
    }
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

#[test]
fn reused_solver_resizes_scratch_and_keeps_best_snapshot_independent() {
    use basin::{Problem, Solver, State};
    use std::convert::Infallible;

    struct Sphere(usize);
    impl CostFunction for Sphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(x.iter().map(|v| v * v).sum())
        }
    }
    impl NonlinearInequalityConstraints for Sphere {
        fn num_constraints(&self) -> usize {
            self.0
        }
        fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            Ok((0..self.0).map(|i| x[i % x.len()] - 4.0).collect())
        }
    }

    let mut solver = Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-6);
    for (n, m) in [(2, 3), (5, 0), (1, 2), (3, 1)] {
        let mut problem = Problem::new(Sphere(m));
        let mut state = solver
            .init(&mut problem, CobylaState::new(vec![1.0; n]))
            .unwrap();
        state.update_best();
        let snapshot = state.best_param().clone();
        let snapshot_cost = state.best_cost();
        // Solver callbacks reuse the current parameter buffer before the
        // executor mirrors the newly selected incumbent into the best snapshot.
        let (mut state, _) = solver.next_iter(&mut problem, state).unwrap();
        assert_eq!(state.best_param(), &snapshot);
        assert_eq!(state.best_cost(), snapshot_cost);
        assert_eq!(state.param().len(), n);
        state.update_best();
        assert_eq!(state.best_param(), state.param());
        assert_eq!(state.best_cost(), state.cost());
    }
}
