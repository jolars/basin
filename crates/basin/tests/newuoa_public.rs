//! Public-API integration tests for the NEWUOA solver.
//!
//! Exercises [`Newuoa`] through the framework: [`Executor`] over a
//! [`NewuoaState`], with framework termination ([`MaxCostEvals`],
//! [`RhoTolerance`]). The algorithm itself is validated bit-against-PRIMA by the
//! in-crate `solver::newuoa::parity` tests; these confirm the public wiring:
//! init/next_iter, the V↔Vec bridge, count mirroring, and the convergence/
//! budget/early-stop termination paths.

use basin::{
    CostFunction, Executor, MaxCostEvals, Newuoa, NewuoaState, RhoTolerance, TerminationReason,
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

#[test]
fn converges_on_rosenbrock_2d() {
    let result = Executor::new(
        Rosenbrock,
        Newuoa::new().with_rho_beg(0.5).with_rho_end(1e-8),
        NewuoaState::new(vec![-1.2, 1.0]),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    // NEWUOA's natural convergence (ρ reached ρ_end), well within budget.
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.best_cost() < 1e-7,
        "best_cost = {}",
        result.best_cost()
    );
    assert!(
        result.cost_evals() < 500,
        "cost_evals = {}",
        result.cost_evals()
    );
    // The reported iterate is the best (monotone), so the two coincide.
    assert_eq!(result.cost(), result.best_cost());
    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "x = {x:?}"
    );
}

#[test]
fn respects_cost_eval_budget() {
    let result = Executor::new(
        Rosenbrock,
        // A tiny ρ_end so the solver would keep going far past the budget.
        Newuoa::new().with_rho_beg(0.5).with_rho_end(1e-12),
        NewuoaState::new(vec![-1.2, 1.0]),
    )
    .terminate_on(MaxCostEvals(20))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert!(
        result.cost_evals() >= 20,
        "cost_evals = {}",
        result.cost_evals()
    );
}

#[test]
fn rho_tolerance_stops_early() {
    let result = Executor::new(
        Rosenbrock,
        // Configured to drive ρ down to 1e-12, but RhoTolerance cuts it off at
        // a coarse ρ first.
        Newuoa::new().with_rho_beg(0.5).with_rho_end(1e-12),
        NewuoaState::new(vec![-1.2, 1.0]),
    )
    .terminate_on(RhoTolerance::new(1e-3))
    .terminate_on(MaxCostEvals(5000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RhoTolerance);
    assert!(result.state.rho() <= 1e-3, "rho = {}", result.state.rho());
}

/// Backend-generic over the parameter vector: drive NEWUOA on `nalgebra`'s
/// `DVector` to prove the `V`-generic solver/state work outside `Vec<f64>`.
#[cfg(feature = "nalgebra")]
#[test]
fn backend_generic_nalgebra() {
    use nalgebra::DVector;

    struct RosenbrockN;
    impl CostFunction for RosenbrockN {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((0..x.len() - 1)
                .map(|i| (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i] * x[i]).powi(2))
                .sum())
        }
    }

    let result = Executor::new(
        RosenbrockN,
        Newuoa::new().with_rho_beg(0.5).with_rho_end(1e-8),
        NewuoaState::new(DVector::from_vec(vec![-1.2, 1.0])),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.best_cost() < 1e-7,
        "best_cost = {}",
        result.best_cost()
    );
}
