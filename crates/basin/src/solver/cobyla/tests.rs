//! End-to-end COBYLA convergence tests on Powell's 1994 problems (Tables 1–2).
//! These assert convergence to the *published* optima — an oracle independent of
//! PRIMA. The PRIMA bit-parity cross-validation lives in [`super::parity`].

use crate::NonlinearInequalityConstraints;
use crate::core::executor::Executor;
use crate::core::problem::CostFunction;
use crate::core::state::CobylaState;
use crate::core::termination::TerminationReason;
use crate::solver::Cobyla;

/// A closure-defined nonlinearly-constrained test problem.
struct Prob {
    f: fn(&[f64]) -> f64,
    c: fn(&[f64]) -> Vec<f64>,
    m: usize,
}
impl CostFunction for Prob {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((self.f)(x))
    }
}
impl NonlinearInequalityConstraints for Prob {
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok((self.c)(x))
    }
    fn num_constraints(&self) -> usize {
        self.m
    }
}

#[test]
fn problem_b_disk_min_product() {
    // (B) min x1·x2 s.t. x1² + x2² ≤ 1 → F* = −1/2.
    let prob = Prob {
        f: |x| x[0] * x[1],
        c: |x| vec![x[0] * x[0] + x[1] * x[1] - 1.0],
        m: 1,
    };
    let solver = Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-4);
    let state = CobylaState::new(vec![1.0, 1.0]);
    let result = Executor::new(prob, solver, state)
        .terminate_on(crate::MaxCostEvals(2000))
        .run()
        .unwrap();
    assert!(
        (result.best_cost() - (-0.5)).abs() < 1e-3,
        "F = {}",
        result.best_cost()
    );
    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[test]
fn problem_c_ellipsoid_min_product3() {
    // (C) min x1·x2·x3 s.t. x1² + 2x2² + 3x3² ≤ 1 → F* ≈ −0.07857.
    let prob = Prob {
        f: |x| x[0] * x[1] * x[2],
        c: |x| vec![x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2] - 1.0],
        m: 1,
    };
    let solver = Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-5);
    let state = CobylaState::new(vec![1.0, 1.0, 1.0]);
    let result = Executor::new(prob, solver, state)
        .terminate_on(crate::MaxCostEvals(3000))
        .run()
        .unwrap();
    assert!(
        (result.best_cost() - (-0.078567)).abs() < 1e-3,
        "F = {}",
        result.best_cost()
    );
}

#[test]
fn problem_f_fletcher() {
    // (F) min −x1 − x2 s.t. x1² ≤ x2 and x1² + x2² ≤ 1 → F* = −√2.
    let prob = Prob {
        f: |x| -x[0] - x[1],
        c: |x| vec![x[0] * x[0] - x[1], x[0] * x[0] + x[1] * x[1] - 1.0],
        m: 2,
    };
    let solver = Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-5);
    let state = CobylaState::new(vec![1.0, 1.0]);
    let result = Executor::new(prob, solver, state)
        .terminate_on(crate::MaxCostEvals(3000))
        .run()
        .unwrap();
    assert!(
        (result.best_cost() - (-2.0_f64.sqrt())).abs() < 1e-3,
        "F = {}",
        result.best_cost()
    );
    // Feasibility at the reported point.
    let x = result.best_param();
    assert!(x[0] * x[0] - x[1] <= 1e-4);
    assert!(x[0] * x[0] + x[1] * x[1] - 1.0 <= 1e-4);
}

#[test]
fn problem_g_fletcher() {
    // (G) min x3 s.t. 5x1 − x2 + x3 ≥ 0, −5x1 − x2 + x3 ≥ 0,
    //     x1² + x2² + 4x2 ≤ x3 → F* = −3.
    // Constraints in ≤ 0 form: −(5x1 − x2 + x3), −(−5x1 − x2 + x3),
    //     x1² + x2² + 4x2 − x3.
    let prob = Prob {
        f: |x| x[2],
        c: |x| {
            vec![
                -(5.0 * x[0] - x[1] + x[2]),
                -(-5.0 * x[0] - x[1] + x[2]),
                x[0] * x[0] + x[1] * x[1] + 4.0 * x[1] - x[2],
            ]
        },
        m: 3,
    };
    let solver = Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-5);
    let state = CobylaState::new(vec![1.0, 1.0, 1.0]);
    let result = Executor::new(prob, solver, state)
        .terminate_on(crate::MaxCostEvals(3000))
        .run()
        .unwrap();
    assert!(
        (result.best_cost() - (-3.0)).abs() < 1e-2,
        "F = {}",
        result.best_cost()
    );
}

#[test]
fn unconstrained_rosenbrock_like() {
    // (D) a mild Rosenbrock; with m = 0 COBYLA still runs (pure trust region).
    let prob = Prob {
        f: |x| 10.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
        c: |_x| vec![],
        m: 0,
    };
    let solver = Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-6);
    let state = CobylaState::new(vec![-1.0, 1.0]);
    let result = Executor::new(prob, solver, state)
        .terminate_on(crate::MaxCostEvals(5000))
        .run()
        .unwrap();
    assert!(result.best_cost() < 1e-3, "F = {}", result.best_cost());
}
