//! Bfgs convergence over the default `Vec<f64>` backend (no feature gate).
//!
//! Mirrors the nalgebra tests in `tests/bfgs.rs` to confirm the generic
//! `Solver` impl runs on the hand-rolled
//! [`DenseMatrix`](basin::DenseMatrix) inverse-Hessian.

use basin::problems::Rosenbrock;
use basin::{
    Bfgs, CostFunction, DenseQuasiNewtonState, Executor, Gradient,
    GradientTolerance, TerminationReason,
};

#[test]
fn bfgs_converges_on_rosenbrock() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];

    let result = Executor::new(
        problem,
        Bfgs::new(),
        DenseQuasiNewtonState::new(initial),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-8,
        "expected near-zero cost, got {}",
        result.cost()
    );
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-4,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn bfgs_terminates_on_gradient_tolerance() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];

    let result = Executor::new(
        problem,
        Bfgs::new(),
        DenseQuasiNewtonState::new(initial),
    )
    .max_iter(200)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
}

/// Strictly convex 5-D quadratic `f(x) = ½ xᵀ A x − bᵀ x` with diagonal
/// `A = diag(1, 2, ..., 5)` and `b = (1, 1, ..., 1)`. Minimum at
/// `x* = A⁻¹ b`. Mirrors the nalgebra `Quadratic` test to exercise an
/// `n > 2` dense inverse-Hessian on the `Vec<f64>` backend.
struct Quadratic {
    diag: Vec<f64>,
}

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok({
            x.iter()
                .enumerate()
                .map(|(i, xi)| 0.5 * self.diag[i] * xi * xi - xi)
                .sum()
        })
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok({
            x.iter()
                .enumerate()
                .map(|(i, xi)| self.diag[i] * xi - 1.0)
                .collect()
        })
    }
}

#[test]
fn bfgs_on_5d_quadratic_converges_quickly() {
    let problem = Quadratic {
        diag: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };
    let initial = vec![0.0; 5];

    let result = Executor::new(
        problem,
        Bfgs::new(),
        DenseQuasiNewtonState::new(initial),
    )
    .max_iter(50)
    .terminate_on(GradientTolerance(1e-8))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    // Optimum: x[i] = 1 / diag[i]; cost = -½ Σ 1/diag[i].
    let expected_cost = -0.5 * (1.0 + 0.5 + 1.0 / 3.0 + 0.25 + 0.2);
    assert!(
        (result.cost() - expected_cost).abs() < 1e-10,
        "cost = {}, expected {}",
        result.cost(),
        expected_cost
    );
    assert!(
        result.iter() <= 15,
        "expected convergence in ≤ 15 iters, got {}",
        result.iter()
    );
}
