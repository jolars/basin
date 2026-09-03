#![cfg(feature = "nalgebra_all")]

use crate::backend_aliases::nalgebra::DVector;
use basin::problems::Rosenbrock;
use basin::{
    Backtracking, BasicState, Bfgs, CostFunction, Executor, Gradient,
    GradientDescent, GradientTolerance, NalgebraQuasiNewtonState,
    TerminationReason,
};

#[test]
fn bfgs_converges_on_rosenbrock() {
    let problem = Rosenbrock::<DVector<f64>>::default();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(
        problem,
        Bfgs::new(),
        NalgebraQuasiNewtonState::new(initial),
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

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

#[test]
fn bfgs_terminates_on_gradient_tolerance() {
    let problem = Rosenbrock::<DVector<f64>>::default();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(
        problem,
        Bfgs::new(),
        NalgebraQuasiNewtonState::new(initial),
    )
    .max_iter(200)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
}

#[test]
fn bfgs_converges_faster_than_gd_with_backtracking() {
    let initial = DVector::from_vec(vec![-1.2, 1.0]);

    let bfgs_result = Executor::new(
        Rosenbrock::<DVector<f64>>::default(),
        Bfgs::new(),
        NalgebraQuasiNewtonState::new(initial.clone()),
    )
    .max_iter(500)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

    let gd_result = Executor::new(
        Rosenbrock::<DVector<f64>>::default(),
        GradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(500)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

    // Bfgs should reach the gradient tolerance while GD won't (or will
    // need many more iters). At the very least Bfgs should have a *much*
    // lower final cost.
    assert!(
        bfgs_result.cost() < gd_result.cost(),
        "Bfgs cost {} not better than GD cost {}",
        bfgs_result.cost(),
        gd_result.cost()
    );
    assert!(
        bfgs_result.iter() < gd_result.iter(),
        "Bfgs iters {} not fewer than GD iters {}",
        bfgs_result.iter(),
        gd_result.iter()
    );
}

/// Strictly convex 5-D quadratic `f(x) = ½ xᵀ A x − bᵀ x` with diagonal
/// `A = diag(1, 2, ..., 5)` and `b = (1, 1, ..., 1)`. Minimum at
/// `x* = A⁻¹ b`, `f(x*) = −½ bᵀ A⁻¹ b`. Bfgs on a quadratic with `n`
/// dimensions converges in `n` steps with exact line search; with strong
/// Wolfe it still converges very fast.
struct Quadratic {
    diag: Vec<f64>,
}

impl CostFunction for Quadratic {
    type Param = DVector<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
        Ok({
            let mut c = 0.0;
            for (i, xi) in x.iter().enumerate() {
                c += 0.5 * self.diag[i] * xi * xi - xi;
            }
            c
        })
    }
}

impl Gradient for Quadratic {
    type Gradient = DVector<f64>;
    fn gradient(
        &self,
        x: &DVector<f64>,
    ) -> Result<DVector<f64>, std::convert::Infallible> {
        Ok({
            let mut g = DVector::zeros(x.len());
            for (i, xi) in x.iter().enumerate() {
                g[i] = self.diag[i] * xi - 1.0;
            }
            g
        })
    }
}

#[test]
fn bfgs_on_5d_quadratic_converges_quickly() {
    let problem = Quadratic {
        diag: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };
    let initial = DVector::from_element(5, 0.0);

    let result = Executor::new(
        problem,
        Bfgs::new(),
        NalgebraQuasiNewtonState::new(initial),
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
    // ~2n iterations is the practical expectation with strong Wolfe (one
    // initial-scaling step + ~n quasi-conjugate steps).
    assert!(
        result.iter() <= 15,
        "expected convergence in ≤ 15 iters, got {}",
        result.iter()
    );
}

/// Confirms the line-search-bail safety net: if a user picks a gradient
/// tolerance below machine precision, Bfgs still terminates (via
/// `SolverConverged`) instead of spinning forever doing wasted line-search
/// work. The fast convergence makes |g| machine-epsilon-small after ~12
/// iterations on this problem.
#[test]
fn bfgs_terminates_via_converged_when_at_machine_precision() {
    let problem = Quadratic {
        diag: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };
    let initial = DVector::from_element(5, 0.0);

    let result = Executor::new(
        problem,
        Bfgs::new(),
        NalgebraQuasiNewtonState::new(initial),
    )
    .max_iter(200)
    .terminate_on(GradientTolerance(1e-30))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let expected_cost = -0.5 * (1.0 + 0.5 + 1.0 / 3.0 + 0.25 + 0.2);
    assert!(
        (result.cost() - expected_cost).abs() < 1e-10,
        "cost = {}, expected {}",
        result.cost(),
        expected_cost
    );
}
