//! L-Bfgs-B convergence tests over the `Vec<f64>` backend.
//!
//! Layer A of S14.7 (per the L-Bfgs-B port plan). Each test exercises
//! a different code path:
//!
//! - **`unbounded_rosenbrock_2d_converges`**: `cnstnd == false`,
//!   skips the GCP after the first iteration and runs L-Bfgs-style.
//! - **`booth_at_corner_converges`**: every variable two-sided
//!   bounded; the optimum is at the upper-bound corner.
//! - **`booth_slack_bounds_recover_unconstrained_minimum`**:
//!   bounds present but inactive at the optimum (sanity check that
//!   the GCP path returns the unconstrained Newton step when nothing
//!   binds).
//! - **`quadratic_5d_diagonal_converges_quickly`**: a strictly-
//!   convex 5-D quadratic where the limited-memory approximation
//!   captures the exact Hessian within `m` iterations.

use basin::problems::BoothBoxed;
use basin::{
    CostFunction, Executor, Gradient, LbfgsState, Lbfgsb, MaxIter, ProjectedGradientTolerance,
};

/// Unbounded Rosenbrock 2D from `(-1.2, 1.0)`. With infinite bounds
/// L-Bfgs-B reduces to L-Bfgs (Fortran's `cnstnd == false` branch
/// skips the GCP entirely on iter ≥ 1).
#[test]
fn unbounded_rosenbrock_2d_converges() {
    struct Rosen {
        l: Vec<f64>,
        u: Vec<f64>,
    }
    impl CostFunction for Rosen {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
        }
    }
    impl Gradient for Rosen {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok({
                let dfdx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
                let dfdx1 = 200.0 * (x[1] - x[0] * x[0]);
                vec![dfdx0, dfdx1]
            })
        }
    }
    impl basin::BoxConstraints for Rosen {
        fn lower(&self) -> &Vec<f64> {
            &self.l
        }
        fn upper(&self) -> &Vec<f64> {
            &self.u
        }
    }

    let problem = Rosen {
        l: vec![f64::NEG_INFINITY; 2],
        u: vec![f64::INFINITY; 2],
    };
    let lower = problem.l.clone();
    let upper = problem.u.clone();
    let state = LbfgsState::new(vec![-1.2, 1.0], 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
        .run()
        .unwrap();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
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

/// `BoothBoxed` with bounds `[-1, 1]²`. The unconstrained minimum
/// `(1, 3)` is outside the box, so the constrained optimum is at
/// the corner `(1, 1)`—both active simultaneously. Verifies that
/// GCP correctly identifies the active set and subsm respects
/// dual feasibility.
#[test]
fn booth_at_corner_converges() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];
    let state = LbfgsState::new(vec![0.0, 0.0], 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(
            lower.clone(),
            upper.clone(),
            1e-8,
        ))
        .run()
        .unwrap();

    // Booth: f(x, y) = (x + 2y − 7)² + (2x + y − 5)². At the upper
    // corner (1, 1): (1 + 2 − 7)² + (2 + 1 − 5)² = 16 + 4 = 20. Both
    // partial derivatives are negative there, so the upward push
    // gets clipped by the upper bounds; KKT satisfied.
    let expected_x = [1.0, 1.0];
    assert!(
        (result.param()[0] - expected_x[0]).abs() < 1e-5,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - expected_x[1]).abs() < 1e-5,
        "x[1] = {}",
        result.param()[1]
    );
    assert!(
        (result.cost() - 20.0).abs() < 1e-8,
        "cost = {}, expected 20",
        result.cost()
    );
}

/// `BoothBoxed` with bounds `[-5, 5]²`—the unconstrained minimum
/// `(1, 3)` lies inside. The projected solver must recover the
/// unconstrained answer, which is the L-Bfgs-style behavior on a
/// well-conditioned 2-D quadratic.
#[test]
fn booth_slack_bounds_recover_unconstrained_minimum() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-5.0, -5.0], vec![5.0, 5.0]);
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let state = LbfgsState::new(vec![0.0, 0.0], 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-10))
        .run()
        .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 3.0).abs() < 1e-4,
        "x[1] = {}",
        result.param()[1]
    );
    assert!(result.cost() < 1e-8, "cost = {}", result.cost());
}

/// Strictly convex 5-D quadratic `f(x) = ½ xᵀ A x − bᵀ x` with
/// diagonal `A = diag(1, 2, …, 5)` and `b = (1, …, 1)`. Unconstrained
/// minimum at `x*[i] = 1/diag[i]`. Bounds `[-2, 2]⁵` are slack; the
/// optimum lies inside. With `m = 5` (matching the dimension), L-Bfgs
/// has enough memory to capture the exact diagonal Hessian.
#[test]
fn quadratic_5d_diagonal_converges_quickly() {
    struct Quadratic {
        diag: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
    }
    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
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
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok({
                let mut g = vec![0.0; x.len()];
                for (i, xi) in x.iter().enumerate() {
                    g[i] = self.diag[i] * xi - 1.0;
                }
                g
            })
        }
    }
    impl basin::BoxConstraints for Quadratic {
        fn lower(&self) -> &Vec<f64> {
            &self.l
        }
        fn upper(&self) -> &Vec<f64> {
            &self.u
        }
    }

    let problem = Quadratic {
        diag: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        l: vec![-2.0; 5],
        u: vec![2.0; 5],
    };
    let lower = problem.l.clone();
    let upper = problem.u.clone();
    let initial = vec![0.0; 5];
    let state = LbfgsState::new(initial, 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(50))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-10))
        .run()
        .unwrap();

    // Optimum: x[i] = 1/diag[i]; cost = -½ Σ 1/diag[i].
    let expected_cost = -0.5 * (1.0 + 0.5 + 1.0 / 3.0 + 0.25 + 0.2);
    assert!(
        (result.cost() - expected_cost).abs() < 1e-8,
        "cost = {}, expected {}",
        result.cost(),
        expected_cost
    );
    for (i, x_i) in result.param().iter().enumerate() {
        let expected = 1.0 / (i + 1) as f64;
        assert!(
            (x_i - expected).abs() < 1e-5,
            "x[{i}] = {x_i}, expected {expected}"
        );
    }
}
