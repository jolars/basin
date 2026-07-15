//! `Vec<f64>`-backend smoke tests for [`MaLsChCma`] (no feature gate).
//! Convergence on Sphere and Rastrigin to confirm the per-backend
//! trait wiring works on the hand-rolled `DenseMatrix` covariance; the
//! deeper algorithmic invariants are covered by the nalgebra mirror
//! test (`tests/ma_ls_ch_cma_nalgebra.rs`).

use basin::problems::RastriginBoxed;
use basin::{CostFunction, DenseMatrix, Executor, MaLsChCma, MaLsChState, MaxCostEvals};

struct BoxedSphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl BoxedSphere {
    fn new(n: usize, half_width: f64) -> Self {
        Self {
            lower: vec![-half_width; n],
            upper: vec![half_width; n],
        }
    }
}

impl CostFunction for BoxedSphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}

impl basin::BoxConstraints for BoxedSphere {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn converges_on_sphere_d10() {
    let problem = BoxedSphere::new(10, 5.0);
    let solver = MaLsChCma::<Vec<f64>, DenseMatrix>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) Vec<f64> cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

#[test]
fn converges_on_rastrigin_d10() {
    let problem = RastriginBoxed::<Vec<f64>>::with_standard_bounds(10);
    let solver = MaLsChCma::<Vec<f64>, DenseMatrix>::new(42).with_pop_size(30);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(50_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1.0,
        "Rastrigin(D=10) Vec<f64> cost {} should be < 1.0 within 50k evals",
        result.cost()
    );
}
