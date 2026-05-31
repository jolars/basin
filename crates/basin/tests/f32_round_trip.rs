//! End-to-end smoke test that solvers, states, and termination criteria
//! compose at `F = f32` over the `Vec<F>` backend. Demonstrates the
//! provisional-choice trigger from `AGENTS.md` is now satisfiable: the
//! whole pipeline runs at a non-`f64` scalar without further refactor.

use basin::GradientDescent;
use basin::core::executor::Executor;
use basin::core::problem::{CostFunction, Gradient};
use basin::core::state::{BasicState, LbfgsState, State};
use basin::core::termination::{
    CostTolerance, GradientTolerance, MaxIter, RelativeCostTolerance, TargetCost,
};
use basin::line_search::{Backtracking, MoreThuente};
use basin::solver::lbfgs::{Lbfgs, Unbounded};

/// `f(x) = ‖x − c‖²` with `c = (1, 2, 3)`. Minimum at `c`, cost 0.
struct ShiftedQuadF32 {
    c: Vec<f32>,
}

impl CostFunction for ShiftedQuadF32 {
    type Param = Vec<f32>;
    type Output = f32;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f32>) -> Result<f32, Self::Error> {
        Ok(x.iter().zip(&self.c).map(|(a, b)| (a - b).powi(2)).sum())
    }
}

impl Gradient for ShiftedQuadF32 {
    type Gradient = Vec<f32>;
    fn gradient(&self, x: &Vec<f32>) -> Result<Vec<f32>, Self::Error> {
        Ok(x.iter().zip(&self.c).map(|(a, b)| 2.0 * (a - b)).collect())
    }
}

#[test]
fn gradient_descent_f32_with_f32_termination_converges() {
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = BasicState::<Vec<f32>, f32>::new(vec![0.0_f32; 3]);
    let solver: GradientDescent<Backtracking<f32>, Vec<f32>, f32> =
        GradientDescent::with_line_search(Backtracking::new());

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(500))
        .terminate_on(GradientTolerance::<f32>(1e-3))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-2);
    assert!((final_x[1] - 2.0).abs() < 1e-2);
    assert!((final_x[2] - 3.0).abs() < 1e-2);
}

#[test]
fn unbounded_lbfgs_f32_round_trips_state_solver_termination() {
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = LbfgsState::<Vec<f32>, f32>::new(vec![0.0_f32; 3], 5);
    let solver: Lbfgs<Unbounded, MoreThuente<f32>, f32> =
        Lbfgs::<Unbounded, MoreThuente<f32>, f32>::with_line_search(MoreThuente::new());

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(100))
        .terminate_on(GradientTolerance::<f32>(1e-3))
        .terminate_on(CostTolerance::<f32>::new(1e-6))
        .terminate_on(RelativeCostTolerance::<f32>::new(1e-6))
        .terminate_on(TargetCost::<f32>(1e-6))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-3);
    assert!((final_x[1] - 2.0).abs() < 1e-3);
    assert!((final_x[2] - 3.0).abs() < 1e-3);
}
