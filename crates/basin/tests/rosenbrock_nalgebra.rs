#![cfg(all(feature = "nalgebra_all", feature = "problems"))]

use crate::backend_aliases::nalgebra::DVector;
use basin::problems::Rosenbrock;
use basin::{
    Backtracking, CostFunction, Executor, FirstOrderState, GradientDescent,
    NelderMead, SimplexProgress,
};

#[test]
fn gradient_descent_with_nalgebra_dvector() {
    let problem = Rosenbrock::<DVector<f64>>::default();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let initial_cost = problem.cost(&initial).unwrap();

    let result = Executor::new(
        problem,
        GradientDescent::new(0.001),
        FirstOrderState::new(initial),
    )
    .max_iter(10_000)
    .run()
    .unwrap();

    assert!(
        result.cost() < initial_cost * 0.1,
        "expected cost to drop by >10x: initial={}, final={}",
        initial_cost,
        result.cost()
    );
}

#[test]
fn gradient_descent_with_nalgebra_dvector_and_backtracking() {
    let problem = Rosenbrock::<DVector<f64>>::default();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let initial_cost = problem.cost(&initial).unwrap();

    let result = Executor::new(
        problem,
        GradientDescent::with_line_search(Backtracking::new()),
        FirstOrderState::new(initial),
    )
    .max_iter(10_000)
    .run()
    .unwrap();

    assert!(
        result.cost() < initial_cost * 0.1,
        "expected cost to drop by >10x: initial={}, final={}",
        initial_cost,
        result.cost()
    );
}

#[test]
fn nelder_mead_with_nalgebra_dvector() {
    let problem = Rosenbrock::<DVector<f64>>::default();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(
        problem,
        NelderMead::adaptive(),
        SimplexProgress::new(initial),
    )
    .max_iter(2_000)
    .run()
    .unwrap();

    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
