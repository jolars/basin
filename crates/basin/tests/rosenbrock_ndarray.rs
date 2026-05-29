#![cfg(feature = "ndarray")]

use basin::problems::Rosenbrock;
use basin::{
    Backtracking, BasicSimplexState, BasicState, CostFunction, Executor, GradientDescent,
    NelderMead,
};
use ndarray::{Array1, array};

#[test]
fn gradient_descent_with_ndarray_array1() {
    let problem = Rosenbrock::<Array1<f64>>::default();
    let initial = array![-1.2, 1.0];
    let initial_cost = problem.cost(&initial).unwrap();

    let result = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(initial),
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
fn gradient_descent_with_ndarray_array1_and_backtracking() {
    let problem = Rosenbrock::<Array1<f64>>::default();
    let initial = array![-1.2, 1.0];
    let initial_cost = problem.cost(&initial).unwrap();

    let result = Executor::new(
        problem,
        GradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
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
fn nelder_mead_with_ndarray_array1() {
    let problem = Rosenbrock::<Array1<f64>>::default();
    let initial = array![-1.2, 1.0];

    let result = Executor::new(
        problem,
        NelderMead::adaptive(),
        BasicSimplexState::new(initial),
    )
    .max_iter(2_000)
    .run()
    .unwrap();

    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
}
