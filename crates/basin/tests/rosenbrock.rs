#![cfg(feature = "problems")]

use basin::problems::Rosenbrock;
use basin::{
    Backtracking, CostFunction, Executor, FirstOrderState, GradientDescent,
    TerminationCode,
};

#[test]
fn gradient_descent_decreases_rosenbrock_cost() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];
    let initial_cost = problem.cost(&initial).unwrap();

    let result = Executor::new(
        problem,
        GradientDescent::new(0.001),
        FirstOrderState::new(initial),
    )
    .max_iter(10_000)
    .run()
    .unwrap();

    assert_eq!(result.iter(), 10_000, "should hit max_iter");
    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    assert!(
        result.cost() < initial_cost * 0.1,
        "expected cost to drop by >10x: initial={}, final={}",
        initial_cost,
        result.cost()
    );
}

#[test]
fn gradient_descent_with_momentum_decreases_rosenbrock_cost() {
    // Heavy-ball momentum is what makes steepest descent behave on the
    // curved Rosenbrock valley (and is what traces the river in the logo
    // example). A modest step with momentum should still drive the cost
    // down sharply from the classical start.
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];
    let initial_cost = problem.cost(&initial).unwrap();

    let result = Executor::new(
        problem,
        GradientDescent::new(0.0008).with_momentum(0.85),
        FirstOrderState::new(initial),
    )
    .max_iter(10_000)
    .run()
    .unwrap();

    assert!(
        result.cost() < initial_cost * 0.1,
        "expected cost to drop by >10x with momentum: initial={}, final={}",
        initial_cost,
        result.cost()
    );
}

#[test]
fn gradient_descent_with_backtracking_decreases_rosenbrock_cost() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];
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
