use basin::NewtonRoot;
use std::convert::Infallible;

fn main() {
    let result = NewtonRoot::new(0.0, 2.0)
        .with_initial_guess(1.0)
        .with_absolute_position_tolerance(1e-10)
        .with_max_iter(100)
        .solve_combined(|x| Ok::<_, Infallible>((x * x - 2.0, 2.0 * x)))
        .unwrap();

    assert!(result.converged());
    assert!((result.root() - 2.0_f64.sqrt()).abs() < 1e-10);
    assert_eq!(result.function_evals(), result.callback_evals());
    assert_eq!(result.derivative_evals(), result.callback_evals());
    assert_eq!(result.second_derivative_evals(), 0);
}
