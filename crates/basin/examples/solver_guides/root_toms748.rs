use basin::Toms748Root;
use std::convert::Infallible;

fn main() {
    let result = Toms748Root::new(0.0, 2.0)
        .with_absolute_position_tolerance(1e-10)
        .with_max_iter(100)
        .solve(|x| Ok::<_, Infallible>(x * x - 2.0))
        .unwrap();

    assert!(result.converged());
    assert!((result.root() - 2.0_f64.sqrt()).abs() < 1e-10);
    assert_eq!(result.value(), result.root() * result.root() - 2.0);
    assert_eq!(result.function_evals(), result.callback_evals());
}
