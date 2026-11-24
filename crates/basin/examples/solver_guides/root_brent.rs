use basin::{BrentRoot, RootBracketer};
use std::convert::Infallible;

fn main() {
    let function = |x| Ok::<_, Infallible>(x * x - 2.0);
    let search = RootBracketer::new(0.0, 1.0)
        .with_lower_bound(0.0)
        .bracket(function)
        .unwrap();
    assert!(search.bracketed());

    let (lower, upper) = search.bracket();
    let result = BrentRoot::new(lower, upper)
        .with_absolute_position_tolerance(1e-10)
        .with_max_iter(100)
        .solve(function)
        .unwrap();

    assert!(result.converged());
    assert!((result.root() - 2.0_f64.sqrt()).abs() < 1e-10);
    assert_eq!(result.value(), result.root() * result.root() - 2.0);
    let total_evaluations = search.function_evals() + result.function_evals();
    assert!(total_evaluations > result.function_evals());
}
