use basin::NewtonRoot;
use std::convert::Infallible;

fn main() {
    let result = NewtonRoot::new(0.0, 2.0)
        .solve(|x| Ok::<_, Infallible>(x * x - 2.0), |x| Ok(2.0 * x))
        .unwrap();

    assert!(result.converged());
    assert!((result.root() - 2.0_f64.sqrt()).abs() < 1e-10);
}
