use basin::HalleyRoot;
use std::convert::Infallible;

fn main() {
    let result = HalleyRoot::new(0.0, 2.0)
        .solve(
            |x| Ok::<_, Infallible>(x * x * x - 2.0),
            |x| Ok(3.0 * x * x),
            |x| Ok(6.0 * x),
        )
        .unwrap();

    assert!(result.converged());
    assert!((result.root() - 2.0_f64.cbrt()).abs() < 1e-10);
}
