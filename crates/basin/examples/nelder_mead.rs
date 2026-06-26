//! Minimize the 2-D Rosenbrock function with Nelder-Mead, the canonical
//! derivative-free testbed. No `Gradient` impl is needed, only `CostFunction`.
//!
//! Run with `cargo run --example nelder_mead`.

use basin::{BasicSimplexState, CostFunction, Executor, NelderMead};

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
    }
}

fn main() {
    let problem = Rosenbrock;
    let x0 = vec![-1.2, 1.0];

    println!("initial param: {:?}", x0);
    println!("initial cost:  {}", problem.cost(&x0).unwrap());

    // `BasicSimplexState::new(x0)` builds a default initial simplex around the
    // starting point (FMINSEARCH/SciPy convention). For full control over
    // the simplex use `BasicSimplexState::with_step` or `::from_simplex`.
    //
    // `NelderMead::adaptive()` infers `n` from the simplex during `init`
    // and uses the dimension-aware parameters of Gao & Han (2012).
    let solver = NelderMead::adaptive();
    let state = BasicSimplexState::new(x0);
    let result = Executor::new(problem, solver, state)
        .max_iter(2_000)
        .run()
        .unwrap();

    println!("final iter:    {}", result.iter());
    println!("final param:   {:?}", result.param());
    println!("final cost:    {}", result.cost());
    println!("termination:   {:?}", result.reason);
}
