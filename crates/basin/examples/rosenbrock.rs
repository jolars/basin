use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent};

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        let a = 1.0;
        let b = 100.0;
        Ok((a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2))
    }
}

impl Gradient for Rosenbrock {
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let a = 1.0;
        let b = 100.0;
        Ok(vec![
            -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0].powi(2)),
            2.0 * b * (x[1] - x[0].powi(2)),
        ])
    }
}

fn main() {
    let problem = Rosenbrock;
    let initial = vec![-1.2, 1.0];

    let initial_cost = problem.cost(&initial).unwrap();
    println!("initial param: {:?}", initial);
    println!("initial cost:  {}", initial_cost);

    let solver = GradientDescent::new(0.001);
    let state = BasicState::new(initial);
    let result = Executor::new(problem, solver, state)
        .max_iter(50_000)
        .run()
        .unwrap();

    println!("final iter:    {}", result.iter());
    println!("final param:   {:?}", result.param());
    println!("final cost:    {}", result.cost());
    println!("termination:   {:?}", result.reason);
}
