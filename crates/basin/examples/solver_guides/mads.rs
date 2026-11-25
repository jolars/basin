use basin::{CostFunction, Executor, Mads, Termination};
use std::convert::Infallible;

struct AbsValley;

impl CostFunction for AbsValley {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 1.0).abs() + 2.0 * (x[1] - 2.0).abs())
    }
}

fn main() {
    let solver = Mads::new()
        .with_initial_poll_size(1.0)
        .with_minimum_poll_size(1e-8);

    let result = Executor::from_start(AbsValley, solver, vec![0.0, 0.0])
        .max_cost_evals(5_000)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < 1e-6);
    assert!((incumbent.param[0] - 1.0).abs() < 1e-6);
    assert!((incumbent.param[1] - 2.0).abs() < 1e-6);
}
