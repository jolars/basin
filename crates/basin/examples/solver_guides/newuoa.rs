use basin::{CostFunction, Executor, Newuoa, Termination};
use std::convert::Infallible;

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2))
    }
}

fn main() {
    let solver = Newuoa::new()
        .with_initial_radius(0.5)
        .with_final_radius(1e-8);

    let result = Executor::from_start(Quadratic, solver, vec![-3.0, 5.0])
        .max_evaluations(basin::EvaluationKind::Cost, 500)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < 1e-12);
    assert!((incumbent.param[0] - 1.0).abs() < 1e-6);
    assert!((incumbent.param[1] + 2.0).abs() < 1e-6);
}
