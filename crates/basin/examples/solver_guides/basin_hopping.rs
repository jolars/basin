use basin::{BasinHopping, CostFunction, Executor, NelderMead};
use std::convert::Infallible;
use std::f64::consts::{E, PI};

struct Ackley;

impl CostFunction for Ackley {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        let n = x.len() as f64;
        let squares = x.iter().map(|v| v * v).sum::<f64>() / n;
        let waves = x.iter().map(|v| (2.0 * PI * v).cos()).sum::<f64>() / n;
        Ok(-20.0 * (-0.2 * squares.sqrt()).exp() - waves.exp() + 20.0 + E)
    }
}

fn main() {
    let inner = NelderMead::adaptive()
        .with_absolute_simplex_size_tolerance(1e-10)
        .with_absolute_simplex_cost_tolerance(1e-10);
    let solver = BasinHopping::new(inner, 7)
        .with_stepsize(1.0)
        .with_temperature(1.0)
        .with_inner_max_iter(1_000);

    let result = Executor::from_start(Ackley, solver, vec![2.5, -2.5])
        .max_iter(200)
        .run()
        .unwrap();

    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < 1e-6);
    assert!(incumbent.param.iter().all(|v| v.abs() < 1e-6));
}
