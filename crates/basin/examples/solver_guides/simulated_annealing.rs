use basin::core::rng::{ChaCha8Rng, RngExt};
use basin::{CostFunction, Executor, SimulatedAnnealing, TemperatureSchedule};
use std::convert::Infallible;

struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}

fn main() {
    let neighbor = |x: &Vec<f64>, temperature: f64, rng: &mut ChaCha8Rng| {
        x.iter()
            .map(|v| v + temperature.sqrt() * (2.0 * rng.random::<f64>() - 1.0))
            .collect::<Vec<_>>()
    };
    let schedule =
        TemperatureSchedule::geometric(0.995).with_steps_per_temperature(8);
    let solver = SimulatedAnnealing::new(neighbor, 2.0, schedule, 42);

    let result = Executor::from_start(Sphere, solver, vec![3.0, -4.0])
        .max_iter(10_000)
        .run()
        .unwrap();

    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < 1e-2);
    assert!(
        incumbent.cost.unwrap()
            <= result.current_record().unwrap().cost.unwrap()
    );
}
