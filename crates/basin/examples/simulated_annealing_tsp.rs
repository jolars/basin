use basin::core::rng::{ChaCha8Rng, RngExt};
use basin::{CostFunction, Executor, SimulatedAnnealing, TemperatureSchedule};
use std::convert::Infallible;

struct TourLength {
    points: Vec<(f64, f64)>,
}

impl CostFunction for TourLength {
    type Param = Vec<usize>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, tour: &Vec<usize>) -> Result<f64, Infallible> {
        Ok((0..tour.len())
            .map(|i| {
                let a = self.points[tour[i]];
                let b = self.points[tour[(i + 1) % tour.len()]];
                (a.0 - b.0).hypot(a.1 - b.1)
            })
            .sum())
    }
}

fn main() {
    let problem = TourLength {
        points: vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
        ],
    };
    let neighbor = |tour: &Vec<usize>, _: f64, rng: &mut ChaCha8Rng| {
        let mut candidate = tour.clone();
        let i = rng.random_range(0..candidate.len());
        let mut j = rng.random_range(0..candidate.len() - 1);
        if j >= i {
            j += 1;
        }
        candidate.swap(i, j);
        candidate
    };
    let solver = SimulatedAnnealing::new(
        neighbor,
        1.0,
        TemperatureSchedule::geometric(0.995).with_steps_per_temperature(4),
        88,
    );
    let result = Executor::from_start(problem, solver, vec![0, 2, 1, 3, 4])
        .max_iter(1_000)
        .run()
        .unwrap();

    println!("tour: {:?}", result.best_param());
    println!("length: {}", result.best_cost());
}
