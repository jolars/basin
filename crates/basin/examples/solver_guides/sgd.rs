use basin::{CostFunction, Executor, MiniBatchGradient, Sgd};
use std::convert::Infallible;

struct MeanFit {
    observations: [f64; 4],
}

impl CostFunction for MeanFit {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        let sum: f64 =
            self.observations.iter().map(|y| (x[0] - y).powi(2)).sum();
        Ok(sum / self.observations.len() as f64)
    }
}

impl MiniBatchGradient for MeanFit {
    type Gradient = Vec<f64>;

    fn n_samples(&self) -> usize {
        self.observations.len()
    }

    fn batch_gradient(
        &self,
        x: &Self::Param,
        batch: &[usize],
    ) -> Result<Self::Gradient, Self::Error> {
        let sum: f64 = batch
            .iter()
            .map(|&i| 2.0 * (x[0] - self.observations[i]))
            .sum();
        Ok(vec![sum / batch.len() as f64])
    }
}

fn main() {
    let problem = MeanFit {
        observations: [1.5, 2.5, 1.0, 3.0],
    };
    let solver = Sgd::new(0.01, 2, 42);

    let result = Executor::from_start(problem, solver, vec![-3.0])
        .max_iter(3_000)
        .run()
        .unwrap();

    assert!((result.param()[0] - 2.0).abs() < 0.05);
    assert!(
        (result.current_record().unwrap().cost.unwrap() - 0.625).abs() < 0.0025
    );
}
