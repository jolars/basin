#![cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]

use basin::core::rng::ChaCha8Rng;
use basin::{CostFunction, Executor, SimulatedAnnealing, TemperatureSchedule};
use std::convert::Infallible;

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_parameter_runs() {
    use crate::backend_aliases::nalgebra::DVector;

    struct Sphere;
    impl CostFunction for Sphere {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, Infallible> {
            Ok(x.iter().map(|value| value * value).sum())
        }
    }

    let solver = SimulatedAnnealing::new(
        |x: &DVector<f64>, _: f64, _: &mut ChaCha8Rng| x * 0.5,
        1.0,
        TemperatureSchedule::reciprocal(),
        1,
    );
    let result = Executor::from_start(
        Sphere,
        solver,
        DVector::from_vec(vec![2.0, -1.0]),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.cost(), 1.25);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_parameter_runs() {
    use crate::backend_aliases::ndarray::Array1;

    struct Sphere;
    impl CostFunction for Sphere {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, Infallible> {
            Ok(x.iter().map(|value| value * value).sum())
        }
    }

    let solver = SimulatedAnnealing::new(
        |x: &Array1<f64>, _: f64, _: &mut ChaCha8Rng| x * 0.5,
        1.0,
        TemperatureSchedule::reciprocal(),
        1,
    );
    let result =
        Executor::from_start(Sphere, solver, Array1::from(vec![2.0, -1.0]))
            .max_iter(1)
            .run()
            .unwrap();
    assert_eq!(result.cost(), 1.25);
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_parameter_runs() {
    use crate::backend_aliases::faer::Col;

    struct Sphere;
    impl CostFunction for Sphere {
        type Param = Col<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, Infallible> {
            Ok((0..x.nrows()).map(|i| x[i] * x[i]).sum())
        }
    }

    let solver = SimulatedAnnealing::new(
        |x: &Col<f64>, _: f64, _: &mut ChaCha8Rng| x * 0.5,
        1.0,
        TemperatureSchedule::reciprocal(),
        1,
    );
    let result = Executor::from_start(
        Sphere,
        solver,
        Col::from_fn(2, |i| [2.0, -1.0][i]),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.cost(), 1.25);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
