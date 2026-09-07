use std::convert::Infallible;
use std::marker::PhantomData;

use basin::{
    CostFunction, Dot, Gradient, HagerZhang, LineSearch, Problem, ScaledAdd,
};

trait TestVector: Clone {
    fn singleton(value: f64) -> Self;
    fn first(&self) -> f64;
}

impl TestVector for Vec<f64> {
    fn singleton(value: f64) -> Self {
        vec![value]
    }

    fn first(&self) -> f64 {
        self[0]
    }
}

#[cfg(feature = "nalgebra_all")]
impl TestVector for crate::backend_aliases::nalgebra::DVector<f64> {
    fn singleton(value: f64) -> Self {
        Self::from_vec(vec![value])
    }

    fn first(&self) -> f64 {
        self[0]
    }
}

#[cfg(feature = "ndarray_all")]
impl TestVector for crate::backend_aliases::ndarray::Array1<f64> {
    fn singleton(value: f64) -> Self {
        Self::from_vec(vec![value])
    }

    fn first(&self) -> f64 {
        self[0]
    }
}

#[cfg(feature = "faer_all")]
impl TestVector for crate::backend_aliases::faer::Col<f64> {
    fn singleton(value: f64) -> Self {
        Self::from_fn(1, |_| value)
    }

    fn first(&self) -> f64 {
        self[0]
    }
}

struct ShiftedQuadratic<V>(PhantomData<V>);

impl<V: TestVector> CostFunction for ShiftedQuadratic<V> {
    type Param = V;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<f64, Self::Error> {
        Ok((x.first() - 3.0).powi(2))
    }
}

impl<V: TestVector> Gradient for ShiftedQuadratic<V> {
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        Ok(V::singleton(2.0 * (x.first() - 3.0)))
    }
}

fn finds_stationary_step<V>()
where
    V: TestVector + Dot<f64> + ScaledAdd<f64>,
{
    let mut problem = Problem::new(ShiftedQuadratic::<V>(PhantomData));
    let mut search = HagerZhang::new();
    let alpha = search
        .next(
            &mut problem,
            &V::singleton(0.0),
            9.0,
            &V::singleton(-6.0),
            &V::singleton(6.0),
        )
        .unwrap();

    assert!((alpha - 0.5).abs() < 1e-12, "alpha = {alpha}");
}

#[test]
fn vec_parameter_runs() {
    finds_stationary_step::<Vec<f64>>();
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_parameter_runs() {
    finds_stationary_step::<crate::backend_aliases::nalgebra::DVector<f64>>();
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_parameter_runs() {
    finds_stationary_step::<crate::backend_aliases::ndarray::Array1<f64>>();
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_parameter_runs() {
    finds_stationary_step::<crate::backend_aliases::faer::Col<f64>>();
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
