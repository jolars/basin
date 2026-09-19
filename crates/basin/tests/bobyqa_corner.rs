use std::convert::Infallible;
use std::ops::{Index, IndexMut};

use basin::core::math::{Scalar, VectorLen};
use basin::{
    Bobyqa, BobyqaState, BoxConstraints, CostFunction, Executor,
    TerminationReason,
};

struct Corner<V> {
    lower: V,
    upper: V,
    coefficients: V,
}

impl<V, F> CostFunction for &Corner<V>
where
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        assert!((0..x.vec_len()).all(|i| {
            x[i].is_finite() && self.lower[i] <= x[i] && x[i] <= self.upper[i]
        }));
        Ok((0..x.vec_len()).map(|i| self.coefficients[i] * x[i]).sum())
    }
}

impl<V, F> BoxConstraints for &Corner<V>
where
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    fn lower(&self) -> &V {
        &self.lower
    }

    fn upper(&self) -> &V {
        &self.upper
    }
}

fn check_corners<V, F>(from_slice: impl Fn(&[f64]) -> V)
where
    V: Clone + VectorLen + IndexMut<usize, Output = F>,
    F: Scalar,
{
    // The first case reproduces issue #100. Reflections exercise lower and
    // mixed bounds, and the last case starts at the optimum.
    for (start, coefficients) in [
        ([0.0, 0.0, 1.0, 1.0], [-1.0; 4]),
        ([1.0, 1.0, 0.0, 0.0], [1.0; 4]),
        ([0.0, 1.0, 0.0, 1.0], [1.0, -1.0, -1.0, 1.0]),
        ([1.0; 4], [-1.0; 4]),
    ] {
        let problem = Corner {
            lower: from_slice(&[0.0; 4]),
            upper: from_slice(&[1.0; 4]),
            coefficients: from_slice(&coefficients),
        };
        let solver = Bobyqa::new()
            .with_initial_radius(F::from_f64(0.05).unwrap())
            .with_final_radius(F::from_f64(5e-5).unwrap());
        let result = Executor::new(
            &problem,
            solver,
            BobyqaState::new(from_slice(&start)),
        )
        .max_cost_evals(300)
        .run()
        .unwrap();

        let tolerance = F::from_f64(1e-6).unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        for (i, &coefficient) in coefficients.iter().enumerate() {
            let expected = if coefficient < 0.0 {
                F::one()
            } else {
                F::zero()
            };
            assert!((result.best_param()[i] - expected).abs() < tolerance);
        }
        let expected_cost = coefficients.iter().map(|c| c.min(0.0)).sum();
        assert!(
            (result.best_cost() - F::from_f64(expected_cost).unwrap()).abs()
                < tolerance
        );
        assert_eq!(
            (&problem).cost(result.best_param()).unwrap(),
            result.best_cost()
        );
    }
}

#[test]
fn linear_objective_reaches_box_corner_vec() {
    check_corners::<_, f64>(|x| x.to_vec());
}

#[test]
fn linear_objective_reaches_box_corner_f32() {
    check_corners::<_, f32>(|x| {
        x.iter().map(|&v| v as f32).collect::<Vec<_>>()
    });
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn linear_objective_reaches_box_corner_nalgebra() {
    check_corners::<_, f64>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
}

#[cfg(feature = "ndarray_all")]
#[test]
fn linear_objective_reaches_box_corner_ndarray() {
    check_corners::<_, f64>(|x| {
        backend_aliases::ndarray::Array1::from(x.to_vec())
    });
}

#[cfg(feature = "faer_all")]
#[test]
fn linear_objective_reaches_box_corner_faer() {
    check_corners::<_, f64>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
