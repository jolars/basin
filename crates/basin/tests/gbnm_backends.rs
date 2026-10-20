#![cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]

use std::convert::Infallible;

use basin::{CostFunction, Executor, Gbnm, PointState};

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_parameter_runs() {
    use crate::backend_aliases::nalgebra::DVector;

    struct Sphere {
        lower: DVector<f64>,
        upper: DVector<f64>,
    }

    impl basin::Constraints for Sphere {
        type Set = basin::ConstraintSet<basin::BoxBounds<DVector<f64>>>;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
                Some(&self.lower),
                Some(&self.upper),
            ))
        }
    }
    impl CostFunction for Sphere {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok(x.iter().map(|value| value * value).sum())
        }
    }

    let result = Executor::new(
        Sphere {
            lower: DVector::from_element(2, -5.0),
            upper: DVector::from_element(2, 5.0),
        },
        Gbnm::new(17),
        PointState::new(DVector::from_element(2, 3.0)),
    )
    .max_iter(250)
    .run_with_solver()
    .unwrap();
    assert!(result.best_cost() < 1e-8);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_parameter_runs() {
    use crate::backend_aliases::ndarray::Array1;

    struct Sphere {
        lower: Array1<f64>,
        upper: Array1<f64>,
    }

    impl basin::Constraints for Sphere {
        type Set = basin::ConstraintSet<basin::BoxBounds<Array1<f64>>>;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
                Some(&self.lower),
                Some(&self.upper),
            ))
        }
    }
    impl CostFunction for Sphere {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok(x.iter().map(|value| value * value).sum())
        }
    }

    let result = Executor::new(
        Sphere {
            lower: Array1::from_elem(2, -5.0),
            upper: Array1::from_elem(2, 5.0),
        },
        Gbnm::new(17),
        PointState::new(Array1::from_elem(2, 3.0)),
    )
    .max_iter(250)
    .run_with_solver()
    .unwrap();
    assert!(result.best_cost() < 1e-8);
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_parameter_runs() {
    use crate::backend_aliases::faer::Col;

    struct Sphere {
        lower: Col<f64>,
        upper: Col<f64>,
    }

    impl basin::Constraints for Sphere {
        type Set = basin::ConstraintSet<basin::BoxBounds<Col<f64>>>;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
                Some(&self.lower),
                Some(&self.upper),
            ))
        }
    }
    impl CostFunction for Sphere {
        type Param = Col<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok((0..x.nrows()).map(|i| x[i] * x[i]).sum())
        }
    }

    let result = Executor::new(
        Sphere {
            lower: Col::from_fn(2, |_| -5.0),
            upper: Col::from_fn(2, |_| 5.0),
        },
        Gbnm::new(17),
        PointState::new(Col::from_fn(2, |_| 3.0)),
    )
    .max_iter(250)
    .run_with_solver()
    .unwrap();
    assert!(result.best_cost() < 1e-8);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
