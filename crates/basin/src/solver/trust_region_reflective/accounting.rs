//! Finite input can still overflow model factorization before any trial.
use super::*;
use crate::{CostFunction, DenseMatrix, Executor};
use std::cell::Cell;
use std::rc::Rc;

struct Overflow<F: Scalar> {
    lower: Vec<F>,
    upper: Vec<F>,
    residuals: Rc<Cell<u32>>,
}

impl<F: Scalar> crate::ProblemConstraints for Overflow<F> {
    type Constraints = crate::Constraints<crate::BoxBounds<Vec<F>>>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as crate::ConstraintDescription>::Borrowed<'_> {
        crate::Constraints::new().with_bounds(crate::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}
impl<F: Scalar> Residual for Overflow<F> {
    type Param = Vec<F>;
    type Output = Vec<F>;
    type Error = std::convert::Infallible;
    fn residual(&self, _: &Vec<F>) -> Result<Vec<F>, Self::Error> {
        self.residuals.set(self.residuals.get() + 1);
        Ok(vec![F::from_f64(0.5).unwrap()])
    }
}
impl<F: Scalar> Jacobian for Overflow<F> {
    type Jacobian = DenseMatrix<F>;
    fn jacobian(&self, _: &Vec<F>) -> Result<DenseMatrix<F>, Self::Error> {
        let large = F::max_value() * F::from_f64(0.9).unwrap();
        Ok(DenseMatrix::from_row_slice(1, 2, &[large, large]))
    }
}
impl<F: Scalar> CostFunction for Overflow<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = std::convert::Infallible;
    fn cost(&self, _: &Vec<F>) -> Result<F, Self::Error> {
        unreachable!()
    }
}

fn check<F: Scalar>() {
    for enabled in [false, true] {
        let residuals = Rc::new(Cell::new(0));
        let x = vec![F::from_f64(0.1).unwrap(); 2];
        let problem = Overflow {
            lower: vec![F::neg_infinity(); 2],
            upper: vec![F::infinity(); 2],
            residuals: residuals.clone(),
        };
        let mut stepper = Executor::from_start(
            problem,
            TrustRegionReflective::<F>::new().with_trial_diagnostics(enabled),
            x.clone(),
        )
        .into_stepper()
        .unwrap();
        stepper.step().unwrap();
        assert_eq!(residuals.get(), 1);
        assert_eq!(stepper.state().param(), &x);
        assert_eq!(stepper.state().cost(), F::from_f64(0.125).unwrap());
        let observations = stepper.solver().least_squares_observations();
        if enabled {
            assert!(observations.iter().any(|o| o.event == "construction"
                && o.outcome == Some("factorization_failed")));
            assert!(observations.iter().all(|o| o.accepted.is_none()
                && !o.trial
                && o.model_solves == 0));
        } else {
            assert!(observations.is_empty());
        }
        assert_eq!(residuals.get(), 1);
    }
}
#[test]
fn failed_construction_f32() {
    check::<f32>();
}
#[test]
fn failed_construction_f64() {
    check::<f64>();
}
