//! Fault injection at the normal-equation backend boundary.
use super::*;
use crate::core::math::{DenseMatrix, LinearSolveError, MaxDiagonal};
use std::cell::Cell;
use std::rc::Rc;

#[derive(Clone)]
struct Matrix<F: Scalar> {
    value: DenseMatrix<F>,
    calls: Rc<Cell<usize>>,
    failures: usize,
}
impl<F: Scalar> GramMatrix for Matrix<F> {
    fn gram(&self) -> Self {
        Self {
            value: self.value.gram(),
            calls: self.calls.clone(),
            failures: self.failures,
        }
    }
}
impl<F: Scalar> MatTransposeVec<Vec<F>> for Matrix<F> {
    fn mat_transpose_vec(&self, v: &Vec<F>) -> Vec<F> {
        self.value.mat_transpose_vec(v)
    }
}
impl<F: Scalar> MatDiagonal<Vec<F>> for Matrix<F> {
    fn diagonal(&self) -> Vec<F> {
        self.value.diagonal()
    }
}
impl<F: Scalar> AddDiagonalVectorInPlace<Vec<F>> for Matrix<F> {
    fn add_diagonal_vector_in_place(&mut self, v: &Vec<F>) {
        self.value.add_diagonal_vector_in_place(v);
    }
}
impl<F: Scalar> MaxDiagonal<F> for Matrix<F> {
    fn max_diagonal(&self) -> F {
        self.value.max_diagonal()
    }
}
impl<F: Scalar> LinearSolveSpd<Vec<F>> for Matrix<F> {
    fn solve_spd(&self, v: &Vec<F>) -> Result<Vec<F>, LinearSolveError> {
        let count = self.calls.get();
        self.calls.set(count + 1);
        if count < self.failures {
            Err(LinearSolveError::NotPositiveDefinite)
        } else {
            self.value.solve_spd(v)
        }
    }
}
struct Linear<F: Scalar> {
    matrix: Matrix<F>,
    residuals: Rc<Cell<usize>>,
}
impl<F: Scalar> Residual for Linear<F> {
    type Param = Vec<F>;
    type Output = Vec<F>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<F>) -> Result<Vec<F>, Self::Error> {
        self.residuals.set(self.residuals.get() + 1);
        Ok(vec![x[0] - F::one()])
    }
}
impl<F: Scalar> Jacobian for Linear<F> {
    type Jacobian = Matrix<F>;
    fn jacobian(&self, _: &Vec<F>) -> Result<Matrix<F>, Self::Error> {
        Ok(self.matrix.clone())
    }
}
fn check<F: Scalar>() {
    for enabled in [false, true] {
        for failures in [2, usize::MAX] {
            for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
                let calls = Rc::new(Cell::new(0));
                let residuals = Rc::new(Cell::new(0));
                let problem = Linear {
                    matrix: Matrix {
                        value: DenseMatrix::from_row_slice(1, 1, &[F::one()]),
                        calls: calls.clone(),
                        failures,
                    },
                    residuals: residuals.clone(),
                };
                let x = vec![F::from_f64(0.1).unwrap()];
                let mut stepper = crate::Executor::from_start(
                    problem,
                    LevenbergMarquardt::<Vec<F>, Matrix<F>, F>::default()
                        .with_damping(damping)
                        .with_max_inner_attempts(4)
                        .with_trial_diagnostics(enabled),
                    x.clone(),
                )
                .max_iter(1)
                .into_stepper()
                .unwrap();
                stepper.step().unwrap();
                let count = calls.get();
                let callbacks = residuals.get();
                let observations =
                    stepper.solver().least_squares_observations();
                if !enabled {
                    assert!(observations.is_empty());
                } else {
                    let attempts: Vec<_> = observations
                        .iter()
                        .filter(|o| o.event == "subproblem")
                        .collect();
                    assert_eq!(attempts.len(), count);
                    assert!(
                        attempts.iter().all(
                            |o| o.accepted.is_none() && o.model_solves == 1
                        )
                    );
                    assert!(
                        attempts
                            .windows(2)
                            .all(|w| w[0].damping_after == w[1].damping)
                    );
                    if failures == usize::MAX {
                        assert_eq!(
                            attempts.last().unwrap().outcome,
                            Some("exhausted")
                        );
                    } else {
                        assert_eq!(
                            observations
                                .iter()
                                .find(|o| o.trial)
                                .unwrap()
                                .model_solves
                                as usize,
                            count
                        );
                    }
                }
                assert_eq!(
                    callbacks,
                    if failures == usize::MAX { 1 } else { 2 }
                );
                if failures == usize::MAX {
                    assert_eq!(stepper.state().param(), &x);
                }
                assert_eq!(calls.get(), count);
                assert_eq!(residuals.get(), callbacks);
            }
        }
    }
}
#[test]
fn normal_equation_attempts_f32() {
    check::<f32>();
}
#[test]
fn normal_equation_attempts_f64() {
    check::<f64>();
}
