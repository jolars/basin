//! Published exponential-fit comparison with matched exposed LM settings.
//!
//! Both solvers use scaled trust regions and pivoted QR. Basin's damping
//! search, acceptance rules, and numerical safeguards still differ from MINPACK.

use std::{cell::RefCell, convert::Infallible, rc::Rc};

use basin::problems::ExponentialFit;
use basin::{
    EvaluationKind, Executor, Jacobian, LevenbergMarquardt, LmDamping,
    PointState, Residual,
};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};

use crate::convergence::{Log, Run, Summary};
use crate::{LM_DEFAULT_TOL, LmExponentialFit};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix as BasinMatrix, DVector as BasinVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix as BasinMatrix, DVector as BasinVector};

const RESIDUAL_BUDGET: usize = 300;
const ITERATION_BUDGET: u64 = 200;
const STEP_BOUND: f64 = 100.0;

#[derive(Clone, Copy)]
pub enum Library {
    Basin,
    Reference,
}

impl Library {
    pub fn name(self) -> &'static str {
        match self {
            Self::Basin => "basin",
            Self::Reference => "levenberg-marquardt",
        }
    }
}

struct BasinFit {
    model: ExponentialFit<BasinVector<f64>>,
    log: Rc<RefCell<Log>>,
    residual_calls: std::cell::Cell<usize>,
    jacobian_calls: std::cell::Cell<usize>,
}

impl basin::Constraints for &BasinFit {
    type Set = basin::NoConstraints;
    fn constraints(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}

impl Residual for &BasinFit {
    type Param = BasinVector<f64>;
    type Output = BasinVector<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Infallible> {
        self.residual_calls.set(self.residual_calls.get() + 1);
        let residual = self.model.residual(x)?;
        self.log.borrow_mut().record(
            0.5 * residual.iter().map(|r| r * r).sum::<f64>(),
            x.as_slice(),
        );
        Ok(residual)
    }
}

impl Jacobian for &BasinFit {
    type Jacobian = BasinMatrix<f64>;
    fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, Infallible> {
        self.jacobian_calls.set(self.jacobian_calls.get() + 1);
        self.model.jacobian(x)
    }
}

struct ReferenceFit {
    model: LmExponentialFit,
    log: Rc<RefCell<Log>>,
    residual_calls: std::cell::Cell<usize>,
    jacobian_calls: std::cell::Cell<usize>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for ReferenceFit {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn params(&self) -> DVector<f64> {
        self.model.params()
    }
    fn set_params(&mut self, x: &DVector<f64>) {
        self.model.set_params(x);
    }
    fn residuals(&self) -> Option<DVector<f64>> {
        self.residual_calls.set(self.residual_calls.get() + 1);
        let residual = self.model.residuals()?;
        self.log.borrow_mut().record(
            0.5 * residual.iter().map(|r| r * r).sum::<f64>(),
            self.model.param_slice(),
        );
        Some(residual)
    }
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        self.jacobian_calls.set(self.jacobian_calls.get() + 1);
        self.model.jacobian()
    }
}

pub fn run(library: Library) -> Run {
    // Fixture preparation stays outside both clocks; construction and solver
    // initialization stay inside. Both adapters retain every residual trial.
    let basin_model = ExponentialFit::sampled(1e5, -1.0, 10, 0.4);
    let reference_model =
        LmExponentialFit::sampled(1e5, -1.0, 10, 0.4, 5e4, -0.3);
    let start = BasinVector::from_vec(vec![5e4, -0.3]);
    let log = Rc::new(RefCell::new(Log::new()));
    let (point, cost, residual_calls, jacobian_calls, status) = match library {
        Library::Basin => {
            let model = BasinFit {
                model: basin_model,
                log: Rc::clone(&log),
                residual_calls: 0.into(),
                jacobian_calls: 0.into(),
            };
            let solver = LevenbergMarquardt::new()
                .with_damping(LmDamping::TrustRegion)
                .with_initial_step_bound(STEP_BOUND)
                .with_absolute_gradient_tolerance(None)
                .with_gradient_orthogonality_tolerance(LM_DEFAULT_TOL)
                .with_relative_model_reduction_tolerance(LM_DEFAULT_TOL)
                .with_relative_step_tolerance(None)
                .with_relative_trust_radius_tolerance(LM_DEFAULT_TOL)
                .with_pivoted_qr();
            let result = Executor::new(&model, solver, PointState::new(start))
                .max_iter(ITERATION_BUDGET)
                .max_evaluations(
                    EvaluationKind::Residual,
                    RESIDUAL_BUDGET as u64,
                )
                .run()
                .unwrap();
            (
                result.param().as_slice().to_vec(),
                result.cost(),
                model.residual_calls.get(),
                model.jacobian_calls.get(),
                format!("{:?}", result.report.code()),
            )
        }
        Library::Reference => {
            let model = ReferenceFit {
                model: reference_model,
                log: Rc::clone(&log),
                residual_calls: 0.into(),
                jacobian_calls: 0.into(),
            };
            let (model, report) =
                levenberg_marquardt::LevenbergMarquardt::new()
                    .with_ftol(LM_DEFAULT_TOL)
                    .with_xtol(LM_DEFAULT_TOL)
                    .with_gtol(LM_DEFAULT_TOL)
                    .with_stepbound(STEP_BOUND)
                    .with_scale_diag(true)
                    .with_patience(RESIDUAL_BUDGET / 3)
                    .minimize(model);
            (
                model.params().as_slice().to_vec(),
                report.objective_function,
                model.residual_calls.get(),
                model.jacobian_calls.get(),
                format!("{:?}", report.termination),
            )
        }
    };
    let elapsed = log.borrow().elapsed();
    let check_model =
        ExponentialFit::<DVector<f64>>::sampled(1e5, -1.0, 10, 0.4);
    let residual = check_model
        .residual(&DVector::from_vec(point.clone()))
        .unwrap();
    let check_cost = 0.5 * residual.iter().map(|r| r * r).sum::<f64>();
    let verified = point.iter().all(|x| x.is_finite())
        && cost.is_finite()
        && (cost - check_cost).abs() <= 1e-12 * check_cost.abs().max(1.0)
        && check_cost <= 1e-8
        && (point[0] - 1e5).abs() <= 1e-3
        && (point[1] + 1.0).abs() <= 1e-8;
    Rc::try_unwrap(log).ok().unwrap().into_inner().finish(
        elapsed,
        Summary {
            point,
            cost,
            violation: 0.0,
            objective_calls: 0,
            residual_calls,
            jacobian_calls,
            constraint_calls: 0,
            status,
            verified,
        },
        None,
        |_| 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_agree_on_residuals_and_analytic_jacobians() {
        let basin =
            ExponentialFit::<BasinVector<f64>>::sampled(1e5, -1.0, 10, 0.4);
        let mut reference =
            LmExponentialFit::sampled(1e5, -1.0, 10, 0.4, 5e4, -0.3);
        for values in [[5e4, -0.3], [9e4, -0.8], [1e5, -1.0]] {
            let x = BasinVector::from_vec(values.to_vec());
            reference.set_params(&DVector::from_vec(values.to_vec()));
            assert_eq!(
                basin.residual(&x).unwrap().as_slice(),
                reference.residuals().unwrap().as_slice()
            );
            assert_eq!(
                basin.jacobian(&x).unwrap().as_slice(),
                reference.jacobian().unwrap().as_slice()
            );
        }
    }

    #[test]
    fn both_fits_recover_the_solution_with_counted_residual_trials() {
        for library in [Library::Basin, Library::Reference] {
            let result = run(library);
            assert!(
                result.summary.verified,
                "{}: {:?}",
                library.name(),
                result.summary
            );
            assert!(result.summary.residual_calls <= RESIDUAL_BUDGET);
            assert_eq!(result.points.len(), result.summary.residual_calls + 1);
            assert!(result.summary.jacobian_calls > 0);
            assert!(
                result
                    .points
                    .windows(2)
                    .all(|w| w[0].0 <= w[1].0 && w[0].1 >= w[1].1)
            );
        }
    }
}
