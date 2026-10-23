//! Published multi-curve exponential-fit comparison with matched exposed LM settings.
//!
//! Both solvers use scaled trust regions and pivoted QR. Basin's damping
//! search, acceptance rules, and numerical safeguards still differ from MINPACK.

use std::{cell::RefCell, convert::Infallible, rc::Rc};

use basin::{
    EvaluationKind, Executor, Jacobian, LevenbergMarquardt, LmDamping,
    PointState, Residual,
};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};

use crate::LM_DEFAULT_TOL;
use crate::convergence::{Log, Run, Summary};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix as BasinMatrix, DVector as BasinVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix as BasinMatrix, DVector as BasinVector};

/// Sixteen exponential response curves, each with an amplitude and decay rate.
pub const CURVES: usize = 16;
pub const OBSERVATIONS: usize = 256;

struct Fit {
    point: Vec<f64>,
}

impl Fit {
    fn new() -> Self {
        Self {
            point: (0..CURVES).flat_map(|_| [5e4, -0.3]).collect(),
        }
    }

    fn residuals_at(x: &[f64]) -> Vec<f64> {
        (0..CURVES * OBSERVATIONS)
            .map(|i| {
                let curve = i / OBSERVATIONS;
                let t =
                    4.0 * (i % OBSERVATIONS) as f64 / (OBSERVATIONS - 1) as f64;
                let amplitude = 1e5 + 1000.0 * curve as f64;
                let decay = -1.0 - 0.02 * curve as f64;
                x[2 * curve] * (x[2 * curve + 1] * t).exp()
                    - amplitude * (decay * t).exp()
            })
            .collect()
    }

    fn jacobian_at(x: &[f64]) -> Vec<f64> {
        let rows = CURVES * OBSERVATIONS;
        let mut j = vec![0.0; rows * 2 * CURVES];
        for i in 0..rows {
            let curve = i / OBSERVATIONS;
            let t = 4.0 * (i % OBSERVATIONS) as f64 / (OBSERVATIONS - 1) as f64;
            let e = (x[2 * curve + 1] * t).exp();
            j[2 * curve * rows + i] = e;
            j[(2 * curve + 1) * rows + i] = x[2 * curve] * t * e;
        }
        j
    }
}

const RESIDUAL_BUDGET: usize = 330;
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
        let residual = BasinVector::from_vec(Fit::residuals_at(x.as_slice()));
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
        Ok(BasinMatrix::from_vec(
            CURVES * OBSERVATIONS,
            2 * CURVES,
            Fit::jacobian_at(x.as_slice()),
        ))
    }
}

struct ReferenceFit {
    model: Fit,
    log: Rc<RefCell<Log>>,
    residual_calls: std::cell::Cell<usize>,
    jacobian_calls: std::cell::Cell<usize>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for ReferenceFit {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn params(&self) -> DVector<f64> {
        DVector::from_vec(self.model.point.clone())
    }
    fn set_params(&mut self, x: &DVector<f64>) {
        self.model.point = x.as_slice().to_vec();
    }
    fn residuals(&self) -> Option<DVector<f64>> {
        self.residual_calls.set(self.residual_calls.get() + 1);
        let residual = DVector::from_vec(Fit::residuals_at(&self.model.point));
        self.log.borrow_mut().record(
            0.5 * residual.iter().map(|r| r * r).sum::<f64>(),
            &self.model.point,
        );
        Some(residual)
    }
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        self.jacobian_calls.set(self.jacobian_calls.get() + 1);
        Some(DMatrix::from_vec(
            CURVES * OBSERVATIONS,
            2 * CURVES,
            Fit::jacobian_at(&self.model.point),
        ))
    }
}

pub fn run(library: Library) -> Run {
    // Fixture preparation stays outside both clocks; construction and solver
    // initialization stay inside. Both adapters retain every residual trial.
    let basin_model = Fit::new();
    let reference_model = Fit::new();
    let start = BasinVector::from_vec(basin_model.point.clone());
    let log = Rc::new(RefCell::new(Log::new()));
    let (point, cost, residual_calls, jacobian_calls, status) = match library {
        Library::Basin => {
            let model = BasinFit {
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
                    .with_patience(RESIDUAL_BUDGET / (2 * CURVES + 1))
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
    let residual = Fit::residuals_at(&point);
    let check_cost = 0.5 * residual.iter().map(|r| r * r).sum::<f64>();
    let verified = point.iter().all(|x| x.is_finite())
        && cost.is_finite()
        && (cost - check_cost).abs() <= 1e-12 * check_cost.abs().max(1.0)
        && check_cost <= 1e-8
        && point.chunks_exact(2).enumerate().all(|(i, x)| {
            (x[0] - (1e5 + 1000.0 * i as f64)).abs() <= 1e-3
                && (x[1] + 1.0 + 0.02 * i as f64).abs() <= 1e-8
        });
    Rc::try_unwrap(log).ok().unwrap().into_inner().finish(
        elapsed,
        Summary {
            point,
            cost,
            violation: 0.0,
            objective_calls: 0,
            residual_calls,
            jacobian_calls,
            gradient_calls: 0,
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
    fn analytic_jacobian_matches_central_differences() {
        let x = Fit::new().point;
        let j = Fit::jacobian_at(&x);
        let rows = CURVES * OBSERVATIONS;
        for col in 0..x.len() {
            let h = 1e-5 * x[col].abs().max(1.0);
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[col] += h;
            minus[col] -= h;
            for (row, (a, b)) in Fit::residuals_at(&plus)
                .iter()
                .zip(Fit::residuals_at(&minus))
                .enumerate()
            {
                let numeric = (a - b) / (2.0 * h);
                let analytic = j[col * rows + row];
                assert!(
                    (numeric - analytic).abs() < 1e-5 * analytic.abs().max(1.0)
                );
            }
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
