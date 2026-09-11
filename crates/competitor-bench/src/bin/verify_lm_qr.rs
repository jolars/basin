//! Reproduce conditioning and calibration comparisons without GlobalSearch.
//! Run `cargo run -p competitor-bench --release --bin verify_lm_qr`.

use basin::{
    Executor, FactorizePivotedQr, Jacobian, LevenbergMarquardt, NllsState,
    RegularizedQrSolve, Residual, TerminationReason,
};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use std::{cell::Cell, convert::Infallible};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix as BasinMatrix, DVector as BasinVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix as BasinMatrix, DVector as BasinVector};

#[path = "../../investigations/cobyla-lm/support/lm_models.rs"]
mod models;
use models::{Model, check_jacobian};

struct Probe<'a> {
    model: &'a Model,
    y: &'a DVector<f64>,
    nr: &'a Cell<usize>,
    nj: &'a Cell<usize>,
    budget: usize,
    x: DVector<f64>,
}
impl Probe<'_> {
    fn residuals_at(&self, x: &DVector<f64>) -> DVector<f64> {
        assert!(
            self.nr.get() < self.budget,
            "residual callback budget exceeded"
        );
        self.nr.set(self.nr.get() + 1);
        self.model.evaluate(x).0 - self.y
    }
    fn jacobian_at(&self, x: &DVector<f64>) -> DMatrix<f64> {
        self.nj.set(self.nj.get() + 1);
        self.model.evaluate(x).1
    }
}
impl Residual for Probe<'_> {
    type Param = BasinVector<f64>;
    type Output = BasinVector<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        let r = self.residuals_at(&DVector::from_column_slice(x.as_slice()));
        Ok(BasinVector::from_column_slice(r.as_slice()))
    }
}
impl Jacobian for Probe<'_> {
    type Jacobian = BasinMatrix<f64>;
    fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, Self::Error> {
        let j = self.jacobian_at(&DVector::from_column_slice(x.as_slice()));
        Ok(BasinMatrix::from_column_slice(
            j.nrows(),
            j.ncols(),
            j.as_slice(),
        ))
    }
}
impl LeastSquaresProblem<f64, Dyn, Dyn> for Probe<'_> {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    fn params(&self) -> DVector<f64> {
        self.x.clone()
    }
    fn set_params(&mut self, x: &DVector<f64>) {
        self.x.copy_from(x);
    }
    fn residuals(&self) -> Option<DVector<f64>> {
        Some(self.residuals_at(&self.x))
    }
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        Some(self.jacobian_at(&self.x))
    }
}

fn compare(name: &str, model: Model, truth: Vec<f64>, starts: Vec<Vec<f64>>) {
    let truth = DVector::from_vec(truth);
    let (y, j) = model.evaluate(&truth);
    let sv = j.svd(false, false).singular_values;
    let condition = sv.max() / sv.min();
    check_jacobian(&model, &truth);
    for (start, x) in starts.into_iter().enumerate() {
        let x = DVector::from_vec(x);
        check_jacobian(&model, &x);
        for kind in ["cholesky", "qr", "minpack"] {
            let nr = Cell::new(0);
            let nj = Cell::new(0);
            let budget = 200 * (x.len() + 1);
            let p = Probe {
                model: &model,
                y: &y,
                nr: &nr,
                nj: &nj,
                budget,
                x: x.clone(),
            };
            let (solution, reason, converged) = if kind == "minpack" {
                let (p, report) =
                    levenberg_marquardt::LevenbergMarquardt::new()
                        .with_tol(1e-12)
                        .with_patience(200)
                        .minimize(p);
                (
                    p.x,
                    format!("{:?}", report.termination),
                    report.termination.was_successful(),
                )
            } else {
                let solver = LevenbergMarquardt::new()
                    .with_absolute_gradient_tolerance(0.)
                    .with_gradient_orthogonality_tolerance(1e-12)
                    .with_relative_model_reduction_tolerance(1e-12)
                    .with_relative_step_tolerance(1e-12);
                let state = NllsState::new(BasinVector::from_column_slice(
                    x.as_slice(),
                ));
                let out = if kind == "qr" {
                    Executor::new(p, solver.with_pivoted_qr(), state)
                        .max_iter((budget - 1) as u64)
                        .run()
                        .unwrap()
                } else {
                    Executor::new(p, solver, state)
                        .max_iter((budget - 1) as u64)
                        .run()
                        .unwrap()
                };
                assert_eq!(out.cost_evals() as usize, nr.get());
                (
                    DVector::from_column_slice(out.param().as_slice()),
                    format!("{:?}", out.reason),
                    out.reason == TerminationReason::SolverConverged,
                )
            };
            assert!(solution.iter().all(|x| x.is_finite()));
            let residual = (model.evaluate(&solution).0 - &y).norm();
            let error = (&solution - &truth).amax();
            let mut equivalent = solution.clone();
            if matches!(model, Model::Svi(_)) {
                equivalent[4] = equivalent[4].abs();
            }
            let equivalent_error = (&equivalent - &truth).amax();
            println!(
                "{name},{start},{kind},{condition:.8e},{converged},{},{},{residual:.8e},{error:.8e},{equivalent_error:.8e},\"{reason}\",\"{:?}\"",
                nr.get(),
                nj.get(),
                solution.as_slice()
            );
        }
    }
}

fn steps() {
    println!("separation,mu,cholesky_error,qr_error,qr_result");
    for eps in [1e-4_f64, 1e-6, 1e-8, 1e-10, 0.] {
        let a = DMatrix::from_row_slice(
            3,
            2,
            &[1., 1., 1., 1. + eps, 1., 1. - eps],
        );
        let b = &a * DVector::from_vec(vec![1., -1.]);
        let gram = a.transpose() * &a;
        let ba = BasinMatrix::from_column_slice(3, 2, a.as_slice());
        let qr = ba
            .factorize_pivoted_qr(&BasinVector::from_column_slice(b.as_slice()))
            .unwrap();
        let d: BasinVector<f64> = qr.column_norms_squared();
        for mu in [1e-3, 1e-10, 1e-16, 1e-20, 0.] {
            let augmented = DMatrix::from_fn(5, 2, |i, j| {
                if i < 3 {
                    a[(i, j)]
                } else if i - 3 == j {
                    (mu * d[j]).sqrt()
                } else {
                    0.
                }
            });
            let rhs = DVector::from_fn(5, |i, _| if i < 3 { b[i] } else { 0. });
            let reference =
                augmented.svd(true, true).solve(&rhs, 1e-30).unwrap();
            let mut damped = gram.clone();
            for j in 0..2 {
                damped[(j, j)] += mu * d[j];
            }
            let chol =
                damped.cholesky().map(|c| c.solve(&(a.transpose() * &b)));
            let step = qr.solve_regularized(mu, &d, None);
            let error = |x: Option<DVector<f64>>| {
                x.map(|x| (x - &reference).norm()).unwrap_or(f64::NAN)
            };
            let qr_error = error(
                step.as_ref()
                    .ok()
                    .map(|x| DVector::from_column_slice(x.as_slice())),
            );
            let result = step
                .map(|_| "ok".to_owned())
                .unwrap_or_else(|e| e.to_string());
            println!(
                "{eps:e},{mu:e},{:.8e},{qr_error:.8e},{result}",
                error(chol)
            );
        }
    }
}
fn main() {
    if std::env::args().any(|x| x == "--steps") {
        steps();
        return;
    }
    println!(
        "case,start,solver,condition,converged,residual_calls,jacobian_calls,residual_norm,parameter_error,equivalent_parameter_error,termination,parameters"
    );
    models::for_each_case(compare);
}
