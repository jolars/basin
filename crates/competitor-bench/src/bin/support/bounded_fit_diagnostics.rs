//! Compare final least-squares evidence using public results and analytic callbacks.

use std::cell::{Cell, RefCell};

use basin::{
    BoundedLevenbergMarquardt, ConvergenceEvidence, DenseMatrix,
    EvaluationKind, Executor, Jacobian, LevenbergMarquardt,
    OptimizationResultWithSolver, PointState, Residual, Solver, Termination,
    TrustRegionReflective,
    solver::least_squares_diagnostics::LeastSquaresDiagnostics,
};
use competitor_bench::{
    lm_defaults::{MAX_EVALUATIONS, MAX_ITERATIONS, Model, Quality},
    nist::{Nist, datasets},
};
use rmpfit::{
    MPConfig, MPError, MPFitter, MPMaxFev, MPPar, MPResult, MPSide, MPSuccess,
};

const TOL: f64 = 1e-10;
const STEP_FACTOR: f64 = 100.0;

struct Fit<'a> {
    model: &'a dyn Model<f64>,
    observations: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
    start: Vec<f64>,
}

impl Fit<'_> {
    fn run(&self, library: Library, iterations: u64) -> Outcome {
        let calls = Calls::default();
        let mut result = match library {
            Library::Lm => {
                let solver = LevenbergMarquardt::new()
                    .with_initial_step_bound(STEP_FACTOR)
                    .with_absolute_gradient_tolerance(None)
                    .with_gradient_orthogonality_tolerance(TOL)
                    .with_relative_model_reduction_tolerance(TOL)
                    .with_relative_trust_radius_tolerance(TOL)
                    .with_trial_diagnostics(true);
                let run = self.basin(solver, &calls, iterations, false);
                Outcome::basin(&run, &run.solver, &calls)
            }
            Library::BoundedLm | Library::ReducedBoundedLm => {
                let solver = BoundedLevenbergMarquardt::new()
                    .with_absolute_scaled_gradient_tolerance(TOL)
                    .with_trial_diagnostics(true)
                    .with_relative_cost_change_tolerance(TOL)
                    .with_relative_step_tolerance(TOL);
                let run = self.basin(
                    solver,
                    &calls,
                    iterations,
                    matches!(library, Library::ReducedBoundedLm),
                );
                Outcome::basin(&run, run.solver.solver(), &calls)
            }
            Library::Trf => {
                let solver = TrustRegionReflective::new()
                    .with_absolute_scaled_gradient_tolerance(TOL)
                    .with_trial_diagnostics(true)
                    .with_relative_cost_change_tolerance(TOL)
                    .with_relative_step_tolerance(TOL);
                let run = self.basin(solver, &calls, iterations, false);
                Outcome::basin(&run, run.solver.solver(), &calls)
            }
            Library::Rmpfit => self.rmpfit(&calls, iterations),
        };
        if let Library::ReducedBoundedLm = library {
            result.point = result
                .point
                .map(|x| lift(&self.start, &self.columns(true), &x));
        }
        result.residual_calls = calls.residuals.get();
        result.jacobian_calls = calls.jacobians.get();
        result.last_jacobian = calls.last_jacobian.into_inner();
        result
    }

    fn columns(&self, reduce_fixed: bool) -> Vec<usize> {
        (0..self.start.len())
            .filter(|&i| !reduce_fixed || self.lower[i] != self.upper[i])
            .collect()
    }

    fn basin<'a, S, E>(
        &'a self,
        solver: S,
        calls: &'a Calls,
        iterations: u64,
        reduce_fixed: bool,
    ) -> OptimizationResultWithSolver<PointState<Vec<f64>>, S>
    where
        S: Solver<BasinFit<'a>, PointState<Vec<f64>>, Error = E>,
        E: std::fmt::Debug,
    {
        let columns = self.columns(reduce_fixed);
        let select =
            |values: &[f64]| columns.iter().map(|&i| values[i]).collect();
        let start: Vec<f64> = select(&self.start);
        let lower = select(&self.lower);
        let upper = select(&self.upper);
        Executor::new(
            BasinFit {
                fit: self,
                calls,
                columns,
                lower,
                upper,
            },
            solver,
            PointState::new(start),
        )
        .max_evaluations(EvaluationKind::Residual, MAX_EVALUATIONS)
        .max_iter(iterations)
        .run_with_solver()
        .unwrap()
    }

    fn rmpfit(&self, calls: &Calls, iterations: u64) -> Outcome {
        let config = MPConfig {
            ftol: TOL,
            xtol: TOL,
            gtol: TOL,
            step_factor: STEP_FACTOR,
            // MPFIT numbers the seed as iteration one and counts acceptances.
            max_iter: iterations as usize + 1,
            // In 2.0.0 n_fev starts at one before the first callback.
            max_fev: MPMaxFev::Limited(MAX_EVALUATIONS as usize + 1),
            ..MPConfig::default()
        };
        let parameters = self
            .lower
            .iter()
            .zip(&self.upper)
            .map(|(&lo, &hi)| MPPar {
                fixed: lo == hi,
                limited_low: lo.is_finite(),
                limited_up: hi.is_finite(),
                limit_low: lo,
                limit_up: hi,
                side: MPSide::User,
                ..MPPar::default()
            })
            .collect();
        let mut fitter = MpFit {
            fit: self,
            calls,
            config,
            parameters,
        };
        let mut point = self.start.clone();
        match fitter.mpfit(&mut point) {
            Ok(status) => {
                assert_eq!(status.n_fev as u64, calls.residuals.get() + 1);
                assert_eq!(status.n_par, point.len());
                assert_eq!(status.n_func, self.observations);
                assert_eq!(
                    status.n_free,
                    self.lower
                        .iter()
                        .zip(&self.upper)
                        .filter(|(l, u)| l != u)
                        .count()
                );
                let residual = self.model.residuals(&point).unwrap();
                assert_eq!(status.resid, residual);
                Outcome {
                    point: Some(point),
                    reported_rss: Some(status.best_norm),
                    iterations: Some(status.n_iter as u64),
                    stop: mp_success(status.success).into(),
                    native_evaluations: Some(status.n_fev),
                    pegged: Some(status.n_pegged),
                    ..Outcome::default()
                }
            }
            Err(error) => Outcome {
                stop: format!("Error({})", mp_error(error)),
                ..Outcome::default()
            },
        }
    }
}

#[derive(Default)]
struct Calls {
    residuals: Cell<u64>,
    jacobians: Cell<u64>,
    last_jacobian: RefCell<Option<Captured>>,
}

struct Captured {
    point: Vec<f64>,
    columns: Vec<Option<Vec<f64>>>,
}

impl Calls {
    fn residual(&self) {
        self.residuals.set(self.residuals.get() + 1);
    }

    fn jacobian(&self, point: Vec<f64>, columns: Vec<Option<Vec<f64>>>) {
        self.jacobians.set(self.jacobians.get() + 1);
        self.last_jacobian
            .replace(Some(Captured { point, columns }));
    }
}

struct BasinFit<'a> {
    fit: &'a Fit<'a>,
    calls: &'a Calls,
    columns: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl BasinFit<'_> {
    fn lift(&self, x: &[f64]) -> Vec<f64> {
        lift(&self.fit.start, &self.columns, x)
    }
}

fn lift(start: &[f64], columns: &[usize], x: &[f64]) -> Vec<f64> {
    let mut full = start.to_vec();
    for (&i, &value) in columns.iter().zip(x) {
        full[i] = value;
    }
    full
}

impl basin::Constraints for BasinFit<'_> {
    type Set = basin::ConstraintSet<basin::BoxBounds<Vec<f64>>>;

    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}

impl Residual for BasinFit<'_> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = String;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, String> {
        self.calls.residual();
        self.fit.model.residuals(&self.lift(x))
    }
}

impl Jacobian for BasinFit<'_> {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, String> {
        let full = self.lift(x);
        let rows = self.fit.model.jacobian_rows(&full)?;
        let entries: Vec<_> = rows
            .iter()
            .flat_map(|row| self.columns.iter().map(|&i| row[i]))
            .collect();
        let jacobian =
            DenseMatrix::from_row_slice(rows.len(), x.len(), &entries);
        let mut supplied = vec![None; full.len()];
        for (j, &i) in self.columns.iter().enumerate() {
            supplied[i] =
                Some((0..rows.len()).map(|r| jacobian.get(r, j)).collect());
        }
        self.calls.jacobian(full, supplied);
        Ok(jacobian)
    }
}

struct MpFit<'a> {
    fit: &'a Fit<'a>,
    calls: &'a Calls,
    config: MPConfig,
    parameters: Vec<MPPar>,
}

impl MPFitter for MpFit<'_> {
    fn eval(&mut self, x: &[f64], residual: &mut [f64]) -> MPResult<()> {
        self.calls.residual();
        residual.copy_from_slice(
            &self.fit.model.residuals(x).map_err(|_| MPError::Eval)?,
        );
        Ok(())
    }

    fn number_of_points(&self) -> usize {
        self.fit.observations
    }

    fn config(&self) -> &MPConfig {
        &self.config
    }

    fn parameters(&self) -> &[MPPar] {
        &self.parameters
    }

    fn jacobian(
        &mut self,
        x: &[f64],
        residual: &mut [f64],
        columns: &mut [Option<Vec<f64>>],
    ) -> MPResult<()> {
        // MPFIT's analytic callback supplies both residuals and derivatives.
        self.eval(x, residual)?;
        let rows =
            self.fit.model.jacobian_rows(x).map_err(|_| MPError::Eval)?;
        for (j, column) in columns.iter_mut().enumerate() {
            if let Some(column) = column {
                for (value, row) in column.iter_mut().zip(&rows) {
                    *value = row[j];
                }
            }
        }
        self.calls.jacobian(x.to_vec(), columns.to_vec());
        Ok(())
    }
}

fn mp_success(status: MPSuccess) -> &'static str {
    match status {
        MPSuccess::Chi => "Chi",
        MPSuccess::Par => "Par",
        MPSuccess::Both => "Both",
        MPSuccess::Dir => "Dir",
        MPSuccess::MaxIter => "MaxIter",
        MPSuccess::Ftol => "Ftol",
        MPSuccess::Xtol => "Xtol",
        MPSuccess::Gtol => "Gtol",
        MPSuccess::NotDone => "NotDone",
    }
}

fn mp_error(error: MPError) -> &'static str {
    match error {
        MPError::Input => "Input",
        MPError::Nan => "Nan",
        MPError::Empty => "Empty",
        MPError::NoFree => "NoFree",
        MPError::InitBounds => "InitBounds",
        MPError::Bounds => "Bounds",
        MPError::DoF => "DoF",
        MPError::Eval => "Eval",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Library {
    Lm,
    BoundedLm,
    ReducedBoundedLm,
    Trf,
    Rmpfit,
}

#[derive(Default)]
struct Outcome {
    point: Option<Vec<f64>>,
    reported_rss: Option<f64>,
    stop: String,
    iterations: Option<u64>,
    residual_calls: u64,
    jacobian_calls: u64,
    native_evaluations: Option<usize>,
    last_jacobian: Option<Captured>,
    pegged: Option<usize>,
    last_accepted_trial: Option<bool>,
    last_model_optimality: Option<(&'static str, f64, ConvergenceEvidence)>,
}

impl Outcome {
    fn basin<S>(
        run: &OptimizationResultWithSolver<PointState<Vec<f64>>, S>,
        diagnostics: &impl LeastSquaresDiagnostics,
        calls: &Calls,
    ) -> Self {
        assert_eq!(run.counts.residual_evals, calls.residuals.get());
        assert_eq!(run.counts.jacobian_evals, calls.jacobians.get());
        let stop = match &run.report.termination {
            Termination::Converged(reason) => {
                format!(
                    "{:?}",
                    reason
                        .criteria()
                        .iter()
                        .map(|c| &c.test)
                        .collect::<Vec<_>>()
                )
            }
            reason => format!("{reason:?}"),
        };
        let batch = diagnostics.least_squares_observations();
        let last_accepted_trial = batch
            .iter()
            .rev()
            .find_map(|observation| observation.accepted);
        let last_model_optimality = batch.iter().rev().find_map(|row| {
            row.checks
                .iter()
                .find(|check| {
                    matches!(check.name, "orthogonality" | "scaled_gradient")
                })
                .and_then(|check| {
                    Some((
                        check.name,
                        2.0 * row.base_cost,
                        check.evidence.clone()?,
                    ))
                })
        });
        let current = run.state.current();
        Self {
            point: current.map(|(point, _)| point.clone()),
            reported_rss: current.map(|(_, cost)| 2.0 * cost),
            iterations: Some(run.iter()),
            stop,
            last_accepted_trial,
            last_model_optimality,
            ..Self::default()
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum JacobianAtFit {
    Full,
    FreeColumns,
    Missing,
}

struct Evidence {
    rss: f64,
    box_violation: f64,
    stationarity: f64,
    complementarity: f64,
    callback_jacobian: JacobianAtFit,
}

impl Evidence {
    fn at(
        fit: &Fit<'_>,
        x: &[f64],
        captured: Option<&Captured>,
    ) -> Option<Self> {
        let residual = fit.model.residuals(x).unwrap();
        let rows = fit.model.jacobian_rows(x).unwrap();
        assert_eq!(rows.len(), fit.observations);
        assert!(rows.iter().all(|row| row.len() == x.len()));
        if x.iter()
            .chain(&residual)
            .chain(rows.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return None;
        }
        let callback_jacobian = match captured.filter(|last| last.point == x) {
            Some(last) => {
                for (i, column) in last.columns.iter().enumerate() {
                    if let Some(column) = column {
                        assert!(
                            column
                                .iter()
                                .copied()
                                .eq(rows.iter().map(|row| row[i]))
                        );
                    } else {
                        assert_eq!(
                            fit.lower[i], fit.upper[i],
                            "missing free derivative"
                        );
                    }
                }
                if last.columns.iter().all(Option::is_some) {
                    JacobianAtFit::Full
                } else {
                    JacobianAtFit::FreeColumns
                }
            }
            None => JacobianAtFit::Missing,
        };
        let mut box_violation: f64 = 0.0;
        let mut stationarity: f64 = 0.0;
        let mut complementarity: f64 = 0.0;
        for i in 0..x.len() {
            let g: f64 =
                rows.iter().zip(&residual).map(|(row, r)| row[i] * r).sum();
            let lo = x[i] - fit.lower[i];
            let hi = fit.upper[i] - x[i];
            box_violation = box_violation.max((-lo).max(-hi));
            stationarity = stationarity.max(g.clamp(-hi, lo).abs());
            // Diagnose box multipliers from the gradient only on near-active
            // faces. At a fixed coordinate both slacks are zero.
            if lo <= 1e-6 {
                complementarity = complementarity.max(lo.abs() * g.max(0.0));
            }
            if hi <= 1e-6 {
                complementarity = complementarity.max(hi.abs() * (-g).max(0.0));
            }
        }
        Some(Self {
            rss: residual.iter().map(|v| v * v).sum(),
            box_violation,
            stationarity,
            complementarity,
            callback_jacobian,
        })
    }
}

struct AnalyticFit;

impl Model<f64> for AnalyticFit {
    fn residuals(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        Ok(vec![x[0] * x[0] - 4.0, x[1] - x[0], x[2] - 3.0])
    }

    fn jacobian_rows(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        Ok(vec![
            vec![2.0 * x[0], 0.0, 0.0],
            vec![-1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
    }
}

fn print(
    case: &str,
    library: Library,
    outcome: &Outcome,
    evidence: Option<&Evidence>,
    recovered: Option<bool>,
) {
    println!(
        "{case} {library:?} stop={} iter={:?} residual_calls={} jacobian_calls={} \
         native_evaluations={:?} reported_rss={:?} pegged={:?} last_accepted_trial={:?} recovered={recovered:?}",
        outcome.stop,
        outcome.iterations,
        outcome.residual_calls,
        outcome.jacobian_calls,
        outcome.native_evaluations,
        outcome.reported_rss,
        outcome.pegged,
        outcome.last_accepted_trial,
    );
    if let Some(e) = evidence {
        if library != Library::Rmpfit {
            let reported = outcome.reported_rss.unwrap();
            assert!((reported - e.rss).abs() <= 1e-12 * e.rss.abs().max(1.0));
        }
        println!(
            "  rebuilt: rss={:.12e} box_violation={:.3e} stationarity={:.3e} \
             box_complementarity={:.3e} last_callback_jacobian={:?}",
            e.rss,
            e.box_violation,
            e.stationarity,
            e.complementarity,
            e.callback_jacobian,
        );
    }
    if let Some((name, rss, evidence)) = &outcome.last_model_optimality {
        println!("  native model: {name} at base_rss={rss:.12e} {evidence:?}");
    }
}

fn check_analytic_cases() {
    let unbounded = Fit {
        model: &AnalyticFit,
        observations: 3,
        lower: vec![f64::NEG_INFINITY; 3],
        upper: vec![f64::INFINITY; 3],
        start: vec![1.8, 1.5, 2.0],
    };
    audit_accepted_step(&unbounded, "unbounded", Library::Lm);
    for fixed in [false, true] {
        let fit = Fit {
            model: &AnalyticFit,
            observations: 3,
            lower: vec![
                0.25,
                f64::NEG_INFINITY,
                if fixed { 2.0 } else { f64::NEG_INFINITY },
            ],
            upper: vec![
                1.0,
                f64::INFINITY,
                if fixed { 2.0 } else { f64::INFINITY },
            ],
            start: vec![0.8, 0.3, 2.0],
        };
        let case = if fixed { "active-fixed" } else { "active" };
        let mut libraries =
            vec![Library::BoundedLm, Library::Trf, Library::Rmpfit];
        if fixed {
            libraries.push(Library::ReducedBoundedLm);
        }
        for library in libraries {
            let outcome = fit.run(library, MAX_ITERATIONS);
            let x = outcome.point.as_ref().unwrap();
            let evidence =
                Evidence::at(&fit, x, outcome.last_jacobian.as_ref()).unwrap();
            let expected_rss = if fixed { 10.0 } else { 9.0 };
            let recovered = (evidence.rss - expected_rss).abs() < 1e-8
                && evidence.box_violation == 0.0
                && evidence.stationarity <= 1e-6
                && evidence.complementarity <= 1e-8
                && (x[0] - 1.0).abs() <= 1e-8
                && (x[1] - 1.0).abs() <= 1e-6
                && if fixed {
                    x[2] == 2.0
                } else {
                    (x[2] - 3.0).abs() <= 1e-6
                };
            print(case, library, &outcome, Some(&evidence), Some(recovered));
            if fixed && library == Library::BoundedLm && !recovered {
                // The direct route currently fails before accepting a point.
                assert!(outcome.stop.contains("non-finite damping"));
                continue;
            }
            assert!(recovered, "{case} {library:?}");
            if library == Library::Rmpfit {
                assert_eq!(outcome.pegged, Some(1 + usize::from(fixed)));
                let expected = if fixed {
                    JacobianAtFit::FreeColumns
                } else {
                    JacobianAtFit::Full
                };
                assert_eq!(evidence.callback_jacobian, expected);
            }
            audit_accepted_step(&fit, case, library);
        }
    }
}

fn audit_accepted_step(fit: &Fit<'_>, case: &str, library: Library) {
    let outcome = fit.run(library, 1);
    let x = outcome.point.as_ref().unwrap();
    let evidence =
        Evidence::at(fit, x, outcome.last_jacobian.as_ref()).unwrap();
    print(
        &format!("{case}-one-step"),
        library,
        &outcome,
        Some(&evidence),
        None,
    );
    let initial: f64 = fit
        .model
        .residuals(&fit.start)
        .unwrap()
        .iter()
        .map(|r| r * r)
        .sum();
    assert!(evidence.rss < initial && evidence.box_violation == 0.0);
    if library == Library::Rmpfit {
        assert_eq!(outcome.stop, "MaxIter");
        assert_eq!(outcome.iterations, Some(2));
    } else {
        assert!(outcome.stop.starts_with("Limit([Iterations"));
        assert_eq!(outcome.last_accepted_trial, Some(true));
    }
    if library == Library::Trf {
        assert_eq!(outcome.jacobian_calls, 2);
        assert_eq!(evidence.callback_jacobian, JacobianAtFit::Full);
    } else {
        assert_eq!(outcome.jacobian_calls, 1);
        assert_eq!(evidence.callback_jacobian, JacobianAtFit::Missing);
        let old = outcome.last_jacobian.as_ref().unwrap();
        let final_first_derivative = fit.model.jacobian_rows(x).unwrap()[0][0];
        assert!(
            (old.columns[0].as_ref().unwrap()[0] - final_first_derivative)
                .abs()
                > 1e-6
        );
    }
}

pub fn run() {
    println!("# Least-squares diagnostics");
    check_analytic_cases();
    let libraries = [
        Library::Lm,
        Library::BoundedLm,
        Library::Trf,
        Library::Rmpfit,
    ];
    let mut results = [(0, 0, 0); 4];
    for dataset in datasets().unwrap() {
        let model = Nist::new(dataset);
        let n = model.dataset.reference.len();
        for start in 0..2 {
            let fit = Fit {
                model: &model,
                observations: model.dataset.observations.len(),
                lower: vec![f64::NEG_INFINITY; n],
                upper: vec![f64::INFINITY; n],
                start: model.dataset.start(start),
            };
            for (library, (total, recoveries, final_jacobians)) in
                libraries.into_iter().zip(&mut results)
            {
                let outcome = fit.run(library, MAX_ITERATIONS);
                let evidence = outcome.point.as_ref().and_then(|x| {
                    Evidence::at(&fit, x, outcome.last_jacobian.as_ref())
                });
                let recovered = outcome.point.as_ref().map(|x| {
                    Quality::at(&model.dataset, x).unwrap().recovered(1e-4)
                });
                *total += 1;
                *recoveries += usize::from(recovered == Some(true));
                *final_jacobians +=
                    usize::from(evidence.as_ref().is_some_and(|e| {
                        e.callback_jacobian == JacobianAtFit::Full
                    }));
                print(
                    &format!("{}-{}", model.dataset.id, start + 1),
                    library,
                    &outcome,
                    evidence.as_ref(),
                    recovered,
                );
            }
        }
    }
    for (library, (total, recoveries, final_jacobians)) in
        libraries.into_iter().zip(results)
    {
        println!(
            "NIST {library:?} recovered={recoveries}/{total} final_callback_jacobian={final_jacobians}/{total}"
        );
    }
}

#[test]
fn analytic_fits_and_accepted_step_jacobians() {
    check_analytic_cases();
}
