//! Diagnose LM stopping without changing the production iteration or callbacks.
//! Run `cargo run -p competitor-bench --release --bin verify_lm_stopping`.
//!
//! Trial steps are reconstructed from callback coordinates, so rounding can make
//! them differ from LM's internal step. Model/SVD diagnostics are offline work,
//! excluded from evaluation counts. They neither infer damping nor identify the
//! native stopping clause. Output retains the first ten boundaries, the first
//! unchanged trial, the last boundary, and a separate final-state row per solve.
//! Relative trial steps use the returned iterate as denominator; SVD steps use
//! the base iterate. Zero denominators give zero for a zero numerator and infinity
//! otherwise. Scaled norms use independently tracked monotone column norms.

use basin::{
    Executor, Jacobian, LevenbergMarquardt, LmDamping, NllsState, Problem,
    Residual, Solver, State, TerminationReason,
};
use nalgebra::{DMatrix, DVector};
use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix as BasinMatrix, DVector as BasinVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix as BasinMatrix, DVector as BasinVector};

#[path = "../../investigations/cobyla-lm/support/lm_models.rs"]
mod models;
use models::Model;

struct Case {
    name: String,
    start: usize,
    model: Model,
    truth: DVector<f64>,
    initial: DVector<f64>,
}

#[derive(Default)]
struct Calls {
    residual: Cell<usize>,
    jacobian: Cell<usize>,
    trial: RefCell<Option<DVector<f64>>>,
}

struct Probe<'a> {
    model: &'a Model,
    observations: &'a DVector<f64>,
    calls: &'a Calls,
    budget: usize,
}

impl Residual for Probe<'_> {
    type Param = BasinVector<f64>;
    type Output = BasinVector<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        assert!(
            self.calls.residual.get() < self.budget,
            "residual budget exceeded"
        );
        self.calls.residual.set(self.calls.residual.get() + 1);
        let x = DVector::from_column_slice(x.as_slice());
        let residual = self.model.evaluate(&x).0 - self.observations;
        self.calls.trial.replace(Some(x));
        Ok(BasinVector::from_column_slice(residual.as_slice()))
    }
}

impl Jacobian for Probe<'_> {
    type Jacobian = BasinMatrix<f64>;

    fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, Self::Error> {
        self.calls.jacobian.set(self.calls.jacobian.get() + 1);
        let j = self
            .model
            .evaluate(&DVector::from_column_slice(x.as_slice()))
            .1;
        Ok(BasinMatrix::from_column_slice(
            j.nrows(),
            j.ncols(),
            j.as_slice(),
        ))
    }
}

fn relative(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0. {
        if numerator == 0. { 0. } else { f64::INFINITY }
    } else {
        numerator / denominator
    }
}

fn scaled_norm(x: &DVector<f64>, diagonal: &DVector<f64>) -> f64 {
    x.component_mul(&diagonal.map(f64::sqrt)).norm()
}

fn column_squares(j: &DMatrix<f64>) -> DVector<f64> {
    DVector::from_iterator(
        j.ncols(),
        j.column_iter().map(|column| column.norm_squared()),
    )
}

struct Diagnostics {
    residual: f64,
    relative_residual: f64,
    gradient: f64,
    cosine: f64,
    parameter_error: f64,
    rank: usize,
    potential: f64,
    svd_step: f64,
    svd_scaled_step: f64,
}

impl Diagnostics {
    fn at(
        case: &Case,
        y: &DVector<f64>,
        x: &DVector<f64>,
        d: &DVector<f64>,
    ) -> Self {
        let (prediction, j) = case.model.evaluate(x);
        let r = prediction - y;
        let residual = r.norm();
        let gradient = (j.transpose() * &r).amax();
        let normalized = if residual > 0. {
            &r / residual
        } else {
            r.clone()
        };
        let cosine = j
            .column_iter()
            .filter_map(|column| {
                let norm = column.norm();
                (norm > 0.).then(|| (column / norm).dot(&normalized).abs())
            })
            .fold(0_f64, f64::max);
        let mut equivalent = x.clone();
        if matches!(case.model, Model::Svi(_)) {
            equivalent[4] = equivalent[4].abs();
        }
        let cutoff_factor = f64::EPSILON * j.nrows().max(j.ncols()) as f64;
        let svd = j.svd(true, true);
        let cutoff = cutoff_factor * svd.singular_values.max();
        let u = svd.u.as_ref().unwrap();
        let vt = svd.v_t.as_ref().unwrap();
        let mut step = DVector::zeros(x.len());
        let mut rank = 0;
        let mut potential = 0.;
        for (index, &sigma) in svd.singular_values.iter().enumerate() {
            if sigma > cutoff {
                rank += 1;
                let coefficient = u.column(index).dot(&r);
                step += vt.row(index).transpose() * (-coefficient / sigma);
                potential += relative(coefficient, residual).powi(2);
            }
        }
        Self {
            residual,
            relative_residual: relative(residual, y.norm()),
            gradient,
            cosine,
            parameter_error: (&equivalent - &case.truth).amax(),
            rank,
            potential,
            svd_step: relative(step.norm(), x.norm()),
            svd_scaled_step: relative(scaled_norm(&step, d), scaled_norm(x, d)),
        }
    }
}

struct Row {
    boundary: usize,
    residual_calls: usize,
    jacobian_calls: usize,
    trial: bool,
    accepted: bool,
    unchanged: bool,
    diagnostics: Diagnostics,
    actual: Option<f64>,
    predicted: Option<f64>,
    step: Option<f64>,
    scaled_step: Option<f64>,
    reason: Option<TerminationReason>,
}

impl Row {
    fn print(&self, case: &Case, route: &str, profile: &str, kind: &str) {
        let d = &self.diagnostics;
        let number = |value: Option<f64>| {
            value.map(|x| format!("{x:.16e}")).unwrap_or_default()
        };
        println!(
            "{},{},{route},{profile},{kind},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:?}",
            case.name,
            case.start,
            self.boundary,
            self.residual_calls,
            self.jacobian_calls,
            self.trial,
            self.accepted,
            self.unchanged,
            d.residual,
            d.relative_residual,
            d.gradient,
            d.cosine,
            d.parameter_error,
            number(self.actual),
            number(self.predicted),
            number(self.step),
            number(self.scaled_step),
            d.rank,
            d.potential,
            d.svd_step,
            d.svd_scaled_step,
            self.reason,
        );
    }
}

struct Recorder<'a, S> {
    inner: S,
    case: &'a Case,
    observations: &'a DVector<f64>,
    calls: &'a Calls,
    diagonal: &'a RefCell<DVector<f64>>,
    rows: &'a RefCell<Vec<Row>>,
}

impl<'a, S> Solver<Probe<'a>, NllsState<BasinVector<f64>>> for Recorder<'a, S>
where
    S: Solver<Probe<'a>, NllsState<BasinVector<f64>>, Error = Infallible>,
{
    type Error = Infallible;

    fn init(
        &mut self,
        problem: &mut Problem<Probe<'a>>,
        state: NllsState<BasinVector<f64>>,
    ) -> Result<NllsState<BasinVector<f64>>, Infallible> {
        self.inner.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Probe<'a>>,
        state: NllsState<BasinVector<f64>>,
    ) -> Result<
        (NllsState<BasinVector<f64>>, Option<TerminationReason>),
        Infallible,
    > {
        let base = DVector::from_column_slice(state.param().as_slice());
        let (prediction, j) = self.case.model.evaluate(&base);
        let r = prediction - self.observations;
        let diagonal = self
            .diagonal
            .borrow()
            .zip_map(&column_squares(&j), f64::max);
        self.diagonal.replace(diagonal);
        let diagnostics = Diagnostics::at(
            self.case,
            self.observations,
            &base,
            &self.diagonal.borrow(),
        );
        let base_cost = state.cost();
        let before = self.calls.residual.get();
        self.calls.trial.take();
        let (state, reason) = self.inner.next_iter(problem, state)?;
        let calls = self.calls.residual.get() - before;
        assert!(calls <= 1, "LM unexpectedly made multiple trial callbacks");
        let trial = self.calls.trial.take();
        assert_eq!(calls == 1, trial.is_some());
        let current = DVector::from_column_slice(state.param().as_slice());
        let mut row = Row {
            boundary: self.rows.borrow().len() + 1,
            residual_calls: self.calls.residual.get(),
            jacobian_calls: self.calls.jacobian.get(),
            trial: trial.is_some(),
            accepted: state.cost() < base_cost,
            unchanged: false,
            diagnostics,
            actual: None,
            predicted: None,
            step: None,
            scaled_step: None,
            reason,
        };
        if let Some(trial) = trial {
            // Coordinate subtraction reflects observable movement, including rounding.
            let h = &trial - &base;
            row.unchanged = trial == base;
            let trial_residual =
                self.case.model.evaluate(&trial).0 - self.observations;
            row.actual = Some(relative(
                base_cost - 0.5 * trial_residual.norm_squared(),
                base_cost,
            ));
            let rnorm = r.norm();
            let linear_delta = &j * &h;
            row.predicted = Some(if rnorm > 0. {
                let normalized_delta = linear_delta / rnorm;
                -2. * (&r / rnorm).dot(&normalized_delta)
                    - normalized_delta.norm_squared()
            } else {
                0.
            });
            row.step = Some(relative(h.norm(), current.norm()));
            row.scaled_step = Some(relative(
                scaled_norm(&h, &self.diagonal.borrow()),
                scaled_norm(&current, &self.diagonal.borrow()),
            ));
        }
        self.rows.borrow_mut().push(row);
        Ok((state, reason))
    }

    fn reset_convergence(&mut self) {
        self.inner.reset_convergence();
    }

    fn check_convergence(
        &mut self,
        problem: &Problem<Probe<'a>>,
        state: &NllsState<BasinVector<f64>>,
    ) -> Option<TerminationReason> {
        self.inner.check_convergence(problem, state)
    }

    fn terminate(
        &self,
        state: &NllsState<BasinVector<f64>>,
    ) -> Option<TerminationReason> {
        self.inner.terminate(state)
    }
}

fn run<S>(case: &Case, route: &str, profile: &str, solver: S)
where
    S: for<'a> Solver<
            Probe<'a>,
            NllsState<BasinVector<f64>>,
            Error = Infallible,
        >,
{
    let observations = case.model.evaluate(&case.truth).0;
    let calls = Calls::default();
    let rows = RefCell::new(Vec::new());
    let budget = 200 * (case.initial.len() + 1);
    let diagonal = column_squares(&case.model.evaluate(&case.initial).1)
        .map(|x| if x == 0. { 1. } else { x });
    let diagonal = RefCell::new(diagonal);
    let recorder = Recorder {
        inner: solver,
        case,
        observations: &observations,
        calls: &calls,
        diagonal: &diagonal,
        rows: &rows,
    };
    let result = Executor::new(
        Probe {
            model: &case.model,
            observations: &observations,
            calls: &calls,
            budget,
        },
        recorder,
        NllsState::new(BasinVector::from_column_slice(case.initial.as_slice())),
    )
    .max_iter((budget - 1) as u64)
    .run()
    .unwrap();
    assert_eq!(result.cost_evals() as usize, calls.residual.get());
    assert_eq!(result.state.jacobian_evals() as usize, calls.jacobian.get());
    assert!(calls.residual.get() <= budget);
    assert!(result.param().iter().all(|x| x.is_finite()));
    let rows = rows.borrow();
    let first_unchanged =
        rows.iter().position(|row| row.trial && row.unchanged);
    for (index, row) in rows.iter().enumerate() {
        if index < 10
            || index + 1 == rows.len()
            || Some(index) == first_unchanged
        {
            row.print(
                case,
                route,
                profile,
                if Some(index) == first_unchanged {
                    "first-unchanged"
                } else {
                    "boundary"
                },
            );
        }
    }
    let x = DVector::from_column_slice(result.param().as_slice());
    let diagnostics =
        Diagnostics::at(case, &observations, &x, &diagonal.borrow());
    if case.name == "tiny-orthogonality" {
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert!(calls.residual.get() > 1);
        assert!((&x - &case.truth).amax() < 1e-12);
        assert!(diagnostics.relative_residual < 1e-12);
    }
    Row {
        boundary: rows.len(),
        residual_calls: calls.residual.get(),
        jacobian_calls: calls.jacobian.get(),
        trial: false,
        accepted: false,
        unchanged: false,
        diagnostics,
        actual: None,
        predicted: None,
        step: None,
        scaled_step: None,
        reason: Some(result.reason),
    }
    .print(case, route, profile, "final");
}

fn compare(case: &Case, route: &str, profile: &str) {
    let mut solver = LevenbergMarquardt::new()
        .with_absolute_gradient_tolerance(0.)
        .with_gradient_orthogonality_tolerance(1e-12)
        .with_relative_model_reduction_tolerance(1e-12)
        .with_relative_step_tolerance(1e-12);
    if route.starts_with("trust-") {
        solver = solver.with_damping(LmDamping::TrustRegion);
    }
    if profile == "gradient-only" {
        solver = solver
            .with_relative_model_reduction_tolerance(None)
            .with_relative_step_tolerance(None);
    }
    if route.ends_with("qr") {
        run(case, route, profile, solver.with_pivoted_qr());
    } else {
        run(case, route, profile, solver);
    }
}

fn main() {
    println!(
        "case,start,solver,profile,row,boundary,residual_calls,jacobian_calls,trial_evaluated,accepted,trial_unchanged,base_residual_norm,base_relative_residual,base_gradient_inf,base_orthogonality,base_equivalent_parameter_error,actual_relative_reduction,reconstructed_linear_relative_reduction,reconstructed_relative_step,reconstructed_scaled_relative_step,svd_rank,svd_relative_model_potential,svd_relative_step,svd_scaled_relative_step,reason"
    );
    models::for_each_case(|name, model, truth, starts| {
        if name != "collinear-1e-8" && name != "svi-width-0.01" {
            return;
        }
        let truth = DVector::from_vec(truth);
        models::check_jacobian(&model, &truth);
        for (start, initial) in starts.into_iter().enumerate() {
            if name == "svi-width-0.01" && start != 2 {
                continue;
            }
            let case = Case {
                name: name.to_owned(),
                start,
                model: model.clone(),
                truth: truth.clone(),
                initial: DVector::from_vec(initial),
            };
            models::check_jacobian(&case.model, &case.initial);
            let routes: &[&str] = if name == "collinear-1e-8" {
                &[
                    "nielsen-cholesky",
                    "nielsen-qr",
                    "trust-cholesky",
                    "trust-qr",
                ]
            } else {
                &["trust-qr"]
            };
            for &route in routes {
                for profile in ["relative", "gradient-only"] {
                    compare(&case, route, profile);
                }
            }
        }
    });
    compare(
        &Case {
            name: "tiny-orthogonality".to_owned(),
            start: 0,
            model: Model::Linear(DMatrix::from_element(1, 1, 1e-100)),
            truth: DVector::from_element(1, -1.),
            initial: DVector::zeros(1),
        },
        "nielsen-qr",
        "gradient-only",
    );
    for scale in [1., 1e13] {
        // Changing the second coordinate's units leaves the physical fit identical.
        compare(
            &Case {
                name: format!("coordinate-scale-{scale:e}"),
                start: 0,
                model: Model::Linear(DMatrix::from_diagonal(
                    &DVector::from_vec(vec![1., 1. / scale]),
                )),
                truth: DVector::from_vec(vec![1., scale]),
                initial: DVector::from_vec(vec![0., scale]),
            },
            "nielsen-qr",
            "relative",
        );
    }
}
