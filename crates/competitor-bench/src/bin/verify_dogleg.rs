//! Compare least-squares solver effectiveness from both NIST StRD starts.
//!
//! Run `cargo run --release -p competitor-bench --bin verify_dogleg`.
//! The default build uses f64 and nalgebra 0.34 throughout. Basin uses
//! nalgebra 0.35 when the benchmark's `basin-latest` feature is enabled.
//! Both dogleg routes use the Gauss–Newton model. Basin solves the Gram
//! system with Cholesky and falls back to a Cauchy step if that solve fails.
//! The reference uses an SVD of the Jacobian. The unscaled reference shares
//! Basin's initial radius and Euclidean trust region. Each solver keeps its
//! own acceptance, radius update, and stopping rules.
//!
//! Recovery requires relative errors within 1e-4 for the independently
//! recomputed RSS and every fitted parameter. Verification calls are excluded
//! from the reported solve counts.

use std::{
    cell::{Cell, RefCell},
    io::{self, Write},
};

use basin::{
    Dogleg, EvaluationKind, Executor, Jacobian, LevenbergMarquardt,
    OptimizationResult, Residual, State, TrustRegion,
};
use competitor_bench::{
    lm_defaults::{MAX_EVALUATIONS, Quality},
    nist::{Nist, datasets},
};
use dogleg::{LeastSquaresProblem, dogleg::InitialTrusRegionRadius};
use nalgebra::{DMatrix, DVector};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::DVector as BasinVector;
#[cfg(feature = "basin-latest")]
use nalgebra_latest::DVector as BasinVector;

const RADIUS: f64 = 1.0;

#[cfg(not(feature = "basin-latest"))]
use nalgebra::DMatrix as BasinMatrix;
#[cfg(feature = "basin-latest")]
use nalgebra_latest::DMatrix as BasinMatrix;

#[derive(Default)]
struct Trace(RefCell<Vec<f64>>);

impl Trace {
    fn record(&self, residual: &[f64]) {
        self.0
            .borrow_mut()
            .push(residual.iter().map(|r| r * r).sum());
    }

    fn points(&self, reference_rss: f64) -> Vec<[f64; 2]> {
        let mut best = f64::INFINITY;
        let mut points = Vec::new();
        for (i, &rss) in self.0.borrow().iter().enumerate() {
            let gap = ((rss - reference_rss) / reference_rss).max(1e-12);
            if rss.is_finite() && gap < best {
                best = gap;
                points.push([(i + 1) as f64, best]);
            }
        }
        let end = self.0.borrow().len() as f64;
        if let Some(last) = points.last()
            && last[0] < end
        {
            points.push([end, best]);
        }
        points
    }
}

struct TracedNist<'a> {
    model: &'a Nist,
    trace: Option<&'a Trace>,
}

impl Residual for TracedNist<'_> {
    type Param = BasinVector<f64>;
    type Output = BasinVector<f64>;
    type Error = String;

    fn residual(&self, x: &Self::Param) -> Result<Self::Output, String> {
        let residual = self.model.residual(x)?;
        if let Some(trace) = self.trace {
            trace.record(residual.as_slice());
        }
        Ok(residual)
    }
}

impl Jacobian for TracedNist<'_> {
    type Jacobian = BasinMatrix<f64>;

    fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, String> {
        self.model.jacobian(x)
    }
}

struct Reference<'a> {
    model: &'a Nist,
    point: DVector<f64>,
    trace: Option<&'a Trace>,
    residual_calls: Cell<u64>,
    jacobian_calls: Cell<u64>,
}

impl LeastSquaresProblem<f64> for Reference<'_> {
    type Parameters = DVector<f64>;
    type Residuals = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    fn params(&self) -> Self::Parameters {
        self.point.clone()
    }

    fn set_params(&mut self, point: Self::Parameters) {
        self.point = point;
    }

    fn residuals(&self) -> Option<Self::Residuals> {
        self.residual_calls.set(self.residual_calls.get() + 1);
        let residual = self.model.residuals(self.point.as_slice()).ok()?;
        if let Some(trace) = self.trace {
            trace.record(&residual);
        }
        Some(DVector::from_vec(residual))
    }

    fn jacobian(&self) -> Option<Self::Jacobian> {
        self.jacobian_calls.set(self.jacobian_calls.get() + 1);
        let rows = self.model.jacobian_rows(self.point.as_slice()).ok()?;
        Some(DMatrix::from_fn(rows.len(), self.point.len(), |i, j| {
            rows[i][j]
        }))
    }
}

#[derive(Default)]
struct Outcome {
    point: Option<Vec<f64>>,
    rss: Option<f64>,
    iterations: Option<u64>,
    residual_calls: Option<u64>,
    jacobian_calls: Option<u64>,
    termination: String,
}

fn basin_outcome<S, E>(result: Result<OptimizationResult<S>, E>) -> Outcome
where
    S: State<Param = BasinVector<f64>, Float = f64>,
    E: std::fmt::Debug,
{
    match result {
        Ok(run) => Outcome {
            point: Some(run.param().as_slice().to_vec()),
            rss: Some(2.0 * run.cost()),
            iterations: Some(run.iter()),
            residual_calls: Some(run.counts.residual_evals),
            jacobian_calls: Some(run.counts.jacobian_evals),
            termination: format!("{:?}", run.report.termination),
        },
        Err(error) => Outcome {
            termination: format!("Error({error:?})"),
            ..Outcome::default()
        },
    }
}

fn reference(
    model: &Nist,
    start: &[f64],
    patience: u64,
    scaled: bool,
    trace: Option<&Trace>,
) -> Outcome {
    let problem = Reference {
        model,
        trace,
        point: DVector::from_column_slice(start),
        residual_calls: Cell::new(0),
        jacobian_calls: Cell::new(0),
    };
    let solver = dogleg::Dogleg::new().with_patience(patience);
    let solver = if scaled {
        solver
    } else {
        solver
            .with_diagonal_scaling(false)
            .with_jacobi_scaling(false)
            .with_stepbound(InitialTrusRegionRadius::Exact(RADIUS))
    };
    let (problem, rss, termination) = match solver.minimize(problem) {
        Ok((problem, report)) => (
            problem,
            Some(2.0 * report.objective_function),
            format!("{:?}", report.termination),
        ),
        Err(error) => (error.problem, None, format!("{:?}", error.failure)),
    };
    Outcome {
        point: Some(problem.point.as_slice().to_vec()),
        rss,
        residual_calls: Some(problem.residual_calls.get()),
        jacobian_calls: Some(problem.jacobian_calls.get()),
        termination,
        ..Outcome::default()
    }
}

fn csv(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn field<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(String::new, |v| v.to_string())
}

fn write_trace() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = datasets()?
        .into_iter()
        .find(|dataset| dataset.id == "BoxBOD")
        .ok_or("BoxBOD fixture missing")?;
    let model = Nist::new(dataset);
    let traces = std::array::from_fn(|_| Trace::default());
    let [basin, unscaled, scaled] = traces.each_ref().map(Some);
    let [_, basin_run, unscaled_run, scaled_run] =
        run_case(&model, 0, [None, basin, unscaled, scaled]);
    let runs = [basin_run, unscaled_run, scaled_run];
    let rss = model.dataset.rss.midpoint;
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "{{")?;
    for (i, ((name, run), trace)) in runs.into_iter().zip(&traces).enumerate() {
        let count = trace.0.borrow().len() as u64;
        if count == 0 || run.residual_calls != Some(count) {
            return Err(format!("{name}: trace and solve counts differ").into());
        }
        let comma = if i == traces.len() - 1 { "" } else { "," };
        writeln!(out, "    {name:?}: {:?}{comma}", trace.points(rss))?;
    }
    writeln!(out, "}}")?;
    out.flush()?;
    Ok(())
}

fn residual_limits(n: usize) -> (u64, u64) {
    let unit = n as u64 + 1;
    let patience = MAX_EVALUATIONS / unit;
    (patience, patience * unit)
}

fn run_case(
    model: &Nist,
    index: usize,
    [lm_trace, dogleg_trace, unscaled_trace, scaled_trace]: [Option<&Trace>; 4],
) -> [(&'static str, Outcome); 4] {
    let start = model.dataset.start::<f64>(index);
    let (patience, budget) = residual_limits(start.len());
    let problem = |trace| TracedNist { model, trace };
    let lm = basin_outcome(
        Executor::from_start(
            problem(lm_trace),
            LevenbergMarquardt::new(),
            BasinVector::from_column_slice(&start),
        )
        .max_evaluations(EvaluationKind::Residual, budget)
        .run(),
    );
    let dogleg = basin_outcome(
        Executor::from_start(
            problem(dogleg_trace),
            TrustRegion::least_squares_with(Dogleg).with_radius(RADIUS),
            BasinVector::from_column_slice(&start),
        )
        .max_evaluations(EvaluationKind::Residual, budget)
        .run(),
    );
    [
        ("basin-lm", lm),
        ("basin-dogleg", dogleg),
        (
            "dogleg-unscaled",
            reference(model, &start, patience, false, unscaled_trace),
        ),
        (
            "dogleg-default",
            reference(model, &start, patience, true, scaled_trace),
        ),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::args().nth(1).as_deref() {
        Some("--trace") => return write_trace(),
        Some(_) => return Err("usage: verify_dogleg [--trace]".into()),
        None => {}
    }

    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(
        out,
        "dataset,m,n,start,solver,residual_budget,iterations,residual_evals,jacobian_evals,reported_rss,recomputed_rss,rss_relative_error,parameter_relative_error,gradient_inf,recovered_1e4,termination,parameters"
    )?;
    for dataset in datasets()? {
        let model = Nist::new(dataset);
        let dataset = &model.dataset;
        let m = dataset.observations.len();
        let n = dataset.reference.len();
        let (_, budget) = residual_limits(n);
        for index in 0..2 {
            for (name, run) in run_case(&model, index, [None; 4]) {
                let quality = match &run.point {
                    Some(point) => Quality::at(dataset, point)?,
                    None => Quality::invalid(),
                };
                writeln!(
                    out,
                    "{},{m},{n},{},{name},{budget},{},{},{},{},{:e},{:e},{:e},{:e},{},{},{}",
                    dataset.id,
                    index + 1,
                    field(run.iterations),
                    field(run.residual_calls),
                    field(run.jacobian_calls),
                    field(run.rss),
                    quality.rss,
                    quality.rss_relative_error,
                    quality.parameter_relative_error,
                    quality.gradient_inf,
                    quality.recovered(1e-4),
                    csv(&run.termination),
                    run.point
                        .map_or_else(String::new, |p| csv(&format!("{p:?}"))),
                )?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
