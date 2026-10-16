//! Fixed, source-informed stopping policies, without a tolerance sweep.
//! Run `cargo run -p competitor-bench --release --bin verify_nlls_defaults`.
//! Both LM policies retain Nielsen damping, and TRF uses Basin's observed
//! cost/step formulas for TRF; neither policy reproduces SciPy trajectories.

use basin::{
    BoxBounds, Constrained, Constraints, EvaluationKind, Executor, Jacobian,
    LevenbergMarquardt, LmDamping, NoConstraints, PointState,
    ProblemConstraints, Residual, Solver, Trf, TrustRegionReflective,
};
use competitor_bench::nist::{Nist, datasets};
use std::fmt::Debug;

#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix, DVector};

#[path = "support/nlls_defaults_cases.rs"]
mod cases;

type Vector = DVector<f64>;
type Matrix = DMatrix<f64>;

trait Fixture:
    Clone
    + Residual<Param = Vector, Output = Vector, Error = String>
    + Jacobian<Jacobian = Matrix>
    + ProblemConstraints<Constraints = NoConstraints>
{
    fn id(&self) -> &str;
    fn starts(&self) -> Vec<Vec<f64>>;
    fn reference_point(&self) -> Vec<f64>;
    fn reference_rss(&self) -> f64;
}

impl Fixture for Nist {
    fn id(&self) -> &str {
        self.dataset.id
    }
    fn starts(&self) -> Vec<Vec<f64>> {
        (0..2).map(|i| self.dataset.start(i)).collect()
    }
    fn reference_point(&self) -> Vec<f64> {
        self.dataset.reference_point()
    }
    fn reference_rss(&self) -> f64 {
        self.dataset.rss.midpoint
    }
}

fn csv(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn relative_error(value: f64, reference: f64) -> f64 {
    if !value.is_finite() || !reference.is_finite() {
        f64::INFINITY
    } else if reference == 0.0 {
        if value == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        ((value - reference) / reference).abs()
    }
}

struct Quality {
    rss: f64,
    parameter_error: f64,
    gradient_inf: f64,
    orthogonality: f64,
}

impl Quality {
    fn at<P: Fixture>(fixture: &P, x: &Vector) -> Result<Self, String> {
        let residual = fixture.residual(x)?;
        let jacobian = fixture.jacobian(x)?;
        let reference = fixture.reference_point();
        if x.len() != reference.len()
            || jacobian.ncols() != x.len()
            || jacobian.nrows() != residual.len()
        {
            return Err("diagnostic dimensions differ".into());
        }
        if !x
            .iter()
            .chain(residual.iter())
            .chain(jacobian.iter())
            .all(|v| v.is_finite())
        {
            return Ok(Self::invalid());
        }
        let parameter_error = x
            .iter()
            .zip(reference)
            .map(|(&v, r)| relative_error(v, r))
            .fold(0.0_f64, f64::max);
        let residual_norm = residual.iter().fold(0.0_f64, |n, &v| n.hypot(v));
        let mut gradient_inf = 0.0_f64;
        let mut orthogonality = 0.0_f64;
        for column in jacobian.column_iter() {
            let gradient: f64 = column
                .iter()
                .zip(residual.iter())
                .map(|(&j, &r)| j * r)
                .sum();
            let column_norm = column.iter().fold(0.0_f64, |n, &v| n.hypot(v));
            let cosine = if residual_norm == 0.0 || column_norm == 0.0 {
                0.0
            } else {
                column
                    .iter()
                    .zip(residual.iter())
                    .map(|(&j, &r)| (j / column_norm) * (r / residual_norm))
                    .sum::<f64>()
                    .abs()
            };
            gradient_inf = if gradient.is_finite() {
                gradient_inf.max(gradient.abs())
            } else {
                f64::INFINITY
            };
            orthogonality = if cosine.is_finite()
                && residual_norm.is_finite()
                && column_norm.is_finite()
            {
                orthogonality.max(cosine)
            } else {
                f64::INFINITY
            };
        }
        Ok(Self {
            rss: residual.iter().map(|r| r * r).sum(),
            parameter_error,
            gradient_inf,
            orthogonality,
        })
    }

    fn invalid() -> Self {
        Self {
            rss: f64::INFINITY,
            parameter_error: f64::INFINITY,
            gradient_inf: f64::INFINITY,
            orthogonality: f64::INFINITY,
        }
    }
}

fn run<P, S, F>(
    fixture: &F,
    problem: P,
    solver: S,
    route: &str,
    policy: &str,
    start: usize,
    x: &[f64],
) where
    F: Fixture,
    S: Solver<P, PointState<Vector>>,
    S::Error: Debug,
{
    let result = Executor::new(
        problem,
        solver,
        PointState::new(Vector::from_column_slice(x)),
    )
    .max_iter(2000)
    .max_evaluations(EvaluationKind::TotalWork, 20000)
    .run();
    match result {
        Ok(result) => {
            let quality = Quality::at(fixture, result.param())
                .unwrap_or_else(|_| Quality::invalid());
            let rss = 2.0 * result.cost();
            let counts = &result.counts;
            println!(
                "{route},{policy},{},{},{:?},{},{},{},{:e},{:e},{:e},{:e},{:e},{:e},{}",
                csv(fixture.id()),
                start + 1,
                result.report.code(),
                result.iter(),
                counts.residual_evals,
                counts.jacobian_evals,
                relative_error(quality.rss, fixture.reference_rss()),
                quality.parameter_error,
                quality.gradient_inf,
                quality.orthogonality,
                rss,
                quality.rss,
                csv(&format!("{:?}", result.report.termination))
            );
        }
        Err(error) => {
            // Errors do not return a state, so missing work counts stay empty.
            println!(
                "{route},{policy},{},{},Error,,,,inf,inf,inf,inf,inf,inf,{}",
                csv(fixture.id()),
                start + 1,
                csv(&format!("{error:?}"))
            );
        }
    }
}

fn compare<P: Fixture>(fixture: P) {
    for (start, x) in fixture.starts().iter().enumerate() {
        for relative in [false, true] {
            let lm = || {
                let solver = LevenbergMarquardt::<Vector, Matrix>::new()
                    .with_damping(LmDamping::Nielsen);
                if relative {
                    solver
                        .with_absolute_gradient_tolerance(None)
                        .with_gradient_orthogonality_tolerance(1e-8)
                        .with_relative_model_reduction_tolerance(1e-8)
                } else {
                    solver
                }
            };
            let lm_policy = if relative {
                "relative-model"
            } else {
                "current"
            };
            run(&fixture, fixture.clone(), lm(), "lm", lm_policy, start, x);
            run(
                &fixture,
                fixture.clone(),
                lm().with_pivoted_qr(),
                "lm-qr",
                lm_policy,
                start,
                x,
            );
            let trf_policy = if relative {
                "observed-relative"
            } else {
                "current"
            };
            let boxed = || {
                Constrained::new(
                    fixture.clone(),
                    Constraints::new().with_bounds(BoxBounds::new(
                        Some(Vector::from_element(x.len(), f64::NEG_INFINITY)),
                        Some(Vector::from_element(x.len(), f64::INFINITY)),
                    )),
                )
            };
            let trf = Trf::<Vector, Matrix>::new();
            let full_trf = TrustRegionReflective::new();
            if relative {
                run(
                    &fixture,
                    boxed(),
                    trf.with_relative_cost_change_tolerance(1e-8)
                        .with_relative_step_tolerance(1e-8),
                    "trf",
                    trf_policy,
                    start,
                    x,
                );
                run(
                    &fixture,
                    boxed(),
                    full_trf
                        .with_relative_cost_change_tolerance(1e-8)
                        .with_relative_step_tolerance(1e-8),
                    "full-trf",
                    trf_policy,
                    start,
                    x,
                );
            } else {
                run(&fixture, boxed(), trf, "trf", trf_policy, start, x);
                run(
                    &fixture,
                    boxed(),
                    full_trf,
                    "full-trf",
                    trf_policy,
                    start,
                    x,
                );
            }
        }
    }
}

fn main() -> Result<(), String> {
    println!(
        "solver,policy,dataset,start,reason,iterations,residual_evals,jacobian_evals,rss_relative_error,max_parameter_relative_error,gradient_inf,orthogonality,state_rss,recomputed_rss,termination"
    );
    for dataset in datasets()? {
        compare(Nist::new(dataset));
    }
    cases::run();
    Ok(())
}
