//! Shared configurations and adapters for the LM default investigation.
//!
//! Only damping and factorization vary. Fixture preparation and independent
//! diagnostics belong outside timing; solver construction, initialization,
//! callbacks, and teardown belong inside it.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use basin::solver::least_squares_diagnostics::LeastSquaresDiagnostics;
use basin::{
    DenseMatrix, EvaluationKind, Executor, Jacobian, LevenbergMarquardt,
    LmDamping, ObserverMode, OptimizationResult, PointState, Residual, Scalar,
    Solver,
};

use crate::nist::{Dataset, Nist};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix, DVector};

pub const MAX_ITERATIONS: u64 = 2_000;
pub const MAX_EVALUATIONS: u64 = 20_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Configuration {
    NielsenCholesky,
    NielsenQr,
    TrustCholesky,
    TrustQr,
}

impl Configuration {
    pub const ALL: [Self; 4] = [
        Self::NielsenCholesky,
        Self::NielsenQr,
        Self::TrustCholesky,
        Self::TrustQr,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::NielsenCholesky => "nielsen-cholesky",
            Self::NielsenQr => "nielsen-qr",
            Self::TrustCholesky => "trust-cholesky",
            Self::TrustQr => "trust-qr",
        }
    }

    fn damping(self) -> LmDamping {
        match self {
            Self::NielsenCholesky | Self::NielsenQr => LmDamping::Nielsen,
            Self::TrustCholesky | Self::TrustQr => LmDamping::TrustRegion,
        }
    }

    fn qr(self) -> bool {
        matches!(self, Self::NielsenQr | Self::TrustQr)
    }
}

/// Slice access keeps native fixture arithmetic identical across backends.
pub trait Model<F: Scalar> {
    fn residuals(&self, parameters: &[F]) -> Result<Vec<F>, String>;
    fn jacobian_rows(&self, parameters: &[F]) -> Result<Vec<Vec<F>>, String>;
}

impl<F: Scalar> Model<F> for Nist<F> {
    fn residuals(&self, parameters: &[F]) -> Result<Vec<F>, String> {
        self.residuals(parameters)
    }

    fn jacobian_rows(&self, parameters: &[F]) -> Result<Vec<Vec<F>>, String> {
        self.jacobian_rows(parameters)
    }
}

pub trait Backend: Sized + 'static {
    type Float: Scalar;
    type Vector: Clone;
    type Matrix;
    const NAME: &'static str;
    const PRECISION: &'static str;

    fn vector(values: Vec<Self::Float>) -> Self::Vector;
    fn values(vector: &Self::Vector) -> Vec<Self::Float>;
    fn matrix(rows: Vec<Vec<Self::Float>>) -> Self::Matrix;

    fn solve<P: Model<Self::Float>>(
        model: &P,
        start: Vec<Self::Float>,
        configuration: Configuration,
        diagnostics: bool,
    ) -> Result<Outcome<Self>, String>;
}

pub struct Adapter<'a, B: Backend, P> {
    pub model: &'a P,
    pub backend: PhantomData<B>,
}

impl<B: Backend, P> basin::Constraints for Adapter<'_, B, P> {
    type Set = basin::NoConstraints;
    fn constraints(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}

impl<B: Backend, P: Model<B::Float>> Residual for Adapter<'_, B, P> {
    type Param = B::Vector;
    type Output = B::Vector;
    type Error = String;
    fn residual(&self, x: &B::Vector) -> Result<B::Vector, String> {
        self.model.residuals(&B::values(x)).map(B::vector)
    }
}

impl<B: Backend, P: Model<B::Float>> Jacobian for Adapter<'_, B, P> {
    type Jacobian = B::Matrix;
    fn jacobian(&self, x: &B::Vector) -> Result<B::Matrix, String> {
        self.model.jacobian_rows(&B::values(x)).map(B::matrix)
    }
}

/// Sum subproblem events only: trial rows repeat their model-solve counts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceTotals {
    pub model_solves: u64,
    pub retries: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub last_damping: Option<f64>,
    pub last_radius: Option<f64>,
    next_sequence: u64,
}

impl TraceTotals {
    fn observe<F: Scalar>(&mut self, solver: &impl LeastSquaresDiagnostics<F>) {
        for row in solver.least_squares_observations() {
            if row.sequence < self.next_sequence {
                continue;
            }
            self.next_sequence = row.sequence + 1;
            if row.event == "subproblem" {
                self.model_solves += u64::from(row.model_solves);
                self.retries += u64::from(matches!(
                    row.outcome,
                    Some("retry" | "exhausted")
                ));
            }
            if let Some(accepted) = row.accepted {
                self.accepted += u64::from(accepted);
                self.rejected += u64::from(!accepted);
            }
            if let Some(mu) = row.damping_after.or(row.damping) {
                self.last_damping = mu.to_f64();
            }
            if let Some(radius) = row.radius_after.or(row.radius_before) {
                self.last_radius = radius.to_f64();
            }
        }
    }
}

pub struct Outcome<B: Backend> {
    pub result: OptimizationResult<PointState<B::Vector, B::Float>>,
    pub trace: Option<TraceTotals>,
}

fn execute<'a, B, P, S>(
    model: &'a P,
    start: Vec<B::Float>,
    solver: S,
    diagnostics: bool,
) -> Result<Outcome<B>, String>
where
    B: Backend,
    P: Model<B::Float>,
    S: Solver<
            Adapter<'a, B, P>,
            PointState<B::Vector, B::Float>,
            Error = String,
        > + LeastSquaresDiagnostics<B::Float>
        + 'static,
{
    let executor = Executor::new(
        Adapter::<B, P> {
            model,
            backend: PhantomData,
        },
        solver,
        PointState::new(B::vector(start)),
    )
    .max_iter(MAX_ITERATIONS)
    .max_evaluations(EvaluationKind::TotalWork, MAX_EVALUATIONS);
    if diagnostics {
        let trace = Rc::new(RefCell::new(TraceTotals::default()));
        let captured = Rc::clone(&trace);
        let result = executor
            .observe_solver(
                move |_, solver, _| captured.borrow_mut().observe(solver),
                ObserverMode::Always,
            )
            .run()?;
        let totals = trace.borrow().clone();
        Ok(Outcome {
            result,
            trace: Some(totals),
        })
    } else {
        executor.run().map(|result| Outcome {
            result,
            trace: None,
        })
    }
}

macro_rules! backend {
    ($name:ident, $label:literal, $float:ty, $vector:ty, $matrix:ty,
     $make_vector:expr, $values:expr, $make_matrix:expr) => {
        pub struct $name;
        impl Backend for $name {
            type Float = $float;
            type Vector = $vector;
            type Matrix = $matrix;
            const NAME: &'static str = $label;
            const PRECISION: &'static str = stringify!($float);
            fn vector(values: Vec<$float>) -> $vector {
                ($make_vector)(values)
            }
            fn values(vector: &$vector) -> Vec<$float> {
                ($values)(vector)
            }
            fn matrix(rows: Vec<Vec<$float>>) -> $matrix {
                ($make_matrix)(rows)
            }
            fn solve<P: Model<$float>>(
                model: &P,
                start: Vec<$float>,
                configuration: Configuration,
                diagnostics: bool,
            ) -> Result<Outcome<Self>, String> {
                let solver =
                    LevenbergMarquardt::<$vector, $matrix, $float>::default()
                        .with_damping(configuration.damping())
                        .with_trial_diagnostics(diagnostics);
                if configuration.qr() {
                    execute::<Self, _, _>(
                        model,
                        start,
                        solver.with_pivoted_qr(),
                        diagnostics,
                    )
                } else {
                    execute::<Self, _, _>(model, start, solver, diagnostics)
                }
            }
        }
    };
}

macro_rules! backends {
    ($float:ty, $vec:ident, $nalgebra:ident, $ndarray:ident, $faer:ident) => {
        backend!(
            $vec,
            "vec",
            $float,
            Vec<$float>,
            DenseMatrix<$float>,
            |v| v,
            |v: &Vec<$float>| v.clone(),
            |r: Vec<Vec<$float>>| DenseMatrix::from_row_slice(
                r.len(),
                r[0].len(),
                &r.concat()
            )
        );
        backend!(
            $nalgebra,
            "nalgebra",
            $float,
            DVector<$float>,
            DMatrix<$float>,
            DVector::from_vec,
            |v: &DVector<$float>| v.as_slice().to_vec(),
            |r: Vec<Vec<$float>>| DMatrix::from_fn(
                r.len(),
                r[0].len(),
                |i, j| r[i][j]
            )
        );
        backend!(
            $ndarray,
            "ndarray",
            $float,
            ndarray::Array1<$float>,
            ndarray::Array2<$float>,
            ndarray::Array1::from_vec,
            |v: &ndarray::Array1<$float>| v.to_vec(),
            |r: Vec<Vec<$float>>| ndarray::Array2::from_shape_fn(
                (r.len(), r[0].len()),
                |(i, j)| r[i][j]
            )
        );
        backend!(
            $faer,
            "faer",
            $float,
            faer::Col<$float>,
            faer::Mat<$float>,
            |v: Vec<$float>| faer::Col::from_fn(v.len(), |i| v[i]),
            |v: &faer::Col<$float>| v.iter().copied().collect(),
            |r: Vec<Vec<$float>>| faer::Mat::from_fn(
                r.len(),
                r[0].len(),
                |i, j| r[i][j]
            )
        );
    };
}

backends!(f32, VecF32, NalgebraF32, NdarrayF32, FaerF32);
backends!(f64, VecF64, NalgebraF64, NdarrayF64, FaerF64);

pub fn relative_error(value: f64, reference: f64) -> f64 {
    if !value.is_finite() || !reference.is_finite() {
        f64::INFINITY
    } else if reference == 0.0 {
        value.abs()
    } else {
        ((value - reference) / reference).abs()
    }
}

/// Reevaluate native parameters against the original fixture in f64.
/// This also exposes f32 rounding error relative to the published problem.
#[derive(Debug)]
pub struct Quality {
    pub rss: f64,
    pub rss_relative_error: f64,
    pub parameter_relative_error: f64,
    pub gradient_inf: f64,
    pub orthogonality: f64,
}

impl Quality {
    pub fn at(dataset: &Dataset, parameters: &[f64]) -> Result<Self, String> {
        let model = Nist::<f64>::new(dataset.clone());
        let residual = model.residuals(parameters)?;
        let rows = model.jacobian_rows(parameters)?;
        if parameters
            .iter()
            .chain(&residual)
            .chain(rows.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Ok(Self::invalid());
        }
        let norm = residual.iter().fold(0.0_f64, |n, &v| n.hypot(v));
        let rss = residual.iter().map(|r| r * r).sum();
        let mut gradient_inf = 0.0_f64;
        let mut orthogonality = 0.0_f64;
        for j in 0..parameters.len() {
            let column_norm =
                rows.iter().fold(0.0_f64, |n, row| n.hypot(row[j]));
            let gradient: f64 =
                rows.iter().zip(&residual).map(|(row, r)| row[j] * r).sum();
            gradient_inf = gradient_inf.max(if gradient.is_finite() {
                gradient.abs()
            } else {
                f64::INFINITY
            });
            let cosine = if norm == 0.0 || column_norm == 0.0 {
                0.0
            } else {
                rows.iter()
                    .zip(&residual)
                    .map(|(row, r)| (row[j] / column_norm) * (r / norm))
                    .sum::<f64>()
                    .abs()
            };
            orthogonality = orthogonality.max(if cosine.is_finite() {
                cosine
            } else {
                f64::INFINITY
            });
        }
        let parameter_relative_error = parameters
            .iter()
            .zip(&dataset.reference)
            .map(|(&p, r)| relative_error(p, r.midpoint))
            .fold(0.0_f64, f64::max);
        Ok(Self {
            rss,
            rss_relative_error: relative_error(rss, dataset.rss.midpoint),
            parameter_relative_error,
            gradient_inf,
            orthogonality,
        })
    }

    pub fn recovered(&self, tolerance: f64) -> bool {
        self.rss_relative_error <= tolerance
            && self.parameter_relative_error <= tolerance
    }

    pub fn invalid() -> Self {
        Self {
            rss: f64::INFINITY,
            rss_relative_error: f64::INFINITY,
            parameter_relative_error: f64::INFINITY,
            gradient_inf: f64::INFINITY,
            orthogonality: f64::INFINITY,
        }
    }
}

/// Use the existing MGH formulas when measuring larger parameter counts.
pub struct VarDimModel;

impl Model<f64> for VarDimModel {
    fn residuals(&self, parameters: &[f64]) -> Result<Vec<f64>, String> {
        let mut out = vec![0.0; parameters.len() + 2];
        crate::vardim_residual(parameters, &mut out);
        Ok(out)
    }
    fn jacobian_rows(
        &self,
        parameters: &[f64],
    ) -> Result<Vec<Vec<f64>>, String> {
        let n = parameters.len();
        let mut out = vec![0.0; (n + 2) * n];
        crate::vardim_jacobian_row_major(parameters, &mut out);
        Ok(out.chunks(n).map(<[f64]>::to_vec).collect())
    }
}
