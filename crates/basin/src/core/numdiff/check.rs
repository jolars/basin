//! Diagnostics for analytic gradients and residual Jacobians.
//!
//! [`DerivativeChecker`] compares derivatives at a single point. Coordinate
//! checks locate individual errors; directional checks compare `gᵀd` or `Jd`
//! with differences along a supplied direction and need only two probes with
//! the default central stencil (plus the base evaluation). A directional
//! check can miss errors orthogonal to its direction. Check several points
//! and directions when validating an implementation.
//!
//! These are numerical diagnostics, not proofs: noise, cancellation, nonsmooth
//! functions, and very narrow bounds can make finite differences unreliable.
//! Adjust the step and tolerances to the function's accuracy. The reported
//! errors measure disagreement, not an estimate of finite-difference accuracy.
//! See also SciPy's [`check_grad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html).

use super::{Method, bounded};
use crate::core::math::{MatrixIndex, Scalar, VectorIndex, VectorLen};
use crate::core::parallel::{MaybeSend, MaybeSync};
use crate::core::problem::{Gradient, Jacobian};

/// The derivative calculation that produced a non-finite value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DerivativeSource {
    /// The supplied analytic derivative, or its directional product.
    Analytic,
    /// Finite-difference arithmetic on finite function evaluations.
    FiniteDifference,
}

/// A failed derivative-check evaluation or invalid input.
///
/// Finite derivative disagreements are returned in [`DerivativeCheckReport`],
/// never as errors. Callback errors retain their original type in `Evaluation`.
#[derive(Debug)]
#[non_exhaustive]
pub enum DerivativeCheckError<E, F: Scalar = f64> {
    /// A cost, residual, gradient, or Jacobian callback failed.
    Evaluation(E),
    /// Invalid point, direction, bounds dimension, or derivative shape.
    InvalidInput(&'static str),
    /// A residual callback returned an unexpected number of entries.
    OutputSize {
        /// Number of rows in the analytic Jacobian.
        expected: usize,
        /// Number of residual entries returned.
        actual: usize,
    },
    /// A cost or residual evaluation returned NaN or infinity.
    NonFiniteEvaluation {
        /// The actual base or probe point evaluated.
        point: Vec<F>,
        /// Residual index, or zero for a scalar objective.
        output: usize,
        /// The non-finite value returned by the callback.
        value: F,
    },
    /// An analytic derivative or finite-difference calculation is non-finite.
    NonFiniteDerivative {
        /// Calculation that produced the non-finite value.
        source: DerivativeSource,
        /// Residual row, or zero for a scalar objective.
        output: usize,
        /// Parameter index, or `None` for a directional product.
        coordinate: Option<usize>,
        /// The non-finite derivative value.
        value: F,
    },
    /// No distinct representable probe fits along the supplied direction.
    /// This includes directions blocked on both sides at a box corner or
    /// directions that move a fixed coordinate.
    NoFeasibleDirection,
}

impl<E: std::fmt::Display, F: Scalar> std::fmt::Display
    for DerivativeCheckError<E, F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Evaluation(e) => {
                write!(f, "derivative-check callback failed: {e}")
            }
            Self::InvalidInput(message) => f.write_str(message),
            Self::OutputSize { expected, actual } => {
                write!(f, "expected {expected} residual entries, got {actual}")
            }
            Self::NonFiniteEvaluation {
                point,
                output,
                value,
            } => write!(
                f,
                "non-finite function value {value:?} at output {output}, point {point:?}"
            ),
            Self::NonFiniteDerivative {
                source,
                output,
                coordinate,
                value,
            } => write!(
                f,
                "non-finite {source:?} derivative {value:?} at output {output}, coordinate {coordinate:?}"
            ),
            Self::NoFeasibleDirection => f.write_str(
                "no feasible representable probe along the direction",
            ),
        }
    }
}
impl<E: std::error::Error + 'static, F: Scalar> std::error::Error
    for DerivativeCheckError<E, F>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation(e) => Some(e),
            _ => None,
        }
    }
}

/// One scalar comparison in a gradient or Jacobian check.
#[derive(Clone, Debug)]
pub struct DerivativeComparison<F: Scalar = f64> {
    /// Residual row, or zero for a scalar objective.
    pub output: usize,
    /// Parameter index, or `None` for a directional check.
    pub coordinate: Option<usize>,
    /// Supplied derivative, or its product with the normalized direction.
    pub analytic: F,
    /// Finite-difference derivative of the function values.
    pub finite_difference: F,
    /// `|analytic - finite_difference|`; may overflow to infinity.
    pub absolute_error: F,
    /// Absolute error divided by the larger derivative magnitude (zero if
    /// both derivatives are zero). Computed without subtraction overflow.
    pub relative_error: F,
    /// Absolute error divided by `atol + rtol * max(|analytic|, |numeric|)`.
    /// Zero for exact agreement, including when both tolerances are zero;
    /// infinity for nonzero disagreement with zero tolerances. Computed in
    /// scaled arithmetic to avoid overflow. Values at most one pass.
    pub scaled_error: F,
}
impl<F: Scalar> DerivativeComparison<F> {
    /// Whether this comparison meets the absolute-plus-relative tolerance.
    pub fn passed(&self) -> bool {
        self.scaled_error <= F::one()
    }
}

/// Per-entry diagnostics from a completed derivative check.
#[derive(Clone, Debug)]
pub struct DerivativeCheckReport<F: Scalar = f64> {
    /// Comparisons in coordinate-major, then output-row order. Directional
    /// checks contain one comparison per output row.
    pub comparisons: Vec<DerivativeComparison<F>>,
    /// Fixed coordinates omitted from a coordinate check. Their derivatives
    /// cannot be inferred within the supplied box, so they are not compared
    /// with the zero columns synthesized by bounded finite differences.
    pub skipped_coordinates: Vec<usize>,
    /// Direction actually checked, normalized to have maximum magnitude one.
    /// `None` for a coordinate check.
    pub direction: Option<Vec<F>>,
}
impl<F: Scalar> DerivativeCheckReport<F> {
    /// Whether every performed comparison passes. An empty report passes
    /// vacuously; inspect `comparisons` and `skipped_coordinates` for coverage.
    pub fn passed(&self) -> bool {
        self.comparisons.iter().all(DerivativeComparison::passed)
    }
}

/// Check analytic gradients and residual Jacobians against finite differences.
///
/// Defaults to central differences, machine function precision, and both
/// absolute and relative tolerances of `sqrt(F::epsilon())`. An entry passes
/// when `|analytic - numeric| <= atol + rtol * max(|analytic|, |numeric|)`.
/// Coordinate steps are `precision.cbrt() * max(1, |x[j]|)` for central
/// differences, or `precision.sqrt() * max(1, |x[j]|)` for forward differences.
///
/// [`with_bounds`](Self::with_bounds) explicitly supplies probe bounds. The
/// checker shares [`super::BoundedFiniteDiff`]'s stencils, including second-order
/// one-sided central stencils at boundaries and first-order fallback when only
/// one distinct probe fits. Fixed coordinates are skipped. Problem-side bounds
/// are not discovered automatically. No solver or evaluation counters are used.
/// With `parallel`, coordinate probes can run concurrently and the problem,
/// parameters, scalar, and callback errors need the corresponding thread-safety
/// bounds. All callbacks should return deterministic values at a given point.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`, for
/// `F = f64` or `f32`. Gradients and residuals need only [`VectorLen`] and
/// [`VectorIndex`]; Jacobians need only [`MatrixIndex`]. The analytic gradient
/// and residual vector types may differ from the parameter type.
///
/// # Example
///
/// ```
/// use basin::{CostFunction, DerivativeChecker, Gradient};
/// struct Square;
/// impl CostFunction for Square {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x[0] * x[0])
///     }
/// }
/// impl Gradient for Square {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(vec![2.0 * x[0]])
///     }
/// }
/// let report = DerivativeChecker::new()
///     .with_bounds(vec![0.0], vec![1.0])
///     .check_gradient(&Square, &vec![1.0])?;
/// assert!(report.passed());
/// # Ok::<(), basin::DerivativeCheckError<std::convert::Infallible>>(())
/// ```
#[derive(Clone, Debug)]
pub struct DerivativeChecker<F: Scalar = f64> {
    method: Method,
    precision: F,
    step: Option<F>,
    absolute_tolerance: F,
    relative_tolerance: F,
    bounds: Option<(Vec<F>, Vec<F>)>,
}
impl<F: Scalar> Default for DerivativeChecker<F> {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Scalar> DerivativeChecker<F> {
    /// Central differences with adaptive steps and scalar-appropriate tolerances.
    pub fn new() -> Self {
        Self {
            method: Method::Central,
            precision: F::epsilon(),
            step: None,
            absolute_tolerance: F::epsilon().sqrt(),
            relative_tolerance: F::epsilon().sqrt(),
            bounds: None,
        }
    }
    /// Select the stencil for all checks. Forward differences generally need
    /// looser tolerances because their truncation error is first order.
    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }
    /// Set finite, nonnegative absolute error tolerance. Zero permits only
    /// the relative tolerance. Panics on invalid values.
    pub fn with_absolute_tolerance(mut self, tolerance: F) -> Self {
        assert!(
            tolerance.is_finite() && tolerance >= F::zero(),
            "invalid absolute tolerance"
        );
        self.absolute_tolerance = tolerance;
        self
    }
    /// Set finite, nonnegative relative error tolerance. Zero permits only
    /// the absolute tolerance. Panics on invalid values.
    pub fn with_relative_tolerance(mut self, tolerance: F) -> Self {
        assert!(
            tolerance.is_finite() && tolerance >= F::zero(),
            "invalid relative tolerance"
        );
        self.relative_tolerance = tolerance;
        self
    }
    /// Set finite, nonnegative relative function precision, floored at machine
    /// epsilon. This changes steps, not acceptance tolerances. Panics on invalid values.
    pub fn function_precision(mut self, precision: F) -> Self {
        assert!(
            precision.is_finite() && precision >= F::zero(),
            "invalid function precision"
        );
        self.precision = precision.max(F::epsilon());
        self
    }
    /// Set a finite, positive absolute step, shortened as needed by bounds.
    /// In directional checks this is displacement along the normalized
    /// direction. Panics on invalid values.
    pub fn with_step(mut self, step: F) -> Self {
        assert!(
            step.is_finite() && step > F::zero(),
            "invalid finite-difference step"
        );
        self.step = Some(step);
        self
    }
    /// Copy explicit probe bounds. As in [`super::BoundedFiniteDiff`], a
    /// non-finite bound means that side is unbounded. Panics if lengths differ
    /// or a finite lower bound exceeds its upper bound. Point/bounds length
    /// mismatches and infeasible points return errors when a check is run.
    pub fn with_bounds<V: VectorLen + VectorIndex<F>>(
        mut self,
        lower: V,
        upper: V,
    ) -> Self {
        assert_eq!(lower.vec_len(), upper.vec_len(), "bound length mismatch");
        let lower: Vec<_> = (0..lower.vec_len())
            .map(|i| {
                let v = lower.get_scalar(i);
                if v.is_finite() { v } else { F::neg_infinity() }
            })
            .collect();
        let upper: Vec<_> = (0..upper.vec_len())
            .map(|i| {
                let v = upper.get_scalar(i);
                if v.is_finite() { v } else { F::infinity() }
            })
            .collect();
        assert!(
            lower.iter().zip(&upper).all(|(l, u)| l <= u),
            "lower bound exceeds upper bound"
        );
        self.bounds = Some((lower, upper));
        self
    }

    fn options(&self) -> bounded::Options<F> {
        bounded::Options {
            method: self.method,
            precision: self.precision,
            step: self.step,
            minpack: false,
        }
    }
    fn bounds(&self) -> Option<(&[F], &[F])> {
        self.bounds
            .as_ref()
            .map(|(l, u)| (l.as_slice(), u.as_slice()))
    }
    fn fixed(&self, j: usize) -> bool {
        self.bounds.as_ref().is_some_and(|(l, u)| l[j] == u[j])
    }
    fn validate_point<V: VectorLen + VectorIndex<F>, E>(
        &self,
        x: &V,
    ) -> Result<(), DerivativeCheckError<E, F>> {
        if let Some((l, _)) = &self.bounds {
            if l.len() != x.vec_len() {
                return Err(DerivativeCheckError::InvalidInput(
                    "point and bounds length mismatch",
                ));
            }
        }
        for j in 0..x.vec_len() {
            let v = x.get_scalar(j);
            if !v.is_finite() {
                return Err(DerivativeCheckError::InvalidInput(
                    "point must be finite",
                ));
            }
            if self
                .bounds
                .as_ref()
                .is_some_and(|(l, u)| v < l[j] || v > u[j])
            {
                return Err(DerivativeCheckError::InvalidInput(
                    "point is outside probe bounds",
                ));
            }
        }
        Ok(())
    }
}

impl<F: Scalar + MaybeSend + MaybeSync> DerivativeChecker<F> {
    /// Compare every non-fixed component of an analytic gradient.
    pub fn check_gradient<P>(
        &self,
        problem: &P,
        x: &P::Param,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<P::Error, F>>
    where
        P: Gradient<Output = F> + MaybeSync,
        P::Param: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        P::Gradient: VectorLen + VectorIndex<F>,
        P::Error: MaybeSend,
    {
        self.gradient(problem, x, None)
    }

    /// Compare `gᵀd` with a finite difference along `d / max(|d|)`.
    /// The direction must be finite, nonzero, and have the point's length.
    /// Bounds may reverse or shorten probes but never project the direction.
    /// The report records the normalized direction. A blocked direction
    /// returns [`DerivativeCheckError::NoFeasibleDirection`].
    ///
    /// The adaptive step is the smallest coordinate step divided by its
    /// nonzero direction magnitude. A maximum-magnitude direction component
    /// anchors representable displacements; bounds narrower than that
    /// component's floating-point resolution can make a direction unavailable.
    pub fn check_gradient_direction<P>(
        &self,
        problem: &P,
        x: &P::Param,
        direction: &P::Param,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<P::Error, F>>
    where
        P: Gradient<Output = F> + MaybeSync,
        P::Param: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        P::Gradient: VectorLen + VectorIndex<F>,
        P::Error: MaybeSend,
    {
        self.gradient(problem, x, Some(direction))
    }

    fn gradient<P>(
        &self,
        problem: &P,
        x: &P::Param,
        direction: Option<&P::Param>,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<P::Error, F>>
    where
        P: Gradient<Output = F> + MaybeSync,
        P::Param: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        P::Gradient: VectorLen + VectorIndex<F>,
        P::Error: MaybeSend,
    {
        self.validate_point(x)?;
        let g = problem
            .gradient(x)
            .map_err(DerivativeCheckError::Evaluation)?;
        if g.vec_len() != x.vec_len() {
            return Err(DerivativeCheckError::InvalidInput(
                "gradient and point length mismatch",
            ));
        }
        let analytic =
            (0..g.vec_len()).map(|j| vec![g.get_scalar(j)]).collect();
        self.run(x, 1, analytic, direction, |x| {
            problem.cost(x).map(|f| vec![f])
        })
    }

    /// Compare every non-fixed column of an analytic residual Jacobian.
    pub fn check_jacobian<P>(
        &self,
        problem: &P,
        x: &P::Param,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<P::Error, F>>
    where
        P: Jacobian + MaybeSync,
        P::Param: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        P::Output: VectorLen + VectorIndex<F>,
        P::Jacobian: MatrixIndex<F>,
        P::Error: MaybeSend,
    {
        self.jacobian(problem, x, None)
    }

    /// Compare `Jd` with differences of residual values along the normalized
    /// direction. Normalization, steps, and bounds follow
    /// [`check_gradient_direction`](Self::check_gradient_direction).
    pub fn check_jacobian_direction<P>(
        &self,
        problem: &P,
        x: &P::Param,
        direction: &P::Param,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<P::Error, F>>
    where
        P: Jacobian + MaybeSync,
        P::Param: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        P::Output: VectorLen + VectorIndex<F>,
        P::Jacobian: MatrixIndex<F>,
        P::Error: MaybeSend,
    {
        self.jacobian(problem, x, Some(direction))
    }

    fn jacobian<P>(
        &self,
        problem: &P,
        x: &P::Param,
        direction: Option<&P::Param>,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<P::Error, F>>
    where
        P: Jacobian + MaybeSync,
        P::Param: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        P::Output: VectorLen + VectorIndex<F>,
        P::Jacobian: MatrixIndex<F>,
        P::Error: MaybeSend,
    {
        self.validate_point(x)?;
        let a = problem
            .jacobian(x)
            .map_err(DerivativeCheckError::Evaluation)?;
        if a.matrix_cols() != x.vec_len() {
            return Err(DerivativeCheckError::InvalidInput(
                "Jacobian column count and point length mismatch",
            ));
        }
        let analytic = (0..a.matrix_cols())
            .map(|j| {
                (0..a.matrix_rows()).map(|i| a.matrix_entry(i, j)).collect()
            })
            .collect();
        self.run(x, a.matrix_rows(), analytic, direction, |x| {
            problem
                .residual(x)
                .map(|r| (0..r.vec_len()).map(|i| r.get_scalar(i)).collect())
        })
    }

    fn run<V, E, C>(
        &self,
        x: &V,
        rows: usize,
        analytic: Vec<Vec<F>>,
        direction: Option<&V>,
        evaluate: C,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<E, F>>
    where
        V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        E: MaybeSend,
        C: Fn(&V) -> Result<Vec<F>, E> + MaybeSync,
    {
        let evaluate = |probe: &V| {
            // Reject arithmetic overflow before passing a probe to user code.
            self.validate_point(probe)?;
            let values =
                evaluate(probe).map_err(DerivativeCheckError::Evaluation)?;
            if values.len() != rows {
                return Err(DerivativeCheckError::OutputSize {
                    expected: rows,
                    actual: values.len(),
                });
            }
            for (output, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(DerivativeCheckError::NonFiniteEvaluation {
                        point: (0..probe.vec_len())
                            .map(|j| probe.get_scalar(j))
                            .collect(),
                        output,
                        value,
                    });
                }
            }
            Ok(values)
        };
        if let Some(d) = direction {
            return self.directional(x, rows, analytic, d, evaluate);
        }
        let mut report = DerivativeCheckReport {
            comparisons: Vec::new(),
            skipped_coordinates: Vec::new(),
            direction: None,
        };
        let (_, numeric) =
            bounded::columns(x, self.options(), self.bounds(), evaluate)?;
        for (j, (a, n)) in analytic.iter().zip(&numeric).enumerate() {
            if self.fixed(j) {
                report.skipped_coordinates.push(j);
                continue;
            }
            for i in 0..rows {
                report.comparisons.push(self.compare(
                    a[i],
                    n[i],
                    i,
                    Some(j),
                )?);
            }
        }
        Ok(report)
    }

    fn directional<V, E, C>(
        &self,
        x: &V,
        rows: usize,
        analytic: Vec<Vec<F>>,
        direction: &V,
        evaluate: C,
    ) -> Result<DerivativeCheckReport<F>, DerivativeCheckError<E, F>>
    where
        V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
        E: MaybeSend,
        C: Fn(&V) -> Result<Vec<F>, DerivativeCheckError<E, F>> + MaybeSync,
    {
        if direction.vec_len() != x.vec_len() {
            return Err(DerivativeCheckError::InvalidInput(
                "direction and point length mismatch",
            ));
        }
        let mut d: Vec<_> = (0..direction.vec_len())
            .map(|j| direction.get_scalar(j))
            .collect();
        let mut magnitude = F::zero();
        let mut pivot = 0;
        for (j, &v) in d.iter().enumerate() {
            if !v.is_finite() {
                return Err(DerivativeCheckError::InvalidInput(
                    "direction must be finite",
                ));
            }
            if v != F::zero() && self.fixed(j) {
                return Err(DerivativeCheckError::NoFeasibleDirection);
            }
            if v.abs() > magnitude {
                magnitude = v.abs();
                pivot = j;
            }
        }
        if magnitude == F::zero() {
            return Err(DerivativeCheckError::InvalidInput(
                "direction must be nonzero",
            ));
        }
        for v in &mut d {
            *v = *v / magnitude;
        }
        let sign = d[pivot];
        let anchor = x.get_scalar(pivot);
        let mut lower = F::neg_infinity();
        let mut upper = F::infinity();
        let scale = if matches!(self.method, Method::Forward) {
            self.precision.sqrt()
        } else {
            self.precision.cbrt()
        };
        let mut step = F::infinity();
        for (j, &v) in d.iter().enumerate() {
            if v == F::zero() {
                continue;
            }
            step =
                step.min(scale * x.get_scalar(j).abs().max(F::one()) / v.abs());
            if let Some((lo, hi)) = &self.bounds {
                let slope = v / sign;
                let a = anchor + (lo[j] - x.get_scalar(j)) / slope;
                let b = anchor + (hi[j] - x.get_scalar(j)) / slope;
                lower = lower.max(a.min(b));
                upper = upper.min(a.max(b));
            }
        }
        if lower >= upper || lower > anchor || upper < anchor {
            return Err(DerivativeCheckError::NoFeasibleDirection);
        }
        let mut options = self.options();
        options.step = Some(self.step.unwrap_or(step));
        // Anchoring at a parameter coordinate makes the common probe machinery
        // account for representability at x, including adjacent-float boxes.
        let (_, numeric) = bounded::columns(
            &vec![anchor],
            options,
            Some((&[lower], &[upper])),
            |s| {
                let t = (s[0] - anchor) / sign;
                let mut probe = x.clone();
                for (j, &v) in d.iter().enumerate() {
                    let value = if j == pivot {
                        s[0]
                    } else if v == F::zero() {
                        x.get_scalar(j)
                    } else {
                        t.mul_add(v, x.get_scalar(j))
                    };
                    if let Some((lo, hi)) = &self.bounds {
                        // The scalar interval can round outward at the anchor's
                        // precision. Clamping would change the checked direction.
                        if value < lo[j] || value > hi[j] {
                            return Err(
                                DerivativeCheckError::NoFeasibleDirection,
                            );
                        }
                    }
                    probe.set_scalar(j, value);
                }
                evaluate(&probe)
            },
        )?;
        let mut report = DerivativeCheckReport {
            comparisons: Vec::new(),
            skipped_coordinates: Vec::new(),
            direction: Some(d.clone()),
        };
        for i in 0..rows {
            let mut product = F::zero();
            for (j, &v) in d.iter().enumerate() {
                if v != F::zero() {
                    finite(
                        analytic[j][i],
                        DerivativeSource::Analytic,
                        i,
                        Some(j),
                    )?;
                    product = product + analytic[j][i] * v;
                }
            }
            report.comparisons.push(self.compare(
                product,
                numeric[0][i] * sign,
                i,
                None,
            )?);
        }
        Ok(report)
    }

    fn compare<E>(
        &self,
        analytic: F,
        numeric: F,
        output: usize,
        coordinate: Option<usize>,
    ) -> Result<DerivativeComparison<F>, DerivativeCheckError<E, F>> {
        finite(analytic, DerivativeSource::Analytic, output, coordinate)?;
        finite(
            numeric,
            DerivativeSource::FiniteDifference,
            output,
            coordinate,
        )?;
        let absolute_error = (analytic - numeric).abs();
        let magnitude = analytic.abs().max(numeric.abs());
        let relative_error = if magnitude == F::zero() {
            F::zero()
        } else if absolute_error.is_finite() {
            absolute_error / magnitude
        } else {
            (analytic / magnitude - numeric / magnitude).abs()
        };
        let scaled_error = if absolute_error == F::zero() {
            F::zero()
        } else {
            // Scaling both the difference and tolerance avoids overflow even
            // when the finite derivatives have opposite signs near F::max_value().
            relative_error
                / (self.absolute_tolerance / magnitude
                    + self.relative_tolerance)
        };
        Ok(DerivativeComparison {
            output,
            coordinate,
            analytic,
            finite_difference: numeric,
            absolute_error,
            relative_error,
            scaled_error,
        })
    }
}

fn finite<E, F: Scalar>(
    value: F,
    source: DerivativeSource,
    output: usize,
    coordinate: Option<usize>,
) -> Result<(), DerivativeCheckError<E, F>> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DerivativeCheckError::NonFiniteDerivative {
            source,
            output,
            coordinate,
            value,
        })
    }
}
