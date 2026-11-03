//! Explicit finite-box coordinate transformations for objective problems.
use crate::{
    BoxBounds, ConstraintSet, Constraints, CostFunction, Gradient,
    HessianProduct, Scalar, VectorIndex, VectorLen,
};

/// Mapping from a unit coordinate to physical parameter units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterScale {
    /// Affine interpolation between physical bounds.
    Linear,
    /// Affine interpolation in logarithmic units; physical bounds must be positive.
    Logarithmic,
}

/// Invalid transform geometry, input or derivative shape.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParameterTransformError {
    /// Bounds, scale list, parameter or derivative dimensions differ.
    Dimension,
    /// Bounds must be finite, strictly ordered, with finite transformed width.
    Bounds,
    /// A logarithmic interval includes zero or negative values.
    NonPositiveLogarithm,
    /// A coordinate is non-finite or outside its declared interval.
    OutsideDomain,
    /// The box-only problem declaration must include both bound vectors.
    MissingBounds,
}
impl std::fmt::Display for ParameterTransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid parameter transform: {self:?}")
    }
}
impl std::error::Error for ParameterTransformError {}

/// Invalid transform input or the wrapped problem's unchanged typed error.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParameterEvaluationError<E> {
    /// Transform validation failed before calling the objective.
    Invalid(ParameterTransformError),
    /// The physical objective or derivative callback failed.
    Evaluation(E),
}
impl<E: std::fmt::Display> std::fmt::Display for ParameterEvaluationError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(e) => e.fmt(f),
            Self::Evaluation(e) => e.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error
    for ParameterEvaluationError<E>
{
}

/// Optimize a physical box-only problem in normalized unit coordinates.
///
/// Construction requires a complete box-only [`Constraints`] declaration;
/// general constraints cannot be silently discarded. Physical bounds are finite
/// and strictly ordered. Fixed coordinates must be eliminated before using this
/// adapter. Each coordinate is linear or logarithmic. The adapter declares unit
/// box bounds, passes physical values to callbacks and applies the chain rule to
/// gradients and exact Hessian-vector products, including the second derivative
/// of logarithmic maps. Results and solver tolerances use normalized units;
/// [`decode`](Self::decode) converts results back to physical units.
///
/// This adapter preserves fused and in-place derivative callbacks. An exact HVP
/// with logarithmic coordinates additionally requests the physical gradient.
/// Basin's outer [`crate::Problem`] counters count adapter requests, not these
/// internal callbacks. Use [`crate::BoundedFiniteDiff`] outside this adapter to
/// synthesize missing derivatives in unit coordinates with explicit unit bounds.
/// Invalid inputs fail before physical callbacks; physical errors remain typed.
///
/// # Backends
///
/// Any cloneable [`VectorLen`] + [`VectorIndex<F>`] vector, including `Vec<F>`,
/// nalgebra `DVector<F>`, ndarray `Array1<F>` and faer `Col<F>` for `f32`/`f64`.
///
/// # Example
///
/// Logarithmic coordinates place the geometric mean halfway along the unit
/// interval. Keep a transform with the same geometry to decode solver results.
///
/// ```
/// use basin::{BoxBounds, Constrained, ConstraintSet, CostFunction,
///             ParameterScale, ParameterTransform};
/// use std::convert::Infallible;
///
/// struct Objective;
/// impl CostFunction for Objective {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
///         Ok((x[0] - 10.0).powi(2))
///     }
/// }
/// let physical = Constrained::new(Objective, ConstraintSet::new()
///     .with_bounds(BoxBounds::new(Some(vec![1.0]), Some(vec![100.0]))));
/// let transform = ParameterTransform::new(physical,
///     vec![ParameterScale::Logarithmic]).unwrap();
/// let unit = transform.encode(&vec![10.0]).unwrap();
/// assert!((unit[0] - 0.5).abs() < 1e-14);
/// let decoded = transform.decode(&unit).unwrap();
/// assert!((decoded[0] - 10.0).abs() < 1e-12);
/// ```
pub struct ParameterTransform<P, V, F: Scalar = f64> {
    problem: P,
    physical_lower: V,
    physical_upper: V,
    lower: V,
    upper: V,
    scales: Vec<ParameterScale>,
    widths: Vec<F>,
}
impl<P, V, F> ParameterTransform<P, V, F>
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
    P: CostFunction<Param = V, Output = F>
        + Constraints<Set = ConstraintSet<BoxBounds<V>>>,
{
    /// Validate physical bounds and one scale per coordinate, without callbacks.
    pub fn new(
        problem: P,
        scales: Vec<ParameterScale>,
    ) -> Result<Self, ParameterTransformError> {
        use ParameterTransformError as E;
        let bounds = problem.constraints();
        let physical_lower = bounds
            .bounds
            .lower()
            .copied()
            .ok_or(E::MissingBounds)?
            .clone();
        let physical_upper = bounds
            .bounds
            .upper()
            .copied()
            .ok_or(E::MissingBounds)?
            .clone();
        let n = physical_lower.vec_len();
        if physical_upper.vec_len() != n || scales.len() != n {
            return Err(E::Dimension);
        }
        let mut widths = Vec::with_capacity(n);
        let mut lower = physical_lower.clone();
        let mut upper = physical_upper.clone();
        for (i, scale) in scales.iter().enumerate() {
            let a = physical_lower.get_scalar(i);
            let b = physical_upper.get_scalar(i);
            if !a.is_finite() || !b.is_finite() || a >= b {
                return Err(E::Bounds);
            }
            if *scale == ParameterScale::Logarithmic && a <= F::zero() {
                return Err(E::NonPositiveLogarithm);
            }
            let width = match scale {
                ParameterScale::Linear => b - a,
                ParameterScale::Logarithmic => b.ln() - a.ln(),
            };
            if !width.is_finite() || width <= F::zero() {
                return Err(E::Bounds);
            }
            widths.push(width);
            lower.set_scalar(i, F::zero());
            upper.set_scalar(i, F::one());
        }
        Ok(Self {
            problem,
            physical_lower,
            physical_upper,
            lower,
            upper,
            scales,
            widths,
        })
    }
}
impl<P, V, F> ParameterTransform<P, V, F>
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
{
    /// Borrow the physical problem.
    pub fn get_ref(&self) -> &P {
        &self.problem
    }
    /// Recover the physical problem.
    pub fn into_inner(self) -> P {
        self.problem
    }
    /// Convert a feasible physical vector to unit coordinates.
    pub fn encode(&self, x: &V) -> Result<V, ParameterTransformError> {
        self.check(x, &self.physical_lower, &self.physical_upper)?;
        let mut out = x.clone();
        for i in 0..x.vec_len() {
            let a = self.physical_lower.get_scalar(i);
            let b = self.physical_upper.get_scalar(i);
            let value = x.get_scalar(i);
            let unit = if value == a {
                F::zero()
            } else if value == b {
                F::one()
            } else {
                match self.scales[i] {
                    ParameterScale::Linear => (value - a) / self.widths[i],
                    ParameterScale::Logarithmic => {
                        (value.ln() - a.ln()) / self.widths[i]
                    }
                }
            };
            out.set_scalar(i, unit.max(F::zero()).min(F::one()));
        }
        Ok(out)
    }
    /// Convert a feasible normalized vector to physical coordinates.
    pub fn decode(&self, x: &V) -> Result<V, ParameterTransformError> {
        self.check(x, &self.lower, &self.upper)?;
        let mut out = x.clone();
        for i in 0..x.vec_len() {
            let a = self.physical_lower.get_scalar(i);
            let b = self.physical_upper.get_scalar(i);
            let unit = x.get_scalar(i);
            let value = if unit == F::zero() {
                a
            } else if unit == F::one() {
                b
            } else {
                match self.scales[i] {
                    ParameterScale::Linear => a + unit * self.widths[i],
                    ParameterScale::Logarithmic => {
                        (a.ln() + unit * self.widths[i]).exp()
                    }
                }
            };
            out.set_scalar(i, value.max(a).min(b));
        }
        Ok(out)
    }
    fn check(
        &self,
        x: &V,
        lo: &V,
        hi: &V,
    ) -> Result<(), ParameterTransformError> {
        if x.vec_len() != self.widths.len() {
            return Err(ParameterTransformError::Dimension);
        }
        if (0..x.vec_len()).any(|i| {
            !x.get_scalar(i).is_finite()
                || x.get_scalar(i) < lo.get_scalar(i)
                || x.get_scalar(i) > hi.get_scalar(i)
        }) {
            return Err(ParameterTransformError::OutsideDomain);
        }
        Ok(())
    }
    fn jacobian(&self, physical: &V, i: usize) -> F {
        match self.scales[i] {
            ParameterScale::Linear => self.widths[i],
            ParameterScale::Logarithmic => {
                physical.get_scalar(i) * self.widths[i]
            }
        }
    }
    fn gradient_chain(
        &self,
        physical: &V,
        g: &mut V,
    ) -> Result<(), ParameterTransformError> {
        if g.vec_len() != self.widths.len() {
            return Err(ParameterTransformError::Dimension);
        }
        for i in 0..g.vec_len() {
            g.set_scalar(i, g.get_scalar(i) * self.jacobian(physical, i));
        }
        Ok(())
    }
}
impl<P, V, F> Constraints for ParameterTransform<P, V, F>
where
    F: Scalar,
{
    type Set = ConstraintSet<BoxBounds<V>>;
    fn constraints(
        &self,
    ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}
impl<P, V, F> CostFunction for ParameterTransform<P, V, F>
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
    P: CostFunction<Param = V, Output = F>,
{
    type Param = V;
    type Output = F;
    type Error = ParameterEvaluationError<P::Error>;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        let physical =
            self.decode(x).map_err(ParameterEvaluationError::Invalid)?;
        self.problem
            .cost(&physical)
            .map_err(ParameterEvaluationError::Evaluation)
    }
}
impl<P, V, F> Gradient for ParameterTransform<P, V, F>
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        let mut out = x.clone();
        self.gradient_into(x, &mut out)?;
        Ok(out)
    }
    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), Self::Error> {
        let physical =
            self.decode(x).map_err(ParameterEvaluationError::Invalid)?;
        self.problem
            .gradient_into(&physical, out)
            .map_err(ParameterEvaluationError::Evaluation)?;
        self.gradient_chain(&physical, out)
            .map_err(ParameterEvaluationError::Invalid)
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(F, V), Self::Error> {
        let mut out = x.clone();
        let f = self.cost_and_gradient_into(x, &mut out)?;
        Ok((f, out))
    }
    fn cost_and_gradient_into(
        &self,
        x: &V,
        out: &mut V,
    ) -> Result<F, Self::Error> {
        let physical =
            self.decode(x).map_err(ParameterEvaluationError::Invalid)?;
        let f = self
            .problem
            .cost_and_gradient_into(&physical, out)
            .map_err(ParameterEvaluationError::Evaluation)?;
        self.gradient_chain(&physical, out)
            .map_err(ParameterEvaluationError::Invalid)?;
        Ok(f)
    }
}
impl<P, V, F> HessianProduct for ParameterTransform<P, V, F>
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + HessianProduct,
{
    fn hessian_product(&self, x: &V, v: &V) -> Result<V, Self::Error> {
        let mut out = x.clone();
        self.hessian_product_into(x, v, &mut out)?;
        Ok(out)
    }
    fn hessian_product_into(
        &self,
        x: &V,
        v: &V,
        out: &mut V,
    ) -> Result<(), Self::Error> {
        use ParameterEvaluationError::{Evaluation, Invalid};
        let physical = self.decode(x).map_err(Invalid)?;
        if v.vec_len() != x.vec_len()
            || (0..v.vec_len()).any(|i| !v.get_scalar(i).is_finite())
        {
            return Err(Invalid(ParameterTransformError::Dimension));
        }
        let mut direction = v.clone();
        for i in 0..v.vec_len() {
            direction
                .set_scalar(i, v.get_scalar(i) * self.jacobian(&physical, i));
        }
        self.problem
            .hessian_product_into(&physical, &direction, out)
            .map_err(Evaluation)?;
        self.gradient_chain(&physical, out).map_err(Invalid)?;
        if self.scales.contains(&ParameterScale::Logarithmic) {
            let g = self.problem.gradient(&physical).map_err(Evaluation)?;
            if g.vec_len() != x.vec_len() {
                return Err(Invalid(ParameterTransformError::Dimension));
            }
            for i in 0..x.vec_len() {
                if self.scales[i] == ParameterScale::Logarithmic {
                    out.set_scalar(
                        i,
                        out.get_scalar(i)
                            + g.get_scalar(i)
                                * physical.get_scalar(i)
                                * self.widths[i]
                                * self.widths[i]
                                * v.get_scalar(i),
                    );
                }
            }
        }
        Ok(())
    }
}
