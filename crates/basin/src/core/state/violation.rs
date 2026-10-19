//! Descriptions of solver-published constraint violations.

use crate::core::math::Scalar;

/// Aggregation of nonnegative constraint violations.
///
/// For the standard measures, equality violations are absolute residuals,
/// inequality violations are their positive parts, and bound violations are
/// distances outside the box. A solver reports the constraints it supports.
/// An empty set has zero violation. Scaling and transformations are described
/// separately by [`ViolationReport`]. These measures do not define feasibility
/// tolerances or certify optimality.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ViolationMeasure {
    /// Sum of the violations.
    L1,
    /// Euclidean norm of the violations.
    L2,
    /// Sum of squared violations, without taking a square root.
    SquaredL2,
    /// Maximum violation.
    LInf,
    /// A different aggregation, described by its formula and conventions.
    Custom(String),
}

/// Scaling applied to constraint residuals before aggregation.
///
/// Unscaled residuals retain the units and any scaling supplied by the
/// problem. This metadata describes additional solver-side scaling; it does
/// not make different problems or measures numerically comparable.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ViolationScaling {
    /// No additional normalization or weighting of the residuals.
    Unscaled,
    /// Solver-side scaling, described with its factors and conventions.
    Custom(String),
}

/// Numerical handling used when computing a reported constraint violation.
///
/// These tags describe existing computations; they do not transform stored
/// values. Ordinary floating-point overflow and non-finite arithmetic still
/// follow the solver's documented behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ViolationTransformation {
    /// No additional moderation or non-finite-value substitution.
    Identity,
    /// PRIMA's componentwise moderation before computing positive parts:
    /// residuals are clipped to `[-2^100, 2^100]`, and NaN becomes `2^100`.
    /// Equalities are first folded into residuals of opposite signs.
    PrimaModeration,
    /// Any non-finite constraint residual makes the aggregate positive infinity.
    NonFiniteToInfinity,
    /// Other handling, described with its order relative to aggregation.
    Custom(String),
}

/// A constraint-violation value together with its numerical meaning.
///
/// Solvers supply already computed values without additional evaluations.
/// Availability describes evaluation, not numerical validity: zero is an
/// evaluated zero violation, while NaN and infinity remain available records.
/// Consumers must inspect the measure, scaling, and transformation before
/// interpreting a value or choosing an application threshold. Reports do not
/// change solver convergence or establish constrained optimality.
///
/// Construct reports with [`Self::new`] and optional builders. Fields remain
/// public for reading and assignment; destructuring must include `..`.
/// Future optional metadata must default to behavior compatible with these
/// constructors throughout Basin 2.x. Serialization compatibility is separate
/// from this source-compatibility contract.
///
/// ```
/// use basin::{SelectedState, ViolationMeasure, ViolationReport};
/// let mut state = SelectedState::new(vec![2.0]);
/// state.replace(vec![2.0], 4.0, ViolationReport::new(1.0, ViolationMeasure::L1));
/// let (_, _, violation) = state.current().unwrap();
/// assert_eq!(violation.value, 1.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ViolationReport<F: Scalar = f64> {
    /// The solver's published aggregate, without conversion or validation.
    pub value: F,
    /// How componentwise violations are aggregated.
    pub measure: ViolationMeasure,
    /// Additional solver-side residual scaling.
    pub scaling: ViolationScaling,
    /// Numerical handling used to compute the value.
    pub transformation: ViolationTransformation,
}

impl<F: Scalar> ViolationReport<F> {
    /// Describe an unscaled aggregate with identity transformation.
    ///
    /// This records the supplied value as is, including NaN and infinity.
    pub fn new(value: F, measure: ViolationMeasure) -> Self {
        Self {
            value,
            measure,
            scaling: ViolationScaling::Unscaled,
            transformation: ViolationTransformation::Identity,
        }
    }

    /// Describe scaling already applied when computing the value.
    pub fn with_scaling(mut self, scaling: ViolationScaling) -> Self {
        self.scaling = scaling;
        self
    }

    /// Describe numerical handling already applied when computing the value.
    pub fn with_transformation(
        mut self,
        transformation: ViolationTransformation,
    ) -> Self {
        self.transformation = transformation;
        self
    }
}
