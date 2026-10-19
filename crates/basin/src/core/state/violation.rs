//! Descriptions of solver-published constraint violations.

use crate::core::math::Scalar;
use std::sync::Arc;

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
/// Construct reports with [`Self::new`] and optional builders. The `value`
/// field remains public; accessors borrow the measure, scaling, and
/// transformation. Built-in metadata uses static storage, so constructing and
/// cloning native reports allocates nothing. Custom metadata is shared across
/// clones and uses copy-on-write when a builder changes it.
///
/// Destructuring must include `..`.
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
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub struct ViolationReport<F: Scalar = f64> {
    /// The solver's published aggregate, without conversion or validation.
    pub value: F,
    metadata: MetadataRef,
}

impl<F: Scalar> std::fmt::Debug for ViolationReport<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ViolationReport")
            .field("value", &self.value)
            .field("measure", self.measure())
            .field("scaling", self.scaling())
            .field("transformation", self.transformation())
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Metadata {
    measure: ViolationMeasure,
    scaling: ViolationScaling,
    transformation: ViolationTransformation,
}

impl Metadata {
    const fn native(
        measure: ViolationMeasure,
        transformation: ViolationTransformation,
    ) -> Self {
        Self {
            measure,
            scaling: ViolationScaling::Unscaled,
            transformation,
        }
    }
}

// Native publications share descriptors rather than copying unused String
// storage through every current, selected, and pending record.
static NATIVE_METADATA: [[Metadata; 3]; 4] = [
    [
        Metadata::native(
            ViolationMeasure::L1,
            ViolationTransformation::Identity,
        ),
        Metadata::native(
            ViolationMeasure::L1,
            ViolationTransformation::PrimaModeration,
        ),
        Metadata::native(
            ViolationMeasure::L1,
            ViolationTransformation::NonFiniteToInfinity,
        ),
    ],
    [
        Metadata::native(
            ViolationMeasure::L2,
            ViolationTransformation::Identity,
        ),
        Metadata::native(
            ViolationMeasure::L2,
            ViolationTransformation::PrimaModeration,
        ),
        Metadata::native(
            ViolationMeasure::L2,
            ViolationTransformation::NonFiniteToInfinity,
        ),
    ],
    [
        Metadata::native(
            ViolationMeasure::SquaredL2,
            ViolationTransformation::Identity,
        ),
        Metadata::native(
            ViolationMeasure::SquaredL2,
            ViolationTransformation::PrimaModeration,
        ),
        Metadata::native(
            ViolationMeasure::SquaredL2,
            ViolationTransformation::NonFiniteToInfinity,
        ),
    ],
    [
        Metadata::native(
            ViolationMeasure::LInf,
            ViolationTransformation::Identity,
        ),
        Metadata::native(
            ViolationMeasure::LInf,
            ViolationTransformation::PrimaModeration,
        ),
        Metadata::native(
            ViolationMeasure::LInf,
            ViolationTransformation::NonFiniteToInfinity,
        ),
    ],
];

#[derive(Clone, Debug, PartialEq)]
enum MetadataRef {
    Native { measure: u8, transformation: u8 },
    Shared(Arc<Metadata>),
}

impl MetadataRef {
    #[inline]
    fn native(metadata: &Metadata) -> Option<Self> {
        if !matches!(metadata.scaling, ViolationScaling::Unscaled) {
            return None;
        }
        let measure = match metadata.measure {
            ViolationMeasure::L1 => 0,
            ViolationMeasure::L2 => 1,
            ViolationMeasure::SquaredL2 => 2,
            ViolationMeasure::LInf => 3,
            ViolationMeasure::Custom(_) => return None,
        };
        let transformation = match metadata.transformation {
            ViolationTransformation::Identity => 0,
            ViolationTransformation::PrimaModeration => 1,
            ViolationTransformation::NonFiniteToInfinity => 2,
            ViolationTransformation::Custom(_) => return None,
        };
        Some(Self::Native {
            measure,
            transformation,
        })
    }

    #[inline]
    fn new(metadata: Metadata) -> Self {
        match Self::native(&metadata) {
            Some(native) => native,
            None => Self::Shared(Arc::new(metadata)),
        }
    }

    #[inline]
    fn get(&self) -> &Metadata {
        match self {
            Self::Native {
                measure,
                transformation,
            } => &NATIVE_METADATA[*measure as usize][*transformation as usize],
            Self::Shared(metadata) => metadata,
        }
    }

    #[cold]
    fn update(self, update: impl FnOnce(&mut Metadata)) -> Self {
        match self {
            native @ Self::Native { .. } => {
                let mut metadata = native.get().clone();
                update(&mut metadata);
                Self::new(metadata)
            }
            Self::Shared(mut metadata) => {
                update(Arc::make_mut(&mut metadata));
                match Self::native(&metadata) {
                    Some(native) => native,
                    None => Self::Shared(metadata),
                }
            }
        }
    }
}

impl<F: Scalar> ViolationReport<F> {
    // SLSQP publishes this fixed descriptor every iteration. Construct it
    // directly so the String-capable public builders stay out of that loop.
    pub(crate) const fn unscaled_l1_nonfinite(value: F) -> Self {
        Self {
            value,
            metadata: MetadataRef::Native {
                measure: 0,
                transformation: 2,
            },
        }
    }

    /// Describe an unscaled aggregate with identity transformation.
    ///
    /// This records the supplied value as is, including NaN and infinity.
    #[inline]
    pub fn new(value: F, measure: ViolationMeasure) -> Self {
        let measure = match measure {
            ViolationMeasure::L1 => 0,
            ViolationMeasure::L2 => 1,
            ViolationMeasure::SquaredL2 => 2,
            ViolationMeasure::LInf => 3,
            ViolationMeasure::Custom(_) => {
                return Self {
                    value,
                    metadata: MetadataRef::new(Metadata::native(
                        measure,
                        ViolationTransformation::Identity,
                    )),
                };
            }
        };
        Self {
            value,
            metadata: MetadataRef::Native {
                measure,
                transformation: 0,
            },
        }
    }

    /// Borrow the aggregation used to compute the value.
    #[inline]
    pub fn measure(&self) -> &ViolationMeasure {
        &self.metadata.get().measure
    }

    /// Borrow the additional solver-side residual scaling.
    #[inline]
    pub fn scaling(&self) -> &ViolationScaling {
        &self.metadata.get().scaling
    }

    /// Borrow the numerical handling used to compute the value.
    #[inline]
    pub fn transformation(&self) -> &ViolationTransformation {
        &self.metadata.get().transformation
    }

    /// Describe the aggregation already applied when computing the value.
    #[inline]
    pub fn with_measure(mut self, measure: ViolationMeasure) -> Self {
        if let MetadataRef::Native {
            measure: native, ..
        } = &mut self.metadata
        {
            match measure {
                ViolationMeasure::L1 => *native = 0,
                ViolationMeasure::L2 => *native = 1,
                ViolationMeasure::SquaredL2 => *native = 2,
                ViolationMeasure::LInf => *native = 3,
                ViolationMeasure::Custom(_) => {
                    self.metadata = self
                        .metadata
                        .update(|metadata| metadata.measure = measure);
                }
            }
            return self;
        }
        self.metadata =
            self.metadata.update(|metadata| metadata.measure = measure);
        self
    }

    /// Describe scaling already applied when computing the value.
    #[inline]
    pub fn with_scaling(mut self, scaling: ViolationScaling) -> Self {
        if matches!(self.metadata, MetadataRef::Native { .. })
            && matches!(scaling, ViolationScaling::Unscaled)
        {
            return self;
        }
        self.metadata =
            self.metadata.update(|metadata| metadata.scaling = scaling);
        self
    }

    /// Describe numerical handling already applied when computing the value.
    #[inline]
    pub fn with_transformation(
        mut self,
        transformation: ViolationTransformation,
    ) -> Self {
        if let MetadataRef::Native {
            transformation: native,
            ..
        } = &mut self.metadata
        {
            match transformation {
                ViolationTransformation::Identity => *native = 0,
                ViolationTransformation::PrimaModeration => *native = 1,
                ViolationTransformation::NonFiniteToInfinity => *native = 2,
                ViolationTransformation::Custom(_) => {
                    self.metadata = self.metadata.update(|metadata| {
                        metadata.transformation = transformation
                    });
                }
            }
            return self;
        }
        self.metadata = self
            .metadata
            .update(|metadata| metadata.transformation = transformation);
        self
    }
}

#[cfg(feature = "serde")]
impl<F: Scalar + serde::Serialize> serde::Serialize for ViolationReport<F> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("ViolationReport", 4)?;
        record.serialize_field("value", &self.value)?;
        record.serialize_field("measure", self.measure())?;
        record.serialize_field("scaling", self.scaling())?;
        record.serialize_field("transformation", self.transformation())?;
        record.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, F: Scalar + serde::Deserialize<'de>> serde::Deserialize<'de>
    for ViolationReport<F>
{
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "ViolationReport")]
        struct Stored<F> {
            value: F,
            measure: ViolationMeasure,
            scaling: ViolationScaling,
            transformation: ViolationTransformation,
        }
        let stored = Stored::<F>::deserialize(deserializer)?;
        Ok(Self {
            value: stored.value,
            metadata: MetadataRef::new(Metadata {
                measure: stored.measure,
                scaling: stored.scaling,
                transformation: stored.transformation,
            }),
        })
    }
}
