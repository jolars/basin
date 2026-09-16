//! Automatic scalar root and minimum bracketing.
//!
//! Bracketing is an explicit preliminary search, with its own limits and
//! evaluation counts. [`RootBracketer`] finds distinct endpoints suitable for
//! [`crate::BrentRoot`] or the other direct root solvers. [`MinimumBracketer`]
//! finds three points suitable for initializing a scalar minimizer. Solvers
//! reevaluate the supplied points; add the two stages' counts when reporting
//! total work. A discovered search interval is not a global-optimality claim.

use std::fmt;

use crate::core::math::Scalar;

/// Automatic minimum bracketing.
pub mod minimum;
/// Automatic root bracketing.
pub mod root;

pub use minimum::{MinimumBracketResult, MinimumBracketer};
pub use root::{RootBracketResult, RootBracketer};

/// Why an automatic bracket search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BracketTerminationReason {
    /// The sampled points meet the requested bracket condition.
    Bracketed,
    /// The configured expansion limit was reached.
    MaxIter,
    /// All applicable directions reached their finite search limits.
    ///
    /// A minimum at a boundary is not an interior minimum bracket.
    BoundsReached,
    /// Finite, distinct trial points could no longer be generated.
    NoProgress,
}

/// Invalid initial geometry or a failed bracket-search evaluation.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum BracketError<E, F: Scalar = f64> {
    /// The supplied points must be finite, strictly ordered, inside the
    /// search limits, and have finite total width.
    InvalidInitialPoints,
    /// The callback returned an application error.
    Evaluation(E),
    /// The function returned a non-finite value.
    NonFiniteValue {
        /// Evaluated point.
        x: F,
        /// Returned value.
        value: F,
    },
}

impl<E: fmt::Display, F: Scalar> fmt::Display for BracketError<E, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInitialPoints => f.write_str("bracket seeds must be finite, strictly ordered, inside the search limits, and have finite width"),
            Self::Evaluation(error) => write!(f, "bracket callback failed: {error}"),
            Self::NonFiniteValue { x, value } => write!(f, "bracket function returned non-finite value {value:?} at {x:?}"),
        }
    }
}

impl<E: std::error::Error + 'static, F: Scalar> std::error::Error
    for BracketError<E, F>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Search<F> {
    lower: Option<F>,
    upper: Option<F>,
    factor: F,
    max_iter: u64,
}

impl<F: Scalar> Default for Search<F> {
    fn default() -> Self {
        Self {
            lower: None,
            upper: None,
            factor: F::from_f64(2.0).unwrap(),
            max_iter: 1000,
        }
    }
}

impl<F: Scalar> Search<F> {
    fn validate<E>(&self, points: &[F]) -> Result<(), BracketError<E, F>> {
        if points.iter().any(|x| !x.is_finite())
            || points.windows(2).any(|p| p[0] >= p[1])
            || !(points[points.len() - 1] - points[0]).is_finite()
            || self.lower.is_some_and(|lower| points[0] < lower)
            || self
                .upper
                .is_some_and(|upper| points[points.len() - 1] > upper)
        {
            Err(BracketError::InvalidInitialPoints)
        } else {
            Ok(())
        }
    }

    fn next(
        &self,
        current: F,
        anchor: F,
        distance: &mut F,
        limit: Option<F>,
    ) -> Option<F> {
        let candidate = if let Some(limit) = limit {
            if current == limit {
                return None;
            }
            let gap = limit - current;
            let candidate = if gap.is_finite() {
                limit - gap / self.factor
            } else {
                // A convex combination avoids overflowing the gap between
                // finite coordinates of opposite sign.
                let weight = F::one() / self.factor;
                (F::one() - weight) * limit + weight * current
            };
            if candidate == current {
                limit
            } else {
                candidate
            }
        } else {
            *distance = *distance * self.factor;
            anchor + *distance
        };
        if candidate.is_finite()
            && candidate != current
            && (candidate - current).is_finite()
        {
            Some(candidate)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sample<F> {
    x: F,
    value: F,
}

fn evaluate<F: Scalar, E>(
    function: &mut impl FnMut(F) -> Result<F, E>,
    x: F,
    evaluations: &mut u64,
) -> Result<Sample<F>, BracketError<E, F>> {
    *evaluations += 1;
    let value = function(x).map_err(BracketError::Evaluation)?;
    if !value.is_finite() {
        return Err(BracketError::NonFiniteValue { x, value });
    }
    Ok(Sample { x, value })
}

macro_rules! search_builders {
    () => {
        /// Set the finite lower search limit, inclusive.
        ///
        /// # Panics
        /// Panics if the limit is non-finite. Seed geometry is checked when
        /// [`bracket`](Self::bracket) is called.
        pub fn with_lower_bound(mut self, value: F) -> Self {
            assert!(value.is_finite(), "lower search limit must be finite");
            self.search.lower = Some(value);
            self
        }

        /// Set the finite upper search limit, inclusive.
        ///
        /// # Panics
        /// Panics if the limit is non-finite. Seed geometry is checked when
        /// [`bracket`](Self::bracket) is called.
        pub fn with_upper_bound(mut self, value: F) -> Self {
            assert!(value.is_finite(), "upper search limit must be finite");
            self.search.upper = Some(value);
            self
        }

        /// Set the geometric expansion factor (default `2`).
        ///
        /// # Panics
        /// Panics unless the factor is finite and greater than one.
        pub fn with_growth_factor(mut self, value: F) -> Self {
            assert!(
                value.is_finite() && value > F::one(),
                "growth factor must be finite and greater than one"
            );
            self.search.factor = value;
            self
        }

        /// Set the maximum number of expansion rounds (default `1000`).
        /// Zero still evaluates the initial points and checks their bracket.
        pub fn with_max_iter(mut self, value: u64) -> Self {
            self.search.max_iter = value;
            self
        }
    };
}
use search_builders;

macro_rules! result_accessors {
    () => {
        /// Number of expansion rounds that evaluated at least one new point.
        pub fn iterations(&self) -> u64 {
            self.iterations
        }

        /// Number of callback invocations, including all initial points.
        pub fn function_evals(&self) -> u64 {
            self.function_evals
        }

        /// Why the search stopped.
        pub fn reason(&self) -> BracketTerminationReason {
            self.reason
        }

        /// Whether the sampled points form a valid bracket.
        pub fn bracketed(&self) -> bool {
            self.reason == BracketTerminationReason::Bracketed
        }
    };
}
use result_accessors;
