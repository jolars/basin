use std::fmt;

use crate::core::math::Scalar;

/// Failure to initialize or evaluate a secant, Newton, Halley, or TOMS 748 solve.
///
/// Iteration limits return a [`super::RootResult`] instead. Zero or non-finite
/// derivatives disable interpolation and are not hard errors.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum RootError<E, F: Scalar = f64> {
    /// A value or derivative callback failed.
    Evaluation(E),
    /// Endpoints must be finite, strictly ordered, and have finite width.
    InvalidInterval {
        /// Supplied lower endpoint.
        lower: F,
        /// Supplied upper endpoint.
        upper: F,
    },
    /// An initial guess must be finite and strictly inside the interval.
    InvalidInitialGuess {
        /// Supplied initial guess.
        x: F,
    },
    /// Endpoint values have the same nonzero sign.
    NotBracketed {
        /// Lower endpoint.
        lower: F,
        /// Upper endpoint.
        upper: F,
        /// Value at the lower endpoint.
        f_lower: F,
        /// Value at the upper endpoint.
        f_upper: F,
    },
    /// The function returned a non-finite value.
    NonFiniteValue {
        /// Evaluated point.
        x: F,
        /// Returned value.
        value: F,
    },
}

impl<E: fmt::Display, F: Scalar> fmt::Display for RootError<E, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation(error) => {
                write!(f, "root callback failed: {error}")
            }
            Self::InvalidInterval { lower, upper } => write!(
                f,
                "root interval must be finite and ordered with finite width, got [{lower:?}, {upper:?}]"
            ),
            Self::InvalidInitialGuess { x } => write!(
                f,
                "root initial guess must be finite and strictly inside the interval, got {x:?}"
            ),
            Self::NotBracketed {
                lower,
                upper,
                f_lower,
                f_upper,
            } => write!(
                f,
                "root is not bracketed on [{lower:?}, {upper:?}]: values are {f_lower:?} and {f_upper:?}"
            ),
            Self::NonFiniteValue { x, value } => write!(
                f,
                "root function returned non-finite value {value:?} at {x:?}"
            ),
        }
    }
}

impl<E: std::error::Error + 'static, F: Scalar> std::error::Error
    for RootError<E, F>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation(error) => Some(error),
            _ => None,
        }
    }
}
