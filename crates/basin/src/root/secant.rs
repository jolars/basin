use crate::core::math::Scalar;

use super::common::{Settings, ValueOnly, hybrid, root_builders};
use super::{RootError, RootResult};

/// Safeguarded secant iteration for a scalar equation `f(x) = 0`.
///
/// The secant through the two most recently evaluated points proposes each
/// trial. Non-finite, out-of-bracket, and non-advancing proposals fall back to
/// bisection. After two trials that fail to halve the bracket, the next trial
/// is a bisection. This retains a sign-changing interval even when ordinary
/// secant iteration would diverge.
///
/// The callback must be deterministic, continuous on the interval, and finite
/// at every evaluated point. Endpoint values must have opposite signs unless
/// one is exactly zero. Convergence means an exact zero or a bracket width
/// at most `absolute + relative * abs(best_endpoint)`, not a small step.
/// The returned estimate is the evaluated endpoint with the smaller absolute
/// residual. Each iteration evaluates one trial. Limits return a clean
/// [`RootResult`]; invalid inputs and callback failures return [`RootError`].
///
/// # Backends
///
/// Scalar `f64` and `f32`; no linear-algebra backend or optional feature.
///
/// # Example
///
/// ```
/// use basin::SecantRoot;
/// use std::convert::Infallible;
/// let root = SecantRoot::new(0.0, 2.0)
///     .solve(|x| Ok::<_, Infallible>(x * x - 2.0)).unwrap();
/// assert!(root.converged());
/// assert!((root.root() - 2.0_f64.sqrt()).abs() < 1e-10);
/// ```
///
/// # References
///
/// SciPy 1.16.3, `scipy.optimize.newton`, secant update. Basin adds explicit
/// bracketing and bisection safeguards and uses bracket-based termination.
/// <https://docs.scipy.org/doc/scipy-1.16.3/reference/generated/scipy.optimize.newton.html>
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SecantRoot<F: Scalar = f64> {
    settings: Settings<F>,
}

impl<F: Scalar> SecantRoot<F> {
    /// Create a bracketed solver with absolute tolerance `1e-12`, relative
    /// tolerance `4ε_F`, and a limit of 100 trial steps.
    /// Interval validity is checked by [`solve`](Self::solve).
    pub fn new(lower: F, upper: F) -> Self {
        Self {
            settings: Settings::new(lower, upper),
        }
    }

    root_builders!();

    /// Solve with a fallible callback, starting afresh on every call.
    pub fn solve<C, E>(
        &self,
        function: C,
    ) -> Result<RootResult<F>, RootError<E, F>>
    where
        C: FnMut(F) -> Result<F, E>,
    {
        hybrid(&self.settings, ValueOnly(function), false)
    }
}
