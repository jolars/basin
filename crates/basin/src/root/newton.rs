use crate::core::math::Scalar;

use super::common::{Combined, Separate, Settings, hybrid, root_builders};
use super::{RootError, RootResult};

/// Safeguarded Newton iteration for a scalar equation `f(x) = 0`.
///
/// Starts at the midpoint or [`with_initial_guess`](Self::with_initial_guess),
/// then proposes `x - f(x)/f'(x)`. A zero or non-finite derivative disables
/// Newton interpolation. Unsafe proposals fall back to secant interpolation
/// and then bisection. After two trials that fail to halve the bracket, the
/// next trial is a bisection.
///
/// The function must satisfy [`super::SecantRoot`]'s continuity and endpoint
/// sign contract. Derivatives must describe that function wherever they are
/// finite. An exact zero or bracket width at most
/// `absolute + relative * abs(best_endpoint)` establishes convergence; a
/// small Newton step alone does not. Each iteration evaluates one trial.
/// Limits return a clean [`RootResult`]; invalid inputs and callback failures
/// return [`RootError`]. Each solve starts with fresh history.
///
/// Separate callbacks compute derivatives only when an interpolation needs
/// them. [`solve_combined`](Self::solve_combined) computes a value and its
/// derivative together, including at endpoints, and caches the result.
/// Both interfaces preserve typed callback errors. Non-finite function values
/// are errors; non-finite derivatives trigger the safeguards instead.
///
/// # Backends
///
/// Scalar `f64` and `f32`; no linear-algebra backend or optional feature.
///
/// # Example
///
/// ```
/// use basin::NewtonRoot;
/// use std::convert::Infallible;
/// let root = NewtonRoot::new(0.0, 2.0)
///     .solve_combined(|x| Ok::<_, Infallible>((x * x - 2.0, 2.0 * x)))
///     .unwrap();
/// assert!(root.converged());
/// assert!((root.root() - 2.0_f64.sqrt()).abs() < 1e-10);
/// ```
///
/// # References
///
/// Boost.Math 1.89.0, "Root Finding With Derivatives: Newton-Raphson, Halley
/// & Schroeder." Basin adds secant fallback, a two-trial contraction safeguard,
/// and bracket-based termination.
/// <https://www.boost.org/doc/libs/1_89_0/libs/math/doc/html/math_toolkit/roots_deriv.html>
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewtonRoot<F: Scalar = f64> {
    settings: Settings<F>,
}

impl<F: Scalar> NewtonRoot<F> {
    /// Create a bracketed solver with absolute tolerance `1e-12`, relative
    /// tolerance `4ε_F`, and a limit of 100 trial steps.
    pub fn new(lower: F, upper: F) -> Self {
        Self {
            settings: Settings::new(lower, upper),
        }
    }

    root_builders!();

    /// Set the first trial point instead of the midpoint.
    ///
    /// Solving rejects non-finite guesses and guesses not strictly inside the
    /// interval before invoking any callback, even with a zero iteration limit.
    pub fn with_initial_guess(mut self, x: F) -> Self {
        self.settings.guess = Some(x);
        self
    }

    /// Solve with separate fallible function and derivative callbacks.
    pub fn solve<C, D, E>(
        &self,
        function: C,
        derivative: D,
    ) -> Result<RootResult<F>, RootError<E, F>>
    where
        C: FnMut(F) -> Result<F, E>,
        D: FnMut(F) -> Result<F, E>,
    {
        hybrid(
            &self.settings,
            Separate {
                function,
                derivative,
                second: None::<fn(F) -> Result<F, E>>,
            },
            true,
        )
    }

    /// Solve with a callback returning `(f(x), f'(x))` in one invocation.
    pub fn solve_combined<C, E>(
        &self,
        mut function: C,
    ) -> Result<RootResult<F>, RootError<E, F>>
    where
        C: FnMut(F) -> Result<(F, F), E>,
    {
        hybrid(
            &self.settings,
            Combined(|x| function(x).map(|(v, d)| (v, d, None))),
            true,
        )
    }
}
