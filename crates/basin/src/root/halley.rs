use crate::core::math::Scalar;

use super::common::{Combined, Separate, Settings, hybrid, root_builders};
use super::{RootError, RootResult};

/// Safeguarded Halley iteration for a scalar equation `f(x) = 0`.
///
/// With `delta = f/f'`, the proposed trial is
/// `x - delta / (1 - delta*f''/(2*f'))`. Use the Halley correction only when
/// its magnitude is less than one, so it cannot reverse the Newton direction.
/// Unusable derivatives or unsafe proposals fall back through Newton and
/// secant interpolation to bisection. After two trials that fail to halve the
/// bracket, the next trial is a bisection.
///
/// Shares [`super::NewtonRoot`]'s function and derivative contracts, caching,
/// typed errors, and convergence criterion. The initial trial is the midpoint
/// unless explicitly supplied. An iteration evaluates one trial, and a small
/// step alone never establishes convergence. Non-finite function values are
/// hard errors; zero or non-finite derivatives disable the affected proposal.
///
/// # Backends
///
/// Scalar `f64` and `f32`; no linear-algebra backend or optional feature.
///
/// # Example
///
/// ```
/// use basin::HalleyRoot;
/// use std::convert::Infallible;
/// let root = HalleyRoot::new(0.0, 2.0)
///     .solve_combined(|x| Ok::<_, Infallible>((x * x - 2.0, 2.0 * x, 2.0)))
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
pub struct HalleyRoot<F: Scalar = f64> {
    settings: Settings<F>,
}

impl<F: Scalar> HalleyRoot<F> {
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

    /// Solve with separate value, first-derivative, and second-derivative callbacks.
    pub fn solve<C, D, DD, E>(
        &self,
        function: C,
        derivative: D,
        second_derivative: DD,
    ) -> Result<RootResult<F>, RootError<E, F>>
    where
        C: FnMut(F) -> Result<F, E>,
        D: FnMut(F) -> Result<F, E>,
        DD: FnMut(F) -> Result<F, E>,
    {
        hybrid(
            &self.settings,
            Separate {
                function,
                derivative,
                second: Some(second_derivative),
            },
            true,
        )
    }

    /// Solve with a callback returning `(f(x), f'(x), f''(x))` in one invocation.
    pub fn solve_combined<C, E>(
        &self,
        mut function: C,
    ) -> Result<RootResult<F>, RootError<E, F>>
    where
        C: FnMut(F) -> Result<(F, F, F), E>,
    {
        hybrid(
            &self.settings,
            Combined(|x| function(x).map(|(v, d, dd)| (v, d, Some(dd)))),
            true,
        )
    }
}
