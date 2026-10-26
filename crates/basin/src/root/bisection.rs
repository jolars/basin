use crate::core::math::Scalar;

use super::common::{
    Bracket, Counts, Evaluator, Settings, ValueOnly, root_builders,
};
use super::{RootError, RootResult};

/// Bisection for a scalar equation `f(x) = 0`.
///
/// Each iteration evaluates the midpoint and retains the half whose endpoints
/// have opposite signs. The bracket width halves at each step, up to
/// floating-point rounding. An exact zero collapses the bracket immediately.
/// No derivatives or interpolation are used.
///
/// The callback must be deterministic, continuous on the interval, and finite
/// at every evaluated point. Endpoints must be finite and strictly ordered
/// with finite width. Their values must have opposite signs unless one is
/// exactly zero. Use [`crate::RootBracketer`] to discover a bracket first.
///
/// Convergence means an exact zero or a bracket width at most
/// `absolute + relative * abs(best_endpoint)`. The result keeps the evaluated
/// endpoint with the smaller absolute residual and its signed function value.
/// Each iteration evaluates one midpoint. Zero iterations evaluates only the
/// endpoints, stopping after the first if it is an exact root. Exhausting the
/// limit returns a non-converged [`RootResult`]; invalid inputs, non-finite
/// values, and callback failures return [`RootError`].
///
/// # Backends
///
/// Scalar `f64` and `f32`; no linear-algebra backend or optional feature.
///
/// # Example
///
/// ```
/// use basin::BisectionRoot;
/// use std::convert::Infallible;
/// let root = BisectionRoot::new(0.0, 2.0)
///     .solve(|x| Ok::<_, Infallible>(x * x - 2.0)).unwrap();
/// assert!(root.converged());
/// assert!((root.root() - 2.0_f64.sqrt()).abs() < 1e-10);
/// ```
///
/// # References
///
/// GNU Scientific Library Reference Manual, "One Dimensional Root-Finding,"
/// "Root Bracketing Algorithms," bisection.
/// <https://www.gnu.org/software/gsl/doc/html/roots.html#root-bracketing-algorithms>.
/// Independently implemented from the interval-halving rule; Basin returns an
/// evaluated endpoint rather than the unevaluated midpoint of the final interval.
/// Reference checks use SciPy 1.16.3, `scipy.optimize.bisect`. Stopping rules
/// and returned estimates differ, so only initial trajectories and final-root
/// accuracy are compared.
/// <https://docs.scipy.org/doc/scipy-1.16.3/reference/generated/scipy.optimize.bisect.html>
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BisectionRoot<F: Scalar = f64> {
    settings: Settings<F>,
}

impl<F: Scalar> BisectionRoot<F> {
    /// Create a bracketed solver with absolute tolerance `1e-12`, relative
    /// tolerance `4ε_F`, and a limit of 100 midpoint evaluations.
    /// Interval validity is checked by [`solve`](Self::solve).
    pub fn new(lower: F, upper: F) -> Self {
        Self {
            settings: Settings::new(lower, upper),
        }
    }

    root_builders!();

    /// Solve with a fallible value callback, starting afresh on every call.
    pub fn solve<C, E>(
        &self,
        function: C,
    ) -> Result<RootResult<F>, RootError<E, F>>
    where
        C: FnMut(F) -> Result<F, E>,
    {
        let mut eval = ValueOnly(function);
        let mut counts = Counts::default();
        let mut bracket =
            Bracket::initialize(&self.settings, &mut eval, &mut counts)?;
        for iteration in 0..self.settings.max_iter {
            if bracket.converged(&self.settings) {
                return Ok(bracket.result(&self.settings, iteration, counts));
            }
            let point = eval.evaluate(bracket.midpoint(), &mut counts)?;
            bracket.update(point);
        }
        Ok(bracket.result(&self.settings, self.settings.max_iter, counts))
    }
}
