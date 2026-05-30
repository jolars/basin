//! Line searches: produce a step size `α` along a caller-supplied descent
//! direction. Used by first-order solvers (gradient descent, BFGS).

/// Backtracking line search (Armijo-only).
pub mod backtracking;
/// Moré–Thuente line search (MINPACK-2 `dcsrch` + `dcstep`).
pub mod more_thuente;
/// Strong-Wolfe line search (Nocedal & Wright algorithms 3.5/3.6).
pub mod wolfe;

pub use backtracking::Backtracking;
pub use more_thuente::MoreThuente;
pub use wolfe::Wolfe;

use crate::core::math::Scalar;
use crate::core::problem::Problem;

/// Compute a step size `α` along a caller-supplied descent direction `d`.
///
/// Convention: `direction` is a *descent* direction (`gᵀd < 0`); the caller
/// applies `x_new = x + α d`. Solvers that descend along `−∇f` (e.g. plain
/// gradient descent) compute `d = −∇f` themselves and pass it in.
///
/// # Eval counting
///
/// Line searches receive `&mut Problem<P>` so every probe (`problem.cost(x)`,
/// `problem.cost_and_gradient(x)`) bumps the wrapper's
/// [`EvalCounts`](crate::core::problem::EvalCounts) automatically. There is
/// no count returned out-of-band — callers read counts off the wrapper if
/// they need them (the [`Executor`](crate::Executor) mirrors them onto the
/// state after the enclosing
/// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)).
///
/// # Error type
///
/// `Error` is the hard-abort error the line search propagates — concrete
/// impls set `type Error = P::Error;` (with `P: CostFunction`) so the
/// user's typed problem error bubbles untouched through the solver out of
/// [`Executor::run`](crate::Executor::run). See the
/// [`problem`](crate::core::problem) module docs for the soft-reject /
/// hard-abort split.
///
/// `F` defaults to `f64` so legacy `LineSearch<P, V>` bounds on
/// gradient-based solvers (still f64-only pending the linalg-tier
/// migration) keep resolving unchanged.
pub trait LineSearch<P, V, F = f64> {
    /// Hard-abort error type, mirroring the underlying problem's `Error`.
    type Error;

    /// Returns the chosen step size `α`. Counts are tracked on the
    /// wrapper via every `problem.cost` / `problem.cost_and_gradient`
    /// probe inside the search. Returns `Err` if any inner problem call
    /// hard-aborts.
    fn next(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<F, Self::Error>;
}

/// Constant step size — returns the wrapped `α` regardless of input.
/// Useful when the caller already knows a good fixed step.
pub struct Constant<F = f64>(pub F);

impl<F: Scalar> Constant<F> {
    /// Constant step of size `alpha`.
    pub fn new(alpha: F) -> Self {
        Self(alpha)
    }
}

impl<P, V, F> LineSearch<P, V, F> for Constant<F>
where
    P: crate::core::problem::CostFunction,
    F: Scalar,
{
    // `Constant` makes no problem calls and so could declare any error
    // type, but solver bounds expect `L::Error = P::Error`; matching here
    // means callers never need a conversion glue layer.
    type Error = P::Error;

    fn next(
        &mut self,
        _problem: &mut Problem<P>,
        _param: &V,
        _cost: F,
        _gradient: &V,
        _direction: &V,
    ) -> Result<F, Self::Error> {
        Ok(self.0)
    }
}
