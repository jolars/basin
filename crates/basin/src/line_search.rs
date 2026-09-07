//! Line searches: produce a step size `α` along a caller-supplied descent
//! direction. Used by first-order solvers (gradient descent, BFGS).

/// Backtracking line search (Armijo-only).
pub mod backtracking;
/// Hager–Zhang line search with approximate-Wolfe safeguards.
pub mod hager_zhang;
/// Moré–Thuente line search (MINPACK-2 `dcsrch` + `dcstep`).
pub mod more_thuente;
/// Strong-Wolfe line search (Nocedal & Wright algorithms 3.5/3.6).
pub mod wolfe;

pub use backtracking::Backtracking;
pub use hager_zhang::HagerZhang;
pub use more_thuente::MoreThuente;
pub use wolfe::Wolfe;

use crate::core::math::Scalar;
use crate::core::problem::Problem;

/// Outcome of a line-search attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LineSearchOutcome<F> {
    /// The line search selected this step size.
    Step(F),
    /// The line search could not produce a usable step.
    Failed,
}

/// Objective and gradient retained at a line search's accepted point.
///
/// A line search returns this alongside [`LineSearchOutcome::Step`] when it has
/// already evaluated the selected point. Solvers can then adopt the point and
/// its values without repeating an expensive fused evaluation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LineSearchEvaluation<V, F> {
    /// Accepted parameter, equal to `x + alpha * direction`.
    pub param: V,
    /// Objective value at [`param`](Self::param).
    pub cost: F,
    /// Gradient at [`param`](Self::param).
    pub gradient: V,
}

impl<V, F> LineSearchEvaluation<V, F> {
    /// Construct an evaluation at an accepted line-search point.
    pub fn new(param: V, cost: F, gradient: V) -> Self {
        Self {
            param,
            cost,
            gradient,
        }
    }
}

/// Outcome of a line search, optionally carrying its accepted evaluation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LineSearchResult<V, F> {
    /// Whether the search selected a step or failed softly.
    pub outcome: LineSearchOutcome<F>,
    /// Objective and gradient at the selected step, when retained.
    pub evaluation: Option<LineSearchEvaluation<V, F>>,
}

impl<V, F> LineSearchResult<V, F> {
    /// Construct a result without a retained evaluation.
    pub fn new(outcome: LineSearchOutcome<F>) -> Self {
        Self {
            outcome,
            evaluation: None,
        }
    }

    /// Construct a selected step with its retained evaluation.
    pub fn with_evaluation(step: F, param: V, cost: F, gradient: V) -> Self {
        Self {
            outcome: LineSearchOutcome::Step(step),
            evaluation: Some(LineSearchEvaluation::new(param, cost, gradient)),
        }
    }
}

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
/// no count returned out-of-band: callers read counts off the wrapper if
/// they need them (the [`Executor`](crate::Executor) mirrors them onto the
/// state after the enclosing
/// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)).
///
/// # Error type
///
/// `Error` is the hard-abort error the line search propagates; concrete
/// impls set `type Error = P::Error;` (with `P: CostFunction`) so the
/// user's typed problem error bubbles untouched through the solver out of
/// [`Executor::run`](crate::Executor::run). See the
/// [`problem`](crate::core::problem) module docs for the soft-reject and
/// hard-abort split.
///
/// `F` defaults to `f64` so legacy `LineSearch<P, V>` bounds on
/// gradient-based solvers (still f64-only pending the linalg-tier
/// migration) keep resolving unchanged.
pub trait LineSearch<P, V, F = f64> {
    /// Hard-abort error type, mirroring the underlying problem's `Error`.
    type Error;

    /// Returns the chosen step size `α`. Counts are tracked on the
    /// wrapper via every `problem.cost`/`problem.cost_and_gradient`
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

    /// Returns the step together with a distinguishable soft-failure outcome.
    ///
    /// The default wraps [`next`](Self::next) in
    /// [`LineSearchOutcome::Step`] for compatibility with existing line-search
    /// implementations. Implementations that can exhaust without a usable step
    /// should override this method and return [`LineSearchOutcome::Failed`].
    fn next_with_outcome(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<LineSearchOutcome<F>, Self::Error> {
        self.next(problem, param, cost, gradient, direction)
            .map(LineSearchOutcome::Step)
    }

    /// Returns the outcome and, when available, the evaluation at its step.
    ///
    /// The default preserves compatibility with line searches that return only
    /// a step. Implementations that evaluate both the objective and gradient at
    /// their selected point can override this method to expose those values and
    /// prevent the calling solver from evaluating the point again.
    ///
    /// When [`LineSearchResult::evaluation`] is [`Some`], the outcome must be
    /// [`LineSearchOutcome::Step`], and the evaluation's parameter must equal
    /// `param + step * direction`.
    fn next_with_evaluation(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<LineSearchResult<V, F>, Self::Error> {
        self.next_with_outcome(problem, param, cost, gradient, direction)
            .map(LineSearchResult::new)
    }
}

/// Constant step size: returns the wrapped `α` regardless of input.
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
