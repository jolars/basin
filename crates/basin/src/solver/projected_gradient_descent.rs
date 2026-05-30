use crate::core::constraint::BoxConstraints;
use crate::core::math::{ClampInPlace, NegInPlace, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;
use crate::line_search::{Constant, LineSearch};

/// Projected gradient descent for box-constrained problems.
///
/// Steepest-descent step along `−∇f` followed by an element-wise
/// projection back into `[lower, upper]`. The first n-D constrained
/// solver in basin and the smallest vehicle for the
/// [`BoxConstraints`] trait — handing this solver an unconstrained
/// problem is a compile error per tenet 4.
///
/// # Algorithm
///
/// At [`init`](Solver::init) the iterate is projected onto the feasible
/// box once, so an infeasible starting point is silently corrected
/// (and downstream termination criteria see a feasible iterate at iter
/// 0). Each [`next_iter`](Solver::next_iter) then computes
///
/// ```text
/// d   ← −∇f(x)
/// α   ← line_search.next(...)        # on the unconstrained step
/// x   ← π_C(x + α d)                  # project after the step
/// ```
///
/// where `π_C` clamps each component into `[lower, upper]`.
///
/// # Contract
///
/// - **Caller must:** implement [`BoxConstraints`] on the problem with
///   `lower[i] ≤ upper[i]` for every component. Equal bounds are
///   allowed (the corresponding component is pinned).
/// - **Caller must:** pair with a feasible **or** infeasible initial
///   param; an infeasible start is projected at `init`.
/// - **Implementor (this solver) must:** maintain feasibility across
///   iterations — once the loop has run, every iterate the executor
///   sees is in the box.
///
/// The line search runs against the *unconstrained* trial step
/// `f(x + α d)`. If the projection moves the post-step iterate
/// substantially, Armijo guarantees on the unconstrained step do not
/// transfer to `f(π_C(x + α d))`. For tighter guarantees use a small
/// fixed step ([`Constant`]) or wait on a constraint-aware (SPG-style)
/// line search.
///
/// # Termination
///
/// No solver-internal optimality test; the canonical first-order
/// metric is provided as a framework-level criterion,
/// [`ProjectedGradientTolerance`](crate::core::termination::ProjectedGradientTolerance),
/// which captures the bounds at construction so it does not need
/// problem access in `check`. The framework's
/// [`MaxIter`](crate::core::termination::MaxIter),
/// [`CostTolerance`](crate::core::termination::CostTolerance),
/// [`ParamTolerance`](crate::core::termination::ParamTolerance), and
/// [`MaxTime`](crate::core::termination::MaxTime) work on
/// [`BasicState`] for free.
///
/// # Backends
///
/// Backend-generic — works with any `V` implementing
/// [`ScaledAdd<F>`](crate::core::math::ScaledAdd) +
/// [`NegInPlace`] + [`ClampInPlace`] + `Clone`. With the default
/// `F = f64` that covers `Vec<f64>`, `nalgebra::DVector<f64>` (feature
/// `nalgebra`), `ndarray::Array1<f64>` (feature `ndarray`), and
/// `faer::Col<f64>` (feature `faer`). The problem must implement
/// [`BoxConstraints`].
///
/// # Examples
///
/// Box-constrained gradient descent. The bounds live on the problem via
/// [`BoxConstraints`]; the projection keeps every iterate feasible. Here
/// the unconstrained minimum of the shifted sphere is at `(2, 2)`, but the
/// box caps it at `(1, 1)`:
///
/// ```
/// use basin::{BasicState, BoxConstraints, CostFunction, Executor, Gradient, ProjectedGradientDescent};
///
/// struct ShiftedSphere {
///     lower: Vec<f64>,
///     upper: Vec<f64>,
/// }
/// impl CostFunction for ShiftedSphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
///     }
/// }
/// impl Gradient for ShiftedSphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)])
///     }
/// }
/// impl BoxConstraints for ShiftedSphere {
///     fn lower(&self) -> &Vec<f64> { &self.lower }
///     fn upper(&self) -> &Vec<f64> { &self.upper }
/// }
///
/// let problem = ShiftedSphere { lower: vec![-1.0, -1.0], upper: vec![1.0, 1.0] };
/// let result = Executor::new(
///     problem,
///     ProjectedGradientDescent::new(0.1),
///     BasicState::new(vec![0.0, 0.0]),
/// )
/// .max_iter(1_000)
/// .run()
/// .unwrap();
/// assert!((result.param()[0] - 1.0).abs() < 1e-6);
/// assert!((result.param()[1] - 1.0).abs() < 1e-6);
/// ```
pub struct ProjectedGradientDescent<S> {
    line_search: S,
}

impl<F: Scalar> ProjectedGradientDescent<Constant<F>> {
    /// Projected gradient descent with a fixed step size `alpha`.
    /// Equivalent to `with_line_search(Constant(alpha))`. Recommended
    /// default — the line search variant has the caveat documented on
    /// the type.
    pub fn new(alpha: F) -> Self {
        Self {
            line_search: Constant(alpha),
        }
    }
}

impl<S> ProjectedGradientDescent<S> {
    /// Projected gradient descent with an explicit line-search strategy.
    ///
    /// Note: the line search runs against the *unconstrained* trial
    /// step (see the type-level rustdoc). Backtracking is honest only
    /// while the projection isn't active; consider a fixed step in
    /// regimes where many components hit their bounds.
    pub fn with_line_search(line_search: S) -> Self {
        Self { line_search }
    }
}

impl<P, V, F, S> Solver<P, BasicState<V, F>> for ProjectedGradientDescent<S>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V> + BoxConstraints,
    V: ScaledAdd<F> + NegInPlace + ClampInPlace + Clone,
    S: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<BasicState<V, F>, Self::Error> {
        // Project an infeasible start once so iter-0 termination checks
        // see a feasible iterate. Subsequent iterations preserve
        // feasibility by construction.
        state
            .param
            .clamp_in_place(problem.inner().lower(), problem.inner().upper());
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error> {
        let grad = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let prev_cost = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");
        let mut direction = grad.clone();
        direction.neg_in_place();
        let alpha = self
            .line_search
            .next(problem, &state.param, prev_cost, &grad, &direction)?;
        state.param.scaled_add(alpha, &direction);
        state
            .param
            .clamp_in_place(problem.inner().lower(), problem.inner().upper());
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok((state, None))
    }
}
