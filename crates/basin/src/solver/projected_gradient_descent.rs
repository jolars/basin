use crate::core::constraint::BoxConstraints;
use crate::core::inner::InitialState;
use crate::core::math::{
    ClampInPlace, NegInPlace, Scalar, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{FirstOrderState, State};
use crate::core::termination::Termination;
use crate::line_search::{Constant, LineSearch, LineSearchOutcome};

/// Projected gradient descent for box-constrained problems.
///
/// Steepest-descent step along `−∇f` followed by an element-wise
/// projection back into `[lower, upper]`. The first n-D constrained
/// solver in basin and the smallest vehicle for the
/// [`BoxConstraints`] trait; handing this solver an unconstrained
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
///   iterations: once the loop has run, every iterate the executor
///   sees is in the box.
///
/// The line search runs against the *unconstrained* trial step
/// `f(x + α d)`. If the projection moves the post-step iterate
/// substantially, Armijo guarantees on the unconstrained step do not
/// transfer to `f(π_C(x + α d))`. For tighter guarantees use a small
/// fixed step ([`Constant`]) or wait on a constraint-aware (SPG-style)
/// line search.
///
/// # Convergence
///
/// Configure [`with_absolute_projected_gradient_tolerance`](Self::with_absolute_projected_gradient_tolerance)
/// for the infinity norm of `x - projection(x - gradient)`, using the current
/// problem's bounds. This optional test is disabled by default; `None`
/// disables it, and zero tests exact stationarity. Observed step and cost
/// checks are also opt-in and combine using OR. Execution budgets belong on
/// the executor.
///
/// Progress uses [`FirstOrderState<V, F>`]. Fresh initialization clears
/// counters and incumbents, resets the line search, and reevaluates the seed.
/// It projects the seed into the box before evaluation. Exact checkpoints
/// retain line-search history. The scalar parameter on
/// `ProjectedGradientDescent<S, F>` lets [`crate::Executor::from_start`] infer
/// first-order progress for either scalar type.
///
/// # Backends
///
/// Backend-generic; works with any `V` implementing
/// [`ScaledAdd<F>`](crate::core::math::ScaledAdd) +
/// [`NegInPlace`] + [`ClampInPlace`] + [`VectorLen`] + `Clone`. Supports
/// `Vec<F>`, `nalgebra::DVector<F>` (feature `nalgebra`),
/// `ndarray::Array1<F>` (feature `ndarray`), and `faer::Col<F>` (feature
/// `faer`) for both `f32` and `f64`. The problem must implement
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
/// use basin::{FirstOrderState, BoxConstraints, CostFunction, Executor, Gradient, ProjectedGradientDescent};
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
///     FirstOrderState::new(vec![0.0, 0.0]),
/// )
/// .max_iter(1_000)
/// .run()
/// .unwrap();
/// assert!((result.param()[0] - 1.0).abs() < 1e-6);
/// assert!((result.param()[1] - 1.0).abs() < 1e-6);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProjectedGradientDescent<S, F: Scalar = f64> {
    scalar: core::marker::PhantomData<F>,
    line_search: S,
}

impl<F: Scalar> ProjectedGradientDescent<Constant<F>, F> {
    /// Projected gradient descent with a fixed step size `alpha`.
    /// Equivalent to `with_line_search(Constant(alpha))`. Recommended
    /// default; the line search variant has the caveat documented on
    /// the type.
    pub fn new(alpha: F) -> Self {
        Self {
            line_search: Constant(alpha),
            scalar: core::marker::PhantomData,
        }
    }
}

impl<S, F: Scalar> ProjectedGradientDescent<S, F> {
    /// Projected gradient descent with an explicit line-search strategy.
    ///
    /// Note: the line search runs against the *unconstrained* trial
    /// step (see the type-level rustdoc). Backtracking is honest only
    /// while the projection isn't active; consider a fixed step in
    /// regimes where many components hit their bounds.
    pub fn with_line_search(line_search: S) -> Self {
        Self {
            line_search,
            scalar: core::marker::PhantomData,
        }
    }
}

impl<S, V, F: Scalar> InitialState<V> for ProjectedGradientDescent<S, F>
where
    V: Clone,
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        FirstOrderState::new(x.clone())
    }
}

impl<P, V, F, S> Solver<P, FirstOrderState<V, F>>
    for ProjectedGradientDescent<S, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + BoxConstraints,
    V: ScaledAdd<F> + NegInPlace + ClampInPlace + Clone + VectorLen,
    S: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        state.reset();
        self.line_search.reset();
        // Project an infeasible start once so iter-0 termination checks
        // see a feasible iterate. Subsequent iterations preserve
        // feasibility by construction.
        state
            .seed_param_mut()
            .clamp_in_place(problem.inner().lower(), problem.inner().upper());
        let (cost, grad) = problem.cost_and_gradient(state.param())?;
        state
            .set_evaluation(cost, grad)
            .expect("gradient dimension differs from parameter");
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<crate::SolverStep<FirstOrderState<V, F>>, Self::Error> {
        let (prev_cost, grad) = state
            .take_evaluation()
            .expect("solver requires initialized progress");
        let mut direction = grad.clone();
        direction.neg_in_place();
        let alpha = match self.line_search.next_with_outcome(
            problem,
            state.param(),
            prev_cost,
            &grad,
            &direction,
        )? {
            LineSearchOutcome::Step(alpha) => alpha,
            LineSearchOutcome::Failed => {
                state
                    .set_evaluation(prev_cost, grad)
                    .expect("gradient dimension differs from parameter");
                return Ok(crate::SolverStep::from((
                    state,
                    Some(Termination::numerical_failure(
                        "Projected gradient-descent line search failed.",
                    )),
                )));
            }
        };
        state.seed_param_mut().scaled_add(alpha, &direction);
        state
            .seed_param_mut()
            .clamp_in_place(problem.inner().lower(), problem.inner().upper());
        let (cost, grad) = problem.cost_and_gradient(state.param())?;
        state
            .set_evaluation(cost, grad)
            .expect("gradient dimension differs from parameter");
        Ok(crate::SolverStep::from((state, None)))
    }
}
