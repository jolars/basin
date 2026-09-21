use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    Dot, NegInPlace, NormInfinity, Scalar, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{FirstOrderState, State};
use crate::core::termination::TerminationReason;
use crate::line_search::{
    HagerZhang, LineSearch, LineSearchEvaluation, LineSearchOutcome,
};

/// Hager–Zhang nonlinear conjugate gradient for smooth, unconstrained objectives.
///
/// Requires a [`CostFunction`] and [`Gradient`] with the same vector type. The
/// solver stores a search direction and scalar bookkeeping, using O(n) memory.
/// Progress lives in [`FirstOrderState`]. This minimizes a nonlinear objective;
/// [`Steihaug`](crate::Steihaug) instead uses linear CG inside a trust-region
/// subproblem. Bounds and other constraints are not enforced by this solver.
///
/// # Update and safeguards
///
/// With `g = ∇f(x)`, `y = g_next − g`, and `q = dᵀy`, the update is
///
/// ```text
/// beta_hz = (yᵀg_next − 2 (yᵀy) (dᵀg_next) / q) / q
/// beta = max(beta_hz, −1 / (‖d‖₂ min(eta, ‖g‖₂)))
/// d_next = −g_next + beta d,       d_0 = −g_0.
/// ```
///
/// The lower bound uses the **previous** gradient norm. `eta` defaults to
/// `0.01`. In exact arithmetic the update satisfies
/// `g_nextᵀd_next ≤ −⅞ ‖g_next‖₂²`. Basin restarts with steepest descent when
/// `|q| ≤ F::epsilon() ‖d‖₂ ‖y‖₂`, the update has non-finite arithmetic, or
/// the computed direction fails that descent test. Optional periodic restarts
/// are disabled by default. A zero coefficient also resets restart history.
///
/// [`new`](Self::new) uses [`HagerZhang`] with its default settings. Custom
/// searches use the existing [`LineSearch`] interface. The paper's global
/// convergence result assumes the Wolfe conditions and its regularity
/// assumptions; an arbitrary custom step rule does not provide that guarantee.
/// This implements the paper's safeguarded recurrence, rather than every
/// heuristic of CG_DESCENT. Solution and stationarity tests use CG_DESCENT C
/// 1.2 as an external reference; differing line-search initialization and
/// approximate-Wolfe switching preclude trajectory parity.
///
/// # Convergence and lifecycle
///
/// An exactly zero gradient at a finite point with finite cost terminates with
/// [`GradientTolerance`](TerminationReason::GradientTolerance). Optional
/// gradient, step, and cost-change tolerances are disabled by default. Configure
/// them with the `with_absolute_*_tolerance` and `with_relative_*_tolerance`
/// setters; distinct enabled checks combine with OR. Execution budgets belong
/// on [`Executor`](crate::Executor).
///
/// Accepted line-search evaluations are reused. If a search fails along a
/// conjugate direction, the solver retries once from the same point along the
/// negative gradient. Failure along steepest descent, an unrepresentable step,
/// or non-finite accepted data returns [`SolverFailed`](TerminationReason::SolverFailed)
/// with the previous current record intact. Non-finite initial data also fail
/// softly on the first attempted iteration. A typed problem error aborts
/// unchanged. Gradients must have the same dimension as their parameters.
///
/// Fresh initialization resets direction and restart history and reevaluates
/// the seed. Exact solver-and-state checkpoints preserve that history and all
/// evaluation counts. Warm starts use a fresh [`FirstOrderState`].
///
/// # Backends
///
/// `Vec<F>`, `nalgebra::DVector<F>`, `ndarray::Array1<F>`, and `faer::Col<F>`
/// for every supported backend release, with `F = f64` or `f32`. Requires
/// [`Dot`], [`ScaledAdd`], [`NegInPlace`], [`NormInfinity`], [`VectorLen`],
/// and `Clone`; optional convergence checks add their own capabilities.
/// No matrix operations or BLAS/LAPACK provider are required.
///
/// # Reference
///
/// William W. Hager and Hongchao Zhang, “A new conjugate gradient method with
/// guaranteed descent and an efficient line search,” *SIAM Journal on
/// Optimization* 16(1), 2005, pp. 170–192, equations (1.3), (1.5), and (1.6).
/// [doi:10.1137/030601880](https://doi.org/10.1137/030601880).
///
/// # Example
///
/// ```
/// use basin::{CostFunction, Executor, Gradient, NonlinearCg};
/// use std::convert::Infallible;
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
///         Ok(x.iter().map(|v| v * v).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
///         Ok(x.iter().map(|v| 2.0 * v).collect())
///     }
/// }
/// let result = Executor::from_start(
///     Sphere,
///     NonlinearCg::new().with_absolute_gradient_tolerance(1e-8),
///     vec![2.0, -1.0],
/// ).max_iter(100).run().unwrap();
/// assert!(result.cost() < 1e-12);
/// ```
pub struct NonlinearCg<L, V, F: Scalar = f64> {
    line_search: L,
    eta: F,
    restart_interval: Option<u64>,
    direction: Option<V>,
    steepest: bool,
    steps_since_restart: u64,
}

impl<V, F: Scalar> Default for NonlinearCg<HagerZhang<F>, V, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, F: Scalar> NonlinearCg<HagerZhang<F>, V, F> {
    /// Construct a solver with the default Hager–Zhang line search.
    pub fn new() -> Self {
        Self::with_line_search(HagerZhang::new())
    }
}

impl<L, V, F: Scalar> NonlinearCg<L, V, F> {
    /// Construct a solver using an explicit line-search strategy.
    pub fn with_line_search(line_search: L) -> Self {
        Self {
            line_search,
            eta: F::from_f64(0.01).unwrap(),
            restart_interval: None,
            direction: None,
            steepest: true,
            steps_since_restart: 0,
        }
    }

    /// Set the positive parameter in the coefficient's lower bound.
    ///
    /// Default `0.01`. This controls the CG update, not convergence.
    ///
    /// # Panics
    ///
    /// Panics unless `eta` is finite and strictly positive.
    pub fn with_eta(mut self, eta: F) -> Self {
        assert!(
            eta.is_finite() && eta > F::zero(),
            "eta must be finite and positive"
        );
        self.eta = eta;
        self
    }

    /// Restart after this many accepted steps since the last restart.
    ///
    /// Default `None` disables periodic restarts. Safety restarts remain active.
    /// `Some(1)` always uses steepest descent.
    ///
    /// # Panics
    ///
    /// Panics for `Some(0)`.
    pub fn with_restart_interval(mut self, interval: Option<u64>) -> Self {
        assert!(interval != Some(0), "restart interval must be positive");
        self.restart_interval = interval;
        self
    }
}

fn negative<V: Clone + NegInPlace>(gradient: &V) -> V {
    let mut direction = gradient.clone();
    direction.neg_in_place();
    direction
}

fn sufficient_descent<V: Dot<F> + NormInfinity<F>, F: Scalar>(
    gradient: &V,
    direction: &V,
) -> bool {
    let squared = gradient.dot(gradient);
    let slope = gradient.dot(direction);
    // A nonzero gradient whose squared norm underflows is not an exact optimum.
    squared.is_finite()
        && squared > F::zero()
        && direction.norm_infinity().is_finite()
        && slope.is_finite()
        && slope <= -F::from_f64(0.875).unwrap() * squared
}

impl<L, V, F> NonlinearCg<L, V, F>
where
    F: Scalar,
    V: Clone + Dot<F> + ScaledAdd<F> + NegInPlace + NormInfinity<F>,
{
    fn restart(&mut self, gradient: &V) -> V {
        self.steepest = true;
        self.steps_since_restart = 0;
        negative(gradient)
    }

    fn update_direction(
        &mut self,
        gradient: &V,
        next_gradient: &V,
        direction: &V,
    ) -> V {
        self.steps_since_restart = self.steps_since_restart.saturating_add(1);
        if self
            .restart_interval
            .is_some_and(|interval| self.steps_since_restart >= interval)
        {
            return self.restart(next_gradient);
        }
        let mut y = next_gradient.clone();
        y.scaled_add(-F::one(), gradient);
        let dd = direction.dot(direction);
        let gg = gradient.dot(gradient);
        let yy = y.dot(&y);
        let dy = direction.dot(&y);
        if !dd.is_finite()
            || dd <= F::zero()
            || !gg.is_finite()
            || gg <= F::zero()
            || !yy.is_finite()
            || yy <= F::zero()
            || !dy.is_finite()
            || dy.abs() <= F::epsilon() * dd.sqrt() * yy.sqrt()
        {
            return self.restart(next_gradient);
        }
        let dg = direction.dot(next_gradient);
        let yg = y.dot(next_gradient);
        let beta_hz = (yg - (F::one() + F::one()) * (yy / dy) * dg) / dy;
        // Sequential divisions avoid underflow in the lower bound's denominator.
        let lower = -F::one() / dd.sqrt() / self.eta.min(gg.sqrt());
        if !beta_hz.is_finite() || !lower.is_finite() {
            return self.restart(next_gradient);
        }
        let beta = beta_hz.max(lower);
        let mut next = negative(next_gradient);
        next.scaled_add(beta, direction);
        if beta == F::zero() || !sufficient_descent(next_gradient, &next) {
            return self.restart(next_gradient);
        }
        self.steepest = false;
        next
    }

    fn search<P>(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<Option<LineSearchEvaluation<V, F>>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: VectorLen,
        L: LineSearch<P, V, F, Error = P::Error>,
    {
        let result = self
            .line_search
            .next_with_evaluation(problem, param, cost, gradient, direction)?;
        let LineSearchOutcome::Step(alpha) = result.outcome else {
            return Ok(None);
        };
        if !alpha.is_finite() || alpha <= F::zero() {
            return Ok(None);
        }
        let evaluation = if let Some(evaluation) = result.evaluation {
            if !usable_displacement(param, &evaluation.param) {
                return Ok(None);
            }
            evaluation
        } else {
            let mut next = param.clone();
            next.scaled_add(alpha, direction);
            if !usable_displacement(param, &next) {
                return Ok(None);
            }
            let (cost, gradient) = problem.cost_and_gradient(&next)?;
            LineSearchEvaluation::new(next, cost, gradient)
        };
        Ok((evaluation.gradient.vec_len() == param.vec_len()
            && evaluation.cost.is_finite()
            && evaluation.gradient.norm_infinity().is_finite())
        .then_some(evaluation))
    }
}

fn usable_displacement<V, F>(param: &V, next: &V) -> bool
where
    F: Scalar,
    V: Clone + ScaledAdd<F> + NormInfinity<F> + VectorLen,
{
    if next.vec_len() != param.vec_len() || !next.norm_infinity().is_finite() {
        return false;
    }
    let mut displacement = next.clone();
    displacement.scaled_add(-F::one(), param);
    let size = displacement.norm_infinity();
    size.is_finite() && size > F::zero()
}

impl<P, L, V, F> Solver<P, FirstOrderState<V, F>> for NonlinearCg<L, V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    V: Clone + Dot<F> + ScaledAdd<F> + NegInPlace + NormInfinity<F> + VectorLen,
    L: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        state.reset();
        self.direction = None;
        self.steepest = true;
        self.steps_since_restart = 0;
        let param = state.param().clone();
        let (cost, gradient) = problem.cost_and_gradient(&param)?;
        self.direction = Some(negative(&gradient));
        state
            .replace(param, cost, gradient)
            .expect("gradient must match parameter dimension");
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<(FirstOrderState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let (param, cost, gradient) = state
            .current()
            .expect("Solver::init must precede next_iter");
        if !cost.is_finite()
            || !param.norm_infinity().is_finite()
            || !gradient.norm_infinity().is_finite()
        {
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        let mut direction = self
            .direction
            .take()
            .expect("Solver::init must seed the direction");
        if !sufficient_descent(gradient, &direction) {
            direction = self.restart(gradient);
        }
        let accepted = loop {
            if !sufficient_descent(gradient, &direction) {
                break None;
            }
            if let Some(evaluation) =
                self.search(problem, param, cost, gradient, &direction)?
            {
                break Some(evaluation);
            }
            if self.steepest {
                break None;
            }
            direction = self.restart(gradient);
        };
        let Some(evaluation) = accepted else {
            self.direction = Some(direction);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        };
        self.direction = Some(self.update_direction(
            gradient,
            &evaluation.gradient,
            &direction,
        ));
        state
            .replace(evaluation.param, evaluation.cost, evaluation.gradient)
            .expect("accepted gradient dimension was checked");
        Ok((state, None))
    }

    fn terminate(
        &self,
        state: &FirstOrderState<V, F>,
    ) -> Option<TerminationReason> {
        let (param, cost, gradient) = state.current()?;
        (cost.is_finite()
            && param.norm_infinity().is_finite()
            && gradient.norm_infinity() == F::zero())
        .then_some(TerminationReason::GradientTolerance)
    }
}

impl<L, V: Clone, F: Scalar> InitialState<V> for NonlinearCg<L, V, F> {
    type State = FirstOrderState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        FirstOrderState::new(x.clone())
    }
}

impl<L, V: Clone, F: Scalar> WarmStart<V> for NonlinearCg<L, V, F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constant;

    #[test]
    fn coefficient_truncation_uses_the_previous_gradient_norm() {
        let mut solver = NonlinearCg::with_line_search(Constant(1.0));
        let gradient = vec![10.0];
        let next_gradient = vec![-20.0];
        let direction = vec![-10.0];
        // The untruncated one-dimensional coefficient is -2.
        assert_eq!(
            solver.update_direction(&gradient, &next_gradient, &direction),
            vec![40.0]
        );
        // eta=100 makes the bound -1/(10*10)=-0.01. Using the new
        // gradient's norm would instead give -0.005 and a direction of 20.05.
        let mut solver = solver.with_eta(100.0);
        let next =
            solver.update_direction(&gradient, &next_gradient, &direction);
        assert!((next[0] - 20.1_f64).abs() < 1e-12);
        assert!(!solver.steepest);
    }

    #[test]
    fn degenerate_curvature_and_overflow_restart() {
        let mut solver = NonlinearCg::with_line_search(Constant(1.0));
        for (gradient, next_gradient, direction) in [
            (vec![1.0, 1.0], vec![1.0, 1.0], vec![-1.0, -1.0]),
            (vec![1.0, 1.0], vec![2.0, 0.0], vec![-1.0, -1.0]),
            (
                vec![1.0, 1.0],
                vec![2.0, f64::EPSILON / 4.0],
                vec![-1.0, -1.0],
            ),
            (vec![1e200, 0.0], vec![1e200, 1.0], vec![-1e200, 0.0]),
        ] {
            solver.steps_since_restart = 4;
            let next =
                solver.update_direction(&gradient, &next_gradient, &direction);
            assert_eq!(next, negative(&next_gradient));
            assert!(solver.steepest);
            assert_eq!(solver.steps_since_restart, 0);
        }
    }

    #[test]
    fn updates_obey_the_papers_sufficient_descent_bound() {
        let gradient = vec![1.0, 2.0];
        let direction = vec![-1.0, -2.0];
        for eta in [0.01, 100.0] {
            let mut solver =
                NonlinearCg::with_line_search(Constant(1.0)).with_eta(eta);
            for a in -4..=4 {
                for b in -4..=4 {
                    let g = vec![f64::from(a), f64::from(b)];
                    let d = solver.update_direction(&gradient, &g, &direction);
                    assert!(d.norm_infinity().is_finite());
                    assert!(g.dot(&d) <= -0.875 * g.dot(&g) + 1e-13);
                }
            }
        }
    }

    #[test]
    fn descent_test_rejects_weak_ascent_and_nonfinite_directions() {
        let gradient = vec![1.0, 0.0];
        for direction in [
            vec![1.0, 0.0],
            vec![-0.5, 1.0],
            vec![-1.0, f64::NAN],
            vec![f64::NEG_INFINITY, 0.0],
        ] {
            assert!(!sufficient_descent(&gradient, &direction));
        }
        assert!(sufficient_descent(&gradient, &vec![-1.0, 0.0]));
        assert!(!sufficient_descent(&vec![1e-200], &vec![-1e-200]));
    }
}
