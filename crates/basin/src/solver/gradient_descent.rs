use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    NegInPlace, Scalar, ScaleInPlace, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{FirstOrderState, State};
use crate::core::termination::Termination;
use crate::line_search::{Constant, LineSearch, LineSearchOutcome};

/// Steepest-descent solver: step in the direction of `−∇f(x)` with a
/// pluggable line search and optional heavy-ball momentum.
///
/// The line search type parameter `L` is the strategy
/// (e.g. [`Constant`], [`Backtracking`](crate::line_search::Backtracking),
/// [`Wolfe`](crate::line_search::Wolfe),
/// [`HagerZhang`](crate::line_search::HagerZhang)). Use
/// [`GradientDescent::new`] for a fixed step or
/// [`GradientDescent::with_line_search`] to pick a strategy explicitly.
///
/// # Momentum
///
/// [`with_momentum`](Self::with_momentum) adds a heavy-ball velocity term
/// (Polyak 1964). With momentum coefficient `β` and the per-step length
/// `αₖ` chosen by the line search, the update becomes
///
/// ```text
/// vₖ₊₁ = β · vₖ − αₖ · ∇f(xₖ)
/// xₖ₊₁ = xₖ + vₖ₊₁
/// ```
///
/// starting from `v₀ = 0`. `β = 0` (the default) is exactly plain
/// steepest descent; `β ∈ (0, 1)` carries momentum, which cancels the
/// oscillating component of the gradient across a narrow valley while
/// accumulating speed along the valley floor. With a [`Constant`] step
/// this is the classical heavy-ball method, well-behaved on the curved,
/// ill-conditioned Rosenbrock valley where plain steepest descent
/// zig-zags. A too-large effective step (roughly `α / (1 − β)` along
/// consistent directions) diverges, so reduce `α` when adding momentum.
///
/// Progress uses [`FirstOrderState<V, F>`]. Fresh initialization clears
/// counters and incumbents, resets the line search, and reevaluates the seed.
/// Momentum starts from rest on each fresh run, including after a dimension
/// change. Exact checkpoints retain velocity and line-search history.
///
/// # Backends
///
/// Backend-generic; works with any `V` implementing
/// [`ScaledAdd<F>`](crate::core::math::ScaledAdd) +
/// [`NegInPlace`] + [`ScaleInPlace<F>`] + [`VectorLen`] + `Clone`. Supports
/// `Vec<F>`, `nalgebra::DVector<F>` (feature `nalgebra`),
/// `ndarray::Array1<F>` (feature `ndarray`), and `faer::Col<F>` (feature
/// `faer`) for both `f32` and `f64`.
///
/// # References
///
/// Polyak, B. T. (1964). "Some methods of speeding up the convergence of
/// iteration methods." *USSR Computational Mathematics and Mathematical
/// Physics*, 4(5), 1–17.
/// [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5).
///
/// # Examples
///
/// Minimize the 2-D sphere `f(x) = x₀² + x₁²` from `(1, 1)`:
///
/// ```
/// use basin::{FirstOrderState, CostFunction, Executor, Gradient, GradientDescent};
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
/// }
///
/// let result = Executor::new(Sphere, (GradientDescent::new(0.1)).with_absolute_gradient_tolerance(1e-8), FirstOrderState::new(vec![1.0, 1.0]))
///     .max_iter(1_000)
///
///     .run()
///     .unwrap();
/// assert!(result.cost() < 1e-12);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientDescent<L, V, F: Scalar = f64> {
    line_search: L,
    /// Momentum coefficient `β`; `0.0` disables momentum and runs the plain
    /// steepest-descent step, which keeps no persistent velocity buffer.
    beta: F,
    /// Heavy-ball velocity `vₖ`. `None` until the first momentum step
    /// (treated as the zero vector) and reset by [`init`](Solver::init) so
    /// a reused solver restarts from rest. Stays `None` when `β = 0`.
    velocity: Option<V>,
}

impl<V, F: Scalar> GradientDescent<Constant<F>, V, F> {
    /// Gradient descent with a fixed step size `alpha`. Equivalent to
    /// `with_line_search(Constant(alpha))`.
    pub fn new(alpha: F) -> Self {
        Self {
            line_search: Constant(alpha),
            beta: F::zero(),
            velocity: None,
        }
    }
}

impl<L, V, F: Scalar> GradientDescent<L, V, F> {
    /// Gradient descent with an explicit line-search strategy
    /// (e.g. [`Backtracking`](crate::line_search::Backtracking),
    /// [`Wolfe`](crate::line_search::Wolfe), or
    /// [`HagerZhang`](crate::line_search::HagerZhang)).
    pub fn with_line_search(line_search: L) -> Self {
        Self {
            line_search,
            beta: F::zero(),
            velocity: None,
        }
    }

    /// Enable heavy-ball momentum with coefficient `beta` (Polyak 1964).
    /// `beta = 0.0` is plain steepest descent; `beta` in `(0, 1)`
    /// (commonly `0.9`) adds momentum. See the [type docs](Self#momentum)
    /// for the update rule and stability caveat.
    pub fn with_momentum(mut self, beta: F) -> Self {
        self.beta = beta;
        self
    }
}

impl<P, V, F, L> Solver<P, FirstOrderState<V, F>> for GradientDescent<L, V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    V: ScaledAdd<F> + NegInPlace + ScaleInPlace<F> + Clone + VectorLen,
    L: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        state.reset();
        self.line_search.reset();
        // Start momentum from rest, even if this solver instance is reused
        // across runs (composition): velocity must not leak between runs.
        self.velocity = None;
        // Seed cost and gradient at the initial param so iter-0 termination
        // checks (e.g. `GradientTolerance` on a near-optimal start) see a
        // complete state. Same work we'd do on iter 1, hoisted.
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
        let line_search_result = self.line_search.next_with_evaluation(
            problem,
            state.param(),
            prev_cost,
            &grad,
            &direction,
        )?;
        let alpha = match line_search_result.outcome {
            LineSearchOutcome::Step(alpha) => alpha,
            LineSearchOutcome::Failed => {
                state
                    .set_evaluation(prev_cost, grad)
                    .expect("gradient dimension differs from parameter");
                return Ok(crate::SolverStep::from((
                    state,
                    Some(Termination::numerical_failure(
                        "Gradient-descent line search failed.",
                    )),
                )));
            }
        };

        // Momentum can change the actual iterate, so only plain descent can
        // adopt the line search's retained evaluation without recomputing it.
        if self.beta == F::zero() {
            if let Some(evaluation) = line_search_result.evaluation {
                state
                    .replace(
                        evaluation.param,
                        evaluation.cost,
                        evaluation.gradient,
                    )
                    .expect("gradient dimension differs from parameter");
                return Ok(crate::SolverStep::from((state, None)));
            }
            state.seed_param_mut().scaled_add(alpha, &direction);
        } else {
            // Heavy ball: v ← β·v + αₖ·direction (direction = −∇f), then
            // x ← x + v. With v₀ = 0 the first step is just αₖ·direction,
            // which we form by consuming `direction` to avoid a zero vector.
            let velocity = match self.velocity.take() {
                Some(mut v) => {
                    v.scale_in_place(self.beta);
                    v.scaled_add(alpha, &direction);
                    v
                }
                None => {
                    direction.scale_in_place(alpha);
                    direction
                }
            };
            state.seed_param_mut().scaled_add(F::one(), &velocity);
            self.velocity = Some(velocity);
        }

        let (cost, grad) = problem.cost_and_gradient(state.param())?;
        state
            .set_evaluation(cost, grad)
            .expect("gradient dimension differs from parameter");
        Ok(crate::SolverStep::from((state, None)))
    }
}

/// Lets [`GradientDescent`] serve as the inner of a composed solver
/// (e.g. [`BarrierMethod`](crate::solver::BarrierMethod) /
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)),
/// seeding a fresh [`FirstOrderState`] at the warm-start point.
impl<L, V, F> InitialState<V> for GradientDescent<L, V, F>
where
    F: Scalar,
    V: Clone,
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, x: &V) -> FirstOrderState<V, F> {
        FirstOrderState::new(x.clone())
    }
}

impl<L, V, F> WarmStart<V> for GradientDescent<L, V, F>
where
    F: Scalar,
    V: Clone,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::State;
    use crate::{Executor, FirstOrderState};

    /// Isotropic quadratic bowl `f(x) = Σ xᵢ²`, gradient `2x`. Minimum at
    /// the origin. Used where conditioning is irrelevant (first-step and
    /// reset checks).
    struct Quadratic;

    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(x.iter().map(|v| v * v).sum())
        }
    }

    impl Gradient for Quadratic {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(x.iter().map(|v| 2.0 * v).collect())
        }
    }

    /// Ill-conditioned quadratic `f(x) = x₀² + 100·x₁²` (condition number
    /// 100), gradient `[2x₀, 200x₁]`. A step small enough to be stable on
    /// the stiff `x₁` axis crawls along the soft `x₀` axis, the regime
    /// where heavy-ball momentum demonstrably accelerates over plain
    /// steepest descent. This is the well-conditioned-vs-ill-conditioned
    /// distinction that matters: momentum is *not* faster on `Quadratic`.
    struct IllConditioned;

    impl CostFunction for IllConditioned {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(x[0] * x[0] + 100.0 * x[1] * x[1])
        }
    }

    impl Gradient for IllConditioned {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![2.0 * x[0], 200.0 * x[1]])
        }
    }

    #[test]
    fn with_momentum_zero_is_plain_descent_first_step() {
        // β = 0 with v₀ = 0: first iterate is x − α·∇f, unaffected by the
        // momentum branch. f = Σx², ∇f = 2x, so x₁ = 1 − 0.1·2 = 0.8.
        let mut solver = GradientDescent::new(0.1).with_momentum(0.0);
        let mut p = Problem::new(Quadratic);
        let state = solver
            .init(&mut p, FirstOrderState::new(vec![1.0]))
            .unwrap();
        let (state, _, reason) =
            solver.next_iter(&mut p, state).unwrap().into_parts();
        assert!(reason.is_none());
        assert!((state.param()[0] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn momentum_accelerates_over_plain_descent_when_ill_conditioned() {
        // Same learning rate and iteration budget; on an ill-conditioned
        // bowl β > 0 must reach a strictly lower cost than β = 0, because
        // momentum accelerates the slow soft-axis convergence that cripples
        // plain steepest descent.
        let start = vec![1.0, 1.0];
        let iters = 200;
        let alpha = 0.004;

        let plain = Executor::new(
            IllConditioned,
            GradientDescent::new(alpha),
            FirstOrderState::new(start.clone()),
        )
        .max_iter(iters)
        .run()
        .unwrap();
        let momentum = Executor::new(
            IllConditioned,
            GradientDescent::new(alpha).with_momentum(0.9),
            FirstOrderState::new(start),
        )
        .max_iter(iters)
        .run()
        .unwrap();

        assert!(
            momentum.cost() < plain.cost(),
            "momentum cost {} should beat plain {}",
            momentum.cost(),
            plain.cost()
        );
    }

    #[test]
    fn momentum_velocity_resets_between_runs() {
        // A reused solver must restart from rest: running twice from the
        // same start gives the same result (init clears the velocity).
        let start = vec![2.0, -1.0];
        let mut solver = GradientDescent::new(0.05).with_momentum(0.8);

        let run = |solver: &mut GradientDescent<Constant, Vec<f64>>| {
            let mut p = Problem::new(Quadratic);
            let mut state = solver
                .init(&mut p, FirstOrderState::new(start.clone()))
                .unwrap();
            for _ in 0..10 {
                let (next, _, _) =
                    solver.next_iter(&mut p, state).unwrap().into_parts();
                state = next;
            }
            state.param().clone()
        };

        let first = run(&mut solver);
        let second = run(&mut solver);
        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a - b).abs() < 1e-12, "first={a}, second={b}");
        }
    }
}
