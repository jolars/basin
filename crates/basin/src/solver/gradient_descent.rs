use crate::core::inner::WarmStart;
use crate::core::math::{NegInPlace, ScaleInPlace, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;
use crate::line_search::{Constant, LineSearch};

/// Steepest-descent solver: step in the direction of `−∇f(x)` with a
/// pluggable line search and optional heavy-ball momentum.
///
/// The line search type parameter `L` is the strategy
/// (e.g. [`Constant`], [`Backtracking`](crate::line_search::Backtracking),
/// [`Wolfe`](crate::line_search::Wolfe)). Use [`GradientDescent::new`]
/// for a fixed step or
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
/// this is the classical heavy-ball method — well-behaved on the curved,
/// ill-conditioned Rosenbrock valley where plain steepest descent
/// zig-zags. A too-large effective step (roughly `α / (1 − β)` along
/// consistent directions) diverges, so reduce `α` when adding momentum.
///
/// # Backends
///
/// Backend-generic — works with any `V` implementing
/// [`ScaledAdd<f64>`](crate::core::math::ScaledAdd) +
/// [`NegInPlace`] + [`ScaleInPlace`] + `Clone`. That covers
/// `Vec<f64>`, `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`).
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
/// use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent, GradientTolerance};
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
/// let result = Executor::new(Sphere, GradientDescent::new(0.1), BasicState::new(vec![1.0, 1.0]))
///     .max_iter(1_000)
///     .terminate_on(GradientTolerance(1e-8))
///     .run()
///     .unwrap();
/// assert!(result.cost() < 1e-12);
/// ```
pub struct GradientDescent<L, V> {
    line_search: L,
    /// Momentum coefficient `β`; `0.0` disables momentum (plain steepest
    /// descent, taking the original allocation-free code path).
    beta: f64,
    /// Heavy-ball velocity `vₖ`. `None` until the first momentum step
    /// (treated as the zero vector) and reset by [`init`](Solver::init) so
    /// a reused solver restarts from rest. Stays `None` when `β = 0`.
    velocity: Option<V>,
}

impl<V> GradientDescent<Constant, V> {
    /// Gradient descent with a fixed step size `alpha`. Equivalent to
    /// `with_line_search(Constant(alpha))`.
    pub fn new(alpha: f64) -> Self {
        Self {
            line_search: Constant(alpha),
            beta: 0.0,
            velocity: None,
        }
    }
}

impl<L, V> GradientDescent<L, V> {
    /// Gradient descent with an explicit line-search strategy
    /// (e.g. [`Backtracking`](crate::line_search::Backtracking),
    /// [`Wolfe`](crate::line_search::Wolfe)).
    pub fn with_line_search(line_search: L) -> Self {
        Self {
            line_search,
            beta: 0.0,
            velocity: None,
        }
    }

    /// Enable heavy-ball momentum with coefficient `beta` (Polyak 1964).
    /// `beta = 0.0` is plain steepest descent; `beta` in `(0, 1)`
    /// (commonly `0.9`) adds momentum. See the [type docs](Self#momentum)
    /// for the update rule and stability caveat.
    pub fn with_momentum(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }
}

impl<P, V, L> Solver<P, BasicState<V>> for GradientDescent<L, V>
where
    P: CostFunction<Param = V, Output = f64> + Gradient<Gradient = V>,
    V: ScaledAdd<f64> + NegInPlace + ScaleInPlace + Clone,
    L: LineSearch<P, V, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V>,
    ) -> Result<BasicState<V>, Self::Error> {
        // Start momentum from rest, even if this solver instance is reused
        // across runs (composition): velocity must not leak between runs.
        self.velocity = None;
        // Seed cost and gradient at the initial param so iter-0 termination
        // checks (e.g. `GradientTolerance` on a near-optimal start) see a
        // complete state. Same work we'd do on iter 1, hoisted.
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V>,
    ) -> Result<(BasicState<V>, Option<TerminationReason>), Self::Error> {
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

        if self.beta == 0.0 {
            // No momentum: the original allocation-free steepest-descent
            // step, bit-identical to the pre-momentum implementation.
            state.param.scaled_add(alpha, &direction);
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
            state.param.scaled_add(1.0, &velocity);
            self.velocity = Some(velocity);
        }

        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok((state, None))
    }
}

/// Lets [`GradientDescent`] serve as the inner of a composed solver
/// (e.g. [`BarrierMethod`](crate::solver::BarrierMethod) /
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)),
/// seeding a fresh [`BasicState`] at the warm-start point.
impl<L, V> WarmStart<V> for GradientDescent<L, V>
where
    V: Clone,
{
    type State = BasicState<V>;
    fn seed(&self, x: &V) -> BasicState<V> {
        BasicState::new(x.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::State;
    use crate::{BasicState, Executor};

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
    /// the stiff `x₁` axis crawls along the soft `x₀` axis — the regime
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
        let state = solver.init(&mut p, BasicState::new(vec![1.0])).unwrap();
        let (state, reason) = solver.next_iter(&mut p, state).unwrap();
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
            BasicState::new(start.clone()),
        )
        .max_iter(iters)
        .run()
        .unwrap();
        let momentum = Executor::new(
            IllConditioned,
            GradientDescent::new(alpha).with_momentum(0.9),
            BasicState::new(start),
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
            let mut state = solver.init(&mut p, BasicState::new(start.clone())).unwrap();
            for _ in 0..10 {
                let (next, _) = solver.next_iter(&mut p, state).unwrap();
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
