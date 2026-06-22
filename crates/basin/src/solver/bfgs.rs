use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    Dot, GeneralRankOneUpdate, MatVec, MatrixIdentity, NegInPlace, NormSquared, Scalar,
    ScaleInPlace, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::QuasiNewtonState;
use crate::core::termination::TerminationReason;
use crate::line_search::{LineSearch, Wolfe};

/// BFGS quasi-Newton solver.
///
/// Maintains a dense inverse-Hessian approximation `H` updated by the
/// rank-2 BFGS formula. The search direction is `d = −H·∇f(x)`; the step
/// length is set by a configurable line search (default: strong Wolfe,
/// which is what guarantees `yᵀs > 0` so each update preserves positive
/// definiteness).
///
/// On the first accepted step we rescale `H ← (sᵀy / yᵀy)·I` (Nocedal &
/// Wright (6.20)) — cheap, large convergence improvement on poorly scaled
/// problems.
///
/// **Curvature failure (`yᵀs ≤ ε · |y| · |s|`):** the H update is skipped
/// for that iteration. Strong Wolfe with `c2 < 1` guarantees `yᵀs > 0` in
/// exact arithmetic, so this branch is a numerical safeguard, not the
/// primary path. (Damped BFGS / Powell's modification is overkill when
/// strong Wolfe is in place — see plan.)
///
/// # Backends
///
/// Runs on `Vec<f64>` (via the hand-rolled
/// [`DenseMatrix`](crate::core::math::DenseMatrix)), nalgebra
/// (`DVector<f64>` / `DMatrix<f64>`), and faer (`Col<f64>` / `Mat<f64>`).
/// The dense inverse-Hessian needs only matvec, an identity constructor,
/// scaling, and the rank-one update `GeneralRankOneUpdate` — no
/// factorization — so it stays backend-generic. `ndarray` is a
/// compile-time error per tenet 5: its `Array2<f64>` implements neither
/// `GeneralRankOneUpdate` nor [`MatrixIdentity`].
///
/// # Examples
///
/// BFGS on the 2-D Rosenbrock function over the dependency-free
/// `Vec<f64>` backend. Quasi-Newton solvers iterate a
/// [`QuasiNewtonState`], parameterised by the
/// param vector and the dense matrix type — here `Vec<f64>` and
/// [`DenseMatrix`](crate::DenseMatrix), bundled by the
/// [`DenseQuasiNewtonState`](crate::DenseQuasiNewtonState) alias so the
/// matrix type needn't be spelled:
///
/// ```
/// use basin::{Bfgs, CostFunction, DenseQuasiNewtonState, Executor, Gradient};
///
/// struct Rosenbrock;
/// impl CostFunction for Rosenbrock {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
///     }
/// }
/// impl Gradient for Rosenbrock {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(vec![
///             -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
///             200.0 * (x[1] - x[0].powi(2)),
///         ])
///     }
/// }
///
/// let result = Executor::new(
///     Rosenbrock,
///     Bfgs::new(),
///     DenseQuasiNewtonState::new(vec![-1.2, 1.0]),
/// )
/// .max_iter(100)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-8);
/// ```
pub struct Bfgs<S = Wolfe, F = f64> {
    line_search: S,
    epsilon: F,
}

impl Default for Bfgs<Wolfe> {
    fn default() -> Self {
        Self::new()
    }
}

impl Bfgs<Wolfe> {
    /// BFGS with the strong-Wolfe line search (Nocedal & Wright defaults)
    /// and `ε = 1e-10` for the curvature-condition guard.
    pub fn new() -> Self {
        Self {
            line_search: Wolfe::new(),
            epsilon: 1e-10,
        }
    }
}

impl<S, F: Scalar> Bfgs<S, F> {
    /// BFGS with an explicit line-search strategy.
    pub fn with_line_search(line_search: S) -> Self {
        Self {
            line_search,
            epsilon: F::from_f64(1e-10).unwrap(),
        }
    }

    /// Relative threshold for the curvature condition `yᵀs > ε · |y| · |s|`.
    /// Iterations where this fails skip the H update (rare with strong
    /// Wolfe). Default `1e-10`.
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        assert!(epsilon >= F::zero(), "epsilon must be ≥ 0");
        self.epsilon = epsilon;
        self
    }
}

impl<P, S, V, M, F> Solver<P, QuasiNewtonState<V, M, F>> for Bfgs<S, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    S: LineSearch<P, V, F, Error = P::Error>,
    V: Clone + Dot<F> + NormSquared<F> + ScaledAdd<F> + ScaleInPlace<F> + NegInPlace + VectorLen,
    M: MatVec<V> + MatrixIdentity + ScaleInPlace<F> + GeneralRankOneUpdate<V, F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: QuasiNewtonState<V, M, F>,
    ) -> Result<QuasiNewtonState<V, M, F>, Self::Error> {
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: QuasiNewtonState<V, M, F>,
    ) -> Result<(QuasiNewtonState<V, M, F>, Option<TerminationReason>), Self::Error> {
        let g = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let cost_old = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");

        // Quasi-Newton direction: d = −H g. With H positive definite this
        // is automatically a descent direction (gᵀd = −gᵀHg < 0).
        let mut direction = state.inverse_hessian.matvec(&g);
        direction.neg_in_place();

        let alpha = self
            .line_search
            .next(problem, &state.param, cost_old, &g, &direction)?;

        // Line search bailed (α = 0): direction wasn't descent, or we're
        // at numerical convergence. Restore gradient/cost so the state
        // stays consistent and report it as a mid-iter termination so the
        // executor halts immediately. NaN routes here too
        // (`NaN > 0.0` is false).
        if !(alpha.is_finite() && alpha > F::zero()) {
            state.gradient = Some(g);
            state.cost = Some(cost_old);
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        // s = α d, x ← x + s.
        let mut s = direction;
        s.scale_in_place(alpha);
        state.param.scaled_add(F::one(), &s);

        // Fused cost+grad at the new iterate — one fused call gives both
        // values consumed below (BFGS update reads g_new; state caches
        // cost_new at the bottom of the iter).
        let (cost_new, g_new) = problem.cost_and_gradient(&state.param)?;

        // y = g_new − g.
        let mut y = g_new.clone();
        y.scaled_add(-F::one(), &g);
        let sy = s.dot(&y);
        let s_norm = s.norm_squared().sqrt();
        let y_norm = y.norm_squared().sqrt();

        if sy > self.epsilon * s_norm * y_norm {
            // Initial-Hessian rescaling: align H₀ with the local curvature
            // before applying the first BFGS update. Without this, the
            // identity-initialized H produces a unit step that's far too
            // large or small on poorly scaled problems.
            if !state.initial_scaling_done {
                let yy = y.dot(&y);
                if yy > F::zero() {
                    let scale = sy / yy;
                    let n = state.param.vec_len();
                    let mut h0 = M::identity(n);
                    h0.scale_in_place(scale);
                    state.inverse_hessian = h0;
                }
                state.initial_scaling_done = true;
            }

            let rho = F::one() / sy;
            let hy = state.inverse_hessian.matvec(&y);
            let yhy = y.dot(&hy);
            let coef = rho * (F::one() + rho * yhy);

            // H ← H + coef · s sᵀ − ρ · (s (Hy)ᵀ + (Hy) sᵀ).
            // Three rank-1 updates, all in place.
            state.inverse_hessian.general_rank_one_update(coef, &s, &s);
            state.inverse_hessian.general_rank_one_update(-rho, &s, &hy);
            state.inverse_hessian.general_rank_one_update(-rho, &hy, &s);
        }
        // else: curvature failure (very rare with strong Wolfe). Skip the
        // H update; the line search still produced a descent step, so we
        // continue. If this persists, max_iter / GradientTolerance halt.

        state.cost = Some(cost_new);
        state.gradient = Some(g_new);
        Ok((state, None))
    }
}

/// Lets [`Bfgs`] serve as the inner of a composed solver
/// (e.g. [`BarrierMethod`](crate::solver::BarrierMethod) /
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)),
/// seeding a fresh [`QuasiNewtonState`] (identity inverse-Hessian) at the
/// warm-start point.
///
/// Implemented for every backend BFGS itself runs on — `Vec<f64>` (via the
/// hand-rolled [`DenseMatrix`](crate::core::math::DenseMatrix)), nalgebra,
/// and faer — so [`Executor::from_start`](crate::Executor::from_start) and
/// the composed (barrier / AL) inners seed uniformly regardless of backend.
/// `ndarray` is excluded for the same reason BFGS itself is: `Array2`
/// implements neither `GeneralRankOneUpdate` nor the rank-one update BFGS
/// needs (see the `# Backends` note above).
impl<S, F> InitialState<Vec<F>> for Bfgs<S, F>
where
    F: Scalar,
{
    type State = QuasiNewtonState<Vec<F>, crate::core::math::DenseMatrix<F>, F>;
    fn seed(&self, x: &Vec<F>) -> Self::State {
        QuasiNewtonState::<Vec<F>, crate::core::math::DenseMatrix<F>, F>::new(x.clone())
    }
}

impl<S, F> WarmStart<Vec<F>> for Bfgs<S, F> where F: Scalar {}

#[cfg(feature = "nalgebra")]
impl<S, F> InitialState<nalgebra::DVector<F>> for Bfgs<S, F>
where
    F: Scalar + nalgebra::Scalar + num_traits::Zero,
{
    type State = QuasiNewtonState<nalgebra::DVector<F>, nalgebra::DMatrix<F>, F>;
    fn seed(&self, x: &nalgebra::DVector<F>) -> Self::State {
        QuasiNewtonState::<nalgebra::DVector<F>, nalgebra::DMatrix<F>, F>::new(x.clone())
    }
}

#[cfg(feature = "nalgebra")]
impl<S, F> WarmStart<nalgebra::DVector<F>> for Bfgs<S, F> where
    F: Scalar + nalgebra::Scalar + num_traits::Zero
{
}

#[cfg(feature = "faer")]
impl<S, F> InitialState<faer::Col<F>> for Bfgs<S, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    type State = QuasiNewtonState<faer::Col<F>, faer::Mat<F>, F>;
    fn seed(&self, x: &faer::Col<F>) -> Self::State {
        QuasiNewtonState::<faer::Col<F>, faer::Mat<F>, F>::new(x.clone())
    }
}

#[cfg(feature = "faer")]
impl<S, F> WarmStart<faer::Col<F>> for Bfgs<S, F> where F: Scalar + faer_traits::ComplexField {}
