use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    DenseBackend, Dot, GeneralRankOneUpdate, MatVec, MatrixIdentity,
    MatrixIndex, NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd,
    VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{FirstOrderState, State};
use crate::core::termination::Termination;
use crate::line_search::{LineSearch, LineSearchOutcome, Wolfe};

mod initial_inverse_hessian;
pub use initial_inverse_hessian::{
    FixedInverseHessian, IdentityInverseHessian, InitialInverseHessian,
};

/// BFGS quasi-Newton solver.
///
/// Maintains a dense inverse-Hessian approximation `H` updated by the
/// rank-2 BFGS formula. The search direction is `d = −H·∇f(x)`; the step
/// length is set by a configurable line search (default: strong Wolfe,
/// which is what guarantees `yᵀs > 0` so each update preserves positive
/// definiteness).
///
/// By default, each fresh solve starts with `H = I`. Immediately before the
/// first accepted curvature update, we rescale `H ← (sᵀy / yᵀy)·I`
/// (Nocedal & Wright (6.20)), improving convergence on poorly scaled problems.
/// [`with_initial_inverse_hessian`](Self::with_initial_inverse_hessian) sets
/// a fixed seed, and [`with_initial_inverse_hessian_initializer`](Self::with_initial_inverse_hessian_initializer)
/// can construct one from the initial point and gradient. Custom seeds skip
/// this rescaling by default so that their scaling survives the first update.
/// Later curvature failures skip updates without replacing the model.
///
/// **Curvature failure (`yᵀs ≤ ε · |y| · |s|`):** the H update is skipped
/// for that iteration. Strong Wolfe with `c2 < 1` guarantees `yᵀs > 0` in
/// exact arithmetic, so this branch is a numerical safeguard, not the
/// primary path. (Damped BFGS and Powell's modification are overkill when
/// strong Wolfe is in place; see plan.)
///
/// # Backends
///
/// Runs on `Vec<F>` (via the hand-rolled
/// [`DenseMatrix`](crate::core::math::DenseMatrix)), nalgebra
/// (`DVector<F>`/`DMatrix<F>`), ndarray (`Array1<F>`/`Array2<F>`), and
/// faer (`Col<F>`/`Mat<F>`), for both `f32` and `f64`. The dense
/// inverse-Hessian needs only matvec, an identity constructor, scaling, and
/// the rank-one update `GeneralRankOneUpdate` (no factorization), so it
/// stays backend-generic. Fixed seeds additionally require `M: Clone +
/// MatrixIndex<F>`; closure initializers require `M: MatrixIndex<F>`, but
/// not `Clone`. These optional requirements do not affect the default seed.
///
/// # Examples
///
/// BFGS on the 2-D Rosenbrock function over the dependency-free
/// `Vec<f64>` backend. Progress uses [`FirstOrderState`]; the solver owns
/// its inverse Hessian and infers the matrix through [`DenseBackend`].
/// Fresh solves reset both the model and line-search history. Exact
/// solver-and-state checkpoints retain them.
///
/// ```
/// use basin::{
///     Bfgs, CostFunction, FirstOrderState, Executor, Gradient};
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
///     FirstOrderState::new(vec![-1.2, 1.0]),
/// )
/// .max_iter(100)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-8);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bfgs<
    V: DenseBackend<F>,
    F: Scalar = f64,
    M = <V as DenseBackend<F>>::Matrix,
    S = Wolfe<F>,
    I = IdentityInverseHessian,
> {
    line_search: S,
    epsilon: F,
    inverse_hessian: Option<M>,
    #[cfg_attr(feature = "serde", serde(skip, default = "Option::default"))]
    gradient_buffer: Option<V>,
    initial_scaling_done: bool,
    initial_inverse_hessian: I,
    param: std::marker::PhantomData<V>,
}

impl<V: DenseBackend<F>, F: Scalar> Default for Bfgs<V, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: DenseBackend<F>, F: Scalar> Bfgs<V, F> {
    /// BFGS with a strong-Wolfe line search and the backend's default matrix.
    /// The model is initialized from the seed's dimension during each fresh run.
    pub fn new() -> Self {
        Self::with_line_search(Wolfe::new())
    }
}

impl<V: DenseBackend<F>, F: Scalar, S>
    Bfgs<V, F, <V as DenseBackend<F>>::Matrix, S>
{
    /// BFGS with a configured line search and the backend's default matrix.
    pub fn with_line_search(line_search: S) -> Self {
        Self::with_matrix_and_line_search(line_search)
    }
}

impl<V: DenseBackend<F>, F: Scalar, M, S> Bfgs<V, F, M, S> {
    /// Construct with an explicit matrix type and a configured line search.
    /// The line search resets its evolving history on each fresh initialization.
    pub fn with_matrix_and_line_search(line_search: S) -> Self {
        Self {
            line_search,
            epsilon: F::from_f64(1e-10).unwrap(),
            inverse_hessian: None,
            gradient_buffer: None,
            initial_scaling_done: false,
            initial_inverse_hessian: IdentityInverseHessian,
            param: std::marker::PhantomData,
        }
    }
}

impl<V: DenseBackend<F>, F: Scalar, M, S, I> Bfgs<V, F, M, S, I> {
    /// Current inverse-Hessian model, available after initialization.
    /// Use [`crate::Executor::run_with_solver`] to retain it after a solve.
    pub fn inverse_hessian(&self) -> Option<&M> {
        self.inverse_hessian.as_ref()
    }

    /// Use a fixed initial inverse Hessian for each fresh solve.
    ///
    /// Initialization checks that the seed matches the starting dimension and
    /// is finite, exactly symmetric, and positive definite, then clones it
    /// into the working model. Invalid seeds panic at initialization. The
    /// check uses `O(n³)` pure-Rust Cholesky work and `O(n²)` scratch storage.
    /// The stored seed is never updated; reuse with a different dimension
    /// requires a new seed or a dimension-aware initializer.
    ///
    /// This disables first-update curvature rescaling. Existing callers using
    /// the default identity seed retain that rescaling. Changing the seed on
    /// an initialized solver affects its next fresh solve; exact continuation
    /// retains the current model and rescaling flag.
    pub fn with_initial_inverse_hessian(
        self,
        matrix: M,
    ) -> Bfgs<V, F, M, S, FixedInverseHessian<M>>
    where
        V: VectorLen,
        M: Clone + MatrixIndex<F>,
    {
        self.with_initial_inverse_hessian_initializer(FixedInverseHessian(
            matrix,
        ))
    }

    /// Construct the seed from the freshly evaluated point and gradient.
    ///
    /// Accepts an [`InitialInverseHessian`] implementation, including a
    /// closure `Fn(&V, &V) -> M`. Called once per fresh solve, even with a
    /// zero iteration budget, without extra objective or gradient requests.
    /// Point warm starts reevaluate the point and reconstruct the seed; exact
    /// continuation retains the model and does not call the initializer.
    /// First-update rescaling is disabled unless a named implementation
    /// explicitly enables it. Changing the initializer affects the next fresh
    /// solve, leaving an initialized model and its rescaling flag intact.
    ///
    /// Closure results are checked like fixed seeds, requiring only
    /// `M: MatrixIndex<F>` beyond the usual BFGS operations. They need not be
    /// `Clone`. Named implementations enforce their own matrix contract and
    /// may support custom storage or serialization. Closures support owned,
    /// nonserialized checkpoints.
    ///
    /// For example, a diagonal seed based on the initial gradient bounds the
    /// first search direction in each coordinate without changing variables:
    ///
    /// ```
    /// use basin::{Bfgs, DenseMatrix};
    /// let solver = Bfgs::<Vec<f64>>::new()
    ///     .with_initial_inverse_hessian_initializer(
    ///         |x: &Vec<f64>, gradient: &Vec<f64>| {
    ///             DenseMatrix::from_fn(x.len(), x.len(), |i, j| {
    ///                 if i == j { 0.1 / gradient[i].abs().max(1e-3) }
    ///                 else { 0.0 }
    ///             })
    ///         },
    ///     );
    /// assert!(solver.inverse_hessian().is_none());
    /// ```
    pub fn with_initial_inverse_hessian_initializer<Init>(
        self,
        initializer: Init,
    ) -> Bfgs<V, F, M, S, Init>
    where
        Init: InitialInverseHessian<V, M, F>,
    {
        Bfgs {
            line_search: self.line_search,
            epsilon: self.epsilon,
            inverse_hessian: self.inverse_hessian,
            gradient_buffer: self.gradient_buffer,
            initial_scaling_done: self.initial_scaling_done,
            initial_inverse_hessian: initializer,
            param: self.param,
        }
    }

    /// Set the relative threshold for accepting a curvature update.
    /// This is an algorithm safeguard, not an optimization stopping test.
    pub fn with_relative_curvature_tolerance(mut self, epsilon: F) -> Self {
        assert!(epsilon >= F::zero(), "epsilon must be ≥ 0");
        self.epsilon = epsilon;
        self
    }
}

impl<P, S, V, M, F, I> Solver<P, FirstOrderState<V, F>> for Bfgs<V, F, M, S, I>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    S: LineSearch<P, V, F, Error = P::Error>,
    I: InitialInverseHessian<V, M, F>,
    V: DenseBackend<F>
        + Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace
        + VectorLen,
    M: MatVec<V>
        + MatrixIdentity
        + ScaleInPlace<F>
        + GeneralRankOneUpdate<V, F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        state.reset();
        self.line_search.reset();
        self.inverse_hessian = None;
        self.initial_scaling_done =
            !self.initial_inverse_hessian.rescale_on_first_update();
        let param = state.param().clone();
        let (cost, grad) = problem.cost_and_gradient_reusing(&param, None)?;
        self.gradient_buffer = None;
        state
            .replace(param, cost, grad)
            .expect("gradient dimension differs from parameter");
        self.inverse_hessian = Some(self.initial_inverse_hessian.initialize(
            state.param(),
            state.current().expect("initialized BFGS progress").2,
        ));
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<crate::SolverStep<FirstOrderState<V, F>>, Self::Error> {
        let (param, cost_old, g) =
            state.current().expect("BFGS requires initialized progress");
        let mut param = param.clone();
        let h = self
            .inverse_hessian
            .as_mut()
            .expect("BFGS model is not initialized");

        // Quasi-Newton direction: d = −H g. With H positive definite this
        // is automatically a descent direction (gᵀd = −gᵀHg < 0).
        let mut direction = (*h).matvec(g);
        direction.neg_in_place();

        let alpha = match self
            .line_search
            .next_with_outcome(problem, &param, cost_old, g, &direction)?
        {
            LineSearchOutcome::Step(alpha) => alpha,
            LineSearchOutcome::Failed => {
                return Ok(crate::SolverStep::from((
                    state,
                    Some(Termination::numerical_failure(
                        "BFGS line search failed.",
                    )),
                )));
            }
        };

        // A nonpositive step may indicate stagnation even away from a
        // stationary point, so it cannot establish convergence on its own.
        if !(alpha.is_finite() && alpha > F::zero()) {
            return Ok(crate::SolverStep::from((
                state,
                Some(if alpha.is_finite() {
                    Termination::numerical_stall(
                        "The line search returned a nonpositive step without establishing stationarity.",
                    )
                } else {
                    Termination::numerical_failure(
                        "The line search returned a non-finite step.",
                    )
                }),
            )));
        }

        // s = α d, x ← x + s.
        let mut s = direction;
        s.scale_in_place(alpha);
        param.scaled_add(F::one(), &s);

        // Fused cost+grad at the new iterate: one fused call gives both
        // values consumed below (BFGS update reads g_new; state caches
        // cost_new at the bottom of the iter).
        let (cost_new, g_new) = problem
            .cost_and_gradient_reusing(&param, self.gradient_buffer.take())?;

        // y = g_new − g.
        let mut y = g_new.clone();
        y.scaled_add(-F::one(), g);
        let sy = s.dot(&y);
        let s_norm = s.norm_squared().sqrt();
        let y_norm = y.norm_squared().sqrt();

        if sy > self.epsilon * s_norm * y_norm {
            // Initial-Hessian rescaling: align H₀ with the local curvature
            // before applying the first BFGS update. Without this, the
            // identity-initialized H produces a unit step that's far too
            // large or small on poorly scaled problems.
            if !self.initial_scaling_done {
                let yy = y.dot(&y);
                if yy > F::zero() {
                    let scale = sy / yy;
                    let n = param.vec_len();
                    let mut h0 = M::identity(n);
                    h0.scale_in_place(scale);
                    (*h) = h0;
                }
                self.initial_scaling_done = true;
            }

            let rho = F::one() / sy;
            let hy = (*h).matvec(&y);
            let yhy = y.dot(&hy);
            let coef = rho * (F::one() + rho * yhy);

            // H ← H + coef · s sᵀ − ρ · (s (Hy)ᵀ + (Hy) sᵀ).
            // Three rank-1 updates, all in place.
            (*h).general_rank_one_update(coef, &s, &s);
            (*h).general_rank_one_update(-rho, &s, &hy);
            (*h).general_rank_one_update(-rho, &hy, &s);
        }
        // else: curvature failure (very rare with strong Wolfe). Skip the
        // H update; the line search still produced a descent step, so we
        // continue. If this persists, max_iter or GradientTolerance halt.

        self.gradient_buffer = state.take_evaluation().map(|(_, g)| g);
        state
            .replace(param, cost_new, g_new)
            .expect("gradient dimension differs from parameter");
        Ok(crate::SolverStep::from((state, None)))
    }
}

/// A point warm start reevaluates the point and constructs the configured seed.
impl<V: DenseBackend<F> + Clone, F: Scalar, M, S, I> InitialState<V>
    for Bfgs<V, F, M, S, I>
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        FirstOrderState::new(x.clone())
    }
}

impl<V: DenseBackend<F> + Clone, F: Scalar, M, S, I> WarmStart<V>
    for Bfgs<V, F, M, S, I>
{
}
