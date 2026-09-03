//! Problem traits the user implements about their objective. Solvers
//! bind on whichever subset they need (e.g. gradient descent requires
//! [`CostFunction`] *and* [`Gradient`]; Nelder-Mead only needs
//! [`CostFunction`]).
//!
//! # Soft reject vs hard abort
//!
//! Every problem trait method returns `Result<_, Self::Error>`. The two
//! ways to signal "something went wrong" are *deliberately* distinct:
//!
//! - **Soft reject (`Ok(f64::INFINITY)`)**: return `+∞` from
//!   [`CostFunction::cost`] to reject a single point. Line searches treat
//!   it as worse and retreat; population solvers treat it as worst
//!   fitness. This is the right channel for "this `x` is outside my
//!   domain, but the solve should continue."
//! - **Hard abort (`Err(_)`)**: return `Err` to terminate the entire
//!   solve. The error bubbles all the way out of
//!   [`Executor::run`](crate::Executor::run) typed as
//!   `Result<_, P::Error>`. Use this when the failure is *not* about a
//!   particular `x`: a downstream service vanished, the user pressed
//!   cancel, an early-stopping criterion in the problem's own state fired.
//!
//! Problems that never fail in this way pick
//! `type Error = std::convert::Infallible;` (or
//! [`!`](https://doc.rust-lang.org/std/primitive.never.html) on nightly).
//! Niche optimization collapses `Result<f64, Infallible>` to `f64` layout,
//! so the happy path stays zero-cost.

/// Scalar-valued objective `f(x): Param → Output`. The smallest
/// problem trait: every solver binds at least on this.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`:
///   evaluating at the same `param` twice must return the same
///   `Output` (or the same `Err`). Solvers cache costs across iterations,
///   line searches reuse evaluations, and termination criteria assume the
///   cost they read from the state matches what a fresh `cost(param)`
///   would return.
/// - **Implementor must not:** assume any particular call order or
///   frequency. Solvers may evaluate at exploratory points outside the
///   accepted iterate sequence (line-search probes, Nelder-Mead
///   reflections/contractions/shrinks, finite-difference probes).
///
/// # Soft reject vs hard abort
///
/// See the [module docs](self#soft-reject-vs-hard-abort). Return
/// `Ok(f64::INFINITY)` to *reject one point*; return `Err(_)` to abort
/// the entire solve.
///
/// # Examples
///
/// A never-fails problem uses [`Infallible`](std::convert::Infallible) as
/// its error:
///
/// ```
/// use basin::CostFunction;
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
///
/// assert_eq!(Sphere.cost(&vec![3.0, 4.0]).unwrap(), 25.0);
/// ```
pub trait CostFunction {
    /// The parameter type the objective is defined over.
    type Param;
    /// Scalar cost type. In practice `f64` (see `CONTRIBUTING.md`'s
    /// provisional choices).
    type Output;
    /// User-chosen hard-abort error. Pick
    /// [`std::convert::Infallible`] when the cost cannot fail: its
    /// niche optimization keeps `Result<f64, Infallible>` the same
    /// layout as bare `f64` on the happy path.
    type Error;

    /// Evaluate the objective at `param`.
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error>;
}

/// Analytic gradient `∇f(x): Param → Gradient`. Required by
/// first-order solvers (gradient descent, BFGS, …).
///
/// `Gradient` is a *subtrait* of [`CostFunction`]: a gradient is the
/// gradient *of* a cost, so the parameter and error types are inherited
/// and the two cannot disagree.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a `Gradient` whose shape matches
///   `param` so solver math (`x ← x − α·∇f(x)`) lines up. Most
///   problems have `Gradient = Param`, which is what the shipped
///   solvers' bounds expect (e.g. `Gradient<Gradient = V>` paired with
///   `CostFunction<Param = V>`).
/// - The gradient must agree with [`CostFunction::cost`]: it is the
///   actual derivative, not a finite-difference approximation unless
///   the implementor is happy taking the loss in solver
///   convergence behavior.
///
/// # Fused evaluation
///
/// When a solver needs *both* `f(x)` and `∇f(x)` at the same point
/// (which it almost always does at the start of every iteration),
/// it calls [`cost_and_gradient`](Self::cost_and_gradient). The default
/// body simply calls [`CostFunction::cost`] and [`Gradient::gradient`]
/// in turn, which is the right answer for most problems and what
/// users get for free.
///
/// **Override `cost_and_gradient` when the two share substantial
/// intermediate work**: autodiff tapes, forward-mode adjoints,
/// neural-net activations, expensive simulation state. The default
/// then becomes a no-op and the solver picks up the fusion savings
/// without any further change.
///
/// Cost-only callers (line searches probing trial steps, cost-only
/// termination criteria, derivative-free solvers) keep calling
/// [`CostFunction::cost`] directly, with no waste from the fused method.
///
/// # Examples
///
/// ```
/// use basin::{CostFunction, Gradient};
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
/// }
///
/// assert_eq!(Sphere.gradient(&vec![1.0, 2.0]).unwrap(), vec![2.0, 4.0]);
/// ```
///
/// Fusion override (cost and gradient share `x * x`):
///
/// ```
/// use basin::{CostFunction, Gradient};
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
///     fn cost_and_gradient(
///         &self,
///         x: &Vec<f64>,
///     ) -> Result<(f64, Vec<f64>), std::convert::Infallible> {
///         // Single pass over x; the per-element work is shared.
///         let mut cost = 0.0;
///         let grad = x
///             .iter()
///             .map(|xi| {
///                 cost += xi * xi;
///                 2.0 * xi
///             })
///             .collect();
///         Ok((cost, grad))
///     }
/// }
/// ```
pub trait Gradient: CostFunction {
    /// The gradient type. Typically the same as
    /// [`CostFunction::Param`].
    type Gradient;

    /// Evaluate the gradient at `param`.
    fn gradient(
        &self,
        param: &Self::Param,
    ) -> Result<Self::Gradient, Self::Error>;

    /// Evaluate cost *and* gradient at `param` in one call. The default
    /// body delegates to [`CostFunction::cost`] and
    /// [`Gradient::gradient`]; override when shared intermediate work
    /// can be amortized across the two.
    ///
    /// **Contract.** The returned `(cost, gradient)` pair must equal
    /// what [`CostFunction::cost`] and [`Gradient::gradient`] would
    /// return separately at the same `param`. Solvers and the framework
    /// switch freely between the fused call and individual calls
    /// depending on what's needed at a given point; divergence breaks
    /// caching invariants.
    ///
    /// **Eval counting.** One fused call counts as one cost evaluation
    /// *and* one gradient evaluation: it produced both values, in the
    /// work of one fused evaluation.
    fn cost_and_gradient(
        &self,
        param: &Self::Param,
    ) -> Result<(Self::Output, Self::Gradient), Self::Error> {
        Ok((self.cost(param)?, self.gradient(param)?))
    }
}

/// Finite-sum gradient `(1/|B|) Σ_{i ∈ B} ∇fᵢ(x)` over a chosen subset
/// `B` of component samples. Required by mini-batch stochastic solvers
/// ([`Sgd`](crate::solver::Sgd)),
/// which call it once per step with a fresh batch of indices the solver
/// drew from its own [`ChaCha8Rng`](crate::core::rng::ChaCha8Rng).
///
/// `MiniBatchGradient` is a *subtrait* of [`CostFunction`]: the batch
/// gradient is the gradient of the *average* per-sample loss restricted
/// to `B`, so the parameter and error types are inherited.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param` and `batch`:
///   evaluating at the same `(param, batch)` twice must return the same
///   `Gradient` (or the same `Err`). Same call-order independence as
///   [`CostFunction::cost`].
/// - **Implementor must:** return the *averaged* batch gradient
///   `(1/|batch|) · Σ_{i ∈ batch} ∇fᵢ(param)`, not the unscaled sum.
///   This convention matches PyTorch/JAX/ensmallen and keeps the
///   solver's learning-rate `α` interpretation independent of batch
///   size: switching `batch_size` does not require rescaling `α`.
/// - **Implementor must:** return a `Gradient` whose shape matches
///   `param`, same as [`Gradient::gradient`].
/// - **Caller (solver) must:** pass a non-empty `batch` whose indices
///   are all in `0..self.n_samples()`. Implementors may rely on this
///   for indexing without bounds checks.
///
/// No fused `batch_cost_and_gradient` is shipped today: vanilla SGD
/// only consumes the gradient, and the only cost it evaluates is the
/// *full* objective via [`CostFunction::cost`] (so the cached
/// `state.cost` reflects the true value at the current iterate, not a
/// noisy batch estimate). Add a fused entry point alongside a solver
/// that consumes per-batch cost.
///
/// # Soft reject vs hard abort
///
/// Same split as the [module docs](self#soft-reject-vs-hard-abort).
/// `MiniBatchGradient::Error` is inherited from [`CostFunction::Error`].
///
/// # Examples
///
/// Linear regression `f(x) = (1/n) Σᵢ (aᵢ·x − bᵢ)²` with per-sample
/// gradient `2 (aᵢ·x − bᵢ) aᵢ`. Storing rows of `A` and entries of
/// `b` on the problem lets `batch_gradient` average over any subset:
///
/// ```
/// use basin::{CostFunction, MiniBatchGradient};
///
/// struct LinReg {
///     rows: Vec<Vec<f64>>,
///     y: Vec<f64>,
/// }
/// impl CostFunction for LinReg {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         let n = self.rows.len() as f64;
///         let mut s = 0.0;
///         for (a, &yi) in self.rows.iter().zip(self.y.iter()) {
///             let r = a.iter().zip(x).map(|(ai, xi)| ai * xi).sum::<f64>() - yi;
///             s += r * r;
///         }
///         Ok(s / n)
///     }
/// }
/// impl MiniBatchGradient for LinReg {
///     type Gradient = Vec<f64>;
///     fn n_samples(&self) -> usize {
///         self.rows.len()
///     }
///     fn batch_gradient(
///         &self,
///         x: &Vec<f64>,
///         batch: &[usize],
///     ) -> Result<Vec<f64>, Self::Error> {
///         let inv = 2.0 / batch.len() as f64;
///         let mut g = vec![0.0; x.len()];
///         for &i in batch {
///             let a = &self.rows[i];
///             let r = a.iter().zip(x).map(|(ai, xi)| ai * xi).sum::<f64>() - self.y[i];
///             for (gj, aj) in g.iter_mut().zip(a) {
///                 *gj += inv * r * aj;
///             }
///         }
///         Ok(g)
///     }
/// }
/// ```
pub trait MiniBatchGradient: CostFunction {
    /// The gradient type. Typically the same as
    /// [`CostFunction::Param`].
    type Gradient;

    /// Number of component samples `n` in the finite-sum objective
    /// `f(x) = (1/n) Σᵢ fᵢ(x)`. Fixed for a given problem instance, so
    /// solvers may cache it once at [`Solver::init`](crate::core::solver::Solver::init).
    fn n_samples(&self) -> usize;

    /// Averaged gradient over a batch of sample indices:
    /// `(1/|batch|) · Σ_{i ∈ batch} ∇fᵢ(param)`.
    ///
    /// `batch` is non-empty and every index satisfies
    /// `i < self.n_samples()` (the solver's responsibility).
    fn batch_gradient(
        &self,
        param: &Self::Param,
        batch: &[usize],
    ) -> Result<Self::Gradient, Self::Error>;
}

/// Vector-valued residual `r(x): Param → Output` for least-squares
/// problems. Required by Gauss-Newton, Levenberg-Marquardt, and any
/// solver that minimizes `½‖r(x)‖²`.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return an `Output` whose length `m` is fixed
///   for a given problem; `m` does not depend on the iterate. Solvers
///   may allocate workspace once based on the first call. `m` is
///   independent of `param.len() = n`.
/// - When [`CostFunction`] is also implemented, the cost must agree
///   with the residual under the convention `cost(x) = ½ Σ rᵢ(x)²`,
///   unless the problem documents an unscaled `Σ rᵢ²` form (see e.g.
///   the existing Rosenbrock cost, which is the published unscaled
///   form).
///
/// # Soft reject vs hard abort
///
/// `Residual` carries its *own* [`Error`](Residual::Error) (independent
/// of [`CostFunction::Error`]); the soft/hard split from the
/// [module docs](self#soft-reject-vs-hard-abort) applies identically.
/// NLLS solvers `?`-propagate residual errors and treat any `Err` as a
/// hard abort.
///
/// # Examples
///
/// ```
/// use basin::Residual;
///
/// // r(x) = (x₀ − 1, x₁ − 2); the least-squares optimum is (1, 2).
/// struct Affine;
/// impl Residual for Affine {
///     type Param = Vec<f64>;
///     type Output = Vec<f64>;
///     type Error = std::convert::Infallible;
///     fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(vec![x[0] - 1.0, x[1] - 2.0])
///     }
/// }
///
/// assert_eq!(
///     Affine.residual(&vec![0.0, 0.0]).unwrap(),
///     vec![-1.0, -2.0]
/// );
/// ```
pub trait Residual {
    /// The parameter type the residual is defined over (matches
    /// [`CostFunction::Param`]).
    type Param;
    /// The residual vector type. Length is the number of residuals `m`,
    /// independent of `param.len() = n`.
    type Output;
    /// User-chosen hard-abort error. Independent of
    /// [`CostFunction::Error`]: the trait families are orthogonal
    /// (NLLS solvers bind on `Residual` + [`Jacobian`]; first-order
    /// solvers bind on `CostFunction` + [`Gradient`]).
    type Error;

    /// Evaluate the residual at `param`.
    fn residual(
        &self,
        param: &Self::Param,
    ) -> Result<Self::Output, Self::Error>;
}

/// Analytic Jacobian `J(x) = ∂r/∂x: Param → Jacobian` for least-squares
/// solvers (Gauss-Newton, LM, TRF). The associated `Jacobian` matrix
/// type is what lets solvers bound on the linear-algebra ops they need
/// ([`MatVec`](crate::core::math::MatVec),
/// [`LinearSolveSpd`](crate::core::math::LinearSolveSpd), …) without
/// baking in a specific backend or assuming density.
///
/// `Jacobian` is a *subtrait* of [`Residual`]: a Jacobian is the
/// Jacobian *of* a residual, so the parameter and error types are
/// inherited.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a matrix of shape `m × n` where
///   `m = residual(param).len()` and `n = param.len()`. The `(i, j)`
///   entry is `∂rᵢ / ∂xⱼ`. Shape is fixed across iterates.
/// - The Jacobian must agree with [`Residual::residual`]: it is the
///   actual derivative, not a finite-difference approximation, unless
///   the implementor accepts the loss in solver convergence behavior.
///
/// # Fused evaluation
///
/// NLLS solvers (Gauss-Newton, LM, TRF) evaluate `r(x)` and `J(x)`
/// together at every accepted iterate, and `r(x)` is usually the
/// dominant cost, with `J(x)` reusing intermediate state (forward-mode
/// AD on the residual graph, FE assembly, simulation adjoints).
/// [`residual_and_jacobian`](Self::residual_and_jacobian) provides the
/// fused entry point. The default body calls [`Residual::residual`] and
/// [`Jacobian::jacobian`] in turn; override when work can be shared.
///
/// # Backends
///
/// Every dense backend pairs its param vector with an honest matrix type:
///
/// - `Param = Vec<f64>` → `Jacobian = `[`DenseMatrix<f64>`](crate::DenseMatrix),
///   the default backend; its [`LinearSolveSpd`](crate::core::math::LinearSolveSpd)
///   is a pure-Rust Cholesky, wasm-clean with no BLAS/LAPACK.
/// - `Param = nalgebra::DVector<f64>` → `Jacobian = nalgebra::DMatrix<f64>`
///   (dense) or `nalgebra_sparse::CscMatrix<f64>` (sparse). Both ride
///   on the `nalgebra` feature.
/// - `Param = faer::Col<f64>` → `Jacobian = faer::Mat<f64>` (dense) or
///   `faer::sparse::SparseColMat<usize, f64>` (sparse). Both ride on
///   the `faer` feature.
/// - `Param = ndarray::Array1<f64>` → `Jacobian = ndarray::Array2<f64>`,
///   on the `ndarray` feature; `Array2` reuses the same pure-Rust Cholesky
///   as `DenseMatrix` (no `ndarray-linalg`/BLAS, so the wasm-default tenet
///   holds).
///
/// Per tenet 5 in `CONTRIBUTING.md`, a backend that lacks the matrix ops a
/// least-squares solver needs is a compile-time error rather than a runtime
/// surprise.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "nalgebra_v0_35")] {
/// use basin::{Jacobian, Residual};
/// use nalgebra::{DMatrix, DVector};
///
/// struct Affine;
/// impl Residual for Affine {
///     type Param = DVector<f64>;
///     type Output = DVector<f64>;
///     type Error = std::convert::Infallible;
///     fn residual(
///         &self,
///         x: &DVector<f64>,
///     ) -> Result<DVector<f64>, std::convert::Infallible> {
///         Ok(DVector::from_vec(vec![x[0] - 1.0, x[1] - 2.0]))
///     }
/// }
/// impl Jacobian for Affine {
///     type Jacobian = DMatrix<f64>;
///     fn jacobian(
///         &self,
///         _x: &DVector<f64>,
///     ) -> Result<DMatrix<f64>, std::convert::Infallible> {
///         Ok(DMatrix::identity(2, 2))
///     }
/// }
///
/// let j = Affine.jacobian(&DVector::from_vec(vec![0.0, 0.0])).unwrap();
/// assert_eq!(j[(0, 0)], 1.0);
/// # }
/// ```
pub trait Jacobian: Residual {
    /// The Jacobian matrix type, shape `m × n`.
    type Jacobian;

    /// Evaluate the Jacobian at `param`.
    fn jacobian(
        &self,
        param: &Self::Param,
    ) -> Result<Self::Jacobian, <Self as Residual>::Error>;

    /// Evaluate residual *and* Jacobian at `param` in one call. The
    /// default body delegates to [`Residual::residual`] and
    /// [`Jacobian::jacobian`]; override when shared intermediate work
    /// can be amortized across the two, common in NLLS where `r(x)`
    /// reuses forward-mode AD state that `J(x)` continues from.
    ///
    /// **Contract.** The returned `(residual, jacobian)` pair must
    /// equal what [`Residual::residual`] and [`Jacobian::jacobian`]
    /// would return separately at the same `param`.
    ///
    /// **Eval counting.** NLLS solvers count one fused call as one
    /// `cost_evals` *and* one `gradient_evals` increment: the same
    /// convention solvers use for separate calls, because `½‖r‖²`
    /// plays the role of cost and `Jᵀr` the role of gradient.
    fn residual_and_jacobian(
        &self,
        param: &Self::Param,
    ) -> Result<
        (<Self as Residual>::Output, Self::Jacobian),
        <Self as Residual>::Error,
    > {
        Ok((self.residual(param)?, self.jacobian(param)?))
    }
}

/// Analytic Hessian `H(x) = ∇²f(x): Param → Hessian` for second-order
/// solvers (Newton, trust-region-Newton). The associated `Hessian`
/// matrix type lets solvers bound on the linear-algebra ops they need
/// ([`LinearSolveSpd`](crate::core::math::LinearSolveSpd),
/// [`SymmetricEigen`](crate::core::math::SymmetricEigen), …) without
/// baking in a backend.
///
/// `Hessian` is a *subtrait* of [`Gradient`] (which is a subtrait of
/// [`CostFunction`]): a Hessian is the second derivative of a cost.
/// The error type is therefore [`CostFunction::Error`].
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a **symmetric** `n × n` matrix where
///   `n = param.len()`. The `(i, j)` entry is `∂²f / ∂xᵢ∂xⱼ`. Shape is
///   fixed across iterates.
/// - The Hessian must agree with [`CostFunction::cost`] and
///   [`Gradient::gradient`]: it is the actual second derivative, not a
///   finite-difference approximation, unless the implementor accepts
///   the loss in solver convergence behavior.
///
/// # Fused evaluation
///
/// Second-order solvers evaluate `f`, `∇f`, and `∇²f` together at
/// every accepted iterate. The
/// [`cost_and_gradient_and_hessian`](Self::cost_and_gradient_and_hessian)
/// method provides the fused entry point. The default body composes
/// [`Gradient::cost_and_gradient`] with [`Hessian::hessian`]; override
/// when all three share intermediate state.
///
/// # Backends
///
/// Wired up for the LA-heavy backends only, mirroring [`Jacobian`]:
///
/// - `Param = nalgebra::DVector<f64>` → `Hessian = nalgebra::DMatrix<f64>`
///   (rides on the `nalgebra` feature).
/// - `Param = faer::Col<f64>` → `Hessian = faer::Mat<f64>` (rides on
///   the `faer` feature).
///
/// `Vec<f64>` and `ndarray::Array1<f64>` deliberately have no `Hessian`
/// impl: there's no honest dense matrix type to pair with them. Per
/// tenet 5 in `CONTRIBUTING.md`, missing backend coverage is a compile-time
/// error rather than a runtime surprise.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "nalgebra_v0_35")] {
/// use basin::{CostFunction, Gradient, Hessian};
/// use nalgebra::{DMatrix, DVector};
///
/// // f(x) = x₀² + x₁² has constant Hessian 2·I.
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = DVector<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.dot(x))
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = DVector<f64>;
///     fn gradient(
///         &self,
///         x: &DVector<f64>,
///     ) -> Result<DVector<f64>, std::convert::Infallible> {
///         Ok(2.0 * x)
///     }
/// }
/// impl Hessian for Sphere {
///     type Hessian = DMatrix<f64>;
///     fn hessian(
///         &self,
///         x: &DVector<f64>,
///     ) -> Result<DMatrix<f64>, std::convert::Infallible> {
///         Ok(2.0 * DMatrix::identity(x.len(), x.len()))
///     }
/// }
///
/// let h = Sphere.hessian(&DVector::from_vec(vec![1.0, 1.0])).unwrap();
/// assert_eq!(h[(0, 0)], 2.0);
/// # }
/// ```
pub trait Hessian: Gradient {
    /// The Hessian matrix type, shape `n × n` and symmetric.
    type Hessian;

    /// Evaluate the Hessian at `param`.
    fn hessian(
        &self,
        param: &Self::Param,
    ) -> Result<Self::Hessian, <Self as CostFunction>::Error>;

    /// Evaluate cost, gradient, *and* Hessian at `param` in one call.
    /// The default body delegates to [`Gradient::cost_and_gradient`]
    /// followed by [`Hessian::hessian`]; override when all three share
    /// intermediate work.
    ///
    /// **Contract.** The returned triple must equal what the three
    /// methods would return separately at the same `param`.
    #[allow(clippy::type_complexity)]
    fn cost_and_gradient_and_hessian(
        &self,
        param: &Self::Param,
    ) -> Result<
        (
            <Self as CostFunction>::Output,
            <Self as Gradient>::Gradient,
            Self::Hessian,
        ),
        <Self as CostFunction>::Error,
    > {
        let (cost, grad) = self.cost_and_gradient(param)?;
        Ok((cost, grad, self.hessian(param)?))
    }
}

/// Matrix-free Hessian-vector products: `v ↦ ∇²f(param) · v`.
///
/// Implement this when the Hessian is too large to form (or a product is
/// simply cheaper), so matrix-free solvers like
/// [`TrustRegion`](crate::solver::TrustRegion) in
/// [`MatrixFree`](crate::solver::trust_region::MatrixFree) mode can drive
/// second-order optimization without a matrix type anywhere. Deliberately
/// *not* a subtrait of [`Hessian`]: the point is problems that cannot (or
/// should not) materialize `∇²f`.
///
/// # Contract
///
/// - The product must equal `∇²f(param) · v` for the *same* `f` whose
///   derivative the [`Gradient`] impl returns.
/// - It must be a pure function of `(param, v)` and linear in `v`.
///
/// Standard sources of an implementation are analytic derivation,
/// reverse-over-forward automatic differentiation, and the
/// gradient-difference approximation `(∇f(x + h v) − ∇f(x)) / h`
/// (Nocedal & Wright, 2nd ed., eq. 8.20); the latter ships as
/// [`forward_difference_hessian_product`](crate::core::numdiff::forward_difference_hessian_product)
/// and its central-difference sibling.
///
/// # Backends
///
/// No matrix type is involved, so there is nothing to wire per backend:
/// any `Param` works, including `Vec<f64>` and `ndarray::Array1<f64>`,
/// which have no [`Hessian`] impl at all.
///
/// A problem that also implements [`Hessian`] with a
/// [`MatVec`](crate::core::math::MatVec)-capable matrix can forward:
/// `fn hessian_product(&self, x, v) { Ok(self.hessian(x)?.matvec(v)) }`.
/// There is deliberately no blanket impl doing this, so such problems can
/// still provide a cheaper hand-written product.
///
/// # Examples
///
/// ```
/// use basin::{CostFunction, Gradient, HessianProduct};
///
/// // f(x) = x₀² + x₁² has constant Hessian 2·I, so ∇²f(x)·v = 2·v.
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
/// }
/// impl HessianProduct for Sphere {
///     fn hessian_product(
///         &self,
///         _x: &Vec<f64>,
///         v: &Vec<f64>,
///     ) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(v.iter().map(|vi| 2.0 * vi).collect())
///     }
/// }
///
/// let hv = Sphere.hessian_product(&vec![3.0, 4.0], &vec![1.0, 0.0]).unwrap();
/// assert_eq!(hv, vec![2.0, 0.0]);
/// ```
pub trait HessianProduct: Gradient {
    /// Evaluate the Hessian-vector product `∇²f(param) · v`.
    fn hessian_product(
        &self,
        param: &Self::Param,
        v: &Self::Param,
    ) -> Result<<Self as Gradient>::Gradient, <Self as CostFunction>::Error>;
}

/// Per-kind evaluation counters carried by [`Problem`].
///
/// One field per problem-trait method family. The
/// [`Executor`](crate::core::executor::Executor) mirrors these onto the
/// solver `State` after every successful
/// [`Solver::next_iter`](crate::core::solver::Solver::next_iter) /
/// [`Solver::init`](crate::core::solver::Solver::init), with the per-state
/// rule defined by the state's [`CountsMirror`](crate::core::state::CountsMirror)
/// impl. The wrapper itself is authoritative; the state mirror is the
/// "available-everywhere" view that termination criteria and
/// [`OptimizationResult`](crate::core::executor::OptimizationResult) read.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalCounts {
    /// [`CostFunction::cost`] calls (including the cost side of fused
    /// [`Gradient::cost_and_gradient`]/[`Hessian::cost_and_gradient_and_hessian`]).
    pub cost_evals: u64,
    /// [`Gradient::gradient`] calls (including the gradient side of fused
    /// calls).
    pub gradient_evals: u64,
    /// [`Residual::residual`] calls (including the residual side of fused
    /// [`Jacobian::residual_and_jacobian`]).
    pub residual_evals: u64,
    /// [`Jacobian::jacobian`] calls (including the Jacobian side of fused
    /// calls).
    pub jacobian_evals: u64,
    /// [`Hessian::hessian`] calls (including the Hessian side of fused
    /// calls).
    pub hessian_evals: u64,
    /// [`HessianProduct::hessian_product`] calls. One call is one
    /// Hessian-vector product — `O(n)` work, roughly one gradient — as
    /// opposed to [`hessian_evals`](Self::hessian_evals), where one call
    /// forms the full `O(n²)` matrix.
    pub hessian_product_evals: u64,
}

impl EvalCounts {
    /// Sum across every counter. Used for state-mirror rules that fold all
    /// problem work into a single `state.cost_evals` (derivative-free outer
    /// states; see [`CountsMirror`](crate::core::state::CountsMirror)).
    pub fn total_work(&self) -> u64 {
        self.cost_evals
            + self.gradient_evals
            + self.residual_evals
            + self.jacobian_evals
            + self.hessian_evals
            + self.hessian_product_evals
    }

    /// Componentwise `self − base`. Used by
    /// [`run_loop`](crate::core::executor::run_loop) to compute the
    /// per-run delta when an outer solver passes its wrapper to an inner.
    pub fn delta_since(&self, base: &EvalCounts) -> EvalCounts {
        EvalCounts {
            cost_evals: self.cost_evals - base.cost_evals,
            gradient_evals: self.gradient_evals - base.gradient_evals,
            residual_evals: self.residual_evals - base.residual_evals,
            jacobian_evals: self.jacobian_evals - base.jacobian_evals,
            hessian_evals: self.hessian_evals - base.hessian_evals,
            hessian_product_evals: self.hessian_product_evals
                - base.hessian_product_evals,
        }
    }

    /// Componentwise `self + other`. Used by composed solvers that drive an
    /// inner against a *different* problem type (an adapter like
    /// [`LogBarrier`](crate::core::barrier::LogBarrier) or
    /// [`AugmentedLagrangian`](crate::core::augmented_lagrangian::AugmentedLagrangian)):
    /// they construct a fresh inner [`Problem`] and merge its counts back
    /// into the outer's wrapper after [`run_loop`](crate::core::executor::run_loop).
    pub fn add(&mut self, other: &EvalCounts) {
        self.cost_evals += other.cost_evals;
        self.gradient_evals += other.gradient_evals;
        self.residual_evals += other.residual_evals;
        self.jacobian_evals += other.jacobian_evals;
        self.hessian_evals += other.hessian_evals;
        self.hessian_product_evals += other.hessian_product_evals;
    }
}

/// Counting wrapper that solvers receive instead of `&P` directly.
///
/// Every problem-trait method on [`Problem`] bumps the relevant
/// [`EvalCounts`] field before delegating to the inner problem, so
/// solvers can't accidentally lose a count: forgetting to count becomes
/// a compile error (the inner is private; the only way to evaluate the
/// problem is through the wrapper). The
/// [`Executor`](crate::core::executor::Executor) wraps the user's
/// problem once in [`Executor::new`](crate::core::executor::Executor::new)
/// and mirrors [`counts`](Self::counts) onto the solver `State` after
/// every successful
/// [`Solver::init`](crate::core::solver::Solver::init) /
/// [`Solver::next_iter`](crate::core::solver::Solver::next_iter).
///
/// # Composition
///
/// Outer solvers that drive an inner solver receive their own
/// `&mut Problem<P>` in `next_iter`. Two shapes are supported:
///
/// - **Same-problem inner** (e.g.
///   [`CmaInject`](crate::solver::CmaInject)): the outer passes its own
///   `&mut Problem<P>` straight through to the inner via
///   [`run_loop`](crate::core::executor::run_loop). Inner counts flow
///   into the outer's wrapper transparently; no explicit roll-up. Inner
///   `state` counts reflect per-run work (snapshot-relative, computed by
///   [`run_loop`](crate::core::executor::run_loop)).
/// - **Adapter-problem inner** (e.g.
///   [`BarrierMethod`](crate::solver::BarrierMethod) /
///   [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)):
///   the outer constructs a fresh `Problem::new(adapter)` around its
///   adapter type, runs the inner against it, then folds the inner's
///   counts back into the outer's wrapper via
///   [`EvalCounts::add`].
///
/// # Error path
///
/// Counters bump **before** the delegated inner call. A mid-call `Err`
/// therefore still leaves the wrapper count incremented: the wrapper is
/// authoritative even on the hard-abort path, where the state mirror may
/// be stale. Observers can read the true count via
/// [`counts`](Self::counts) regardless.
pub struct Problem<P> {
    inner: P,
    counts: EvalCounts,
}

impl<P> Problem<P> {
    /// Wrap `inner` with fresh zero counters.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            counts: EvalCounts::default(),
        }
    }

    /// Read-only access to the wrapped problem. Used by composed solvers
    /// to build adapter problems (`LogBarrier::new(wrapper.inner(), μ)`,
    /// `AugmentedLagrangian::new(wrapper.inner(), …)`) and by callers that
    /// need to read non-evaluation methods (e.g.
    /// [`BoxConstraints::lower`](crate::core::constraint::BoxConstraints::lower)).
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Current evaluation counters. The wrapper is authoritative;
    /// observers can read counts here even on `Err` paths where the
    /// state mirror has not refreshed.
    pub fn counts(&self) -> &EvalCounts {
        &self.counts
    }

    /// Mutable access to the wrapper's counters. Used by composed solvers
    /// that drive an inner against an *adapter problem*: after
    /// [`run_loop`](crate::core::executor::run_loop) returns the inner
    /// wrapper, the outer calls
    /// `outer.counts_mut().add(inner.counts())` to fold the inner's
    /// per-run work into its own.
    pub fn counts_mut(&mut self) -> &mut EvalCounts {
        &mut self.counts
    }

    /// Consume the wrapper and return the inner problem.
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P: CostFunction> Problem<P> {
    /// Counted [`CostFunction::cost`].
    pub fn cost(&mut self, param: &P::Param) -> Result<P::Output, P::Error> {
        self.counts.cost_evals += 1;
        self.inner.cost(param)
    }

    /// Counted batch cost evaluation: evaluate the inner
    /// [`CostFunction::cost`] at every `param` and collect the outputs in
    /// input order.
    ///
    /// Bumps [`EvalCounts::cost_evals`] by `params.len()` once, on the
    /// calling thread, then fans the evaluations across the `rayon` pool
    /// when the `parallel` feature is on (sequential otherwise). The counter
    /// touch happens before the fan-out and the parallel closure only borrows
    /// the inner problem immutably, so there is no shared mutable state across
    /// threads; the total it adds is identical to one
    /// [`cost`](Self::cost) per element. Results are collected in slice order,
    /// so the returned costs are bit-identical to a serial loop whether or not
    /// `parallel` is enabled.
    ///
    /// Population solvers use this for per-generation fitness evaluation:
    /// the λ candidates of one generation are independent, so they evaluate
    /// concurrently. Short-circuits on the first `Err` (hard abort).
    pub fn cost_batch(
        &mut self,
        params: &[P::Param],
    ) -> Result<Vec<P::Output>, P::Error>
    where
        P: crate::core::parallel::MaybeSync,
        P::Param: crate::core::parallel::MaybeSync,
        P::Output: crate::core::parallel::MaybeSend,
        P::Error: crate::core::parallel::MaybeSend,
    {
        self.counts.cost_evals += params.len() as u64;
        let inner = &self.inner;
        crate::core::parallel::try_map_slice_with(
            params,
            || (),
            |(), p| inner.cost(p),
        )
    }
}

impl<P: Gradient> Problem<P> {
    /// Counted [`Gradient::gradient`].
    pub fn gradient(
        &mut self,
        param: &P::Param,
    ) -> Result<P::Gradient, P::Error> {
        self.counts.gradient_evals += 1;
        self.inner.gradient(param)
    }

    /// Counted [`Gradient::cost_and_gradient`]: bumps both
    /// [`EvalCounts::cost_evals`] and
    /// [`EvalCounts::gradient_evals`] in one place, so the
    /// "one fused call counts as one of each" rule lives in exactly one
    /// spot.
    pub fn cost_and_gradient(
        &mut self,
        param: &P::Param,
    ) -> Result<(P::Output, P::Gradient), P::Error> {
        self.counts.cost_evals += 1;
        self.counts.gradient_evals += 1;
        self.inner.cost_and_gradient(param)
    }
}

impl<P: MiniBatchGradient> Problem<P> {
    /// Counted [`MiniBatchGradient::batch_gradient`]: one call bumps
    /// [`EvalCounts::gradient_evals`] by one, regardless of batch
    /// size, the same convention as [`Gradient::gradient`] (one *call* is
    /// one gradient evaluation; the per-sample work is implementation
    /// detail).
    pub fn batch_gradient(
        &mut self,
        param: &P::Param,
        batch: &[usize],
    ) -> Result<<P as MiniBatchGradient>::Gradient, P::Error> {
        self.counts.gradient_evals += 1;
        self.inner.batch_gradient(param, batch)
    }
}

impl<P: Residual> Problem<P> {
    /// Counted [`Residual::residual`].
    pub fn residual(
        &mut self,
        param: &P::Param,
    ) -> Result<<P as Residual>::Output, <P as Residual>::Error> {
        self.counts.residual_evals += 1;
        self.inner.residual(param)
    }
}

impl<P: Jacobian> Problem<P> {
    /// Counted [`Jacobian::jacobian`].
    pub fn jacobian(
        &mut self,
        param: &P::Param,
    ) -> Result<P::Jacobian, <P as Residual>::Error> {
        self.counts.jacobian_evals += 1;
        self.inner.jacobian(param)
    }

    /// Counted [`Jacobian::residual_and_jacobian`]: bumps both
    /// [`EvalCounts::residual_evals`] and
    /// [`EvalCounts::jacobian_evals`].
    pub fn residual_and_jacobian(
        &mut self,
        param: &P::Param,
    ) -> Result<(<P as Residual>::Output, P::Jacobian), <P as Residual>::Error>
    {
        self.counts.residual_evals += 1;
        self.counts.jacobian_evals += 1;
        self.inner.residual_and_jacobian(param)
    }
}

impl<P: Hessian> Problem<P> {
    /// Counted [`Hessian::hessian`].
    pub fn hessian(
        &mut self,
        param: &P::Param,
    ) -> Result<P::Hessian, <P as CostFunction>::Error> {
        self.counts.hessian_evals += 1;
        self.inner.hessian(param)
    }

    /// Counted [`Hessian::cost_and_gradient_and_hessian`]: bumps
    /// [`EvalCounts::cost_evals`],
    /// [`EvalCounts::gradient_evals`], and
    /// [`EvalCounts::hessian_evals`].
    #[allow(clippy::type_complexity)]
    pub fn cost_and_gradient_and_hessian(
        &mut self,
        param: &P::Param,
    ) -> Result<
        (
            <P as CostFunction>::Output,
            <P as Gradient>::Gradient,
            P::Hessian,
        ),
        <P as CostFunction>::Error,
    > {
        self.counts.cost_evals += 1;
        self.counts.gradient_evals += 1;
        self.counts.hessian_evals += 1;
        self.inner.cost_and_gradient_and_hessian(param)
    }
}

impl<P: HessianProduct> Problem<P> {
    /// Counted [`HessianProduct::hessian_product`].
    pub fn hessian_product(
        &mut self,
        param: &P::Param,
        v: &P::Param,
    ) -> Result<<P as Gradient>::Gradient, <P as CostFunction>::Error> {
        self.counts.hessian_product_evals += 1;
        self.inner.hessian_product(param, v)
    }
}

#[cfg(test)]
mod problem_wrapper_tests {
    use super::*;

    struct Sphere;
    impl CostFunction for Sphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x.iter().map(|xi| xi * xi).sum())
        }
    }
    impl Gradient for Sphere {
        type Gradient = Vec<f64>;
        fn gradient(
            &self,
            x: &Vec<f64>,
        ) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(x.iter().map(|xi| 2.0 * xi).collect())
        }
    }

    #[test]
    fn counts_cost_calls() {
        let mut p = Problem::new(Sphere);
        let _ = p.cost(&vec![1.0, 2.0]).unwrap();
        let _ = p.cost(&vec![0.0, 0.0]).unwrap();
        assert_eq!(p.counts().cost_evals, 2);
        assert_eq!(p.counts().gradient_evals, 0);
    }

    #[test]
    fn counts_gradient_calls() {
        let mut p = Problem::new(Sphere);
        let _ = p.gradient(&vec![1.0, 2.0]).unwrap();
        assert_eq!(p.counts().cost_evals, 0);
        assert_eq!(p.counts().gradient_evals, 1);
    }

    #[test]
    fn fused_counts_as_one_of_each() {
        let mut p = Problem::new(Sphere);
        let _ = p.cost_and_gradient(&vec![1.0, 2.0]).unwrap();
        assert_eq!(p.counts().cost_evals, 1);
        assert_eq!(p.counts().gradient_evals, 1);
    }

    #[test]
    fn delta_since_subtracts_componentwise() {
        let mut p = Problem::new(Sphere);
        let _ = p.cost(&vec![1.0]);
        let base = *p.counts();
        let _ = p.cost_and_gradient(&vec![1.0]).unwrap();
        let _ = p.gradient(&vec![1.0]).unwrap();
        let delta = p.counts().delta_since(&base);
        assert_eq!(delta.cost_evals, 1);
        assert_eq!(delta.gradient_evals, 2);
    }

    #[test]
    fn total_work_sums_all_kinds() {
        let mut p = Problem::new(Sphere);
        let _ = p.cost(&vec![1.0]).unwrap();
        let _ = p.gradient(&vec![1.0]).unwrap();
        assert_eq!(p.counts().total_work(), 2);
    }

    #[test]
    fn add_merges_componentwise() {
        let mut a = EvalCounts {
            cost_evals: 3,
            gradient_evals: 1,
            ..EvalCounts::default()
        };
        let b = EvalCounts {
            cost_evals: 2,
            jacobian_evals: 5,
            ..EvalCounts::default()
        };
        a.add(&b);
        assert_eq!(a.cost_evals, 5);
        assert_eq!(a.gradient_evals, 1);
        assert_eq!(a.jacobian_evals, 5);
    }

    impl HessianProduct for Sphere {
        fn hessian_product(
            &self,
            _x: &Vec<f64>,
            v: &Vec<f64>,
        ) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(v.iter().map(|vi| 2.0 * vi).collect())
        }
    }

    #[test]
    fn counts_hessian_product_calls() {
        let mut p = Problem::new(Sphere);
        let hv = p.hessian_product(&vec![1.0, 2.0], &vec![1.0, 0.0]).unwrap();
        assert_eq!(hv, vec![2.0, 0.0]);
        assert_eq!(p.counts().hessian_product_evals, 1);
        assert_eq!(p.counts().cost_evals, 0);
        assert_eq!(p.counts().gradient_evals, 0);
        assert_eq!(p.counts().hessian_evals, 0);
        assert_eq!(p.counts().total_work(), 1);
    }
}
