//! Finite-difference derivative synthesis.
//!
//! [`FiniteDiff`] wraps a problem that only exposes function values
//! ([`CostFunction`] and/or [`Residual`]) and *adds* the derivative traits
//! the solvers want: [`Gradient`] (for first-order solvers), [`Jacobian`]
//! (for least-squares solvers), [`Hessian`] (for second-order solvers), and
//! [`HessianProduct`] (for matrix-free second-order solvers).
//! Each derivative is approximated by finite differences of the wrapped
//! problem, so a values-only problem flows straight into the existing
//! solvers via basin's type-system dispatch.
//!
//! Separately from the wrapper, the [`forward_difference_hessian_product`] /
//! [`central_difference_hessian_product`] free functions are bounded on
//! `P: Gradient`, so a problem with an *analytic* gradient can implement
//! [`HessianProduct`] as a one-liner (2 gradient evaluations per product)
//! without stacking finite differences.
//!
//! The wrapper *forwards* [`BoxConstraints`] when the inner problem carries
//! box bounds: adding derivatives must not silently un-constrain a problem
//! (tenet 4 in `CONTRIBUTING.md`: an adapter that *adds* a capability preserves
//! the rest).
//!
//! # Step sizes (paper-anchored)
//!
//! - **Gradient** and **Hessian** use Numerical-Recipes-style adaptive
//!   steps `hⱼ = scale · max(|xⱼ|, 1)`, with `scale = eps_f^{1/3}` (central
//!   gradient), `eps_f^{1/2}` (forward gradient), or `eps_f^{1/4}` (Hessian),
//!   where `eps_f = function_precision.max(f64::EPSILON)` is the assumed
//!   relative accuracy of the function. The step is re-rounded so `xⱼ + hⱼ`
//!   is exactly representable.
//! - **Forward Jacobian** reproduces MINPACK `fdjac2` exactly:
//!   `eps = sqrt(eps_f)`, `hⱼ = eps · |xⱼ|`, and `hⱼ = eps` when `xⱼ = 0`.
//!   This is the function-values-only counterpart of MINPACK `lmder`, i.e.
//!   the `lmdif` Jacobian, chosen as the default so least-squares fits via
//!   `FiniteDiff` match `lmdif`'s convergence lineage.
//!
//! # Caveats
//!
//! - **Evaluation counting.** One counted [`Gradient::gradient`],
//!   [`Jacobian::jacobian`], or [`Hessian::hessian`] call performs *many*
//!   internal cost or residual evaluations (`2n` central/`n+1` forward
//!   gradient, `n+1`/`2n` Jacobian columns, `~2n²` Hessian). Eval counting
//!   lives on the solver `State` and is incremented by the solver per
//!   *derivative* call, so a problem wrapper has no way to record the inner
//!   calls, so `result.cost_evals()` will **not** reflect them. Users who need
//!   true cost-evaluation budgets should account for the `O(n)`/`O(n²)`
//!   multiplier themselves.
//! - **Domain & hard-abort.** Each probe `?`-propagates the inner
//!   function's [`Error`](CostFunction::Error). If the inner `cost` /
//!   `residual` returns `Err`, the derivative call returns the same `Err`:
//!   no swallowing, no partial result. A point where the inner function
//!   *soft-rejects* (`Ok(f64::INFINITY)`) poisons the derivative with `±∞`
//!   rather than aborting; keeping probes inside the strict domain
//!   (e.g. via [`FiniteDiff::function_precision`] or
//!   [`FiniteDiff::with_step`]) is the caller's responsibility.
//!
//! # Parallelism (`parallel` feature)
//!
//! The probes of one derivative are independent, so with the `parallel`
//! feature they fan across a `rayon` thread pool: the `n` gradient/Jacobian
//! columns, and the `n(n+1)/2` upper-triangular Hessian entries, are evaluated
//! concurrently (the shared `f(x)`/`r(x)` base evaluation runs once up
//! front). The result is **bit-identical** to the sequential path (each
//! output slot is written independently and collected in order, with no
//! floating-point reduction), so enabling the feature never changes the
//! numbers, only the wall-clock.
//!
//! This is a win when the wrapped `cost`/`residual` is expensive; for a
//! cheap function at small `n` the thread-pool overhead can exceed the saving,
//! so it is opt-in. Under the feature, `FiniteDiff`'s [`Gradient`] /
//! [`Jacobian`]/[`Hessian`] impls additionally require the wrapped problem
//! (and its param vector/error type) to be [`Sync`]/[`Send`]; without the
//! feature there is no such bound and the evaluation is sequential.

use crate::core::constraint::BoxConstraints;
use crate::core::math::{DenseMatrixFromFn, VectorIndex, VectorLen};
use crate::core::parallel::{MaybeSend, MaybeSync, try_map_range_with, try_map_slice_with};
use crate::core::problem::{CostFunction, Gradient, Hessian, HessianProduct, Jacobian, Residual};

/// Which finite-difference stencil to use for a given derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// One-sided difference. Cheaper (`n+1` evals for a gradient), `O(h)`
    /// truncation error. The MINPACK `fdjac2` stencil for Jacobians.
    Forward,
    /// Two-sided (central) difference. More accurate (`O(h²)` truncation
    /// error) at twice the cost.
    Central,
}

/// Wraps a problem to synthesize its derivatives by finite differences.
///
/// Construct with [`FiniteDiff::new`] (central gradient and Hessian, forward
/// Jacobian; see the [module docs](self)) and adjust with the builder
/// methods. The wrapper delegates [`CostFunction`]/[`Residual`] /
/// [`BoxConstraints`] to the inner problem and implements [`Gradient`] /
/// [`Jacobian`]/[`Hessian`] via finite differences.
///
/// # Backends
///
/// [`Gradient`] and [`HessianProduct`] are backend-generic (any `V: Clone +
/// VectorLen + VectorIndex`; no matrix type is involved). [`Jacobian`] and
/// [`Hessian`] additionally require `V: DenseMatrixFromFn`, so they are
/// available only for the matrix backends (nalgebra `DVector → DMatrix`,
/// faer `Col → Mat`); `Vec<f64>` and `ndarray` produce a compile-time
/// error, mirroring the analytic [`Jacobian`]/[`Hessian`] coverage
/// (tenet 5).
///
/// # Examples
///
/// Run a gradient solver against a problem that only implements
/// [`CostFunction`]: wrapping it in [`FiniteDiff`] synthesizes the
/// [`Gradient`] by central differences.
///
/// ```
/// use basin::{BasicState, CostFunction, Executor, FiniteDiff, GradientDescent, GradientTolerance};
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
/// let result = Executor::new(
///     FiniteDiff::new(Sphere),
///     GradientDescent::new(0.1),
///     BasicState::new(vec![1.0, 1.0]),
/// )
/// .max_iter(1_000)
/// .terminate_on(GradientTolerance(1e-8))
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FiniteDiff<P> {
    problem: P,
    gradient_method: Method,
    jacobian_method: Method,
    hessian_method: Method,
    function_precision: f64,
    fixed_step: Option<f64>,
}

impl<P> FiniteDiff<P> {
    /// Wrap `problem` with default settings: central-difference gradient
    /// and Hessian, forward-difference (MINPACK `fdjac2`) Jacobian,
    /// `function_precision = f64::EPSILON`, adaptive step sizes.
    pub fn new(problem: P) -> Self {
        Self {
            problem,
            gradient_method: Method::Central,
            jacobian_method: Method::Forward,
            hessian_method: Method::Central,
            function_precision: f64::EPSILON,
            fixed_step: None,
        }
    }

    /// Set the stencil used for the gradient (default [`Method::Central`]).
    pub fn gradient_method(mut self, method: Method) -> Self {
        self.gradient_method = method;
        self
    }

    /// Set the stencil used for the Jacobian (default [`Method::Forward`],
    /// the MINPACK `fdjac2` parity choice).
    pub fn jacobian_method(mut self, method: Method) -> Self {
        self.jacobian_method = method;
        self
    }

    /// Set the stencil used for the Hessian (default [`Method::Central`]).
    pub fn hessian_method(mut self, method: Method) -> Self {
        self.hessian_method = method;
        self
    }

    /// Set the assumed relative accuracy of the wrapped function (MINPACK's
    /// `epsfcn`). Larger values widen the step, which helps when the
    /// function is noisy. Floored at `f64::EPSILON`. Default `f64::EPSILON`.
    pub fn function_precision(mut self, epsfcn: f64) -> Self {
        self.function_precision = epsfcn;
        self
    }

    /// Override the adaptive step rule with a fixed absolute step `h` used
    /// for *every* coordinate. Escape hatch; most callers should leave the
    /// adaptive `|xⱼ|`-scaled rule in place.
    pub fn with_step(mut self, h: f64) -> Self {
        self.fixed_step = Some(h);
        self
    }

    /// Borrow the wrapped problem.
    pub fn get_ref(&self) -> &P {
        &self.problem
    }

    /// Unwrap and return the inner problem.
    pub fn into_inner(self) -> P {
        self.problem
    }
}

// ----------------------------------------------------------------------
// Delegated function-value traits.
// ----------------------------------------------------------------------

impl<P: CostFunction> CostFunction for FiniteDiff<P> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.problem.cost(param)
    }
}

impl<P: Residual> Residual for FiniteDiff<P> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn residual(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.problem.residual(param)
    }
}

// Forward box bounds so a `CostFunction + BoxConstraints` problem wrapped
// in `FiniteDiff` still routes to the constrained solvers (ProjectedGD,
// TRF). Per tenet 4 this is correct precisely because `FiniteDiff` *adds* a
// capability and preserves the constraint: it does not consume it.
impl<P: BoxConstraints> BoxConstraints for FiniteDiff<P> {
    fn lower(&self) -> &Self::Param {
        self.problem.lower()
    }
    fn upper(&self) -> &Self::Param {
        self.problem.upper()
    }
}

// ----------------------------------------------------------------------
// Synthesized derivative traits.
// ----------------------------------------------------------------------

impl<P, V> Gradient for FiniteDiff<P>
where
    P: CostFunction<Param = V, Output = f64> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + MaybeSync,
    <P as CostFunction>::Error: MaybeSend,
{
    type Gradient = V;
    fn gradient(&self, param: &V) -> Result<V, P::Error> {
        match self.gradient_method {
            Method::Forward => forward_difference_gradient(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
            ),
            Method::Central => central_difference_gradient(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
            ),
        }
    }
}

impl<P, V> Jacobian for FiniteDiff<P>
where
    P: Residual<Param = V, Output = V> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + MaybeSync + MaybeSend,
    <P as Residual>::Error: MaybeSend,
{
    type Jacobian = <V as DenseMatrixFromFn>::Matrix;
    fn jacobian(&self, param: &V) -> Result<Self::Jacobian, P::Error> {
        match self.jacobian_method {
            Method::Forward => forward_difference_jacobian(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
            ),
            Method::Central => central_difference_jacobian(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
            ),
        }
    }
}

impl<P, V> Hessian for FiniteDiff<P>
where
    P: CostFunction<Param = V, Output = f64> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + MaybeSync,
    <P as CostFunction>::Error: MaybeSend,
{
    type Hessian = <V as DenseMatrixFromFn>::Matrix;
    fn hessian(&self, param: &V) -> Result<Self::Hessian, P::Error> {
        match self.hessian_method {
            Method::Forward => forward_difference_hessian(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
            ),
            Method::Central => central_difference_hessian(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
            ),
        }
    }
}

impl<P, V> HessianProduct for FiniteDiff<P>
where
    P: CostFunction<Param = V, Output = f64> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + MaybeSync,
    <P as CostFunction>::Error: MaybeSend,
{
    /// Hessian-vector product by differencing this wrapper's own
    /// finite-difference gradients along `v` (stencil set by
    /// [`hessian_method`](FiniteDiff::hessian_method)). Note this stacks two
    /// truncation errors (FD of FD): it is the no-analytic-anything
    /// fallback. With an analytic [`Gradient`], skip the wrapper and call
    /// [`forward_difference_hessian_product`] /
    /// [`central_difference_hessian_product`] directly (2 gradient
    /// evaluations per product). Cost here is 2 synthesized gradients:
    /// `~4n` cost evaluations with the central/central default.
    fn hessian_product(&self, param: &V, v: &V) -> Result<V, P::Error> {
        match self.hessian_method {
            Method::Forward => forward_difference_hessian_product(
                self,
                param,
                v,
                self.function_precision,
                self.fixed_step,
            ),
            Method::Central => central_difference_hessian_product(
                self,
                param,
                v,
                self.function_precision,
                self.fixed_step,
            ),
        }
    }
}

// ----------------------------------------------------------------------
// Step-size helper.
// ----------------------------------------------------------------------

/// Numerical-Recipes adaptive step `scale · max(|xⱼ|, 1)`, re-rounded so
/// `xⱼ + h` is exactly representable (removes a rounding error in the
/// denominator). `fixed_step`, when set, overrides the rule.
fn nr_step(xj: f64, scale: f64, fixed_step: Option<f64>) -> f64 {
    if let Some(h) = fixed_step {
        return h;
    }
    let h = scale * xj.abs().max(1.0);
    let reround = (xj + h) - xj;
    if reround == 0.0 { h } else { reround }
}

// ----------------------------------------------------------------------
// Free functions: the tested numerics core, reused by the wrapper.
// ----------------------------------------------------------------------

/// Central-difference gradient `∇f(x)ⱼ ≈ (f(x+hⱼeⱼ) − f(x−hⱼeⱼ)) / 2hⱼ`.
///
/// `function_precision` is the assumed relative accuracy of `f` (floored at
/// `f64::EPSILON`); `fixed_step`, when `Some`, overrides the adaptive step.
/// `2n` cost evaluations. Returns `Err` if any probe's `cost` does.
pub fn central_difference_gradient<P, V>(
    problem: &P,
    x: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<V, P::Error>
where
    P: CostFunction<Param = V, Output = f64> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + MaybeSync,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    let scale = function_precision.max(f64::EPSILON).cbrt();
    // Each task perturbs coordinate `j` of its own probe buffer, evaluates,
    // and resets it, so the buffer stays equal to `x` between indices.
    let cols = try_map_range_with(
        n,
        || x.clone(),
        |probe, j| {
            let xj = x.get_scalar(j);
            let h = nr_step(xj, scale, fixed_step);
            probe.set_scalar(j, xj + h);
            let fp = problem.cost(probe)?;
            probe.set_scalar(j, xj - h);
            let fm = problem.cost(probe)?;
            probe.set_scalar(j, xj);
            Ok((fp - fm) / (2.0 * h))
        },
    )?;
    let mut grad = x.clone();
    for (j, g) in cols.into_iter().enumerate() {
        grad.set_scalar(j, g);
    }
    Ok(grad)
}

/// Forward-difference gradient `∇f(x)ⱼ ≈ (f(x+hⱼeⱼ) − f(x)) / hⱼ`.
///
/// Cheaper than [`central_difference_gradient`] (`n+1` cost evaluations) at
/// the cost of `O(h)` rather than `O(h²)` accuracy. Returns `Err` if any
/// probe's `cost` does.
pub fn forward_difference_gradient<P, V>(
    problem: &P,
    x: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<V, P::Error>
where
    P: CostFunction<Param = V, Output = f64> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + MaybeSync,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    let scale = function_precision.max(f64::EPSILON).sqrt();
    let f0 = problem.cost(x)?;
    let cols = try_map_range_with(
        n,
        || x.clone(),
        |probe, j| {
            let xj = x.get_scalar(j);
            let h = nr_step(xj, scale, fixed_step);
            probe.set_scalar(j, xj + h);
            let fp = problem.cost(probe)?;
            probe.set_scalar(j, xj);
            Ok((fp - f0) / h)
        },
    )?;
    let mut grad = x.clone();
    for (j, g) in cols.into_iter().enumerate() {
        grad.set_scalar(j, g);
    }
    Ok(grad)
}

/// Forward-difference Jacobian, column `j ≈ (r(x+hⱼeⱼ) − r(x)) / hⱼ`.
///
/// Reproduces MINPACK `fdjac2`: `eps = sqrt(function_precision)`,
/// `hⱼ = eps·|xⱼ|`, `hⱼ = eps` when `xⱼ = 0`. `n+1` residual evaluations.
/// The result is an `m × n` matrix (`m = r(x).len()`, `n = x.len()`).
/// Returns `Err` if any probe's `residual` does.
pub fn forward_difference_jacobian<P, V>(
    problem: &P,
    x: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<<V as DenseMatrixFromFn>::Matrix, P::Error>
where
    P: Residual<Param = V, Output = V> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    let r0 = problem.residual(x)?;
    let m = r0.vec_len();
    let eps = function_precision.max(f64::EPSILON).sqrt();
    let columns = try_map_range_with(
        n,
        || x.clone(),
        |probe, j| {
            let xj = x.get_scalar(j);
            let h = match fixed_step {
                Some(h) => h,
                None => {
                    let h = eps * xj.abs();
                    if h == 0.0 { eps } else { h }
                }
            };
            probe.set_scalar(j, xj + h);
            let mut col = problem.residual(probe)?;
            probe.set_scalar(j, xj);
            for i in 0..m {
                col.set_scalar(i, (col.get_scalar(i) - r0.get_scalar(i)) / h);
            }
            Ok(col)
        },
    )?;
    Ok(V::dense_from_fn(m, n, |i, j| columns[j].get_scalar(i)))
}

/// Central-difference Jacobian, column `j ≈ (r(x+hⱼeⱼ) − r(x−hⱼeⱼ)) / 2hⱼ`.
///
/// More accurate than [`forward_difference_jacobian`] (`O(h²)`) at `2n+1`
/// residual evaluations. Result shape is `m × n`. Returns `Err` if any
/// probe's `residual` does.
pub fn central_difference_jacobian<P, V>(
    problem: &P,
    x: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<<V as DenseMatrixFromFn>::Matrix, P::Error>
where
    P: Residual<Param = V, Output = V> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    let m = problem.residual(x)?.vec_len();
    let scale = function_precision.max(f64::EPSILON).cbrt();
    let columns = try_map_range_with(
        n,
        || x.clone(),
        |probe, j| {
            let xj = x.get_scalar(j);
            let h = nr_step(xj, scale, fixed_step);
            probe.set_scalar(j, xj + h);
            let mut col = problem.residual(probe)?;
            probe.set_scalar(j, xj - h);
            let rm = problem.residual(probe)?;
            probe.set_scalar(j, xj);
            for i in 0..m {
                col.set_scalar(i, (col.get_scalar(i) - rm.get_scalar(i)) / (2.0 * h));
            }
            Ok(col)
        },
    )?;
    Ok(V::dense_from_fn(m, n, |i, j| columns[j].get_scalar(i)))
}

/// Central-difference Hessian (Numerical Recipes second differences).
///
/// Diagonal `Hᵢᵢ = (f(x+hᵢeᵢ) − 2f(x) + f(x−hᵢeᵢ)) / hᵢ²`; off-diagonal the
/// four-point stencil `(f₊₊ − f₊₋ − f₋₊ + f₋₋) / 4hᵢhⱼ`. Symmetric `n × n`
/// by construction (the upper triangle is mirrored). `~2n²` cost
/// evaluations. Returns `Err` if any probe's `cost` does.
pub fn central_difference_hessian<P, V>(
    problem: &P,
    x: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<<V as DenseMatrixFromFn>::Matrix, P::Error>
where
    P: CostFunction<Param = V, Output = f64> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + MaybeSync,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    let scale = function_precision.max(f64::EPSILON).powf(0.25);
    let f0 = problem.cost(x)?;
    let h: Vec<f64> = (0..n)
        .map(|j| nr_step(x.get_scalar(j), scale, fixed_step))
        .collect();
    // Upper-triangular entries (i ≤ j); each is independent given `f0`/`h`.
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let values = try_map_slice_with(
        &pairs,
        || x.clone(),
        |probe, &(i, j)| {
            let xi = x.get_scalar(i);
            if i == j {
                probe.set_scalar(i, xi + h[i]);
                let fp = problem.cost(probe)?;
                probe.set_scalar(i, xi - h[i]);
                let fm = problem.cost(probe)?;
                probe.set_scalar(i, xi);
                Ok((fp - 2.0 * f0 + fm) / (h[i] * h[i]))
            } else {
                let xj = x.get_scalar(j);
                probe.set_scalar(i, xi + h[i]);
                probe.set_scalar(j, xj + h[j]);
                let fpp = problem.cost(probe)?;
                probe.set_scalar(j, xj - h[j]);
                let fpm = problem.cost(probe)?;
                probe.set_scalar(i, xi - h[i]);
                let fmm = problem.cost(probe)?;
                probe.set_scalar(j, xj + h[j]);
                let fmp = problem.cost(probe)?;
                probe.set_scalar(i, xi);
                probe.set_scalar(j, xj);
                Ok((fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j]))
            }
        },
    )?;
    let mut hess = vec![0.0_f64; n * n];
    for (&(i, j), &v) in pairs.iter().zip(&values) {
        hess[i * n + j] = v;
        hess[j * n + i] = v;
    }
    Ok(V::dense_from_fn(n, n, |i, j| hess[i * n + j]))
}

/// Forward-difference Hessian (one-sided second differences).
///
/// `Hᵢⱼ = (f(x+hᵢeᵢ+hⱼeⱼ) − f(x+hᵢeᵢ) − f(x+hⱼeⱼ) + f(x)) / hᵢhⱼ`, with the
/// diagonal as the `i = j` case (`f(x+2hᵢeᵢ)`). Symmetric `n × n` by
/// construction; `1 + n + n(n+1)/2` cost evaluations. Returns `Err` if any
/// probe's `cost` does.
pub fn forward_difference_hessian<P, V>(
    problem: &P,
    x: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<<V as DenseMatrixFromFn>::Matrix, P::Error>
where
    P: CostFunction<Param = V, Output = f64> + MaybeSync,
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + MaybeSync,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    let scale = function_precision.max(f64::EPSILON).powf(0.25);
    let f0 = problem.cost(x)?;
    let h: Vec<f64> = (0..n)
        .map(|j| nr_step(x.get_scalar(j), scale, fixed_step))
        .collect();
    // f(x + hₖeₖ) for each k, independent across k.
    let fi = try_map_range_with(
        n,
        || x.clone(),
        |probe, k| {
            let xk = x.get_scalar(k);
            probe.set_scalar(k, xk + h[k]);
            let f = problem.cost(probe)?;
            probe.set_scalar(k, xk);
            Ok(f)
        },
    )?;
    // Upper-triangular entries (i ≤ j); each is independent given `fi`/`f0`/`h`.
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let values = try_map_slice_with(
        &pairs,
        || x.clone(),
        |probe, &(i, j)| {
            let xi = x.get_scalar(i);
            let cross = if i == j {
                probe.set_scalar(i, xi + 2.0 * h[i]);
                let c = problem.cost(probe)?;
                probe.set_scalar(i, xi);
                c
            } else {
                let xj = x.get_scalar(j);
                probe.set_scalar(i, xi + h[i]);
                probe.set_scalar(j, xj + h[j]);
                let c = problem.cost(probe)?;
                probe.set_scalar(i, xi);
                probe.set_scalar(j, xj);
                c
            };
            Ok((cross - fi[i] - fi[j] + f0) / (h[i] * h[j]))
        },
    )?;
    let mut hess = vec![0.0_f64; n * n];
    for (&(i, j), &v) in pairs.iter().zip(&values) {
        hess[i * n + j] = v;
        hess[j * n + i] = v;
    }
    Ok(V::dense_from_fn(n, n, |i, j| hess[i * n + j]))
}

/// Euclidean norm via the scalar accessors; shared by the
/// Hessian-vector-product step rules.
fn vec_norm<V: VectorLen + VectorIndex>(x: &V) -> f64 {
    (0..x.vec_len())
        .map(|j| {
            let xj = x.get_scalar(j);
            xj * xj
        })
        .sum::<f64>()
        .sqrt()
}

/// `x + h·v`, built through the scalar accessors.
fn probe_along<V: Clone + VectorLen + VectorIndex>(x: &V, v: &V, h: f64) -> V {
    let mut probe = x.clone();
    for j in 0..x.vec_len() {
        probe.set_scalar(j, x.get_scalar(j) + h * v.get_scalar(j));
    }
    probe
}

/// Forward-difference Hessian-vector product
/// `∇²f(x)·v ≈ (∇f(x + h v) − ∇f(x)) / h` (Nocedal & Wright, *Numerical
/// Optimization*, 2e, eq. 8.20).
///
/// This is the standard route to a matrix-free
/// [`HessianProduct`] when only a
/// (possibly analytic) [`Gradient`] is available: 2 gradient evaluations
/// per product, no matrix type anywhere. A problem with an analytic
/// gradient implements the trait as a one-liner calling this.
///
/// The step is `h = √eps_f · max(‖x‖, 1) / ‖v‖` with
/// `eps_f = function_precision.max(f64::EPSILON)`, so the *perturbation*
/// `h·v` has magnitude `√eps_f · max(‖x‖, 1)` regardless of the scale of
/// `v`. `fixed_step`, when `Some`, overrides `h` with the raw step along
/// `v`. `‖v‖ = 0` short-circuits to the zero vector (exact by linearity).
/// Returns `Err` if either gradient evaluation does.
pub fn forward_difference_hessian_product<P, V>(
    problem: &P,
    x: &V,
    v: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<V, <P as CostFunction>::Error>
where
    P: Gradient<Param = V, Gradient = V, Output = f64>,
    V: Clone + VectorLen + VectorIndex,
{
    let v_norm = vec_norm(v);
    if v_norm == 0.0 {
        let mut hv = v.clone();
        for j in 0..hv.vec_len() {
            hv.set_scalar(j, 0.0);
        }
        return Ok(hv);
    }
    let h = match fixed_step {
        Some(h) => h,
        None => {
            let eps = function_precision.max(f64::EPSILON);
            eps.sqrt() * vec_norm(x).max(1.0) / v_norm
        }
    };
    let g0 = problem.gradient(x)?;
    let gp = problem.gradient(&probe_along(x, v, h))?;
    let mut hv = v.clone();
    for j in 0..hv.vec_len() {
        hv.set_scalar(j, (gp.get_scalar(j) - g0.get_scalar(j)) / h);
    }
    Ok(hv)
}

/// Central-difference Hessian-vector product
/// `∇²f(x)·v ≈ (∇f(x + h v) − ∇f(x − h v)) / 2h`: the two-sided variant of
/// [`forward_difference_hessian_product`] (Nocedal & Wright, 2e, eq. 8.20),
/// `O(h²)` truncation error at the same 2-gradient cost (the base gradient
/// is not needed).
///
/// The step is `h = ∛eps_f · max(‖x‖, 1) / ‖v‖`; `fixed_step` and `‖v‖ = 0`
/// behave as in the forward variant. Returns `Err` if either gradient
/// evaluation does.
pub fn central_difference_hessian_product<P, V>(
    problem: &P,
    x: &V,
    v: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<V, <P as CostFunction>::Error>
where
    P: Gradient<Param = V, Gradient = V, Output = f64>,
    V: Clone + VectorLen + VectorIndex,
{
    let v_norm = vec_norm(v);
    if v_norm == 0.0 {
        let mut hv = v.clone();
        for j in 0..hv.vec_len() {
            hv.set_scalar(j, 0.0);
        }
        return Ok(hv);
    }
    let h = match fixed_step {
        Some(h) => h,
        None => {
            let eps = function_precision.max(f64::EPSILON);
            eps.cbrt() * vec_norm(x).max(1.0) / v_norm
        }
    };
    let gp = problem.gradient(&probe_along(x, v, h))?;
    let gm = problem.gradient(&probe_along(x, v, -h))?;
    let mut hv = v.clone();
    for j in 0..hv.vec_len() {
        hv.set_scalar(j, (gp.get_scalar(j) - gm.get_scalar(j)) / (2.0 * h));
    }
    Ok(hv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f(x) = Σ aᵢ xᵢ²`: gradient `2aᵢxᵢ`, diagonal Hessian `2aᵢ`. Used
    /// for backend-agnostic checks that don't need the `problems` feature.
    struct DiagQuadratic {
        a: Vec<f64>,
    }

    impl CostFunction for DiagQuadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x.iter().zip(&self.a).map(|(xi, ai)| ai * xi * xi).sum())
        }
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn central_gradient_matches_analytic_quadratic() {
        let a = vec![1.0, 2.0, 0.5];
        let x = vec![-1.2, 0.7, 3.0];
        let g = FiniteDiff::new(DiagQuadratic { a: a.clone() })
            .gradient(&x)
            .unwrap();
        for i in 0..3 {
            let want = 2.0 * a[i] * x[i];
            assert!(approx(g[i], want, 1e-7), "i={i} got {} want {want}", g[i]);
        }
    }

    #[test]
    fn forward_gradient_matches_analytic_quadratic() {
        let a = vec![1.0, 2.0, 0.5];
        let x = vec![-1.2, 0.7, 3.0];
        let g = FiniteDiff::new(DiagQuadratic { a: a.clone() })
            .gradient_method(Method::Forward)
            .gradient(&x)
            .unwrap();
        for i in 0..3 {
            let want = 2.0 * a[i] * x[i];
            assert!(approx(g[i], want, 1e-5), "i={i} got {} want {want}", g[i]);
        }
    }

    #[test]
    fn gradient_handles_zero_coordinate_and_singletons() {
        // x = 0 exercises the |xⱼ| = 0 step branch.
        let g = FiniteDiff::new(DiagQuadratic { a: vec![3.0] })
            .gradient(&vec![0.0])
            .unwrap();
        assert!(approx(g[0], 0.0, 1e-9), "got {}", g[0]);
        assert_eq!(g.len(), 1);
    }

    // The probe fan-out (under `parallel`) writes each output slot
    // independently and collects in order, so the result must be identical
    // across repeated calls, with no reduction-order nondeterminism. `n = 16` is
    // large enough that the rayon path spreads across worker threads.
    #[test]
    fn gradient_is_bitwise_reproducible_across_calls() {
        let a: Vec<f64> = (0..16).map(|k| 0.5 + k as f64).collect();
        let x: Vec<f64> = (0..16).map(|k| -2.0 + 0.37 * k as f64).collect();
        let fd = FiniteDiff::new(DiagQuadratic { a });
        let g1 = fd.gradient(&x).unwrap();
        let g2 = fd.gradient(&x).unwrap();
        assert_eq!(g1, g2, "finite-difference gradient is not reproducible");
        // And it still tracks the analytic gradient 2·aᵢ·xᵢ.
        for (k, gk) in g1.iter().enumerate() {
            let want = 2.0 * (0.5 + k as f64) * x[k];
            assert!(approx(*gk, want, 1e-6), "k={k} got {gk} want {want}");
        }
    }

    #[test]
    fn gradient_of_empty_param_is_empty() {
        let g = FiniteDiff::new(DiagQuadratic { a: vec![] })
            .gradient(&Vec::<f64>::new())
            .unwrap();
        assert!(g.is_empty());
    }

    #[test]
    fn defaults_are_central_gradient_forward_jacobian() {
        let fd = FiniteDiff::new(DiagQuadratic { a: vec![1.0] });
        assert_eq!(fd.gradient_method, Method::Central);
        assert_eq!(fd.jacobian_method, Method::Forward);
        assert_eq!(fd.hessian_method, Method::Central);
    }

    /// `DiagQuadratic` with its analytic gradient, for the `P: Gradient`
    /// Hessian-product free functions.
    struct DiagQuadraticGrad {
        a: Vec<f64>,
    }

    impl CostFunction for DiagQuadraticGrad {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x.iter().zip(&self.a).map(|(xi, ai)| ai * xi * xi).sum())
        }
    }
    impl Gradient for DiagQuadraticGrad {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(x.iter()
                .zip(&self.a)
                .map(|(xi, ai)| 2.0 * ai * xi)
                .collect())
        }
    }

    #[test]
    fn hessian_product_free_fns_match_analytic_quadratic() {
        // ∇²f = diag(2a), so ∇²f·v = 2a∘v exactly (the quadratic makes the
        // forward difference exact up to rounding).
        let a = vec![1.0, 2.0, 0.5];
        let p = DiagQuadraticGrad { a: a.clone() };
        let x = vec![-1.2, 0.7, 3.0];
        let v = vec![1.0, -2.0, 0.25];
        let fwd = forward_difference_hessian_product(&p, &x, &v, f64::EPSILON, None).unwrap();
        let cen = central_difference_hessian_product(&p, &x, &v, f64::EPSILON, None).unwrap();
        for i in 0..3 {
            let want = 2.0 * a[i] * v[i];
            assert!(approx(fwd[i], want, 1e-5), "fwd i={i} {} vs {want}", fwd[i]);
            assert!(approx(cen[i], want, 1e-7), "cen i={i} {} vs {want}", cen[i]);
        }
    }

    #[test]
    fn hessian_product_of_zero_direction_is_zero() {
        let p = DiagQuadraticGrad { a: vec![1.0, 2.0] };
        let x = vec![-1.2, 0.7];
        let v = vec![0.0, 0.0];
        let hv = central_difference_hessian_product(&p, &x, &v, f64::EPSILON, None).unwrap();
        assert_eq!(hv, vec![0.0, 0.0]);
    }

    #[test]
    fn finite_diff_hessian_product_matches_rosenbrock_analytic() {
        // FD-of-FD (the wrapper differences its own synthesized gradients),
        // so the tolerance is looser than for the analytic-gradient path.
        // Analytic 2D Rosenbrock Hessian at a generic point:
        //   H₀₀ = 2 − 400(x₁ − 3x₀²), H₀₁ = H₁₀ = −400x₀, H₁₁ = 200.
        struct Rosenbrock2;
        impl CostFunction for Rosenbrock2 {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
                Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
            }
        }
        let (x0, x1) = (-1.2, 1.0);
        let x = vec![x0, x1];
        let v = vec![0.3, -0.7];
        let h = [
            [2.0 - 400.0 * (x1 - 3.0 * x0 * x0), -400.0 * x0],
            [-400.0 * x0, 200.0],
        ];
        let want = [
            h[0][0] * v[0] + h[0][1] * v[1],
            h[1][0] * v[0] + h[1][1] * v[1],
        ];
        let hv = FiniteDiff::new(Rosenbrock2)
            .hessian_product(&x, &v)
            .unwrap();
        for i in 0..2 {
            let rel = (hv[i] - want[i]).abs() / want[i].abs().max(1.0);
            assert!(rel < 1e-3, "i={i} {} vs {}", hv[i], want[i]);
        }
    }

    #[test]
    fn finite_diff_trust_region_matrix_free_converges() {
        // End-to-end: a values-only problem through FiniteDiff's synthesized
        // HessianProduct into the matrix-free trust region, on the
        // dependency-free Vec<f64> backend (which has no Hessian impl at all).
        use crate::solver::TrustRegion;
        use crate::{BasicState, Executor, GradientTolerance};

        let result = Executor::new(
            FiniteDiff::new(DiagQuadratic {
                a: vec![1.0, 50.0, 2.5],
            }),
            TrustRegion::matrix_free(),
            BasicState::new(vec![5.0, 1.0, -3.0]),
        )
        .max_iter(100)
        .terminate_on(GradientTolerance(1e-6))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    }

    #[cfg(feature = "problems")]
    mod with_problems {
        use super::*;
        use crate::problems::rosenbrock::{Rosenbrock, rosenbrock_gradient};
        use crate::problems::sphere::{Sphere, sphere_gradient};

        #[test]
        fn central_gradient_matches_sphere_analytic() {
            let x = vec![-1.2, 1.0, 0.7, 0.4];
            let g = FiniteDiff::new(Sphere::<Vec<f64>>::new())
                .gradient(&x)
                .unwrap();
            let mut want = vec![0.0; x.len()];
            sphere_gradient(&x, &mut want);
            for i in 0..x.len() {
                assert!(approx(g[i], want[i], 1e-8), "i={i} {} vs {}", g[i], want[i]);
            }
        }

        #[test]
        fn central_gradient_matches_rosenbrock_analytic() {
            let x = vec![-1.2, 1.0, 0.7, 0.4];
            let g = FiniteDiff::new(Rosenbrock::<Vec<f64>>::new())
                .gradient(&x)
                .unwrap();
            let mut want = vec![0.0; x.len()];
            rosenbrock_gradient(&x, &mut want);
            for i in 0..x.len() {
                assert!(approx(g[i], want[i], 1e-4), "i={i} {} vs {}", g[i], want[i]);
            }
        }
    }

    // BoxConstraints forwarding: a wrapped boxed problem stays constrained
    // and still satisfies the gradient-solver bound.
    #[cfg(feature = "problems")]
    #[test]
    fn box_constrained_is_forwarded() {
        use crate::core::constraint::BoxConstraints;
        use crate::problems::rastrigin::RastriginBoxed;

        fn _assert_constrained_and_differentiable<T: BoxConstraints + Gradient>() {}
        _assert_constrained_and_differentiable::<FiniteDiff<RastriginBoxed<Vec<f64>>>>();

        let lower = vec![-5.12, -5.12];
        let upper = vec![5.12, 5.12];
        let p = RastriginBoxed::new(lower.clone(), upper.clone());
        let wrapped = FiniteDiff::new(p);
        assert_eq!(wrapped.lower(), &lower);
        assert_eq!(wrapped.upper(), &upper);
    }

    #[cfg(all(feature = "problems", feature = "nalgebra"))]
    mod nalgebra_matrix {
        use super::*;
        use crate::problems::rosenbrock::{Rosenbrock, RosenbrockResiduals};
        use crate::problems::sphere::Sphere;
        use nalgebra::DVector;

        #[test]
        fn forward_jacobian_matches_analytic() {
            let x = DVector::from_vec(vec![-1.2, 1.0]);
            let analytic = RosenbrockResiduals::<DVector<f64>>::new()
                .jacobian(&x)
                .unwrap();
            let fd = FiniteDiff::new(RosenbrockResiduals::<DVector<f64>>::new())
                .jacobian(&x)
                .unwrap();
            assert_eq!(fd.shape(), (2, 2));
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        approx(fd[(i, j)], analytic[(i, j)], 1e-5),
                        "({i},{j}) {} vs {}",
                        fd[(i, j)],
                        analytic[(i, j)]
                    );
                }
            }
        }

        #[test]
        fn central_jacobian_is_tighter_than_forward() {
            let x = DVector::from_vec(vec![-1.2, 1.0]);
            let analytic = RosenbrockResiduals::<DVector<f64>>::new()
                .jacobian(&x)
                .unwrap();
            let central = FiniteDiff::new(RosenbrockResiduals::<DVector<f64>>::new())
                .jacobian_method(Method::Central)
                .jacobian(&x)
                .unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    assert!(approx(central[(i, j)], analytic[(i, j)], 1e-7));
                }
            }
        }

        #[test]
        fn hessian_of_sphere_is_two_times_identity() {
            let x = DVector::from_vec(vec![0.5, -1.5, 2.0]);
            let h = FiniteDiff::new(Sphere::<DVector<f64>>::new())
                .hessian(&x)
                .unwrap();
            assert_eq!(h.shape(), (3, 3));
            for i in 0..3 {
                for j in 0..3 {
                    let want = if i == j { 2.0 } else { 0.0 };
                    assert!(approx(h[(i, j)], want, 1e-3), "({i},{j}) {}", h[(i, j)]);
                }
            }
        }

        #[test]
        fn hessian_matches_rosenbrock_analytic_2d() {
            // Analytic 2D Rosenbrock Hessian (cost form, n=2):
            //   H₀₀ = 2 − 400(x₁ − 3x₀²),  H₀₁ = H₁₀ = −400x₀,  H₁₁ = 200.
            let x = DVector::from_vec(vec![-1.2, 1.0]);
            let h = FiniteDiff::new(Rosenbrock::<DVector<f64>>::new())
                .hessian(&x)
                .unwrap();
            let (x0, x1) = (x[0], x[1]);
            let want = [
                [2.0 - 400.0 * (x1 - 3.0 * x0 * x0), -400.0 * x0],
                [-400.0 * x0, 200.0],
            ];
            for i in 0..2 {
                for j in 0..2 {
                    let rel = (h[(i, j)] - want[i][j]).abs() / want[i][j].abs().max(1.0);
                    assert!(rel < 1e-3, "({i},{j}) {} vs {}", h[(i, j)], want[i][j]);
                    // symmetry
                    assert!(approx(h[(i, j)], h[(j, i)], 1e-9));
                }
            }
        }
    }

    #[cfg(all(feature = "problems", feature = "faer"))]
    mod faer_matrix {
        use super::*;
        use crate::problems::rosenbrock::RosenbrockResiduals;
        use crate::problems::sphere::Sphere;
        use faer::Col;

        #[test]
        fn forward_jacobian_matches_analytic() {
            let x = Col::<f64>::from_fn(2, |i| [-1.2, 1.0][i]);
            let analytic = RosenbrockResiduals::<Col<f64>>::new().jacobian(&x).unwrap();
            let fd = FiniteDiff::new(RosenbrockResiduals::<Col<f64>>::new())
                .jacobian(&x)
                .unwrap();
            assert_eq!((fd.nrows(), fd.ncols()), (2, 2));
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        approx(fd[(i, j)], analytic[(i, j)], 1e-5),
                        "({i},{j}) {} vs {}",
                        fd[(i, j)],
                        analytic[(i, j)]
                    );
                }
            }
        }

        #[test]
        fn hessian_of_sphere_is_two_times_identity() {
            let x = Col::<f64>::from_fn(3, |i| [0.5, -1.5, 2.0][i]);
            let h = FiniteDiff::new(Sphere::<Col<f64>>::new())
                .hessian(&x)
                .unwrap();
            assert_eq!((h.nrows(), h.ncols()), (3, 3));
            for i in 0..3 {
                for j in 0..3 {
                    let want = if i == j { 2.0 } else { 0.0 };
                    assert!(approx(h[(i, j)], want, 1e-3), "({i},{j}) {}", h[(i, j)]);
                }
            }
        }
    }
}
