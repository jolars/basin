//! Finite-difference derivative synthesis and analytic derivative checks.
//!
//! [`DerivativeChecker`] diagnoses analytic gradients and residual Jacobians
//! with scale-aware coordinate or directional comparisons and optional bounds.
//!
//! [`FiniteDiff`] wraps a problem that only exposes function values
//! ([`CostFunction`] and/or [`Residual`]) and *adds* the derivative traits
//! the solvers want: [`Gradient`] (for first-order solvers), [`Jacobian`]
//! (for least-squares solvers), [`Hessian`] (for second-order solvers), and
//! [`HessianProduct`] (for matrix-free second-order solvers). Nonlinear
//! constraint derivatives use [`crate::ConstraintJacobian`].
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
//! The wrapper forwards all problem-side constraint traits: adding derivatives must not silently un-constrain a problem
//! (tenet 4 in `CONTRIBUTING.md`: an adapter that *adds* a capability preserves
//! the rest).
//!
//! # Step sizes (paper-anchored)
//!
//! - **Gradient** and **Hessian** use Numerical-Recipes-style adaptive
//!   steps `hⱼ = scale · max(|xⱼ|, 1)`, with `scale = eps_f^{1/3}` (central
//!   gradient), `eps_f^{1/2}` (forward gradient), or `eps_f^{1/4}` (Hessian),
//!   where `eps_f = function_precision.max(F::epsilon())` is the assumed
//!   relative accuracy of the function, with `F` the scalar type. The step is re-rounded so `xⱼ + hⱼ`
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
//!   gradient, `n+1`/`2n` Jacobian probes, `~2n²` Hessian). As specified by
//!   [`EvalCounts`](crate::EvalCounts), counters record requests at the
//!   [`Problem`](crate::Problem) boundary. A derivative request charges its
//!   category once; its internal probes add no cost or residual counts.
//!   Evaluation budgets, including [`EvaluationKind::TotalWork`](crate::EvaluationKind::TotalWork),
//!   use those request counts and exclude the probes. Instrument the underlying
//!   callbacks separately to measure their actual evaluations. A change in
//!   stencil, fixed coordinates, or future Jacobian coloring can change probe
//!   work without changing the reported derivative-request count.
//!   Fused cost/gradient and cost/gradient/Hessian calls share one base cost
//!   evaluation, with identical probes for owned and `*_into` methods.
//! - **Domain & hard-abort.** Each probe `?`-propagates the inner
//!   function's [`Error`](CostFunction::Error). If the inner `cost` /
//!   `residual` returns `Err`, the derivative call returns the same `Err`:
//!   no swallowing, no partial result. A point where the inner function
//!   *soft-rejects* (returning positive infinity) poisons the derivative with `±∞`
//!   rather than aborting; keeping probes inside the strict domain
//!   is the caller's responsibility. Use [`FiniteDiff::with_bounds`] or
//!   [`BoundedFiniteDiff`] for gradients and Jacobians that respect box bounds.
//!
//! # Parallelism (`parallel` feature)
//!
//! The probes of one derivative are independent, so with the `parallel`
//! feature they fan across a `rayon` thread pool: the `n` gradient/Jacobian
//! columns, and the `n(n+1)/2` upper-triangular Hessian entries, are evaluated
//! concurrently. Forward Hessians schedule the base, single-coordinate, and
//! two-coordinate probes together, then combine their values. Other stencils
//! evaluate a shared `f(x)`/`r(x)` base once up front when needed; unbounded
//! central Jacobians infer their output shape from the probes instead.
//! The result is **bit-identical** to the
//! sequential path (each output slot is written independently and collected in order, with no
//! floating-point reduction), so enabling the feature never changes the
//! numbers, only the wall-clock.
//!
//! This is a win when the wrapped `cost`/`residual` is expensive; for a
//! cheap function at small `n` the thread-pool overhead can exceed the saving,
//! so it is opt-in. Under the feature, `FiniteDiff`'s [`Gradient`] /
//! [`Jacobian`]/[`Hessian`] impls additionally require the wrapped problem
//! (and its param vector/error type) to be [`Sync`]/[`Send`]; without the
//! feature there is no such bound and the evaluation is sequential.

use crate::core::math::{
    DenseBackend, DenseMatrixFromFn, Scalar, SquareMatrixFromFn, VectorIndex,
    VectorLen,
};
use crate::core::parallel::{MaybeSend, MaybeSync, try_map_range_with};
use crate::core::problem::{
    CostFunction, Gradient, Hessian, HessianProduct, Jacobian, Residual,
};

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

impl Method {
    fn first_derivative_scale<F: Scalar>(self, precision: F) -> F {
        let precision = precision.max(F::epsilon());
        match self {
            Self::Forward => precision.sqrt(),
            Self::Central => precision.cbrt(),
        }
    }
}

/// Wraps a problem to synthesize its derivatives by finite differences.
///
/// Construct with [`FiniteDiff::new`] for `f64` or use
/// `FiniteDiff::<_, false, f32>::new_with_scalar(problem)` for `f32`.
/// The default stencils are central for gradients and Hessians and
/// forward for Jacobians. See the [module docs](self) for step rules. The wrapper delegates [`CostFunction`]/[`Residual`] /
/// all constraint traits to the inner problem and implements [`Gradient`],
/// [`Jacobian`], [`crate::ConstraintJacobian`], and [`Hessian`] via finite differences.
///
/// # Backends
///
/// [`Gradient`] and [`HessianProduct`] are backend-generic (any `V: Clone +
/// VectorLen + VectorIndex`; no matrix type is involved). [`Jacobian`] and
/// the default [`Hessian`] additionally require `V: DenseMatrixFromFn`, so they are
/// available for `Vec<F>`/`DenseMatrix`, nalgebra `DVector<F>`/`DMatrix`,
/// ndarray `Array1<F>`/`Array2`, and faer `Col<F>`/`Mat` with `F` equal
/// to `f32` or `f64`. Nalgebra `SVector<F, N>` parameters use
/// `OMatrix<F, Dyn, Const<N>>` and may be paired with `SVector<F, M>` or
/// `DVector<F>` residuals. The Jacobian row count comes from the residual length. Use [`FiniteDiff::with_square_hessian`] for a
/// square fixed-size Hessian.
/// [`crate::ConstraintJacobian`] uses the parameter backend's dense matrix,
/// independently of the linear block matrices.
///
/// # Solver stopping tolerances
///
/// The synthesized gradient has truncation and rounding error, so it can be
/// nonzero at an exact minimum. Choose the solver's gradient stopping tolerance
/// to account for that error in the problem's units. A threshold below the
/// derivative error can cause extra evaluations or a line-search failure even
/// after reaching an accurate solution. Central differences often reduce
/// truncation error, but do not guarantee a particular stopping accuracy.
/// [`function_precision`](Self::function_precision) controls probe step sizes;
/// it is not a bound on gradient error. Objective scaling, curvature, parameter
/// magnitudes, and evaluation noise also affect that error. See
/// [`Lbfgs`](crate::Lbfgs) for a forward-difference stopping example.
///
/// # Examples
///
/// Run a gradient solver against a problem that only implements
/// [`CostFunction`]: wrapping it in [`FiniteDiff`] synthesizes the
/// [`Gradient`] by central differences.
///
/// ```
/// use basin::{
///     FirstOrderState, CostFunction, Executor, FiniteDiff, GradientDescent};
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
///     GradientDescent::new(0.1).with_absolute_gradient_tolerance(1e-8),
///     FirstOrderState::new(vec![1.0, 1.0]),
/// )
/// .max_iter(1_000)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FiniteDiff<P, const SQUARE: bool = false, F: Scalar = f64> {
    problem: P,
    gradient_method: Method,
    jacobian_method: Method,
    hessian_method: Method,
    function_precision: F,
    fixed_step: Option<F>,
}

impl<P> FiniteDiff<P> {
    /// Wrap `problem` with default settings: central-difference gradient
    /// and Hessian, forward-difference (MINPACK `fdjac2`) Jacobian,
    /// `function_precision = f64::EPSILON`, adaptive step sizes.
    pub fn new(problem: P) -> Self {
        Self::new_with_scalar(problem)
    }
}

impl<P, F: Scalar> FiniteDiff<P, false, F> {
    /// Wrap a problem with the default stencils and machine epsilon for `F`.
    pub fn new_with_scalar(problem: P) -> Self {
        Self {
            problem,
            gradient_method: Method::Central,
            jacobian_method: Method::Forward,
            hessian_method: Method::Central,
            function_precision: F::epsilon(),
            fixed_step: None,
        }
    }

    /// Use the parameter backend's square matrix for the Hessian.
    ///
    /// For nalgebra `SVector<F, N>`, the Hessian is an
    /// `SMatrix<F, N, N>`, suitable for [`crate::Dogleg`] and
    /// [`crate::MoreSorensen`]. Jacobian types are independent of this option.
    ///
    /// ```
    /// # #[cfg(feature = "nalgebra_v0_35")] {
    /// use basin::{CostFunction, Dogleg, Executor, FiniteDiff, TrustRegion};
    /// use nalgebra::SVector;
    ///
    /// struct Sphere;
    /// impl CostFunction for Sphere {
    ///     type Param = SVector<f64, 2>;
    ///     type Output = f64;
    ///     type Error = std::convert::Infallible;
    ///     fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
    ///         Ok(x.norm_squared())
    ///     }
    /// }
    ///
    /// let result = Executor::from_start(
    ///     FiniteDiff::new(Sphere).with_square_hessian(),
    ///     TrustRegion::with_subproblem(Dogleg),
    ///     SVector::<f64, 2>::new(1.0, 2.0),
    /// ).max_iter(20).run().unwrap();
    /// assert!(result.cost() < 1e-12);
    /// # }
    /// ```
    pub fn with_square_hessian(self) -> FiniteDiff<P, true, F>
    where
        P: CostFunction<Output = F>,
        P::Param: DenseBackend<F, Matrix: SquareMatrixFromFn<F>>,
    {
        FiniteDiff {
            problem: self.problem,
            gradient_method: self.gradient_method,
            jacobian_method: self.jacobian_method,
            hessian_method: self.hessian_method,
            function_precision: self.function_precision,
            fixed_step: self.fixed_step,
        }
    }
}

impl<P, const SQUARE: bool, F: Scalar> FiniteDiff<P, SQUARE, F> {
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
    /// function is noisy. Defaults to the machine epsilon of `F`, which also
    /// serves as the lower bound.
    pub fn function_precision(mut self, epsfcn: F) -> Self {
        self.function_precision = epsfcn;
        self
    }

    /// Override the adaptive step rule with a fixed absolute step `h` used
    /// for *every* coordinate. Escape hatch; most callers should leave the
    /// adaptive `|xⱼ|`-scaled rule in place.
    pub fn with_step(mut self, h: F) -> Self {
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

// Delegated function-value traits.

impl<P: CostFunction, const SQUARE: bool, F: Scalar> CostFunction
    for FiniteDiff<P, SQUARE, F>
{
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.problem.cost(param)
    }
}

impl<P: Residual, const SQUARE: bool, F: Scalar> Residual
    for FiniteDiff<P, SQUARE, F>
{
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn residual(
        &self,
        param: &Self::Param,
    ) -> Result<Self::Output, Self::Error> {
        self.problem.residual(param)
    }
    fn residual_into(
        &self,
        param: &Self::Param,
        out: &mut Self::Output,
    ) -> Result<(), Self::Error> {
        self.problem.residual_into(param, out)
    }
}

// Forward box bounds so a `CostFunction + BoxBounds` problem wrapped
// in `FiniteDiff` still routes to the constrained solvers (ProjectedGD,
// TRF). Per tenet 4 this is correct precisely because `FiniteDiff` *adds* a
// capability and preserves the constraint: it does not consume it.

// Synthesized derivative traits.

impl<P, V, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync> Gradient
    for FiniteDiff<P, SQUARE, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    <P as CostFunction>::Error: MaybeSend,
{
    type Gradient = V;
    fn gradient(&self, param: &V) -> Result<V, P::Error> {
        let mut out = param.clone();
        self.gradient_into(param, &mut out)?;
        Ok(out)
    }
    fn gradient_into(&self, param: &V, out: &mut V) -> Result<(), P::Error> {
        match self.gradient_method {
            Method::Forward => forward_difference_gradient_into(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                out,
            ),
            Method::Central => central_difference_gradient_into(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                out,
            ),
        }
    }

    fn cost_and_gradient(&self, param: &V) -> Result<(F, V), P::Error> {
        let mut grad = param.clone();
        let cost = self.cost_and_gradient_into(param, &mut grad)?;
        Ok((cost, grad))
    }

    fn cost_and_gradient_into(
        &self,
        param: &V,
        grad: &mut V,
    ) -> Result<F, P::Error> {
        let cost = self.cost(param)?;
        match self.gradient_method {
            Method::Forward => forward_difference_gradient_with_base_into(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                cost,
                grad,
            ),
            Method::Central => self.gradient_into(param, grad),
        }?;
        Ok(cost)
    }
}

impl<P, V, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync> Jacobian
    for FiniteDiff<P, SQUARE, F>
where
    P: Residual<Param = V> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + DenseMatrixFromFn<F>
        + MaybeSync
        + MaybeSend,
    P::Output: Clone + VectorLen + VectorIndex<F> + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    type Jacobian = <V as DenseMatrixFromFn<F>>::Matrix;
    fn jacobian(&self, param: &V) -> Result<Self::Jacobian, P::Error> {
        let mut out =
            V::dense_from_fn(0, param.vec_len(), |_, _| unreachable!());
        self.jacobian_into(param, &mut out)?;
        Ok(out)
    }
    fn jacobian_into(
        &self,
        param: &V,
        out: &mut Self::Jacobian,
    ) -> Result<(), P::Error> {
        match self.jacobian_method {
            Method::Forward => forward_difference_jacobian_into(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                out,
            ),
            Method::Central => central_difference_jacobian_into(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                None,
                out,
            ),
        }
    }

    fn residual_and_jacobian(
        &self,
        param: &V,
    ) -> Result<(P::Output, Self::Jacobian), P::Error> {
        let res = residual_base(&self.problem, param)?;
        let mut jac =
            V::dense_from_fn(0, param.vec_len(), |_, _| unreachable!());
        self.jacobian_with_base_into(param, &res, &mut jac)?;
        Ok((res, jac))
    }

    fn residual_and_jacobian_into(
        &self,
        param: &V,
        res: &mut P::Output,
        jac: &mut Self::Jacobian,
    ) -> Result<(), P::Error> {
        self.residual_into(param, res)?;
        self.jacobian_with_base_into(param, res, jac)
    }
}

impl<P, V, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync>
    FiniteDiff<P, SQUARE, F>
where
    P: Residual<Param = V> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + DenseMatrixFromFn<F>
        + MaybeSync
        + MaybeSend,
    P::Output: Clone + VectorLen + VectorIndex<F> + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    fn jacobian_with_base_into(
        &self,
        param: &V,
        res: &P::Output,
        jac: &mut V::Matrix,
    ) -> Result<(), P::Error> {
        match self.jacobian_method {
            Method::Forward => forward_difference_jacobian_with_base_into(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                res,
                jac,
            ),
            Method::Central => central_difference_jacobian_into(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                Some(res),
                jac,
            ),
        }
    }
}

trait HessianOutput<const SQUARE: bool, F: Scalar> {
    type Matrix;

    fn build(n: usize, entry: impl FnMut(usize, usize) -> F) -> Self::Matrix;
    fn fill(
        n: usize,
        entry: impl FnMut(usize, usize) -> F,
        out: &mut Self::Matrix,
    );
}

impl<F: Scalar, V: DenseMatrixFromFn<F>> HessianOutput<false, F> for V {
    type Matrix = V::Matrix;

    fn build(n: usize, entry: impl FnMut(usize, usize) -> F) -> Self::Matrix {
        V::dense_from_fn(n, n, entry)
    }

    fn fill(
        n: usize,
        entry: impl FnMut(usize, usize) -> F,
        out: &mut Self::Matrix,
    ) {
        V::dense_from_fn_into(n, n, entry, out);
    }
}

impl<F: Scalar, V: DenseBackend<F, Matrix: SquareMatrixFromFn<F>>>
    HessianOutput<true, F> for V
{
    type Matrix = V::Matrix;

    fn build(n: usize, entry: impl FnMut(usize, usize) -> F) -> Self::Matrix {
        V::Matrix::square_from_fn(n, entry)
    }

    fn fill(
        n: usize,
        entry: impl FnMut(usize, usize) -> F,
        out: &mut Self::Matrix,
    ) {
        *out = Self::build(n, entry);
    }
}

impl<P, V, M, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync> Hessian
    for FiniteDiff<P, SQUARE, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + HessianOutput<SQUARE, F, Matrix = M>
        + MaybeSync,
    P::Error: MaybeSend,
{
    type Hessian = M;

    fn hessian(&self, param: &V) -> Result<M, P::Error> {
        self.evaluate_hessian(param, None, |triangle, values| {
            V::build(param.vec_len(), |i, j| {
                values[triangle.symmetric_index(i, j)]
            })
        })
    }

    fn hessian_into(&self, param: &V, out: &mut M) -> Result<(), P::Error> {
        self.evaluate_hessian(param, None, |triangle, values| {
            V::fill(
                param.vec_len(),
                |i, j| values[triangle.symmetric_index(i, j)],
                out,
            );
        })
    }

    fn cost_and_gradient_and_hessian(
        &self,
        param: &V,
    ) -> Result<(F, V, M), P::Error> {
        let (cost, grad) = self.cost_and_gradient(param)?;
        let hess =
            self.evaluate_hessian(param, Some(cost), |triangle, values| {
                V::build(param.vec_len(), |i, j| {
                    values[triangle.symmetric_index(i, j)]
                })
            })?;
        Ok((cost, grad, hess))
    }

    fn cost_and_gradient_and_hessian_into(
        &self,
        param: &V,
        grad: &mut V,
        out: &mut M,
    ) -> Result<F, P::Error> {
        let cost = self.cost_and_gradient_into(param, grad)?;
        self.evaluate_hessian(param, Some(cost), |triangle, values| {
            V::fill(
                param.vec_len(),
                |i, j| values[triangle.symmetric_index(i, j)],
                out,
            );
        })?;
        Ok(cost)
    }
}

impl<P, V, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync>
    FiniteDiff<P, SQUARE, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    P::Error: MaybeSend,
{
    fn evaluate_hessian<R>(
        &self,
        param: &V,
        base: Option<F>,
        publish: impl FnOnce(&HessianTriangle, &[F]) -> R,
    ) -> Result<R, P::Error> {
        match self.hessian_method {
            Method::Forward => forward_hessian(
                &self.problem,
                param,
                self.function_precision,
                self.fixed_step,
                base,
                publish,
            ),
            Method::Central => {
                let cost = match base {
                    Some(cost) => cost,
                    None => self.cost(param)?,
                };
                central_hessian(
                    &self.problem,
                    param,
                    self.function_precision,
                    self.fixed_step,
                    cost,
                    publish,
                )
            }
        }
    }
}

impl<P, V, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync> HessianProduct
    for FiniteDiff<P, SQUARE, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
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
        let mut out = param.clone();
        self.hessian_product_into(param, v, &mut out)?;
        Ok(out)
    }
    fn hessian_product_into(
        &self,
        param: &V,
        v: &V,
        out: &mut V,
    ) -> Result<(), P::Error> {
        match self.hessian_method {
            Method::Forward => forward_difference_hessian_product_into(
                self,
                param,
                v,
                self.function_precision,
                self.fixed_step,
                out,
            ),
            Method::Central => central_difference_hessian_product_into(
                self,
                param,
                v,
                self.function_precision,
                self.fixed_step,
                out,
            ),
        }
    }
}

impl<P, V, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync>
    FiniteDiff<P, SQUARE, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    P::Error: MaybeSend,
{
    /// Evaluate a Hessian-vector product with caller-owned probe and gradient
    /// storage. Uses this wrapper's Hessian stencil, precision, and step, just
    /// like [`HessianProduct::hessian_product_into`].
    ///
    /// Reuse `scratch` across products to avoid allocating these two buffers
    /// again. Numerical gradient evaluations may still allocate their own
    /// workspace. Scratch retains no evaluated values, so changing the point,
    /// direction, or wrapper settings always recomputes both gradients.
    /// See [`HessianProductScratch`] for an example and the buffer contracts.
    pub fn hessian_product_with_scratch_into(
        &self,
        param: &V,
        v: &V,
        out: &mut V,
        scratch: &mut HessianProductScratch<V>,
    ) -> Result<(), P::Error> {
        scratch.difference_into(
            self,
            param,
            v,
            HessianProductOptions {
                method: self.hessian_method,
                function_precision: self.function_precision,
                fixed_step: self.fixed_step,
            },
            out,
        )
    }
}

// Step-size helper.

/// Numerical-Recipes adaptive step `scale · max(|xⱼ|, 1)`, re-rounded so
/// `xⱼ + h` is exactly representable (removes a rounding error in the
/// denominator). `fixed_step`, when set, overrides the rule.
fn nr_step<F: Scalar>(xj: F, scale: F, fixed_step: Option<F>) -> F {
    if let Some(h) = fixed_step {
        return h;
    }
    let h = scale * xj.abs().max(F::one());
    let reround = (xj + h) - xj;
    if reround == F::zero() { h } else { reround }
}

fn minpack_step<F: Scalar>(value: F, scale: F) -> F {
    let h = scale * value.abs();
    if h == F::zero() { scale } else { h }
}

// Free functions: the tested numerics core, reused by the wrapper.

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
    let mut out = x.clone();
    central_difference_gradient_into(
        problem,
        x,
        function_precision,
        fixed_step,
        &mut out,
    )?;
    Ok(out)
}

fn gradient_from_probes_into<V, F, E, C>(
    x: &V,
    out: &mut V,
    evaluate: C,
) -> Result<(), E>
where
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    F: Scalar + MaybeSend,
    E: MaybeSend,
    C: Fn(&mut V, usize) -> Result<F, E> + MaybeSync,
{
    let n = x.vec_len();
    // Stage until every probe succeeds to preserve the output on error. Small
    // gradients need no heap storage; larger ones allocate their final size once.
    let mut inline = [F::zero(); 32];
    let mut heap = Vec::new();
    let values = if n <= inline.len() {
        &mut inline[..n]
    } else {
        heap.resize(n, F::zero());
        &mut heap[..]
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        values.par_iter_mut().enumerate().try_for_each_init(
            || x.clone(),
            |probe, (j, value)| {
                *value = evaluate(probe, j)?;
                Ok::<(), E>(())
            },
        )?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut probe = x.clone();
        for (j, value) in values.iter_mut().enumerate() {
            *value = evaluate(&mut probe, j)?;
        }
    }
    if out.vec_len() != n {
        *out = x.clone();
    }
    for (j, &value) in values.iter().enumerate() {
        out.set_scalar(j, value);
    }
    Ok(())
}

fn central_difference_gradient_into<P, V, F: Scalar + MaybeSend + MaybeSync>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    out: &mut V,
) -> Result<(), P::Error>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    P::Error: MaybeSend,
{
    let scale = Method::Central.first_derivative_scale(function_precision);
    // Each task perturbs coordinate `j` of its own probe buffer, evaluates,
    // and resets it, so the buffer stays equal to `x` between indices.
    gradient_from_probes_into(x, out, |probe, j| {
        let xj = x.get_scalar(j);
        let h = nr_step(xj, scale, fixed_step);
        probe.set_scalar(j, xj + h);
        let fp = problem.cost(probe)?;
        probe.set_scalar(j, xj - h);
        let fm = problem.cost(probe)?;
        probe.set_scalar(j, xj);
        Ok((fp - fm) / ((F::one() + F::one()) * h))
    })
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
    let mut out = x.clone();
    forward_difference_gradient_into(
        problem,
        x,
        function_precision,
        fixed_step,
        &mut out,
    )?;
    Ok(out)
}

fn forward_difference_gradient_into<P, V, F: Scalar + MaybeSend + MaybeSync>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    out: &mut V,
) -> Result<(), P::Error>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    P::Error: MaybeSend,
{
    let f0 = problem.cost(x)?;
    forward_difference_gradient_with_base_into(
        problem,
        x,
        function_precision,
        fixed_step,
        f0,
        out,
    )
}

fn forward_difference_gradient_with_base_into<
    P,
    V,
    F: Scalar + MaybeSend + MaybeSync,
>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    f0: F,
    out: &mut V,
) -> Result<(), P::Error>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    P::Error: MaybeSend,
{
    let scale = Method::Forward.first_derivative_scale(function_precision);
    gradient_from_probes_into(x, out, |probe, j| {
        let xj = x.get_scalar(j);
        let h = nr_step(xj, scale, fixed_step);
        probe.set_scalar(j, xj + h);
        let fp = problem.cost(probe)?;
        probe.set_scalar(j, xj);
        Ok((fp - f0) / h)
    })
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
    P: Residual<Param = V> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex
        + DenseMatrixFromFn
        + MaybeSync
        + MaybeSend,
    P::Output: Clone + VectorLen + VectorIndex + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    let mut out = V::dense_from_fn(0, x.vec_len(), |_, _| unreachable!());
    forward_difference_jacobian_into(
        problem,
        x,
        function_precision,
        fixed_step,
        &mut out,
    )?;
    Ok(out)
}

fn forward_difference_jacobian_into<P, V, F: Scalar + MaybeSend + MaybeSync>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    out: &mut <V as DenseMatrixFromFn<F>>::Matrix,
) -> Result<(), P::Error>
where
    P: Residual<Param = V> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + DenseMatrixFromFn<F>
        + MaybeSync
        + MaybeSend,
    P::Output: Clone + VectorLen + VectorIndex<F> + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    let r0 = residual_base(problem, x)?;
    forward_difference_jacobian_with_base_into(
        problem,
        x,
        function_precision,
        fixed_step,
        &r0,
        out,
    )
}

// Stage sequential Jacobians in padded scalar storage instead of cloning a
// backend vector per column. Parallel block staging regressed measured cases;
// retain the existing parallel kernels until their scheduling is improved.
struct JacobianBlock<F> {
    rows: Option<usize>,
    columns: usize,
    values: Vec<F>,
}

impl<F: Scalar> JacobianBlock<F> {
    fn push_column(
        &mut self,
        rows: usize,
        value: impl FnMut(usize) -> F,
    ) -> &mut [F] {
        if let Some(expected) = self.rows {
            assert_eq!(
                rows, expected,
                "finite-difference output length changed"
            );
        } else {
            self.rows = Some(rows);
            if rows != 0 {
                self.values.reserve_exact(
                    rows.checked_add(1)
                        .and_then(|stride| stride.checked_mul(self.columns))
                        .expect("Jacobian size overflow"),
                );
            }
        }
        let start = self.values.len();
        self.values.extend((0..rows).map(value));
        // Break power-of-two strides when row-major backends read across columns.
        if rows != 0 {
            self.values.push(F::zero());
        }
        &mut self.values[start..start + rows]
    }

    fn write_into<V: DenseMatrixFromFn<F>>(
        self,
        rows: usize,
        columns: usize,
        out: &mut V::Matrix,
    ) {
        assert!(
            self.rows.is_none() || self.rows == Some(rows),
            "finite-difference output length changed"
        );
        V::dense_from_fn_into(
            rows,
            columns,
            |i, j| self.values[j * (rows + 1) + i],
            out,
        );
    }
}

fn jacobian_block<S, E, F>(
    n: usize,
    init: impl FnOnce() -> S,
    evaluate: impl Fn(&mut S, usize, &mut JacobianBlock<F>) -> Result<(), E>,
) -> Result<JacobianBlock<F>, E> {
    let mut scratch = init();
    let mut result = JacobianBlock {
        rows: None,
        columns: n,
        values: Vec::new(),
    };
    for j in 0..n {
        evaluate(&mut scratch, j, &mut result)?;
    }
    Ok(result)
}

fn forward_difference_jacobian_with_base_into<
    P,
    V,
    F: Scalar + MaybeSend + MaybeSync,
>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    r0: &P::Output,
    out: &mut <V as DenseMatrixFromFn<F>>::Matrix,
) -> Result<(), P::Error>
where
    P: Residual<Param = V> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + DenseMatrixFromFn<F>
        + MaybeSync
        + MaybeSend,
    P::Output: Clone + VectorLen + VectorIndex<F> + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    let eps = Method::Forward.first_derivative_scale(function_precision);
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        let n = x.vec_len();
        let m = r0.vec_len();
        let columns = try_map_range_with(
            n,
            || (x.clone(), r0.clone()),
            |(probe, residual), j| {
                let xj = x.get_scalar(j);
                let h = fixed_step.unwrap_or_else(|| minpack_step(xj, eps));
                probe.set_scalar(j, xj + h);
                problem.residual_into(probe, residual)?;
                assert_eq!(
                    residual.vec_len(),
                    m,
                    "finite-difference output length changed"
                );
                let mut col = residual.clone();
                probe.set_scalar(j, xj);
                for i in 0..m {
                    col.set_scalar(
                        i,
                        (col.get_scalar(i) - r0.get_scalar(i)) / h,
                    );
                }
                Ok(col)
            },
        )?;
        V::dense_from_fn_into(m, n, |i, j| columns[j].get_scalar(i), out);
        return Ok(());
    }
    {
        let n = x.vec_len();
        let m = r0.vec_len();
        let columns = jacobian_block(
            n,
            || (x.clone(), r0.clone()),
            |(probe, residual), j, block| {
                let xj = x.get_scalar(j);
                let h = fixed_step.unwrap_or_else(|| minpack_step(xj, eps));
                probe.set_scalar(j, xj + h);
                problem.residual_into(probe, residual)?;
                assert_eq!(
                    residual.vec_len(),
                    m,
                    "finite-difference output length changed"
                );
                probe.set_scalar(j, xj);
                block.push_column(m, |i| {
                    (residual.get_scalar(i) - r0.get_scalar(i)) / h
                });
                Ok(())
            },
        )?;
        columns.write_into::<V>(m, n, out);
        Ok(())
    }
}

/// Central-difference Jacobian, column `j ≈ (r(x+hⱼeⱼ) − r(x−hⱼeⱼ)) / 2hⱼ`.
///
/// More accurate than [`forward_difference_jacobian`] (`O(h²)`) at `2n`
/// residual evaluations (`1` when `n = 0`, to determine the output shape).
/// Result shape is `m × n`. Returns `Err` if any probe's `residual` does.
pub fn central_difference_jacobian<P, V>(
    problem: &P,
    x: &V,
    function_precision: f64,
    fixed_step: Option<f64>,
) -> Result<<V as DenseMatrixFromFn>::Matrix, P::Error>
where
    P: Residual<Param = V> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex
        + DenseMatrixFromFn
        + MaybeSync
        + MaybeSend,
    P::Output: Clone + VectorLen + VectorIndex + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    let mut out = V::dense_from_fn(0, x.vec_len(), |_, _| unreachable!());
    central_difference_jacobian_into(
        problem,
        x,
        function_precision,
        fixed_step,
        None,
        &mut out,
    )?;
    Ok(out)
}

fn central_difference_jacobian_into<P, V, F: Scalar + MaybeSend + MaybeSync>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    base: Option<&P::Output>,
    out: &mut <V as DenseMatrixFromFn<F>>::Matrix,
) -> Result<(), P::Error>
where
    P: Residual<Param = V> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + DenseMatrixFromFn<F>
        + MaybeSync
        + MaybeSend,
    P::Output: Clone + VectorLen + VectorIndex<F> + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    let scale = Method::Central.first_derivative_scale(function_precision);
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        let n = x.vec_len();
        if n == 0 {
            // With no probes, only a base evaluation can reveal the output shape.
            let m = if let Some(base) = base {
                base.vec_len()
            } else {
                residual_base(problem, x)?.vec_len()
            };
            V::dense_from_fn_into(m, 0, |_, _| unreachable!(), out);
            return Ok(());
        }
        let columns = try_map_range_with(
            n,
            || (x.clone(), base.cloned().or_else(P::Output::residual_buffer)),
            |(probe, residual), j| {
                let xj = x.get_scalar(j);
                let h = nr_step(xj, scale, fixed_step);
                probe.set_scalar(j, xj + h);
                let mut col = residual_probe(problem, probe, residual)?.clone();
                probe.set_scalar(j, xj - h);
                let residual = residual_probe(problem, probe, residual)?;
                probe.set_scalar(j, xj);
                let m = col.vec_len();
                assert_eq!(
                    residual.vec_len(),
                    m,
                    "finite-difference output length changed"
                );
                for i in 0..m {
                    col.set_scalar(
                        i,
                        (col.get_scalar(i) - residual.get_scalar(i))
                            / ((F::one() + F::one()) * h),
                    );
                }
                Ok(col)
            },
        )?;
        let m = base.map_or_else(|| columns[0].vec_len(), VectorLen::vec_len);
        assert!(
            columns.iter().all(|col| col.vec_len() == m),
            "finite-difference output length changed"
        );
        V::dense_from_fn_into(m, n, |i, j| columns[j].get_scalar(i), out);
        return Ok(());
    }
    {
        let n = x.vec_len();
        if n == 0 {
            // With no probes, only a base evaluation can reveal the output shape.
            let m = if let Some(base) = base {
                base.vec_len()
            } else {
                residual_base(problem, x)?.vec_len()
            };
            V::dense_from_fn_into(m, 0, |_, _| unreachable!(), out);
            return Ok(());
        }
        let columns = jacobian_block(
            n,
            || (x.clone(), base.cloned().or_else(P::Output::residual_buffer)),
            |(probe, residual), j, block| {
                let xj = x.get_scalar(j);
                let h = nr_step(xj, scale, fixed_step);
                probe.set_scalar(j, xj + h);
                let positive = residual_probe(problem, probe, residual)?;
                let m = positive.vec_len();
                let col = block.push_column(m, |i| positive.get_scalar(i));
                probe.set_scalar(j, xj - h);
                let residual = residual_probe(problem, probe, residual)?;
                probe.set_scalar(j, xj);
                assert_eq!(
                    residual.vec_len(),
                    m,
                    "finite-difference output length changed"
                );
                for (i, value) in col.iter_mut().enumerate() {
                    *value = (*value - residual.get_scalar(i))
                        / ((F::one() + F::one()) * h);
                }
                Ok(())
            },
        )?;
        let m = base.map_or_else(|| columns.rows.unwrap(), VectorLen::vec_len);
        columns.write_into::<V>(m, n, out);
        Ok(())
    }
}

fn residual_base<P: Residual>(
    problem: &P,
    x: &P::Param,
) -> Result<P::Output, P::Error>
where
    P::Output: VectorLen,
{
    if let Some(mut out) = P::Output::residual_buffer() {
        problem.residual_into(x, &mut out)?;
        Ok(out)
    } else {
        problem.residual(x)
    }
}

fn residual_probe<'a, P: Residual>(
    problem: &P,
    x: &P::Param,
    scratch: &'a mut Option<P::Output>,
) -> Result<&'a P::Output, P::Error> {
    if let Some(out) = scratch {
        problem.residual_into(x, out)?;
    } else {
        *scratch = Some(problem.residual(x)?);
    }
    Ok(scratch.as_ref().unwrap())
}

struct HessianTriangle {
    row_offsets: Vec<usize>,
    len: usize,
}

impl HessianTriangle {
    fn new(n: usize) -> Self {
        let mut len = 0;
        let row_offsets = (0..n)
            .map(|i| {
                let offset = len;
                len += n - i;
                offset
            })
            .collect();
        Self { row_offsets, len }
    }

    fn coordinates(&self, index: usize, row: &mut usize) -> (usize, usize) {
        // Consecutive probes usually share a row, including within Rayon tasks.
        let start = self.row_offsets[*row];
        if index < start || index - start >= self.row_offsets.len() - *row {
            *row =
                self.row_offsets.partition_point(|&offset| offset <= index) - 1;
        }
        let i = *row;
        (i, i + index - self.row_offsets[i])
    }

    fn symmetric_index(&self, i: usize, j: usize) -> usize {
        let row = i.min(j);
        self.row_offsets[row] + i.max(j) - row
    }
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
    let f0 = problem.cost(x)?;
    central_hessian(
        problem,
        x,
        function_precision,
        fixed_step,
        f0,
        |triangle, values| {
            V::dense_from_fn(x.vec_len(), x.vec_len(), |i, j| {
                values[triangle.symmetric_index(i, j)]
            })
        },
    )
}

fn central_hessian<P, V, F: Scalar + MaybeSend + MaybeSync, R>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    f0: F,
    publish: impl FnOnce(&HessianTriangle, &[F]) -> R,
) -> Result<R, P::Error>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    let two = F::one() + F::one();
    let scale = function_precision
        .max(F::epsilon())
        .powf(F::from_f64(0.25).unwrap());
    let h: Vec<F> = (0..n)
        .map(|j| nr_step(x.get_scalar(j), scale, fixed_step))
        .collect();
    // Row offsets avoid storing a pair of indices for every packed entry.
    let triangle = HessianTriangle::new(n);
    let values = try_map_range_with(
        triangle.len,
        || (x.clone(), 0),
        |(probe, row), index| {
            let (i, j) = triangle.coordinates(index, row);
            let xi = x.get_scalar(i);
            if i == j {
                probe.set_scalar(i, xi + h[i]);
                let fp = problem.cost(probe)?;
                probe.set_scalar(i, xi - h[i]);
                let fm = problem.cost(probe)?;
                probe.set_scalar(i, xi);
                Ok((fp - two * f0 + fm) / (h[i] * h[i]))
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
                Ok((fpp - fpm - fmp + fmm) / ((two + two) * h[i] * h[j]))
            }
        },
    )?;
    Ok(publish(&triangle, &values))
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
    forward_hessian(
        problem,
        x,
        function_precision,
        fixed_step,
        None,
        |triangle, values| {
            V::dense_from_fn(x.vec_len(), x.vec_len(), |i, j| {
                values[triangle.symmetric_index(i, j)]
            })
        },
    )
}

fn forward_hessian<P, V, F: Scalar + MaybeSend + MaybeSync, R>(
    problem: &P,
    x: &V,
    function_precision: F,
    fixed_step: Option<F>,
    base: Option<F>,
    publish: impl FnOnce(&HessianTriangle, &[F]) -> R,
) -> Result<R, P::Error>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    P::Error: MaybeSend,
{
    let n = x.vec_len();
    if n == 0 {
        if base.is_none() {
            problem.cost(x)?;
        }
        return Ok(publish(&HessianTriangle::new(0), &[]));
    }
    let scale = function_precision
        .max(F::epsilon())
        .powf(F::from_f64(0.25).unwrap());
    let h: Vec<F> = (0..n)
        .map(|j| nr_step(x.get_scalar(j), scale, fixed_step))
        .collect();
    let triangle = HessianTriangle::new(n);
    let base_probes = usize::from(base.is_none());
    let pair_start = base_probes + n;
    // Probe evaluations are independent. Only the final differences need the
    // base and singles, so one map avoids barriers between the three groups.
    let mut probes = try_map_range_with(
        pair_start + triangle.len,
        || (x.clone(), 0),
        |(probe, row), index| {
            if index < base_probes {
                return problem.cost(x);
            }
            if index < pair_start {
                let k = index - base_probes;
                let xk = x.get_scalar(k);
                probe.set_scalar(k, xk + h[k]);
                let f = problem.cost(probe)?;
                probe.set_scalar(k, xk);
                return Ok(f);
            }
            let (i, j) = triangle.coordinates(index - pair_start, row);
            let xi = x.get_scalar(i);
            let cross = if i == j {
                probe.set_scalar(i, xi + (F::one() + F::one()) * h[i]);
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
            Ok(cross)
        },
    )?;
    let (prefix, values) = probes.split_at_mut(pair_start);
    let f0 = base.unwrap_or_else(|| prefix[0]);
    let fi = &prefix[base_probes..];
    for i in 0..n {
        for j in i..n {
            let value = &mut values[triangle.symmetric_index(i, j)];
            *value = (*value - fi[i] - fi[j] + f0) / (h[i] * h[j]);
        }
    }
    Ok(publish(&triangle, values))
}

/// Euclidean norm via the scalar accessors; shared by the
/// Hessian-vector-product step rules.
fn vec_norm<F: Scalar, V: VectorLen + VectorIndex<F>>(x: &V) -> F {
    (0..x.vec_len())
        .map(|j| {
            let xj = x.get_scalar(j);
            xj * xj
        })
        .sum::<F>()
        .sqrt()
}

fn probe_along_into<F: Scalar, V: VectorLen + VectorIndex<F>>(
    x: &V,
    v: &V,
    h: F,
    probe: &mut V,
) {
    for j in 0..x.vec_len() {
        probe.set_scalar(j, x.get_scalar(j) + h * v.get_scalar(j));
    }
}

fn probe_along<F: Scalar, V: Clone + VectorLen + VectorIndex<F>>(
    x: &V,
    v: &V,
    h: F,
) -> V {
    let mut probe = x.clone();
    probe_along_into(x, v, h, &mut probe);
    probe
}

/// Caller-owned buffers for finite-difference Hessian-vector products.
///
/// Reuse this scratch with [`Self::forward_into`], [`Self::central_into`], or
/// [`FiniteDiff::hessian_product_with_scratch_into`] for repeated products.
/// The first nonzero product initializes one probe vector and one temporary
/// gradient. Later calls reuse them, resizing buffers when dimensions change.
/// A zero direction needs no scratch and makes no gradient evaluations.
///
/// Every successful nonzero product calls [`Gradient::gradient_into`] twice. Only storage
/// is retained between calls; no gradient or point is cached. Callbacks that
/// override `gradient_into` to reuse their output can avoid all allocations
/// after warm-up when the output and scratch have the correct shape. The
/// default callback may still allocate, as may numerical gradients.
///
/// The point and direction must have the same shape. The output is resized
/// when needed, and its contents are unspecified on error. Errors propagate
/// from the gradient callback, and scratch can be reused after an error.
/// Independent scratch buffers can share the same immutable problem across
/// threads when its type permits it. The scratch adds no thread-safety bounds
/// to the gradient problem or parameter vector.
///
/// # Backends
///
/// Products require only `V: Clone + VectorLen + VectorIndex`, supporting
/// `Vec<f64>`, nalgebra `DVector<f64>`, ndarray `Array1<f64>`, and faer
/// `Col<f64>`, without matrix operations.
///
/// # Examples
///
/// ```
/// use basin::{CostFunction, FiniteDiff, HessianProductScratch};
///
/// #[derive(Clone, Copy)]
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
///
/// let fd = FiniteDiff::new(Sphere).with_step(1e-3);
/// let x = vec![1.0, 2.0];
/// let mut scratch = HessianProductScratch::new();
/// let mut out = Vec::new();
/// for direction in [vec![1.0, 0.0], vec![0.0, 1.0]] {
///     fd.hessian_product_with_scratch_into(
///         &x, &direction, &mut out, &mut scratch,
///     ).unwrap();
///     for (actual, v) in out.iter().zip(direction) {
///         assert!((actual - 2.0 * v).abs() < 1e-6);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct HessianProductScratch<V> {
    buffers: Option<(V, V)>,
}

impl<V> Default for HessianProductScratch<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> HessianProductScratch<V> {
    /// Create empty scratch. Buffers are initialized lazily by the first
    /// nonzero product, so construction does not allocate or clone a vector.
    pub const fn new() -> Self {
        Self { buffers: None }
    }
}

struct HessianProductOptions<F> {
    method: Method,
    function_precision: F,
    fixed_step: Option<F>,
}

impl<V: Clone + VectorLen> HessianProductScratch<V> {
    /// Evaluate [`forward_difference_hessian_product`] into `out`, reusing
    /// probe and temporary-gradient storage. Step sizes, zero directions, and
    /// callback counts follow the free function's contract.
    pub fn forward_into<P>(
        &mut self,
        problem: &P,
        x: &V,
        v: &V,
        function_precision: f64,
        fixed_step: Option<f64>,
        out: &mut V,
    ) -> Result<(), P::Error>
    where
        V: VectorIndex,
        P: Gradient<Param = V, Gradient = V, Output = f64>,
    {
        self.difference_into(
            problem,
            x,
            v,
            HessianProductOptions {
                method: Method::Forward,
                function_precision,
                fixed_step,
            },
            out,
        )
    }

    /// Evaluate [`central_difference_hessian_product`] into `out`, reusing
    /// one probe for both directions and temporary-gradient storage. Step
    /// sizes, zero directions, and callback counts follow the free function's
    /// contract.
    pub fn central_into<P>(
        &mut self,
        problem: &P,
        x: &V,
        v: &V,
        function_precision: f64,
        fixed_step: Option<f64>,
        out: &mut V,
    ) -> Result<(), P::Error>
    where
        V: VectorIndex,
        P: Gradient<Param = V, Gradient = V, Output = f64>,
    {
        self.difference_into(
            problem,
            x,
            v,
            HessianProductOptions {
                method: Method::Central,
                function_precision,
                fixed_step,
            },
            out,
        )
    }

    fn difference_into<P, F: Scalar>(
        &mut self,
        problem: &P,
        x: &V,
        v: &V,
        options: HessianProductOptions<F>,
        out: &mut V,
    ) -> Result<(), P::Error>
    where
        V: VectorIndex<F>,
        P: Gradient<Param = V, Gradient = V, Output = F>,
    {
        if out.vec_len() != v.vec_len() {
            *out = v.clone();
        }
        let v_norm = vec_norm(v);
        if v_norm == F::zero() {
            for j in 0..out.vec_len() {
                out.set_scalar(j, F::zero());
            }
            return Ok(());
        }
        let h = options.fixed_step.unwrap_or_else(|| {
            let scale = options
                .method
                .first_derivative_scale(options.function_precision);
            scale * vec_norm(x).max(F::one()) / v_norm
        });
        let (probe, gradient) =
            self.buffers.get_or_insert_with(|| (x.clone(), x.clone()));
        if probe.vec_len() != x.vec_len() {
            probe.clone_from(x);
        }
        if gradient.vec_len() != x.vec_len() {
            gradient.clone_from(x);
        }
        match options.method {
            Method::Forward => {
                problem.gradient_into(x, gradient)?;
                probe_along_into(x, v, h, probe);
                problem.gradient_into(probe, out)?;
                for j in 0..out.vec_len() {
                    out.set_scalar(
                        j,
                        (out.get_scalar(j) - gradient.get_scalar(j)) / h,
                    );
                }
            }
            Method::Central => {
                probe_along_into(x, v, h, probe);
                problem.gradient_into(probe, out)?;
                // Rebuild from x to preserve rounding rather than subtracting
                // twice the step from the positive probe.
                probe_along_into(x, v, -h, probe);
                problem.gradient_into(probe, gradient)?;
                for j in 0..out.vec_len() {
                    out.set_scalar(
                        j,
                        (out.get_scalar(j) - gradient.get_scalar(j))
                            / ((F::one() + F::one()) * h),
                    );
                }
            }
        }
        Ok(())
    }
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
/// Use [`HessianProductScratch::forward_into`] to reuse buffers across products.
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
    let mut out = x.clone();
    forward_difference_hessian_product_into(
        problem,
        x,
        v,
        function_precision,
        fixed_step,
        &mut out,
    )?;
    Ok(out)
}

fn forward_difference_hessian_product_into<P, V, F: Scalar>(
    problem: &P,
    x: &V,
    v: &V,
    function_precision: F,
    fixed_step: Option<F>,
    out: &mut V,
) -> Result<(), <P as CostFunction>::Error>
where
    P: Gradient<Param = V, Gradient = V, Output = F>,
    V: Clone + VectorLen + VectorIndex<F>,
{
    if out.vec_len() != v.vec_len() {
        *out = v.clone();
    }
    let v_norm = vec_norm(v);
    if v_norm == F::zero() {
        for j in 0..out.vec_len() {
            out.set_scalar(j, F::zero());
        }
        return Ok(());
    }
    let h = match fixed_step {
        Some(h) => h,
        None => {
            Method::Forward.first_derivative_scale(function_precision)
                * vec_norm(x).max(F::one())
                / v_norm
        }
    };
    let mut g0 = x.clone();
    problem.gradient_into(x, &mut g0)?;
    problem.gradient_into(&probe_along(x, v, h), out)?;
    for j in 0..out.vec_len() {
        out.set_scalar(j, (out.get_scalar(j) - g0.get_scalar(j)) / h);
    }
    Ok(())
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
/// Use [`HessianProductScratch::central_into`] to reuse buffers across products.
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
    let mut out = x.clone();
    central_difference_hessian_product_into(
        problem,
        x,
        v,
        function_precision,
        fixed_step,
        &mut out,
    )?;
    Ok(out)
}

fn central_difference_hessian_product_into<P, V, F: Scalar>(
    problem: &P,
    x: &V,
    v: &V,
    function_precision: F,
    fixed_step: Option<F>,
    out: &mut V,
) -> Result<(), <P as CostFunction>::Error>
where
    P: Gradient<Param = V, Gradient = V, Output = F>,
    V: Clone + VectorLen + VectorIndex<F>,
{
    HessianProductScratch::new().difference_into(
        problem,
        x,
        v,
        HessianProductOptions {
            method: Method::Central,
            function_precision,
            fixed_step,
        },
        out,
    )
}

/// Bound-aware first-derivative adapters.
pub mod bounded;
pub use bounded::BoundedFiniteDiff;
pub mod check;
pub use check::{
    DerivativeCheckError, DerivativeCheckReport, DerivativeChecker,
    DerivativeComparison, DerivativeSource,
};

impl<P, const SQUARE: bool, F: Scalar> crate::Constraints
    for FiniteDiff<P, SQUARE, F>
where
    P: crate::Constraints,
{
    type Set = P::Set;
    fn constraints(
        &self,
    ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
        self.problem.constraints()
    }
}

impl<P, const SQUARE: bool, F: Scalar> crate::NonlinearEqualityValues
    for FiniteDiff<P, SQUARE, F>
where
    P: crate::NonlinearEqualityValues,
{
    type Param = P::Param;
    type Values = P::Values;
    type Error = P::Error;
    fn nonlinear_equalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, crate::ConstraintSolveError<Self::Error>> {
        self.problem.nonlinear_equalities(x)
    }
}

impl<P, const SQUARE: bool, F: Scalar> crate::NonlinearInequalityValues
    for FiniteDiff<P, SQUARE, F>
where
    P: crate::NonlinearInequalityValues,
{
    type Param = P::Param;
    type Values = P::Values;
    type Error = P::Error;
    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, crate::ConstraintSolveError<Self::Error>> {
        self.problem.nonlinear_inequalities(x)
    }
}

impl<P, V, const SQUARE: bool, F: Scalar + MaybeSend + MaybeSync>
    crate::ConstraintJacobian for FiniteDiff<P, SQUARE, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    P: crate::Constraints,
    P::Set: crate::core::constraint::evaluation::NonlinearSet<P, V, F>,
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F> + MaybeSync,
    P::Error: MaybeSend,
{
    type Param = V;
    type Jacobian = <V as DenseMatrixFromFn<F>>::Matrix;
    type Error = P::Error;
    fn constraint_jacobian(
        &self,
        x: &V,
    ) -> Result<Self::Jacobian, crate::ConstraintSolveError<P::Error>> {
        use crate::core::constraint::evaluation::NonlinearSet;
        let expected = P::Set::counts(self.problem.constraints());
        let m = expected.0.checked_add(expected.1).ok_or(
            crate::ConstraintSolveError::InvalidConstraints(
                crate::ConstraintError::CountOverflow,
            ),
        )?;
        let options = bounded::Options {
            method: self.jacobian_method,
            precision: self.function_precision,
            step: self.fixed_step,
            minpack: true,
        };
        let (_, cols) = bounded::columns(x, options, None, |point| {
            bounded::constraint_values(&self.problem, point, expected)
        })?;
        Ok(V::dense_from_fn(m, x.vec_len(), |i, j| cols[j][i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f(x) = Σ aᵢ xᵢ²`: gradient `2aᵢxᵢ`, diagonal Hessian `2aᵢ`. Used
    /// for backend-agnostic checks that don't need the `problems` feature.
    struct DiagQuadratic {
        a: Vec<f64>,
    }

    impl crate::Constraints for DiagQuadratic {
        type Set = crate::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
            crate::ConstraintSet::new()
        }
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

    impl crate::Constraints for DiagQuadraticGrad {
        type Set = crate::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
            crate::ConstraintSet::new()
        }
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
        fn gradient(
            &self,
            x: &Vec<f64>,
        ) -> Result<Vec<f64>, std::convert::Infallible> {
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
        let fwd =
            forward_difference_hessian_product(&p, &x, &v, f64::EPSILON, None)
                .unwrap();
        let cen =
            central_difference_hessian_product(&p, &x, &v, f64::EPSILON, None)
                .unwrap();
        for i in 0..3 {
            let want = 2.0 * a[i] * v[i];
            assert!(
                approx(fwd[i], want, 1e-5),
                "fwd i={i} {} vs {want}",
                fwd[i]
            );
            assert!(
                approx(cen[i], want, 1e-7),
                "cen i={i} {} vs {want}",
                cen[i]
            );
        }
    }

    #[test]
    fn hessian_product_of_zero_direction_is_zero() {
        let p = DiagQuadraticGrad { a: vec![1.0, 2.0] };
        let x = vec![-1.2, 0.7];
        let v = vec![0.0, 0.0];
        let hv =
            central_difference_hessian_product(&p, &x, &v, f64::EPSILON, None)
                .unwrap();
        assert_eq!(hv, vec![0.0, 0.0]);
    }

    #[test]
    fn finite_diff_hessian_product_matches_rosenbrock_analytic() {
        // FD-of-FD (the wrapper differences its own synthesized gradients),
        // so the tolerance is looser than for the analytic-gradient path.
        // Analytic 2D Rosenbrock Hessian at a generic point:
        //   H₀₀ = 2 − 400(x₁ − 3x₀²), H₀₁ = H₁₀ = −400x₀, H₁₁ = 200.
        struct Rosenbrock2;

        impl crate::Constraints for Rosenbrock2 {
            type Set = crate::NoConstraints;
            fn constraints(
                &self,
            ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_>
            {
                crate::ConstraintSet::new()
            }
        }
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
        use crate::{Executor, FirstOrderState};

        let result = Executor::new(
            FiniteDiff::new(DiagQuadratic {
                a: vec![1.0, 50.0, 2.5],
            }),
            (TrustRegion::matrix_free()).with_absolute_gradient_tolerance(1e-6),
            FirstOrderState::new(vec![5.0, 1.0, -3.0]),
        )
        .max_iter(100)
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
                assert!(
                    approx(g[i], want[i], 1e-8),
                    "i={i} {} vs {}",
                    g[i],
                    want[i]
                );
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
                assert!(
                    approx(g[i], want[i], 1e-4),
                    "i={i} {} vs {}",
                    g[i],
                    want[i]
                );
            }
        }
    }

    // BoxBounds forwarding: a wrapped boxed problem stays constrained
    // and still satisfies the gradient-solver bound.
    #[cfg(feature = "problems")]
    #[test]
    fn box_constrained_is_forwarded() {
        use crate::core::constraint::evaluation::BoxProblem;
        use crate::problems::rastrigin::RastriginBoxed;

        fn _assert_constrained_and_differentiable<
            T: crate::core::constraint::evaluation::BoxProblem + Gradient,
        >() {
        }
        _assert_constrained_and_differentiable::<
            FiniteDiff<RastriginBoxed<Vec<f64>>>,
        >();

        let lower = vec![-5.12, -5.12];
        let upper = vec![5.12, 5.12];
        let p = RastriginBoxed::new(lower.clone(), upper.clone());
        let wrapped = FiniteDiff::new(p);
        assert_eq!(wrapped.lower(), &lower);
        assert_eq!(wrapped.upper(), &upper);
    }

    #[cfg(all(feature = "problems", feature = "nalgebra_all"))]
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
            let fd =
                FiniteDiff::new(RosenbrockResiduals::<DVector<f64>>::new())
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
            let central =
                FiniteDiff::new(RosenbrockResiduals::<DVector<f64>>::new())
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
                    assert!(
                        approx(h[(i, j)], want, 1e-3),
                        "({i},{j}) {}",
                        h[(i, j)]
                    );
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
                    let rel = (h[(i, j)] - want[i][j]).abs()
                        / want[i][j].abs().max(1.0);
                    assert!(
                        rel < 1e-3,
                        "({i},{j}) {} vs {}",
                        h[(i, j)],
                        want[i][j]
                    );
                    // symmetry
                    assert!(approx(h[(i, j)], h[(j, i)], 1e-9));
                }
            }
        }
    }

    #[cfg(all(feature = "problems", feature = "faer_all"))]
    mod faer_matrix {
        use super::*;
        use crate::problems::rosenbrock::RosenbrockResiduals;
        use crate::problems::sphere::Sphere;
        use faer::Col;

        #[test]
        fn forward_jacobian_matches_analytic() {
            let x = Col::<f64>::from_fn(2, |i| [-1.2, 1.0][i]);
            let analytic =
                RosenbrockResiduals::<Col<f64>>::new().jacobian(&x).unwrap();
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
                    assert!(
                        approx(h[(i, j)], want, 1e-3),
                        "({i},{j}) {}",
                        h[(i, j)]
                    );
                }
            }
        }
    }
}
