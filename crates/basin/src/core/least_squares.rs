//! Robust objectives for nonlinear least squares.
//!
//! [`RobustLeastSquares`] combines raw residuals with a scalar [`LossFunction`].
//! The objective is `C(x) = s²/2 Σ ρ((rᵢ(x)/s)²)`. The scale `s` describes
//! residuals, independently of parameter scaling and solver damping.
//!
//! Least-squares solvers use the safeguarded Gauss–Newton model described in
//! [SciPy's least-squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#notes)
//! and Triggs et al. (1999), *Bundle Adjustment — A Modern Synthesis*,
//! <https://doi.org/10.1007/3-540-44480-7_21>. For `zᵢ = (rᵢ/s)²`, they form
//! `aᵢ = sqrt(max(ρ′(zᵢ) + 2zᵢρ″(zᵢ), ε))`, `J̃ᵢ = aᵢJᵢ`, and
//! `r̃ᵢ = ρ′(zᵢ)rᵢ/aᵢ`. Thus `J̃ᵀr̃` is the robust gradient, while `J̃ᵀJ̃`
//! safeguards the curvature. No second derivatives of the raw residuals are
//! required. The model residual norm is **not** the objective: reported costs
//! and actual reductions always use `C`.
//!
//! Put numerical differentiation inside the adapter, for example
//! `RobustLeastSquares::new(FiniteDiff::new(fit), HuberLoss)`. The underlying
//! [`Residual`] and [`Jacobian`] then continue to describe the original model.
//! The adapter instead exposes [`CostFunction`] and [`Gradient`], so the same
//! objective also works with general optimizers. Compatible constraint traits
//! are forwarded. Losses and scale must remain fixed during a solve or exact
//! continuation; changing them requires a fresh solve.

use crate::SharedEvaluationBuffers;
use crate::core::math::{
    MatTransposeVec, MatTransposeVecInto, Scalar, VectorIndex, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Jacobian, Residual};

mod constraints;
pub(crate) mod evaluation;
#[cfg(test)]
mod iterative_trf_api_prototype;
mod losses;
#[cfg(test)]
mod tests;
pub use losses::{ArctanLoss, CauchyLoss, HuberLoss, SoftL1Loss, SquaredLoss};

pub(crate) fn squared_loss_reduction<F: Scalar>(before: F, after: F) -> (F, F) {
    let half = F::from_f64(0.5).unwrap();
    let before = before.abs();
    let after = after.abs();
    if before == after {
        return (F::zero(), F::zero());
    }
    // TwoSum recovers the error in each factor. The factored difference
    // also avoids subtracting two separately rounded squares.
    let two_sum = |a: F, b: F| {
        let sum = a + b;
        let b_rounded = sum - a;
        (sum, (a - (sum - b_rounded)) + (b - b_rounded))
    };
    let (sum, sum_error) = two_sum(before, after);
    let (difference, difference_error) = two_sum(before, -after);
    let average = half * sum;
    let average_error = half * sum_error;
    let term = difference * average;
    if !term.is_finite() {
        return (term, F::zero());
    }
    // Expand the product of the corrected factors. Their first-order
    // corrections can cancel, leaving the product of the two errors.
    let error = difference.mul_add(average, -term)
        + difference * average_error
        + difference_error * average
        + difference_error * average_error;
    (term, error)
}

pub(crate) struct ReductionSum<F> {
    sum: F,
    correction: F,
}

impl<F: Scalar> ReductionSum<F> {
    pub(crate) fn new() -> Self {
        Self {
            sum: F::zero(),
            correction: F::zero(),
        }
    }

    pub(crate) fn add(&mut self, term: F) {
        let sum = self.sum + term;
        if sum.is_finite() {
            // Neumaier compensation recovers rounding error even when the
            // new term is larger in magnitude than the running sum.
            let error = if self.sum.abs() >= term.abs() {
                (self.sum - sum) + term
            } else {
                (term - sum) + self.sum
            };
            self.correction = self.correction + error;
        }
        self.sum = sum;
    }

    pub(crate) fn total(self) -> F {
        self.sum + self.correction
    }
}

/// A scalar loss value and its first two derivatives.
///
/// The differentiation variable is the squared normalized residual for
/// [`LossFunction::evaluate`] and the raw residual for
/// [`LossFunction::evaluate_scaled`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossEvaluation<F: Scalar = f64> {
    /// The loss value.
    pub value: F,
    /// The first derivative.
    pub first_derivative: F,
    /// The second derivative.
    pub second_derivative: F,
}

/// A componentwise loss `ρ(z)`, where `z = (residual / scale)²`.
///
/// Implementations must be pure, with `ρ(0) = 0`, nonnegative values and first
/// derivatives, and consistent finite derivatives for finite `z >= 0` in their
/// domain. Negative second derivatives are allowed. An undefined evaluation
/// may return non-finite fields; the adapter treats it as invalid, never as a
/// stationary point. At a second-derivative discontinuity, document the branch
/// used (Huber uses the quadratic branch at `z = 1`).
pub trait LossFunction<F: Scalar = f64> {
    /// Evaluate the loss and both derivatives at a squared normalized residual.
    fn evaluate(&self, z: F) -> LossEvaluation<F>;

    /// Evaluate `scale² ρ((residual / scale)²) / 2` and its first two
    /// derivatives with respect to `residual`.
    ///
    /// The adapter supplies a finite residual and a finite, positive scale.
    /// The default uses [`evaluate`](Self::evaluate), so its intermediate
    /// normalized square and loss must be representable. Override this method
    /// to handle a wider range of scales. The built-in losses do so without
    /// forming an unbounded normalized square.
    fn evaluate_scaled(&self, residual: F, scale: F) -> LossEvaluation<F> {
        let t = residual / scale;
        let z = t * t;
        let invalid = || LossEvaluation {
            value: F::infinity(),
            first_derivative: F::nan(),
            second_derivative: F::nan(),
        };
        if !z.is_finite() || (z == F::zero() && residual != F::zero()) {
            return invalid();
        }
        let value = self.evaluate(z);
        if !value.value.is_finite()
            || value.value < F::zero()
            || !value.first_derivative.is_finite()
            || value.first_derivative < F::zero()
            || !value.second_derivative.is_finite()
        {
            return invalid();
        }
        // Taking the square root before rescaling avoids forming scale² or
        // halving a subnormal loss before a large scale restores its magnitude.
        let root = scale * value.value.sqrt();
        LossEvaluation {
            value: (F::from_f64(0.5).unwrap() * root) * root,
            first_derivative: value.first_derivative * residual,
            second_derivative: value.first_derivative
                + F::from_f64(2.0).unwrap() * (value.second_derivative * z),
        }
    }

    /// Compute the signed decrease in scaled loss from `before` to `after`.
    ///
    /// The adapter supplies finite residuals, a finite positive scale, and
    /// their validated costs from [`evaluate_scaled`](Self::evaluate_scaled).
    /// The default subtracts the costs. Override this method to preserve
    /// small reductions that would round away in the individual costs.
    fn reduction_scaled(
        &self,
        _before: F,
        _after: F,
        _scale: F,
        before_cost: F,
        after_cost: F,
    ) -> F {
        before_cost - after_cost
    }
    /// Return a reduction and its rounding correction for compensated summation.
    #[doc(hidden)]
    fn reduction_parts(
        &self,
        before: F,
        after: F,
        scale: F,
        before_cost: F,
        after_cost: F,
    ) -> (F, F) {
        (
            self.reduction_scaled(
                before,
                after,
                scale,
                before_cost,
                after_cost,
            ),
            F::zero(),
        )
    }
}

impl<F: Scalar, L: LossFunction<F> + ?Sized> LossFunction<F> for &L {
    fn evaluate(&self, z: F) -> LossEvaluation<F> {
        (**self).evaluate(z)
    }

    fn evaluate_scaled(&self, residual: F, scale: F) -> LossEvaluation<F> {
        (**self).evaluate_scaled(residual, scale)
    }

    fn reduction_scaled(
        &self,
        before: F,
        after: F,
        scale: F,
        before_cost: F,
        after_cost: F,
    ) -> F {
        (**self).reduction_scaled(before, after, scale, before_cost, after_cost)
    }
    fn reduction_parts(
        &self,
        before: F,
        after: F,
        scale: F,
        before_cost: F,
        after_cost: F,
    ) -> (F, F) {
        (**self).reduction_parts(before, after, scale, before_cost, after_cost)
    }
}

/// An explicit robust objective built from a residual problem and a loss.
///
/// Supports [`crate::GaussNewton`], [`crate::LevenbergMarquardt`],
/// [`crate::LevenbergMarquardtQr`], [`crate::BoundedLevenbergMarquardt`],
/// [`crate::TrustRegionReflective`], and
/// [`TrustRegion`'s least-squares mode](crate::TrustRegion::least_squares_with).
/// Redescending losses such as Cauchy and arctangent can have several local
/// minima.
///
/// This type does not implement `Residual` or `Jacobian`: its robust cost
/// differs from the squared norm of the raw residuals. Access those through
/// [`inner`](Self::inner). Its `CostFunction` implementation derives its cost
/// from the residuals and does not call the wrapped problem's `cost` method.
/// Invalid numeric evaluations return an infinite cost; invalid gradients
/// contain NaNs. Underlying problem errors propagate unchanged.
///
/// # Backends
///
/// `Vec<F>`/`DenseMatrix<F>`, nalgebra `DVector<F>`/`DMatrix<F>`, ndarray
/// `Array1<F>`/`Array2<F>`, and faer `Col<F>`/`Mat<F>`, for `f32` and `f64`.
/// Gauss–Newton, normal-equations LM, and bounded LM also support nalgebra
/// `CscMatrix<F>` and faer `SparseColMat<usize, F>`. Custom Jacobians need
/// [`crate::ScaleRowsInPlace`] for these solver integrations, in addition to
/// their existing solver capabilities. The scalar `CostFunction`/`Gradient`
/// route only needs vector indexing and, for gradients, a transpose matvec.
///
/// # Examples
///
/// ```
/// use basin::{DenseMatrix, Executor, HuberLoss, Jacobian,
///     LevenbergMarquardt, Residual, RobustLeastSquares};
/// struct Location;
/// impl Residual for Location {
///     type Param = Vec<f64>;
///     type Output = Vec<f64>;
///     type Error = std::convert::Infallible;
///     fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(vec![x[0], x[0], x[0], x[0] - 10.0])
///     }
/// }
/// impl Jacobian for Location {
///     type Jacobian = DenseMatrix;
///     fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
///         Ok(DenseMatrix::from_row_slice(4, 1, &[1.0; 4]))
///     }
/// }
/// let objective = RobustLeastSquares::new(Location, HuberLoss).with_scale(1.0);
/// let result = Executor::from_start(objective, LevenbergMarquardt::new(), vec![0.0])
///     .max_iter(100).run().unwrap();
/// assert!((result.param()[0] - 1.0 / 3.0).abs() < 1e-7);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RobustLeastSquares<P, L, F: Scalar = f64, W = ()> {
    problem: P,
    loss: L,
    scale: F,
    workspace: W,
}

impl<P, L, F: Scalar> RobustLeastSquares<P, L, F> {
    /// Wrap a residual problem with a loss and unit residual scale.
    pub fn new(problem: P, loss: L) -> Self {
        Self {
            problem,
            loss,
            scale: F::one(),
            workspace: (),
        }
    }

    /// Retain raw residuals, Jacobians, and loss derivatives between requests.
    ///
    /// The first request seeds backend storage through the owned callbacks;
    /// later requests call their in-place counterparts. Fused gradient requests
    /// preserve the raw fused callback. Concurrent calls and clones share
    /// synchronized scratch space, but every call reevaluates the problem.
    /// The buffered gradient path requires [`MatTransposeVecInto`].
    #[allow(clippy::type_complexity)]
    pub fn with_evaluation_buffers(
        self,
    ) -> RobustLeastSquares<
        P,
        L,
        F,
        SharedEvaluationBuffers<P::Output, P::Jacobian, F>,
    >
    where
        P: Jacobian,
    {
        RobustLeastSquares {
            problem: self.problem,
            loss: self.loss,
            scale: self.scale,
            workspace: SharedEvaluationBuffers::default(),
        }
    }
}

impl<P, L, F: Scalar, W> RobustLeastSquares<P, L, F, W> {
    /// Set the residual scale. Panics unless it is finite and strictly positive.
    pub fn with_scale(mut self, scale: F) -> Self {
        assert!(
            scale.is_finite() && scale > F::zero(),
            "residual scale must be finite and positive"
        );
        self.scale = scale;
        self
    }

    /// The underlying raw residual problem.
    pub fn inner(&self) -> &P {
        &self.problem
    }
    /// The loss function.
    pub fn loss(&self) -> &L {
        &self.loss
    }
    /// The fixed residual scale.
    pub fn scale(&self) -> F {
        self.scale
    }
    /// Consume the adapter and recover the raw residual problem.
    pub fn into_inner(self) -> P {
        self.problem
    }
}

struct Term<F> {
    cost: F,
    derivative: F,
    curvature: F,
}

impl<P, L: LossFunction<F>, F: Scalar, W> RobustLeastSquares<P, L, F, W> {
    fn term(&self, r: F) -> Option<Term<F>> {
        if !r.is_finite() {
            return None;
        }
        let value = self.loss.evaluate_scaled(r, self.scale);
        if !value.value.is_finite()
            || value.value < F::zero()
            || !value.first_derivative.is_finite()
            || (r > F::zero() && value.first_derivative < F::zero())
            || (r < F::zero() && value.first_derivative > F::zero())
            || !value.second_derivative.is_finite()
        {
            return None;
        }
        Some(Term {
            cost: value.value,
            derivative: value.first_derivative,
            curvature: value.second_derivative,
        })
    }

    pub(crate) fn residual_cost<V: VectorLen + VectorIndex<F>>(
        &self,
        r: &V,
    ) -> F {
        let mut sum = F::zero();
        for i in 0..r.vec_len() {
            let Some(term) = self.term(r.get_scalar(i)) else {
                return F::infinity();
            };
            sum = sum + term.cost;
        }
        sum
    }

    fn residual_cost_and_reduction<V: VectorLen + VectorIndex<F>>(
        &self,
        current: &V,
        trial: &V,
    ) -> (F, F) {
        let mut cost = F::zero();
        let mut reduction = ReductionSum::new();
        for i in 0..trial.vec_len() {
            let before = current.get_scalar(i);
            let after = trial.get_scalar(i);
            let Some(next) = self.term(after) else {
                return (F::infinity(), F::neg_infinity());
            };
            cost = cost + next.cost;
            if before != after {
                let (term, error) = self.term(before).map_or(
                    (F::nan(), F::zero()),
                    |previous| {
                        self.loss.reduction_parts(
                            before,
                            after,
                            self.scale,
                            previous.cost,
                            next.cost,
                        )
                    },
                );
                reduction.add(term);
                reduction.add(error);
            }
        }
        (cost, reduction.total())
    }

    fn model_residual<V: Clone + VectorLen + VectorIndex<F>>(
        &self,
        r: &V,
    ) -> Option<(V, Vec<F>)> {
        let mut model = r.clone();
        let mut factors = Vec::with_capacity(r.vec_len());
        let mut cost = F::zero();
        for i in 0..r.vec_len() {
            let term = self.term(r.get_scalar(i))?;
            cost = cost + term.cost;
            let factor = term.curvature.max(F::epsilon()).sqrt();
            let value = term.derivative / factor;
            if !value.is_finite() {
                return None;
            }
            model.set_scalar(i, value);
            factors.push(factor);
        }
        cost.is_finite().then_some((model, factors))
    }

    fn gradient_from<V, R, M>(&self, mut r: R, j: M) -> V
    where
        V: VectorLen + VectorIndex<F>,
        R: VectorLen + VectorIndex<F>,
        M: MatTransposeVec<V, R>,
    {
        let mut valid = true;
        for i in 0..r.vec_len() {
            let derivative = self.term(r.get_scalar(i)).map_or_else(
                || {
                    valid = false;
                    F::nan()
                },
                |t| t.derivative,
            );
            r.set_scalar(i, derivative);
        }
        let mut gradient = j.mat_transpose_vec(&r);
        if !valid {
            // An invalid residual on an empty sparse row must still invalidate
            // the gradient, even though that row contributes no stored entries.
            for i in 0..gradient.vec_len() {
                gradient.set_scalar(i, F::nan());
            }
        }
        gradient
    }
}

impl<P, L, F> CostFunction for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual,
    P::Output: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    type Param = P::Param;
    type Output = F;
    type Error = P::Error;
    fn cost(&self, x: &P::Param) -> Result<F, P::Error> {
        Ok(self.residual_cost(&self.problem.residual(x)?))
    }
}

impl<P, L, V, F, R> Gradient for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = R> + Jacobian,
    P::Jacobian: MatTransposeVec<V, R>,
    R: VectorLen + VectorIndex<F>,
    V: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, P::Error> {
        let (r, j) = self.problem.residual_and_jacobian(x)?;
        Ok(self.gradient_from(r, j))
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(F, V), P::Error> {
        let (r, j) = self.problem.residual_and_jacobian(x)?;
        Ok((self.residual_cost(&r), self.gradient_from(r, j)))
    }
}

impl<P, L, F, W> crate::Constraints for RobustLeastSquares<P, L, F, W>
where
    P: crate::Constraints,
    F: crate::Scalar,
{
    type Set = P::Set;
    fn constraints(
        &self,
    ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
        self.problem.constraints()
    }
}

impl<P, L, R, M, F> CostFunction
    for RobustLeastSquares<P, L, F, SharedEvaluationBuffers<R, M, F>>
where
    F: Scalar,
    P: Residual<Output = R>,
    R: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    type Param = P::Param;
    type Output = F;
    type Error = P::Error;
    fn cost(&self, x: &P::Param) -> Result<F, P::Error> {
        self.workspace.with_buffers(|buffers, _| {
            if let Some(r) = &mut buffers.vector {
                self.problem.residual_into(x, r)?;
            } else {
                buffers.vector = Some(self.problem.residual(x)?);
            }
            Ok(self.residual_cost(buffers.vector.as_ref().unwrap()))
        })
    }
}

impl<P, L, V, M, F, R> Gradient
    for RobustLeastSquares<P, L, F, SharedEvaluationBuffers<R, M, F>>
where
    F: Scalar,
    P: Residual<Param = V, Output = R> + Jacobian<Jacobian = M>,
    M: MatTransposeVecInto<V, F>,
    R: VectorLen + VectorIndex<F>,
    V: Clone + VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, P::Error> {
        let mut out = x.clone();
        self.gradient_into(x, &mut out)?;
        Ok(out)
    }
    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), P::Error> {
        self.cost_and_gradient_into(x, out).map(|_| ())
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(F, V), P::Error> {
        let mut out = x.clone();
        let cost = self.cost_and_gradient_into(x, &mut out)?;
        Ok((cost, out))
    }
    fn cost_and_gradient_into(
        &self,
        x: &V,
        out: &mut V,
    ) -> Result<F, P::Error> {
        self.workspace.with_buffers(|buffers, derivatives| {
            match (&mut buffers.vector, &mut buffers.matrix) {
                (Some(r), Some(j)) => {
                    self.problem.residual_and_jacobian_into(x, r, j)?
                }
                _ => {
                    let (r, j) = self.problem.residual_and_jacobian(x)?;
                    buffers.vector = Some(r);
                    buffers.matrix = Some(j);
                }
            }
            let r = buffers.vector.as_ref().unwrap();
            let j = buffers.matrix.as_ref().unwrap();
            let cost = self.residual_cost(r);
            derivatives.clear();
            let mut valid = true;
            for i in 0..r.vec_len() {
                derivatives.push(self.term(r.get_scalar(i)).map_or_else(
                    || {
                        valid = false;
                        F::nan()
                    },
                    |t| t.derivative,
                ));
            }
            if out.vec_len() != x.vec_len() {
                *out = x.clone();
            }
            j.mat_transpose_vec_into(derivatives, out);
            if !valid {
                for i in 0..out.vec_len() {
                    out.set_scalar(i, F::nan());
                }
            }
            Ok(cost)
        })
    }
}
