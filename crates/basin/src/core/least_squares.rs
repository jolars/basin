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

use crate::core::math::{MatTransposeVec, Scalar, VectorIndex, VectorLen};
use crate::core::problem::{CostFunction, Gradient, Jacobian, Residual};

mod constraints;
pub(crate) mod evaluation;
mod losses;
#[cfg(test)]
mod tests;
pub use losses::{ArctanLoss, CauchyLoss, HuberLoss, SoftL1Loss, SquaredLoss};

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
}

impl<F: Scalar, L: LossFunction<F> + ?Sized> LossFunction<F> for &L {
    fn evaluate(&self, z: F) -> LossEvaluation<F> {
        (**self).evaluate(z)
    }

    fn evaluate_scaled(&self, residual: F, scale: F) -> LossEvaluation<F> {
        (**self).evaluate_scaled(residual, scale)
    }
}

/// An explicit robust objective built from a residual problem and a loss.
///
/// Supports [`crate::GaussNewton`], [`crate::LevenbergMarquardt`],
/// [`crate::LevenbergMarquardtQr`], [`crate::Trf`], and
/// [`crate::TrustRegionReflective`]. Their normal algorithmic limitations
/// remain: in particular, Gauss–Newton takes full steps, and redescending
/// losses such as Cauchy and arctangent can have several local minima.
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
/// Gauss–Newton, normal-equations LM, and legacy TRF also support nalgebra
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
pub struct RobustLeastSquares<P, L, F: Scalar = f64> {
    problem: P,
    loss: L,
    scale: F,
}

impl<P, L, F: Scalar> RobustLeastSquares<P, L, F> {
    /// Wrap a residual problem with a loss and unit residual scale.
    pub fn new(problem: P, loss: L) -> Self {
        Self {
            problem,
            loss,
            scale: F::one(),
        }
    }

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

impl<P, L: LossFunction<F>, F: Scalar> RobustLeastSquares<P, L, F> {
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

    fn gradient_from<V: VectorLen + VectorIndex<F>, M: MatTransposeVec<V>>(
        &self,
        mut r: V,
        j: M,
    ) -> V {
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

impl<P, L, V, F> Gradient for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian,
    P::Jacobian: MatTransposeVec<V>,
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
