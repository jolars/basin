//! Shared first-derivative probes with explicit domain bounds.

use super::{FiniteDiff, Method};
use crate::core::constraint::{
    BoxConstraints, ConstraintJacobian, NonlinearConstraints,
};
use crate::core::math::{DenseMatrixFromFn, Scalar, VectorIndex, VectorLen};
use crate::core::parallel::{MaybeSend, MaybeSync, try_map_range_with};
use crate::core::problem::{CostFunction, Gradient, Jacobian, Residual};

/// Finite-difference first derivatives whose probes stay inside supplied bounds.
///
/// Bounds constrain differentiation probes; they do not replace the wrapped
/// problem's mathematical constraints. Pass its box bounds here when its
/// callbacks are defined only inside that box. All problem-side constraint
/// traits are forwarded unchanged. Non-finite bound entries mean an unbounded
/// side, matching [`NonlinearConstraints`]. Bounds must have matching lengths
/// and be ordered; differentiation points must be finite and inside them.
///
/// Defaults match [`FiniteDiff`]: central gradients, forward Jacobians, and
/// machine-precision adaptive steps. Near a boundary, central differentiation
/// uses a second-order one-sided stencil. If only one distinct probe fits,
/// it uses a first-order difference. Fixed coordinates have zero derivative
/// columns, representing derivatives on the remaining free coordinates.
/// Actual representable displacements are used in denominators.
///
/// This adapter synthesizes [`Gradient`], [`Jacobian`], and
/// [`ConstraintJacobian`], but does not advertise Hessian capabilities.
/// User errors propagate unchanged; non-finite values yield non-finite
/// derivatives. Like `FiniteDiff`, counters charge derivative calls rather
/// than internal probes. The optional `parallel` feature evaluates independent
/// columns concurrently without changing their arithmetic or result ordering.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for `F = f64` or `f32`. Jacobians use each vector's [`DenseMatrixFromFn`]
/// matrix type; the constraint trait's `Matrix` must match it.
///
/// # Examples
/// ```
/// use basin::{BoundedFiniteDiff, CostFunction, Gradient};
/// struct Square;
/// impl CostFunction for Square {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = &'static str;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         if !(0.0..=1.0).contains(&x[0]) { return Err("outside domain"); }
///         Ok(x[0]*x[0])
///     }
/// }
/// let p = BoundedFiniteDiff::new(Square, vec![0.0], vec![1.0]);
/// assert!((p.gradient(&vec![1.0]).unwrap()[0] - 2.0).abs() < 1e-8);
/// ```
#[derive(Clone, Debug)]
pub struct BoundedFiniteDiff<P, F: Scalar = f64> {
    pub(super) problem: P,
    lower: Vec<F>,
    upper: Vec<F>,
    gradient_method: Method,
    jacobian_method: Method,
    precision: F,
    step: Option<F>,
}
impl<P, F: Scalar> BoundedFiniteDiff<P, F> {
    /// Copy bounds from any supported vector backend and wrap the problem.
    /// Panics for mismatched lengths or a finite lower bound above its upper.
    pub fn new<V: VectorLen + VectorIndex<F>>(
        problem: P,
        lower: V,
        upper: V,
    ) -> Self {
        assert_eq!(
            lower.vec_len(),
            upper.vec_len(),
            "finite-difference bound length mismatch"
        );
        let lower: Vec<_> = (0..lower.vec_len())
            .map(|i| {
                let v = lower.get_scalar(i);
                if v.is_finite() { v } else { F::neg_infinity() }
            })
            .collect();
        let upper: Vec<_> = (0..upper.vec_len())
            .map(|i| {
                let v = upper.get_scalar(i);
                if v.is_finite() { v } else { F::infinity() }
            })
            .collect();
        assert!(
            lower.iter().zip(&upper).all(|(l, u)| l <= u),
            "finite-difference lower bound exceeds upper bound"
        );
        Self {
            problem,
            lower,
            upper,
            gradient_method: Method::Central,
            jacobian_method: Method::Forward,
            precision: F::epsilon(),
            step: None,
        }
    }
    /// Choose the objective-gradient stencil (default: central).
    pub fn gradient_method(mut self, method: Method) -> Self {
        self.gradient_method = method;
        self
    }
    /// Choose the residual/constraint Jacobian stencil (default: forward).
    pub fn jacobian_method(mut self, method: Method) -> Self {
        self.jacobian_method = method;
        self
    }
    /// Set finite nonnegative relative function precision, floored at epsilon.
    pub fn function_precision(mut self, precision: F) -> Self {
        assert!(
            precision.is_finite() && precision >= F::zero(),
            "invalid function precision"
        );
        self.precision = precision;
        self
    }
    /// Set a finite positive desired absolute step. Bounds may shorten it.
    pub fn with_step(mut self, step: F) -> Self {
        assert!(
            step.is_finite() && step > F::zero(),
            "finite-difference step must be finite and positive"
        );
        self.step = Some(step);
        self
    }
    /// Borrow the wrapped problem.
    pub fn get_ref(&self) -> &P {
        &self.problem
    }
    /// Recover the wrapped problem.
    pub fn into_inner(self) -> P {
        self.problem
    }
    fn options(&self, jacobian: bool) -> Options<F> {
        Options {
            method: if jacobian {
                self.jacobian_method
            } else {
                self.gradient_method
            },
            precision: self.precision,
            step: self.step,
            minpack: jacobian,
        }
    }
}
impl<P> FiniteDiff<P> {
    /// Transfer first-derivative settings into a [`BoundedFiniteDiff`].
    /// The resulting adapter supports gradients and Jacobians only. Hessian
    /// settings have no effect on its first derivatives.
    pub fn with_bounds<V: VectorLen + VectorIndex>(
        self,
        lower: V,
        upper: V,
    ) -> BoundedFiniteDiff<P> {
        let mut result = BoundedFiniteDiff::new(self.problem, lower, upper)
            .gradient_method(self.gradient_method)
            .jacobian_method(self.jacobian_method)
            .function_precision(self.function_precision);
        if let Some(step) = self.fixed_step {
            result = result.with_step(step);
        }
        result
    }
}

impl<P: CostFunction, F: Scalar> CostFunction for BoundedFiniteDiff<P, F> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn cost(&self, x: &P::Param) -> Result<P::Output, P::Error> {
        self.problem.cost(x)
    }
}
impl<P: Residual, F: Scalar> Residual for BoundedFiniteDiff<P, F> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn residual(&self, x: &P::Param) -> Result<P::Output, P::Error> {
        self.problem.residual(x)
    }
}
impl<P: BoxConstraints, F: Scalar> BoxConstraints for BoundedFiniteDiff<P, F> {
    fn lower(&self) -> &P::Param {
        self.problem.lower()
    }
    fn upper(&self) -> &P::Param {
        self.problem.upper()
    }
}

pub(super) struct Options<F> {
    pub method: Method,
    pub precision: F,
    pub step: Option<F>,
    pub minpack: bool,
}

fn probes<F: Scalar>(
    x: F,
    h: F,
    lo: F,
    hi: F,
    method: Method,
) -> (F, Option<F>) {
    let zero = F::zero();
    let two = F::one() + F::one();
    if lo == hi {
        return (x, None);
    }
    let (left, right) = (x - lo, hi - x);
    if matches!(method, Method::Central) && left >= h && right >= h {
        return ((x + h).min(hi), Some((x - h).max(lo)));
    }
    let forward = if right >= h && matches!(method, Method::Forward) {
        true
    } else {
        right >= left
    };
    let (room, sign, end) = if forward {
        (right, F::one(), hi)
    } else {
        (left, -F::one(), lo)
    };
    let central = matches!(method, Method::Central);
    let step = h.min(if central { room / two } else { room });
    let mut first = (x + sign * step).max(lo).min(hi);
    let second = if central {
        Some((x + sign * (two * step)).max(lo).min(hi))
    } else {
        None
    };
    if first == x {
        // An adjacent representable endpoint may be the only usable probe.
        first = if end.is_finite() {
            end
        } else {
            x + sign * (F::epsilon() * x.abs().max(F::one()))
        };
    }
    let second = second.filter(|&p| p != x && p != first);
    debug_assert!(first >= lo && first <= hi);
    if (first - x) == zero {
        (x, None)
    } else {
        (first, second)
    }
}

pub(super) fn columns<V, F, E, C>(
    x: &V,
    options: Options<F>,
    bounds: Option<(&[F], &[F])>,
    evaluate: C,
) -> Result<(usize, Vec<Vec<F>>), E>
where
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    F: Scalar + MaybeSync + MaybeSend,
    E: MaybeSend,
    C: Fn(&V) -> Result<Vec<F>, E> + MaybeSync,
{
    if let Some((lo, hi)) = bounds {
        assert_eq!(
            x.vec_len(),
            lo.len(),
            "finite-difference point and bounds length mismatch"
        );
        assert_eq!(
            lo.len(),
            hi.len(),
            "finite-difference bound length mismatch"
        );
        assert!(
            (0..x.vec_len()).all(|i| {
                let v = x.get_scalar(i);
                v.is_finite() && v >= lo[i] && v <= hi[i]
            }),
            "finite-difference point is outside its bounds"
        );
    }
    let base = evaluate(x)?;
    let m = base.len();
    let precision = options.precision.max(F::epsilon());
    let scale = if matches!(options.method, Method::Forward) {
        precision.sqrt()
    } else {
        precision.cbrt()
    };
    let evaluate = &evaluate;
    let columns = try_map_range_with(
        x.vec_len(),
        || x.clone(),
        |probe, j| {
            let value = x.get_scalar(j);
            let mut h = options.step.unwrap_or_else(|| {
                if options.minpack && matches!(options.method, Method::Forward)
                {
                    let h = scale * value.abs();
                    if h == F::zero() { scale } else { h }
                } else {
                    scale * value.abs().max(F::one())
                }
            });
            if value + h == value || value - h == value {
                h = scale * value.abs().max(F::one());
            }
            let (lo, hi) = bounds
                .map_or((F::neg_infinity(), F::infinity()), |(lo, hi)| {
                    (lo[j], hi[j])
                });
            let (first, second) = probes(value, h, lo, hi, options.method);
            if first == value {
                return Ok(vec![F::zero(); m]);
            }
            probe.set_scalar(j, first);
            let f1 = evaluate(probe)?;
            assert_eq!(f1.len(), m, "finite-difference output length changed");
            let h1 = first - value;
            let mut col: Vec<_> =
                f1.iter().zip(&base).map(|(&f, &b)| (f - b) / h1).collect();
            if let Some(second) = second {
                probe.set_scalar(j, second);
                let f2 = evaluate(probe)?;
                assert_eq!(
                    f2.len(),
                    m,
                    "finite-difference output length changed"
                );
                let h2 = second - value;
                for i in 0..m {
                    let slope = (f2[i] - base[i]) / h2;
                    col[i] = col[i] + h1 / (h2 - h1) * (col[i] - slope);
                }
            }
            probe.set_scalar(j, value);
            Ok(col)
        },
    )?;
    Ok((m, columns))
}

impl<P, V, F> Gradient for BoundedFiniteDiff<P, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    F: Scalar + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, P::Error> {
        let (_, cols) = columns(
            x,
            self.options(false),
            Some((&self.lower, &self.upper)),
            |x| self.problem.cost(x).map(|f| vec![f]),
        )?;
        let mut result = x.clone();
        for (j, col) in cols.into_iter().enumerate() {
            result.set_scalar(j, col[0]);
        }
        Ok(result)
    }
}
impl<P, V, F> Jacobian for BoundedFiniteDiff<P, F>
where
    P: Residual<Param = V, Output = V> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F> + MaybeSync,
    F: Scalar + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    type Jacobian = V::Matrix;
    fn jacobian(&self, x: &V) -> Result<V::Matrix, P::Error> {
        let (m, cols) = columns(
            x,
            self.options(true),
            Some((&self.lower, &self.upper)),
            |x| {
                self.problem.residual(x).map(|r| {
                    (0..r.vec_len()).map(|i| r.get_scalar(i)).collect()
                })
            },
        )?;
        Ok(V::dense_from_fn(m, x.vec_len(), |i, j| cols[j][i]))
    }
}

pub(super) fn constraint_values<P, V, F>(
    p: &P,
    x: &V,
) -> Result<Vec<F>, P::Error>
where
    P: NonlinearConstraints<Param = V, Output = F>,
    V: VectorLen + VectorIndex<F>,
    F: Scalar,
{
    let mut out = Vec::new();
    if p.num_nonlinear_equalities() > 0 {
        let values = p
            .nonlinear_equalities(x)?
            .expect("declared nonlinear equalities must return a vector");
        assert_eq!(
            values.vec_len(),
            p.num_nonlinear_equalities(),
            "nonlinear equality length mismatch"
        );
        out.extend((0..values.vec_len()).map(|i| values.get_scalar(i)));
    }
    if p.num_nonlinear_constraints() > 0 {
        let values = p.nonlinear_constraints(x)?;
        assert_eq!(
            values.vec_len(),
            p.num_nonlinear_constraints(),
            "nonlinear inequality length mismatch"
        );
        out.extend((0..values.vec_len()).map(|i| values.get_scalar(i)));
    }
    Ok(out)
}
impl<P, V, F> ConstraintJacobian for BoundedFiniteDiff<P, F>
where
    P: NonlinearConstraints<Param = V, Output = F> + MaybeSync,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + DenseMatrixFromFn<F, Matrix = P::Matrix>
        + MaybeSync,
    F: Scalar + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    fn constraint_jacobian(&self, x: &V) -> Result<P::Matrix, P::Error> {
        let (m, cols) = columns(
            x,
            self.options(true),
            Some((&self.lower, &self.upper)),
            |x| constraint_values(&self.problem, x),
        )?;
        Ok(V::dense_from_fn(m, x.vec_len(), |i, j| cols[j][i]))
    }
}
