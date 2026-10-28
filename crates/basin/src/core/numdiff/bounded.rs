//! Shared first-derivative probes with explicit domain bounds.

use super::{FiniteDiff, Method, gradient_from_probes_into};
use crate::core::math::{DenseMatrixFromFn, Scalar, VectorIndex, VectorLen};
use crate::core::parallel::{MaybeSend, MaybeSync, try_map_range_with};
use crate::core::problem::{CostFunction, Gradient, Jacobian, Residual};

/// Finite-difference first derivatives whose probes stay inside supplied bounds.
///
/// Bounds constrain differentiation probes; they do not replace the wrapped
/// problem's mathematical constraints. Pass its box bounds here when its
/// callbacks are defined only inside that box. The canonical mathematical constraint
/// set is forwarded unchanged. Negative-infinite lower and positive-infinite
/// upper endpoints denote unbounded sides. NaN and wrong-sign infinities are
/// invalid. Bounds must have matching lengths and be ordered; differentiation points must be finite and inside them.
///
/// Defaults match [`FiniteDiff`]: central gradients, forward Jacobians, and
/// machine-precision adaptive steps. Near a boundary, central differentiation
/// uses a second-order one-sided stencil. If only one distinct probe fits,
/// it uses a first-order difference. Fixed coordinates have zero derivative
/// columns, representing derivatives on the remaining free coordinates.
/// Actual representable displacements are used in denominators.
///
/// This adapter synthesizes [`Gradient`], [`Jacobian`], and
/// [`crate::ConstraintJacobian`], but does not advertise Hessian capabilities.
/// User errors propagate unchanged; non-finite values yield non-finite
/// derivatives. Counters and evaluation budgets follow [`crate::EvalCounts`]:
/// they charge derivative requests at the [`crate::Problem`] boundary and
/// exclude internal probes. Probe counts depend on the chosen stencils and
/// fixed coordinates. Fused cost/gradient calls reuse the returned cost as
/// the base value for either stencil, in both owned and `*_into` methods.
/// The optional `parallel` feature evaluates independent
/// columns concurrently without changing their arithmetic or result ordering.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for `F = f64` or `f32`. Jacobians use each vector's [`DenseMatrixFromFn`]
/// matrix type; the constraint trait's `Matrix` must match it.
///
/// # Solver stopping tolerances
///
/// As with [`FiniteDiff`], choose the solver's gradient stopping tolerance to
/// account for derivative error in the problem's units. Bounds keep probes
/// inside the domain; they do not eliminate truncation or rounding error.
/// In particular, a first-order stencil near a bound can be less accurate than
/// an interior central stencil. Native `f32` arithmetic generally needs a
/// larger tolerance than `f64`. See [`Lbfgs`](crate::Lbfgs) for an example.
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
    /// Panics for invalid bounds. Use [`Self::try_new`] for typed configuration errors.
    pub fn new<V: VectorLen + VectorIndex<F>>(
        problem: P,
        lower: V,
        upper: V,
    ) -> Self {
        Self::try_new(problem, lower, upper)
            .expect("valid finite-difference bounds")
    }
    /// Copy and validate differentiation bounds, preserving typed configuration errors.
    pub fn try_new<V: VectorLen + VectorIndex<F>>(
        problem: P,
        lower: V,
        upper: V,
    ) -> Result<Self, crate::ConstraintError> {
        crate::core::constraint::model::dimension(
            crate::ConstraintBlock::Bounds,
            crate::ConstraintAxis::Length,
            lower.vec_len(),
            upper.vec_len(),
        )?;
        let lower: Vec<_> =
            (0..lower.vec_len()).map(|i| lower.get_scalar(i)).collect();
        let upper: Vec<_> =
            (0..upper.vec_len()).map(|i| upper.get_scalar(i)).collect();
        for i in 0..lower.len() {
            crate::core::constraint::model::validate_bound(
                lower[i],
                crate::BoundSide::Lower,
                i,
            )?;
            crate::core::constraint::model::validate_bound(
                upper[i],
                crate::BoundSide::Upper,
                i,
            )?;
            if lower[i] > upper[i] {
                return Err(crate::ConstraintError::InvertedBounds {
                    coordinate: i,
                });
            }
        }
        Ok(Self {
            problem,
            lower,
            upper,
            gradient_method: Method::Central,
            jacobian_method: Method::Forward,
            precision: F::epsilon(),
            step: None,
        })
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
    fn residual_into(
        &self,
        x: &P::Param,
        out: &mut P::Output,
    ) -> Result<(), P::Error> {
        self.problem.residual_into(x, out)
    }
}

pub(super) struct Options<F> {
    pub method: Method,
    pub precision: F,
    pub step: Option<F>,
    pub minpack: bool,
}

impl<F: Scalar> Options<F> {
    fn scale(&self) -> F {
        let precision = self.precision.max(F::epsilon());
        if matches!(self.method, Method::Forward) {
            precision.sqrt()
        } else {
            precision.cbrt()
        }
    }

    fn probes(&self, value: F, scale: F, lo: F, hi: F) -> (F, Option<F>) {
        let mut h = self.step.unwrap_or_else(|| {
            if self.minpack && matches!(self.method, Method::Forward) {
                let h = scale * value.abs();
                if h == F::zero() { scale } else { h }
            } else {
                scale * value.abs().max(F::one())
            }
        });
        if value + h == value || value - h == value {
            h = scale * value.abs().max(F::one());
        }
        probes(value, h, lo, hi, self.method)
    }
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
    columns_into(
        x,
        options,
        bounds,
        Vec::new(),
        |_| Vec::new(),
        |x, out| {
            *out = evaluate(x)?;
            Ok(())
        },
    )
}

fn validate_point<V, F>(x: &V, bounds: Option<(&[F], &[F])>)
where
    V: VectorLen + VectorIndex<F>,
    F: Scalar,
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
}

fn columns_into<V, R, F, E, C>(
    x: &V,
    options: Options<F>,
    bounds: Option<(&[F], &[F])>,
    mut base: R,
    scratch: impl Fn(&R) -> R + MaybeSync,
    evaluate: C,
) -> Result<(usize, Vec<Vec<F>>), E>
where
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    R: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    F: Scalar + MaybeSync + MaybeSend,
    E: MaybeSend,
    C: Fn(&V, &mut R) -> Result<(), E> + MaybeSync,
{
    validate_point(x, bounds);
    evaluate(x, &mut base)?;
    columns_with_base(x, options, bounds, &base, scratch, evaluate)
}

fn columns_with_base<V, R, F, E, C>(
    x: &V,
    options: Options<F>,
    bounds: Option<(&[F], &[F])>,
    base: &R,
    scratch: impl Fn(&R) -> R + MaybeSync,
    evaluate: C,
) -> Result<(usize, Vec<Vec<F>>), E>
where
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    R: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    F: Scalar + MaybeSync + MaybeSend,
    E: MaybeSend,
    C: Fn(&V, &mut R) -> Result<(), E> + MaybeSync,
{
    let m = base.vec_len();
    let scale = options.scale();
    let evaluate = &evaluate;
    let columns = try_map_range_with(
        x.vec_len(),
        || (x.clone(), scratch(base)),
        |(probe, residual), j| {
            let value = x.get_scalar(j);
            let (lo, hi) = bounds
                .map_or((F::neg_infinity(), F::infinity()), |(lo, hi)| {
                    (lo[j], hi[j])
                });
            let (first, second) = options.probes(value, scale, lo, hi);
            if first == value {
                return Ok(vec![F::zero(); m]);
            }
            probe.set_scalar(j, first);
            evaluate(probe, residual)?;
            assert_eq!(
                residual.vec_len(),
                m,
                "finite-difference output length changed"
            );
            let h1 = first - value;
            let mut col: Vec<_> = (0..m)
                .map(|i| (residual.get_scalar(i) - base.get_scalar(i)) / h1)
                .collect();
            if let Some(second) = second {
                probe.set_scalar(j, second);
                evaluate(probe, residual)?;
                assert_eq!(
                    residual.vec_len(),
                    m,
                    "finite-difference output length changed"
                );
                let h2 = second - value;
                for (i, value) in col.iter_mut().enumerate() {
                    let slope =
                        (residual.get_scalar(i) - base.get_scalar(i)) / h2;
                    *value = *value + h1 / (h2 - h1) * (*value - slope);
                }
            }
            probe.set_scalar(j, value);
            Ok(col)
        },
    )?;
    Ok((m, columns))
}

impl<P, V, F> BoundedFiniteDiff<P, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
    F: Scalar + MaybeSync + MaybeSend,
    P::Error: MaybeSend,
{
    fn gradient_with_base_into(
        &self,
        x: &V,
        cost: F,
        out: &mut V,
    ) -> Result<(), P::Error> {
        let options = self.options(false);
        let scale = options.scale();
        gradient_from_probes_into(x, out, |probe, j| {
            let value = x.get_scalar(j);
            let (first, second) =
                options.probes(value, scale, self.lower[j], self.upper[j]);
            if first == value {
                return Ok(F::zero());
            }
            probe.set_scalar(j, first);
            let first_cost = self.problem.cost(probe)?;
            let h1 = first - value;
            let mut derivative = (first_cost - cost) / h1;
            if let Some(second) = second {
                probe.set_scalar(j, second);
                let second_cost = self.problem.cost(probe)?;
                let h2 = second - value;
                let slope = (second_cost - cost) / h2;
                derivative = derivative + h1 / (h2 - h1) * (derivative - slope);
            }
            probe.set_scalar(j, value);
            Ok(derivative)
        })
    }
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
        let mut out = x.clone();
        self.gradient_into(x, &mut out)?;
        Ok(out)
    }
    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), P::Error> {
        validate_point(x, Some((&self.lower, &self.upper)));
        let cost = self.cost(x)?;
        self.gradient_with_base_into(x, cost, out)
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(F, V), P::Error> {
        let mut grad = x.clone();
        let cost = self.cost_and_gradient_into(x, &mut grad)?;
        Ok((cost, grad))
    }
    fn cost_and_gradient_into(
        &self,
        x: &V,
        grad: &mut V,
    ) -> Result<F, P::Error> {
        let cost = self.cost(x)?;
        validate_point(x, Some((&self.lower, &self.upper)));
        self.gradient_with_base_into(x, cost, grad)?;
        Ok(cost)
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
        let mut out = V::dense_from_fn(0, 0, |_, _| unreachable!());
        self.jacobian_into(x, &mut out)?;
        Ok(out)
    }
    fn jacobian_into(
        &self,
        x: &V,
        out: &mut V::Matrix,
    ) -> Result<(), P::Error> {
        let (m, cols) = columns_into(
            x,
            self.options(true),
            Some((&self.lower, &self.upper)),
            x.clone(),
            Clone::clone,
            |x, out| self.problem.residual_into(x, out),
        )?;
        V::dense_from_fn_into(m, x.vec_len(), |i, j| cols[j][i], out);
        Ok(())
    }
    fn residual_and_jacobian(
        &self,
        x: &V,
    ) -> Result<(V, Self::Jacobian), P::Error> {
        let mut res = x.clone();
        let mut jac = V::dense_from_fn(0, 0, |_, _| unreachable!());
        self.residual_and_jacobian_into(x, &mut res, &mut jac)?;
        Ok((res, jac))
    }
    fn residual_and_jacobian_into(
        &self,
        x: &V,
        res: &mut V,
        jac: &mut Self::Jacobian,
    ) -> Result<(), P::Error> {
        let bounds = Some((&self.lower[..], &self.upper[..]));
        validate_point(x, bounds);
        self.residual_into(x, res)?;
        let (m, cols) = columns_with_base(
            x,
            self.options(true),
            bounds,
            res,
            Clone::clone,
            |x, out| self.problem.residual_into(x, out),
        )?;
        V::dense_from_fn_into(m, x.vec_len(), |i, j| cols[j][i], jac);
        Ok(())
    }
}

pub(super) fn constraint_values<P, V, F>(
    p: &P,
    x: &V,
    expected: (usize, usize),
) -> Result<Vec<F>, crate::ConstraintSolveError<P::Error>>
where
    P: CostFunction<Param = V, Output = F>,
    P: crate::Constraints,
    P::Set: crate::core::constraint::evaluation::NonlinearSet<P, V, F>,
    F: Scalar,
{
    use crate::core::constraint::evaluation::{NonlinearSet, invalid};
    if P::Set::counts(p.constraints()) != expected {
        return Err(invalid(crate::ConstraintError::LayoutChanged));
    }
    let mut counts = crate::EvalCounts::default();
    let (mut eq, ineq) = P::Set::values(p, x, &mut counts)?;
    eq.extend(ineq);
    Ok(eq)
}

impl<P, F> crate::Constraints for BoundedFiniteDiff<P, F>
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

impl<P, F> crate::NonlinearEqualityValues for BoundedFiniteDiff<P, F>
where
    P: crate::NonlinearEqualityValues,
    F: crate::Scalar,
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

impl<P, F> crate::NonlinearInequalityValues for BoundedFiniteDiff<P, F>
where
    P: crate::NonlinearInequalityValues,
    F: crate::Scalar,
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

impl<P, V, F> crate::ConstraintJacobian for BoundedFiniteDiff<P, F>
where
    P: CostFunction<Param = V, Output = F> + MaybeSync,
    P: crate::Constraints,
    P::Set: crate::core::constraint::evaluation::NonlinearSet<P, V, F>,
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F> + MaybeSync,
    P::Error: MaybeSend,
    F: Scalar + MaybeSend + MaybeSync,
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
        let options = self.options(true);
        let (_, cols) =
            columns(x, options, Some((&self.lower, &self.upper)), |point| {
                constraint_values(&self.problem, point, expected)
            })?;
        Ok(V::dense_from_fn(m, x.vec_len(), |i, j| cols[j][i]))
    }
}
