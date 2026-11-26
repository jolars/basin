//! Shared first-derivative probes with explicit domain bounds.

use super::{
    FiniteDiff, Method, ProbeEvaluation, gradient_from_probes_into,
    minpack_step,
};
use crate::SerialEvaluation;
use crate::core::math::{DenseMatrixFromFn, Scalar, VectorIndex, VectorLen};
use crate::core::problem::{CostFunction, Gradient, Jacobian, Residual};
use std::marker::PhantomData;

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
/// Probes run on the calling thread by default. With the `parallel` feature,
/// call `with_parallel_evaluation()` to evaluate independent probes on the
/// active Rayon pool.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for `F = f64` or `f32`. Jacobians use each vector's [`DenseMatrixFromFn`]
/// matrix type. For fixed-size nalgebra parameters `SVector<F, N>`, the
/// residual may be `SVector<F, M>` or `DVector<F>` and the Jacobian is an
/// `OMatrix<F, Dyn, Const<N>>`.
/// The constraint trait's `Matrix` must match the parameter backend's matrix.
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
pub struct BoundedFiniteDiff<P, F: Scalar = f64, Evaluation = SerialEvaluation>
{
    evaluation: PhantomData<fn() -> Evaluation>,
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
            evaluation: PhantomData,
            problem,
            lower,
            upper,
            gradient_method: Method::Central,
            jacobian_method: Method::Forward,
            precision: F::epsilon(),
            step: None,
        })
    }
    #[cfg(feature = "parallel")]
    /// Evaluate independent finite-difference probes on the active Rayon pool.
    ///
    /// The derivative traits require shareable probe inputs and sendable
    /// probe results and errors. Bounds and stencil settings carry over.
    pub fn with_parallel_evaluation(
        self,
    ) -> BoundedFiniteDiff<P, F, crate::ParallelEvaluation> {
        self.with_evaluation()
    }
}

impl<P, F: Scalar, Evaluation> BoundedFiniteDiff<P, F, Evaluation> {
    fn with_evaluation<NewEvaluation>(
        self,
    ) -> BoundedFiniteDiff<P, F, NewEvaluation> {
        BoundedFiniteDiff {
            evaluation: PhantomData,
            problem: self.problem,
            lower: self.lower,
            upper: self.upper,
            gradient_method: self.gradient_method,
            jacobian_method: self.jacobian_method,
            precision: self.precision,
            step: self.step,
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
impl<P, const SQUARE: bool, F: Scalar, J, Evaluation>
    FiniteDiff<P, SQUARE, F, J, Evaluation>
{
    /// Transfer first-derivative settings and the evaluation mode into a
    /// [`BoundedFiniteDiff`].
    /// The resulting adapter supports gradients and Jacobians only. Hessian
    /// settings have no effect on its first derivatives.
    pub fn with_bounds<V: VectorLen + VectorIndex<F>>(
        self,
        lower: V,
        upper: V,
    ) -> BoundedFiniteDiff<P, F, Evaluation> {
        let mut result = BoundedFiniteDiff::new(self.problem, lower, upper)
            .gradient_method(self.gradient_method)
            .jacobian_method(self.jacobian_method)
            .function_precision(self.function_precision);
        if let Some(step) = self.fixed_step {
            result = result.with_step(step);
        }
        result.with_evaluation()
    }
}

impl<P: CostFunction, F: Scalar, Evaluation> CostFunction
    for BoundedFiniteDiff<P, F, Evaluation>
{
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn cost(&self, x: &P::Param) -> Result<P::Output, P::Error> {
        self.problem.cost(x)
    }
}
impl<P: Residual, F: Scalar, Evaluation> Residual
    for BoundedFiniteDiff<P, F, Evaluation>
{
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
        self.method.first_derivative_scale(self.precision)
    }

    fn probes(&self, value: F, scale: F, lo: F, hi: F) -> (F, Option<F>) {
        let mut h = self.step.unwrap_or_else(|| {
            if self.minpack && matches!(self.method, Method::Forward) {
                minpack_step(value, scale)
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

type Bounds<'a, F> = (&'a [F], &'a [F]);

pub(super) struct ColumnProbes<'a, C, V, R, F> {
    inputs: &'a C,
    point: &'a V,
    base: &'a R,
    options: Options<F>,
    bounds: Option<Bounds<'a, F>>,
    scale: F,
}

type GradientProbes<'a, 'b, P, V, F> =
    (&'a V, (&'b P, Options<F>, Bounds<'b, F>, F, F));

pub(super) fn columns<Evaluation, V, F, E, C>(
    x: &V,
    options: Options<F>,
    bounds: Option<Bounds<'_, F>>,
    inputs: &C,
    evaluate: impl Fn(&C, &V) -> Result<Vec<F>, E> + Sync,
) -> Result<(usize, Vec<Vec<F>>), E>
where
    V: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
    for<'a> Evaluation:
        ProbeEvaluation<ColumnProbes<'a, C, V, Vec<F>, F>, Vec<F>, E>,
{
    validate_point(x, bounds);
    let base = evaluate(inputs, x)?;
    columns_with_base::<Evaluation, _, _, _, _, _>(
        x,
        options,
        bounds,
        &base,
        |_| Vec::new(),
        inputs,
        |inputs, x, out| {
            *out = evaluate(inputs, x)?;
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

fn columns_with_base<Evaluation, V, R, F, E, C>(
    x: &V,
    options: Options<F>,
    bounds: Option<Bounds<'_, F>>,
    base: &R,
    scratch: impl Fn(&R) -> R + Sync,
    inputs: &C,
    evaluate: impl Fn(&C, &V, &mut R) -> Result<(), E> + Sync,
) -> Result<(usize, Vec<Vec<F>>), E>
where
    V: Clone + VectorLen + VectorIndex<F>,
    R: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
    for<'a> Evaluation:
        ProbeEvaluation<ColumnProbes<'a, C, V, R, F>, Vec<F>, E>,
{
    let m = base.vec_len();
    let probes = ColumnProbes {
        inputs,
        point: x,
        base,
        scale: options.scale(),
        options,
        bounds,
    };
    let columns = Evaluation::try_map(
        x.vec_len(),
        &probes,
        |probes| (probes.point.clone(), scratch(probes.base)),
        |probes, (probe, residual), j| {
            let x = probes.point;
            let base = probes.base;
            let value = x.get_scalar(j);
            let (lo, hi) = probes
                .bounds
                .map_or((F::neg_infinity(), F::infinity()), |(lo, hi)| {
                    (lo[j], hi[j])
                });
            let (first, second) =
                probes.options.probes(value, probes.scale, lo, hi);
            if first == value {
                return Ok(vec![F::zero(); m]);
            }
            probe.set_scalar(j, first);
            evaluate(probes.inputs, probe, residual)?;
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
                evaluate(probes.inputs, probe, residual)?;
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

impl<P, V, F, Evaluation> BoundedFiniteDiff<P, F, Evaluation>
where
    P: CostFunction<Param = V, Output = F>,
    V: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
{
    fn gradient_with_base_into(
        &self,
        x: &V,
        cost: F,
        out: &mut V,
    ) -> Result<(), P::Error>
    where
        for<'a, 'b> Evaluation:
            ProbeEvaluation<GradientProbes<'a, 'b, P, V, F>, F, P::Error>,
    {
        let options = self.options(false);
        let scale = options.scale();
        gradient_from_probes_into::<Evaluation, _, _, _, _>(
            x,
            out,
            (
                &self.problem,
                options,
                (&self.lower[..], &self.upper[..]),
                scale,
                cost,
            ),
            |x, (problem, options, (lower, upper), scale, cost), probe, j| {
                let value = x.get_scalar(j);
                let (first, second) =
                    options.probes(value, *scale, lower[j], upper[j]);
                if first == value {
                    return Ok(F::zero());
                }
                probe.set_scalar(j, first);
                let first_cost = problem.cost(probe)?;
                let h1 = first - value;
                let mut derivative = (first_cost - *cost) / h1;
                if let Some(second) = second {
                    probe.set_scalar(j, second);
                    let second_cost = problem.cost(probe)?;
                    let h2 = second - value;
                    let slope = (second_cost - *cost) / h2;
                    derivative =
                        derivative + h1 / (h2 - h1) * (derivative - slope);
                }
                probe.set_scalar(j, value);
                Ok(derivative)
            },
        )
    }
}

impl<P, V, F, Evaluation> Gradient for BoundedFiniteDiff<P, F, Evaluation>
where
    P: CostFunction<Param = V, Output = F>,
    V: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
    for<'a, 'b> Evaluation:
        ProbeEvaluation<GradientProbes<'a, 'b, P, V, F>, F, P::Error>,
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
impl<P, V, F, Evaluation> Jacobian for BoundedFiniteDiff<P, F, Evaluation>
where
    P: Residual<Param = V>,
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F>,
    P::Output: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
    for<'a> Evaluation:
        ProbeEvaluation<ColumnProbes<'a, P, V, P::Output, F>, Vec<F>, P::Error>,
{
    type Jacobian = V::Matrix;
    fn jacobian(&self, x: &V) -> Result<V::Matrix, P::Error> {
        let mut out = V::dense_from_fn(0, x.vec_len(), |_, _| unreachable!());
        self.jacobian_into(x, &mut out)?;
        Ok(out)
    }
    fn jacobian_into(
        &self,
        x: &V,
        out: &mut V::Matrix,
    ) -> Result<(), P::Error> {
        validate_point(x, Some((&self.lower, &self.upper)));
        let res = super::residual_base(&self.problem, x)?;
        self.jacobian_with_base_into(x, &res, out)
    }
    fn residual_and_jacobian(
        &self,
        x: &V,
    ) -> Result<(P::Output, Self::Jacobian), P::Error> {
        validate_point(x, Some((&self.lower, &self.upper)));
        let res = super::residual_base(&self.problem, x)?;
        let mut jac = V::dense_from_fn(0, x.vec_len(), |_, _| unreachable!());
        self.jacobian_with_base_into(x, &res, &mut jac)?;
        Ok((res, jac))
    }
    fn residual_and_jacobian_into(
        &self,
        x: &V,
        res: &mut P::Output,
        jac: &mut Self::Jacobian,
    ) -> Result<(), P::Error> {
        validate_point(x, Some((&self.lower, &self.upper)));
        self.residual_into(x, res)?;
        self.jacobian_with_base_into(x, res, jac)
    }
}

impl<P, V, F, Evaluation> BoundedFiniteDiff<P, F, Evaluation>
where
    P: Residual<Param = V>,
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F>,
    P::Output: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
{
    fn jacobian_with_base_into(
        &self,
        x: &V,
        res: &P::Output,
        jac: &mut V::Matrix,
    ) -> Result<(), P::Error>
    where
        for<'a> Evaluation: ProbeEvaluation<
                ColumnProbes<'a, P, V, P::Output, F>,
                Vec<F>,
                P::Error,
            >,
    {
        let (m, cols) = columns_with_base::<Evaluation, _, _, _, _, _>(
            x,
            self.options(true),
            Some((&self.lower, &self.upper)),
            res,
            Clone::clone,
            &self.problem,
            |problem, x, out| problem.residual_into(x, out),
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

impl<P, F, Evaluation> crate::Constraints
    for BoundedFiniteDiff<P, F, Evaluation>
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

impl<P, F, Evaluation> crate::NonlinearEqualityValues
    for BoundedFiniteDiff<P, F, Evaluation>
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

impl<P, F, Evaluation> crate::NonlinearInequalityValues
    for BoundedFiniteDiff<P, F, Evaluation>
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

impl<P, V, F, Evaluation> crate::ConstraintJacobian
    for BoundedFiniteDiff<P, F, Evaluation>
where
    P: CostFunction<Param = V, Output = F>,
    P: crate::Constraints,
    P::Set: crate::core::constraint::evaluation::NonlinearSet<P, V, F>,
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F>,
    F: Scalar,
    for<'a> Evaluation: ProbeEvaluation<
            ColumnProbes<'a, P, V, Vec<F>, F>,
            Vec<F>,
            crate::ConstraintSolveError<P::Error>,
        >,
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
        let (_, cols) = columns::<Evaluation, _, _, _, _>(
            x,
            options,
            Some((&self.lower, &self.upper)),
            &self.problem,
            |problem, point| constraint_values(problem, point, expected),
        )?;
        Ok(V::dense_from_fn(m, x.vec_len(), |i, j| cols[j][i]))
    }
}
