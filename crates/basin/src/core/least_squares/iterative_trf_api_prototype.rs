//! Operator-only TRF API experiment. Nothing in this module is exported.
//!
//! This is a model consumer, not an LSMR implementation or a new solver.
//! See TODO.md's iterative-TRF API prototype for the proposed public contract.

use super::*;
use crate::core::constraint::evaluation::BoxProblem;
use crate::{BoxBounds, Constraints, EvalCounts, Problem, ProblemConstraints};
use std::{cell::RefCell, rc::Rc};

/// Independent capabilities: neither requires a Jacobian associated type.
/// Products are derivatives of the raw residual at x, pure and linear in v.
/// Parameter and residual spaces may have different types and lengths.
trait JacobianProduct: Residual {
    fn jacobian_product(
        &self,
        x: &Self::Param,
        v: &Self::Param,
    ) -> Result<Self::Output, Self::Error>;
}
trait TransposeJacobianProduct: Residual {
    fn transpose_jacobian_product(
        &self,
        x: &Self::Param,
        v: &Self::Output,
    ) -> Result<Self::Param, Self::Error>;
}

/// Models adding two fields to the non-exhaustive production EvalCounts.
/// Production must put these fields in EvalCounts itself, so states, budgets,
/// checkpoints, serialization, and inner roll-ups retain one source of truth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OperatorCounts {
    existing: EvalCounts,
    jacobian_product_evals: u64,
    transpose_jacobian_product_evals: u64,
}
impl OperatorCounts {
    fn total_work(&self) -> u64 {
        self.existing.total_work()
            + self.jacobian_product_evals
            + self.transpose_jacobian_product_evals
    }
    fn delta_since(&self, base: &Self) -> Self {
        Self {
            existing: self.existing.delta_since(&base.existing),
            jacobian_product_evals: self.jacobian_product_evals
                - base.jacobian_product_evals,
            transpose_jacobian_product_evals: self
                .transpose_jacobian_product_evals
                - base.transpose_jacobian_product_evals,
        }
    }
    fn add(&mut self, other: &Self) {
        self.existing.add(&other.existing);
        self.jacobian_product_evals += other.jacobian_product_evals;
        self.transpose_jacobian_product_evals +=
            other.transpose_jacobian_product_evals;
    }
}

struct OperatorProblem<P> {
    problem: Problem<P>,
    products: OperatorCounts,
}
impl<P> OperatorProblem<P> {
    fn new(problem: P) -> Self {
        Self {
            problem: Problem::new(problem),
            products: OperatorCounts::default(),
        }
    }
    fn counts(&self) -> OperatorCounts {
        OperatorCounts {
            existing: *self.problem.counts(),
            ..self.products
        }
    }
}

/// Separate raw and robust implementations mirror the existing dense
/// Evaluation capability, without making RobustLeastSquares a Residual.
trait OperatorObjective<F: Scalar> {
    type Raw: JacobianProduct + TransposeJacobianProduct;
    fn raw(&self) -> &Self::Raw;
    fn cost(&self, r: &<Self::Raw as Residual>::Output) -> F;
    fn model(
        &self,
        r: &<Self::Raw as Residual>::Output,
    ) -> Option<(<Self::Raw as Residual>::Output, Vec<F>)>;
}
impl<P, F> OperatorObjective<F> for P
where
    P: JacobianProduct + TransposeJacobianProduct,
    P::Output: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
{
    type Raw = P;
    fn raw(&self) -> &P {
        self
    }
    fn cost(&self, r: &P::Output) -> F {
        (0..r.vec_len())
            .map(|i| num::<F>(0.5) * r.get_scalar(i) * r.get_scalar(i))
            .sum()
    }
    fn model(&self, r: &P::Output) -> Option<(P::Output, Vec<F>)> {
        finite(r).then(|| (r.clone(), vec![F::one(); r.vec_len()]))
    }
}
impl<P, L, F> OperatorObjective<F> for RobustLeastSquares<P, L, F>
where
    P: JacobianProduct + TransposeJacobianProduct,
    P::Output: Clone + VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
    F: Scalar,
{
    type Raw = P;
    fn raw(&self) -> &P {
        self.inner()
    }
    fn cost(&self, r: &P::Output) -> F {
        self.residual_cost(r)
    }
    fn model(&self, r: &P::Output) -> Option<(P::Output, Vec<F>)> {
        self.model_residual(r)
    }
}

type Param<P, F> = <<P as OperatorObjective<F>>::Raw as Residual>::Param;
type Residuals<P, F> = <<P as OperatorObjective<F>>::Raw as Residual>::Output;
type Error<P, F> = <<P as OperatorObjective<F>>::Raw as Residual>::Error;
type AugmentedOutput<P, F> = (Residuals<P, F>, Param<P, F>);

/// The borrow ties products to one objective and accepted point. Rejected
/// trial evaluations cannot replace that point or its frozen loss factors.
struct Linearization<'a, P: OperatorObjective<F>, F: Scalar = f64> {
    problem: &'a mut OperatorProblem<P>,
    x: Param<P, F>,
    residual: Residuals<P, F>,
    factors: Vec<F>,
    cost: F,
}
fn linearize<P, F>(
    problem: &mut OperatorProblem<P>,
    x: Param<P, F>,
) -> Result<Option<Linearization<'_, P, F>>, Error<P, F>>
where
    P: OperatorObjective<F>,
    P::Raw: BoxProblem<Param = Param<P, F>>,
    Param<P, F>: Clone + VectorLen + VectorIndex<F>,
    Residuals<P, F>: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
{
    // Constraint validation remains the future driver's responsibility. This
    // consumer only demonstrates the typed box capability and model boundary.
    let raw = problem.problem.inner().raw();
    assert_eq!(raw.lower().vec_len(), x.vec_len());
    assert_eq!(raw.upper().vec_len(), x.vec_len());
    problem.problem.counts_mut().residual_evals += 1;
    let r = problem.problem.inner().raw().residual(&x)?;
    let objective = problem.problem.inner();
    let cost = objective.cost(&r);
    let Some((residual, factors)) = objective.model(&r) else {
        return Ok(None);
    };
    if !cost.is_finite() {
        return Ok(None);
    }
    Ok(Some(Linearization {
        problem,
        x,
        residual,
        factors,
        cost,
    }))
}
impl<P, F> Linearization<'_, P, F>
where
    P: OperatorObjective<F>,
    Param<P, F>: Clone + VectorLen + VectorIndex<F>,
    Residuals<P, F>: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
{
    fn product(
        &mut self,
        v: &Param<P, F>,
    ) -> Result<Residuals<P, F>, Error<P, F>> {
        assert_eq!(v.vec_len(), self.x.vec_len());
        self.problem.products.jacobian_product_evals += 1;
        let mut out = self
            .problem
            .problem
            .inner()
            .raw()
            .jacobian_product(&self.x, v)?;
        assert_eq!(out.vec_len(), self.factors.len());
        multiply(&mut out, &self.factors);
        assert!(finite(&out), "invalid derivative product");
        Ok(out)
    }
    fn transpose(
        &mut self,
        v: &Residuals<P, F>,
    ) -> Result<Param<P, F>, Error<P, F>> {
        assert_eq!(v.vec_len(), self.factors.len());
        let mut weighted = v.clone();
        multiply(&mut weighted, &self.factors);
        self.problem.products.transpose_jacobian_product_evals += 1;
        let out = self
            .problem
            .problem
            .inner()
            .raw()
            .transpose_jacobian_product(&self.x, &weighted)?;
        assert_eq!(out.vec_len(), self.x.vec_len());
        assert!(finite(&out), "invalid derivative product");
        Ok(out)
    }
    fn gradient(&mut self) -> Result<Param<P, F>, Error<P, F>> {
        self.transpose(&self.residual.clone())
    }
    /// LSMR needs both actions of A = [diag(a) J diag(d); diag(sqrt(c))],
    /// not entries or column norms. Fixed coordinates use d = c = 0.
    fn augmented_product(
        &mut self,
        v: &Param<P, F>,
        d: &[F],
        c: &[F],
    ) -> Result<AugmentedOutput<P, F>, Error<P, F>> {
        let mut scaled = v.clone();
        multiply(&mut scaled, d);
        let upper = self.product(&scaled)?;
        let mut lower = v.clone();
        multiply(&mut lower, &c.iter().map(|v| v.sqrt()).collect::<Vec<_>>());
        Ok((upper, lower))
    }
    fn augmented_transpose(
        &mut self,
        upper: &Residuals<P, F>,
        lower: &Param<P, F>,
        d: &[F],
        c: &[F],
    ) -> Result<Param<P, F>, Error<P, F>> {
        let mut out = self.transpose(upper)?;
        multiply(&mut out, d);
        assert_eq!(lower.vec_len(), out.vec_len());
        assert_eq!(c.len(), out.vec_len());
        for (i, ci) in c.iter().enumerate() {
            out.set_scalar(
                i,
                out.get_scalar(i) + ci.sqrt() * lower.get_scalar(i),
            );
        }
        Ok(out)
    }
    fn trial_cost(&mut self, x: &Param<P, F>) -> Result<F, Error<P, F>> {
        self.problem.problem.counts_mut().residual_evals += 1;
        let r = self.problem.problem.inner().raw().residual(x)?;
        assert_eq!(r.vec_len(), self.factors.len());
        if !finite(&r) {
            return Ok(F::infinity());
        }
        Ok(self.problem.problem.inner().cost(&r))
    }
}
fn num<F: Scalar>(x: f64) -> F {
    F::from_f64(x).unwrap()
}
fn finite<V: VectorLen + VectorIndex<F>, F: Scalar>(x: &V) -> bool {
    (0..x.vec_len()).all(|i| x.get_scalar(i).is_finite())
}
fn multiply<V: VectorLen + VectorIndex<F>, F: Scalar>(
    x: &mut V,
    factors: &[F],
) {
    assert_eq!(x.vec_len(), factors.len());
    for (i, factor) in factors.iter().enumerate() {
        x.set_scalar(i, x.get_scalar(i) * *factor);
    }
}

// No CostFunction, Jacobian, matrix storage, MatrixIndex, or ScaleRowsInPlace.
// The nonlinear second residual makes stale-point products observable.
struct Fixture<V, R, F> {
    make_param: fn(&[F]) -> V,
    make_residual: fn(&[F]) -> R,
    lower: V,
    upper: V,
    trace: Rc<RefCell<Vec<(u8, F)>>>,
    fail: Option<u8>,
    invalid: bool,
}
impl<V, R, F: Scalar> Fixture<V, R, F> {
    fn new(make_param: fn(&[F]) -> V, make_residual: fn(&[F]) -> R) -> Self {
        Self {
            make_param,
            make_residual,
            lower: make_param(&[num(-5.0); 2]),
            upper: make_param(&[num(5.0); 2]),
            trace: Rc::default(),
            fail: None,
            invalid: false,
        }
    }
    fn request(&self, kind: u8, x: F) -> Result<(), &'static str> {
        self.trace.borrow_mut().push((kind, x));
        if self.fail == Some(kind) {
            Err("operator callback failed")
        } else {
            Ok(())
        }
    }
}
impl<V, R, F> ProblemConstraints for Fixture<V, R, F> {
    type Constraints = Constraints<BoxBounds<V>>;
    fn constraint_set(&self) -> Constraints<BoxBounds<&V>> {
        Constraints::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}
impl<V: VectorIndex<F>, R, F: Scalar> Residual for Fixture<V, R, F> {
    type Param = V;
    type Output = R;
    type Error = &'static str;
    fn residual(&self, x: &V) -> Result<R, Self::Error> {
        let x0 = x.get_scalar(0);
        let x1 = x.get_scalar(1);
        self.request(0, x0)?;
        Ok((self.make_residual)(&[
            if self.invalid || x0 > num(4.0) {
                F::infinity()
            } else {
                x0 + num::<F>(2.0) * x1 - F::one()
            },
            x0 * x0 + x1 - num(2.0),
            num::<F>(4.0) * x1 + num(3.0),
        ]))
    }
}
impl<V: VectorIndex<F>, R, F: Scalar> JacobianProduct for Fixture<V, R, F> {
    fn jacobian_product(&self, x: &V, v: &V) -> Result<R, Self::Error> {
        self.request(1, x.get_scalar(0))?;
        Ok((self.make_residual)(&[
            if self.invalid {
                F::nan()
            } else {
                v.get_scalar(0) + num::<F>(2.0) * v.get_scalar(1)
            },
            num::<F>(2.0) * x.get_scalar(0) * v.get_scalar(0) + v.get_scalar(1),
            num::<F>(4.0) * v.get_scalar(1),
        ]))
    }
}
impl<V: VectorIndex<F>, R: VectorIndex<F>, F: Scalar> TransposeJacobianProduct
    for Fixture<V, R, F>
{
    fn transpose_jacobian_product(
        &self,
        x: &V,
        v: &R,
    ) -> Result<V, Self::Error> {
        self.request(2, x.get_scalar(0))?;
        Ok((self.make_param)(&[
            v.get_scalar(0) + num::<F>(2.0) * x.get_scalar(0) * v.get_scalar(1),
            num::<F>(2.0) * v.get_scalar(0)
                + v.get_scalar(1)
                + num::<F>(4.0) * v.get_scalar(2),
        ]))
    }
}

fn near<F: Scalar>(actual: F, expected: F) {
    assert!(
        (actual - expected).abs()
            <= num::<F>(300.0) * F::epsilon() * (F::one() + expected.abs()),
        "{actual:?} != {expected:?}"
    );
}
fn dot<V: VectorLen + VectorIndex<F>, F: Scalar>(a: &V, b: &V) -> F {
    assert_eq!(a.vec_len(), b.vec_len());
    (0..a.vec_len())
        .map(|i| a.get_scalar(i) * b.get_scalar(i))
        .sum()
}
fn cases<V, R, F>(make_param: fn(&[F]) -> V, make_residual: fn(&[F]) -> R)
where
    V: Clone + VectorLen + VectorIndex<F>,
    R: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
{
    let x = make_param(&[F::one(), num(-0.5)]);
    let v = make_param(&[num(0.25), -F::one()]);
    let u = make_residual(&[num(2.0), num(-3.0), F::one()]);
    let mut problem =
        OperatorProblem::new(Fixture::new(make_param, make_residual));
    let mut model =
        linearize::<_, F>(&mut problem, x.clone()).unwrap().unwrap();
    near(model.cost, num(2.125));
    let jv = model.product(&v).unwrap();
    let jtu = model.transpose(&u).unwrap();
    near(dot(&jv, &u), dot(&v, &jtu));
    let d = [num(2.0), num(0.5)];
    let c = [num(3.0), num(4.0)];
    let (upper, lower) = model.augmented_product(&v, &d, &c).unwrap();
    let normal = model.augmented_transpose(&upper, &lower, &d, &c).unwrap();
    near(normal.get_scalar(0), num(1.75));
    near(normal.get_scalar(1), num(-8.25));
    let d = [num(2.0), F::zero()];
    let c = [num(3.0), F::zero()];
    let (upper, lower) = model.augmented_product(&v, &d, &c).unwrap();
    let normal = model.augmented_transpose(&upper, &lower, &d, &c).unwrap();
    near(normal.get_scalar(0), num(5.75));
    assert_eq!(normal.get_scalar(1), F::zero());
    let counts = model.problem.counts();
    assert_eq!(counts.existing.residual_evals, 1);
    assert_eq!(counts.existing.jacobian_evals, 0);
    assert_eq!(counts.jacobian_product_evals, 3);
    assert_eq!(counts.transpose_jacobian_product_evals, 3);
    assert_eq!(counts.total_work(), 7);

    for loss in [0, 1, 2, 3, 4] {
        match loss {
            0 => robust_case(make_param, make_residual, SquaredLoss),
            1 => robust_case(make_param, make_residual, SoftL1Loss),
            2 => robust_case(make_param, make_residual, HuberLoss),
            3 => robust_case(make_param, make_residual, CauchyLoss),
            _ => robust_case(make_param, make_residual, ArctanLoss),
        }
    }
}
fn robust_case<V, R, F, L>(
    make_param: fn(&[F]) -> V,
    make_residual: fn(&[F]) -> R,
    loss: L,
) where
    V: Clone + VectorLen + VectorIndex<F>,
    R: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
    L: LossFunction<F>,
{
    let fixture = Fixture::new(make_param, make_residual);
    let trace = fixture.trace.clone();
    let objective = RobustLeastSquares::new(fixture, loss).with_scale(num(0.7));
    let mut problem = OperatorProblem::new(objective);
    let x = make_param(&[F::one(), num(-0.5)]);
    let mut model =
        linearize::<_, F>(&mut problem, x.clone()).unwrap().unwrap();
    let raw_r = [num(-1.0), num(-1.5), F::one()];
    let terms = raw_r.map(|r| {
        model
            .problem
            .problem
            .inner()
            .loss()
            .evaluate_scaled(r, num(0.7))
    });
    near(model.cost, terms.iter().map(|t| t.value).sum());
    let gradient = model.gradient().unwrap();
    near(
        gradient.get_scalar(0),
        terms[0].first_derivative + num::<F>(2.0) * terms[1].first_derivative,
    );
    near(
        gradient.get_scalar(1),
        num::<F>(2.0) * terms[0].first_derivative
            + terms[1].first_derivative
            + num::<F>(4.0) * terms[2].first_derivative,
    );
    let v = make_param(&[num(0.25), -F::one()]);
    let jv = model.product(&v).unwrap();
    let normal = model.transpose(&jv).unwrap();
    let q = [num(-1.75), num(-0.5), num(-4.0)];
    let weighted: Vec<_> = terms
        .iter()
        .zip(q)
        .map(|(t, v)| t.second_derivative.max(F::epsilon()) * v)
        .collect();
    near(
        normal.get_scalar(0),
        weighted[0] + num::<F>(2.0) * weighted[1],
    );
    near(
        normal.get_scalar(1),
        num::<F>(2.0) * weighted[0] + weighted[1] + num::<F>(4.0) * weighted[2],
    );
    let before = model.problem.counts();
    assert!(
        model
            .trial_cost(&make_param(&[num(5.0), F::zero()]))
            .unwrap()
            .is_infinite()
    );
    let trial = make_param(&[num(2.0), F::zero()]);
    let trial_cost = model.trial_cost(&trial).unwrap();
    let expected: F = [F::one(), num(2.0), num(3.0)]
        .iter()
        .map(|r| {
            model
                .problem
                .problem
                .inner()
                .loss()
                .evaluate_scaled(*r, num(0.7))
                .value
        })
        .sum();
    near(trial_cost, expected);
    let repeated = model.gradient().unwrap();
    near(repeated.get_scalar(0), gradient.get_scalar(0));
    near(repeated.get_scalar(1), gradient.get_scalar(1));
    let delta = model.problem.counts().delta_since(&before);
    assert_eq!(delta.existing.residual_evals, 2);
    assert_eq!(delta.transpose_jacobian_product_evals, 1);
    assert_eq!(delta.total_work(), 3);
    assert!(
        trace
            .borrow()
            .iter()
            .filter(|(kind, _)| *kind != 0)
            .all(|(_, x)| *x == F::one())
    );
    let mut rollup = before;
    rollup.add(&delta);
    assert_eq!(rollup, model.problem.counts());
    let counts = model.problem.counts();
    assert_eq!(counts.existing.cost_evals, 0);
    assert_eq!(counts.existing.jacobian_evals, 0);
    // A newly accepted point rebuilds loss factors and uses its own products.
    let mut next = linearize::<_, F>(&mut problem, trial).unwrap().unwrap();
    near(next.cost, expected);
    let terms = [F::one(), num(2.0), num(3.0)].map(|r| {
        next.problem
            .problem
            .inner()
            .loss()
            .evaluate_scaled(r, num(0.7))
    });
    let gradient = next.gradient().unwrap();
    near(
        gradient.get_scalar(0),
        terms[0].first_derivative + num::<F>(4.0) * terms[1].first_derivative,
    );
    near(
        gradient.get_scalar(1),
        num::<F>(2.0) * terms[0].first_derivative
            + terms[1].first_derivative
            + num::<F>(4.0) * terms[2].first_derivative,
    );
    assert_eq!(trace.borrow().last().unwrap(), &(2, num(2.0)));
}

#[test]
fn operator_vec_both_precisions() {
    cases::<_, _, f32>(|x| x.to_vec(), |x| x.to_vec());
    cases::<_, _, f64>(|x| x.to_vec(), |x| x.to_vec());
}

// Different Rust types prove that Param and Output are not conflated.
#[derive(Clone)]
struct Parameters(Vec<f64>);
impl VectorLen for Parameters {
    fn vec_len(&self) -> usize {
        self.0.len()
    }
}
impl VectorIndex<f64> for Parameters {
    fn get_scalar(&self, i: usize) -> f64 {
        self.0[i]
    }
    fn set_scalar(&mut self, i: usize, v: f64) {
        self.0[i] = v;
    }
}
#[test]
fn distinct_parameter_and_residual_spaces() {
    cases::<_, _, f64>(|x| Parameters(x.to_vec()), |x| x.to_vec());
}

#[test]
fn failed_requests_stay_counted_and_invalid_models_fail() {
    for robust in [false, true] {
        for kind in 0..3 {
            let mut fixture =
                Fixture::new(|x: &[f64]| x.to_vec(), |x: &[f64]| x.to_vec());
            fixture.fail = Some(kind);
            fn check<
                P: OperatorObjective<f64, Raw = Fixture<Vec<f64>, Vec<f64>, f64>>,
            >(
                p: P,
                kind: u8,
            ) {
                let mut p = OperatorProblem::new(p);
                let result = linearize::<_, f64>(&mut p, vec![1.0, -0.5]);
                if kind == 0 {
                    assert!(matches!(result, Err("operator callback failed")));
                } else {
                    let mut model = result.unwrap().unwrap();
                    let error = if kind == 1 {
                        model.product(&vec![1.0, 0.0])
                    } else {
                        model.transpose(&vec![1.0; 3])
                    };
                    assert_eq!(error.unwrap_err(), "operator callback failed");
                }
                let counts = p.counts();
                assert_eq!(counts.existing.residual_evals, 1);
                assert_eq!(counts.jacobian_product_evals, u64::from(kind == 1));
                assert_eq!(
                    counts.transpose_jacobian_product_evals,
                    u64::from(kind == 2)
                );
                assert_eq!(counts.total_work(), if kind == 0 { 1 } else { 2 });
            }
            if robust {
                check(RobustLeastSquares::new(fixture, SoftL1Loss), kind);
            } else {
                check(fixture, kind);
            }
        }
    }
    let mut fixture =
        Fixture::new(|x: &[f64]| x.to_vec(), |x: &[f64]| x.to_vec());
    fixture.invalid = true;
    let mut p = OperatorProblem::new(fixture);
    assert!(
        linearize::<_, f64>(&mut p, vec![1.0, -0.5])
            .unwrap()
            .is_none()
    );
    let mut p = OperatorProblem::new(RobustLeastSquares::new(
        p.problem.into_inner(),
        CauchyLoss,
    ));
    assert!(
        linearize::<_, f64>(&mut p, vec![1.0, -0.5])
            .unwrap()
            .is_none()
    );
}

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
macro_rules! backend_cases {
    ($name:ident, $make:expr) => {
        #[test]
        fn $name() {
            cases::<_, _, f32>($make, $make);
            cases::<_, _, f64>($make, $make);
        }
    };
}
#[cfg(feature = "nalgebra_v0_35")]
backend_cases!(operator_nalgebra_035, |x| {
    nalgebra::DVector::from_column_slice(x)
});
#[cfg(feature = "nalgebra_v0_34")]
backend_cases!(operator_nalgebra_034, |x| {
    nalgebra_0_34::DVector::from_column_slice(x)
});
#[cfg(feature = "nalgebra_v0_33")]
backend_cases!(operator_nalgebra_033, |x| {
    nalgebra_0_33::DVector::from_column_slice(x)
});
#[cfg(feature = "nalgebra_v0_32")]
backend_cases!(operator_nalgebra_032, |x| {
    nalgebra_0_32::DVector::from_column_slice(x)
});
#[cfg(feature = "ndarray_v0_17")]
backend_cases!(operator_ndarray_017, |x| ndarray::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "ndarray_v0_16")]
backend_cases!(operator_ndarray_016, |x| ndarray_0_16::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "ndarray_v0_15")]
backend_cases!(operator_ndarray_015, |x| ndarray_0_15::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "faer_v0_24")]
backend_cases!(operator_faer_024, |x| faer::Col::from_fn(x.len(), |i| x[i]));
#[cfg(feature = "faer_v0_23")]
backend_cases!(operator_faer_023, |x| faer_0_23::Col::from_fn(
    x.len(),
    |i| x[i]
));
#[cfg(feature = "faer_v0_22")]
backend_cases!(operator_faer_022, |x| faer_0_22::Col::from_fn(
    x.len(),
    |i| x[i]
));

#[test]
fn shape_and_derivative_guards_preserve_attempted_counts() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let fixture = Fixture::new(|x: &[f64]| x.to_vec(), |x: &[f64]| x.to_vec());
    let mut problem = OperatorProblem::new(fixture);
    let mut model = linearize::<_, f64>(&mut problem, vec![1.0, -0.5])
        .unwrap()
        .unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.product(&vec![1.0]))).is_err()
    );
    assert_eq!(model.problem.counts().jacobian_product_evals, 0);
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.transpose(&vec![1.0]))).is_err()
    );
    assert_eq!(model.problem.counts().transpose_jacobian_product_evals, 0);
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.product(&vec![f64::NAN, 0.0])))
            .is_err()
    );
    assert_eq!(model.problem.counts().jacobian_product_evals, 1);
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.transpose(&vec![f64::NAN; 3])))
            .is_err()
    );
    assert_eq!(model.problem.counts().transpose_jacobian_product_evals, 1);
}

struct ProbingProducts(Fixture<Vec<f64>, Vec<f64>, f64>);
impl ProblemConstraints for ProbingProducts {
    type Constraints = Constraints<BoxBounds<Vec<f64>>>;
    fn constraint_set(&self) -> Constraints<BoxBounds<&Vec<f64>>> {
        self.0.constraint_set()
    }
}
impl Residual for ProbingProducts {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.0.residual(x)
    }
}
impl JacobianProduct for ProbingProducts {
    fn jacobian_product(
        &self,
        x: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Self::Error> {
        // Stand in for a directional-difference adapter's internal probes.
        self.0.residual(x)?;
        self.0.jacobian_product(x, v)
    }
}
impl TransposeJacobianProduct for ProbingProducts {
    fn transpose_jacobian_product(
        &self,
        x: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Self::Error> {
        self.0.residual(x)?;
        self.0.transpose_jacobian_product(x, v)
    }
}
#[test]
fn internal_probes_do_not_charge_boundary_residual_counts() {
    let fixture = Fixture::new(|x: &[f64]| x.to_vec(), |x: &[f64]| x.to_vec());
    let trace = fixture.trace.clone();
    let mut problem = OperatorProblem::new(RobustLeastSquares::new(
        ProbingProducts(fixture),
        SoftL1Loss,
    ));
    let mut model = linearize::<_, f64>(&mut problem, vec![1.0, -0.5])
        .unwrap()
        .unwrap();
    model.product(&vec![1.0, 0.0]).unwrap();
    model.gradient().unwrap();
    assert_eq!(model.problem.counts().existing.residual_evals, 1);
    assert_eq!(model.problem.counts().total_work(), 3);
    assert_eq!(
        trace.borrow().iter().filter(|(kind, _)| *kind == 0).count(),
        3
    );
}

struct DenseOnly(Fixture<Vec<f64>, Vec<f64>, f64>);
impl ProblemConstraints for DenseOnly {
    type Constraints = Constraints<BoxBounds<Vec<f64>>>;
    fn constraint_set(&self) -> Constraints<BoxBounds<&Vec<f64>>> {
        self.0.constraint_set()
    }
}
impl Residual for DenseOnly {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.0.residual(x)
    }
}
impl Jacobian for DenseOnly {
    type Jacobian = crate::DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<Self::Jacobian, Self::Error> {
        Ok(crate::DenseMatrix::from_row_slice(
            3,
            2,
            &[1.0, 2.0, 2.0 * x[0], 1.0, 0.0, 4.0],
        ))
    }
}
#[test]
fn dense_trf_still_accepts_problems_without_product_capabilities() {
    use crate::{Executor, TrustRegionReflective};
    let fixture = || DenseOnly(Fixture::new(|x| x.to_vec(), |x| x.to_vec()));
    let result = Executor::from_start(
        fixture(),
        TrustRegionReflective::new(),
        vec![1.0, -0.5],
    )
    .max_iter(0)
    .run()
    .unwrap();
    near(result.cost(), 2.125);
    let result = Executor::from_start(
        RobustLeastSquares::new(fixture(), SoftL1Loss).with_scale(0.7),
        TrustRegionReflective::new(),
        vec![1.0, -0.5],
    )
    .max_iter(0)
    .run()
    .unwrap();
    let expected: f64 = [-1.0, -1.5, 1.0]
        .iter()
        .map(|r| SoftL1Loss.evaluate_scaled(*r, 0.7).value)
        .sum();
    near(result.cost(), expected);
}

#[test]
fn capabilities_do_not_imply_matrix_or_residual_adapter_traits() {
    // Ambiguous inference makes these checks fail if the forbidden impl appears.
    macro_rules! assert_not {
        ($ty:ty, $bound:path) => {{
            trait AmbiguousIfImpl<Marker> {
                fn check() {}
            }
            impl<T> AmbiguousIfImpl<()> for T {}
            impl<T: $bound> AmbiguousIfImpl<u8> for T {}
            let _ = <$ty as AmbiguousIfImpl<_>>::check;
        }};
    }
    assert_not!(Fixture<Vec<f64>, Vec<f64>, f64>, Jacobian);
    assert_not!(DenseOnly, JacobianProduct);
    assert_not!(DenseOnly, TransposeJacobianProduct);
    assert_not!(
        RobustLeastSquares<Fixture<Vec<f64>, Vec<f64>, f64>, SoftL1Loss>,
        Residual
    );
}
