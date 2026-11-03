use basin::{
    BoxBounds, ConstraintDescription, ConstraintSet, Constraints, CostFunction,
    Gradient, HessianProduct, ParameterScale, ParameterTransform, Scalar,
    VectorIndex, VectorLen,
};
use std::cell::Cell;

#[path = "support/backend_aliases.rs"]
mod backends;
struct Coupled<V, F> {
    lo: V,
    hi: V,
    scalar: std::marker::PhantomData<F>,
}
impl<V, F> Constraints for Coupled<V, F> {
    type Set = ConstraintSet<BoxBounds<V>>;
    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lo), Some(&self.hi)))
    }
}
impl<V: VectorLen + VectorIndex<F>, F: Scalar> CostFunction for Coupled<V, F> {
    type Param = V;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        Ok(x.get_scalar(0) * x.get_scalar(1)
            + x.get_scalar(0) * x.get_scalar(0))
    }
}
impl<V: Clone + VectorLen + VectorIndex<F>, F: Scalar> Gradient
    for Coupled<V, F>
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        let mut g = x.clone();
        g.set_scalar(
            0,
            x.get_scalar(1) + (F::one() + F::one()) * x.get_scalar(0),
        );
        g.set_scalar(1, x.get_scalar(0));
        Ok(g)
    }
}
impl<V: Clone + VectorLen + VectorIndex<F>, F: Scalar> HessianProduct
    for Coupled<V, F>
{
    fn hessian_product(&self, _: &V, v: &V) -> Result<V, Self::Error> {
        let mut out = v.clone();
        out.set_scalar(
            0,
            (F::one() + F::one()) * v.get_scalar(0) + v.get_scalar(1),
        );
        out.set_scalar(1, v.get_scalar(0));
        Ok(out)
    }
}
fn exercise<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
{
    let f = |v: f64| F::from_f64(v).unwrap();
    let p = ParameterTransform::new(
        Coupled {
            lo: make(&[f(1.0), f(-2.0)]),
            hi: make(&[f(4.0), f(3.0)]),
            scalar: std::marker::PhantomData,
        },
        vec![ParameterScale::Logarithmic, ParameterScale::Linear],
    )
    .unwrap();
    let x = make(&[f(0.4), f(0.6)]);
    let v = make(&[f(0.3), f(-0.7)]);
    let physical = p.decode(&x).unwrap();
    let encoded = p.encode(&physical).unwrap();
    let tol = F::epsilon().sqrt() * f(100.0);
    for i in 0..2 {
        assert!((encoded.get_scalar(i) - x.get_scalar(i)).abs() < tol);
    }
    let step = F::epsilon().powf(f(1.0 / 3.0));
    let hv = p.hessian_product(&x, &v).unwrap();
    let mut plus = x.clone();
    let mut minus = x.clone();
    for i in 0..2 {
        plus.set_scalar(i, x.get_scalar(i) + step * v.get_scalar(i));
        minus.set_scalar(i, x.get_scalar(i) - step * v.get_scalar(i));
    }
    let gp = p.gradient(&plus).unwrap();
    let gm = p.gradient(&minus).unwrap();
    for i in 0..2 {
        let finite = (gp.get_scalar(i) - gm.get_scalar(i)) / (f(2.0) * step);
        assert!((hv.get_scalar(i) - finite).abs() < tol * f(20.0));
    }
    let (cost, g) = p.cost_and_gradient(&x).unwrap();
    assert_eq!(cost, p.cost(&x).unwrap());
    let separate = p.gradient(&x).unwrap();
    for i in 0..2 {
        assert_eq!(g.get_scalar(i), separate.get_scalar(i));
    }
    assert!(p.cost(&make(&[f(-0.1), f(0.5)])).is_err());
    exercise_extreme_mappings(make);
}

struct LinearSlope<V, F> {
    origin: F,
    slope: F,
    marker: std::marker::PhantomData<V>,
}
impl<V: VectorIndex<F>, F: Scalar> CostFunction for LinearSlope<V, F> {
    type Param = V;
    type Output = F;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        Ok(self.slope * (x.get_scalar(0) - self.origin))
    }
}
impl<V: Clone + VectorIndex<F>, F: Scalar> Gradient for LinearSlope<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        let mut g = x.clone();
        g.set_scalar(0, self.slope);
        Ok(g)
    }
}
impl<V: Clone + VectorIndex<F>, F: Scalar> HessianProduct
    for LinearSlope<V, F>
{
    fn hessian_product(&self, _: &V, v: &V) -> Result<V, Self::Error> {
        let mut out = v.clone();
        out.set_scalar(0, F::zero());
        Ok(out)
    }
}

fn exercise_extreme_mappings<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
{
    let f = |v: f64| F::from_f64(v).unwrap();
    let lower = F::max_value() / f(4.0);
    let upper = lower + lower * (f(32.0) * F::epsilon());
    let transform = ParameterTransform::new(
        basin::Constrained::new(
            LinearSlope {
                origin: lower,
                slope: f(10.0),
                marker: std::marker::PhantomData,
            },
            ConstraintSet::new().with_bounds(BoxBounds::new(
                Some(make(&[lower])),
                Some(make(&[upper])),
            )),
        ),
        vec![ParameterScale::Logarithmic],
    )
    .unwrap();
    let point = make(&[f(0.5)]);
    let physical = transform.decode(&point).unwrap().get_scalar(0);
    let width = ((upper - lower) / lower).ln_1p();
    let expected_gradient = f(10.0) * (physical * width);
    let gradient = transform.gradient(&point).unwrap().get_scalar(0);
    let tolerance = f(100.0) * F::epsilon();
    assert!(gradient.is_finite());
    assert!(
        (gradient - expected_gradient).abs()
            <= tolerance * expected_gradient.abs()
    );
    let product = transform
        .hessian_product(&point, &make(&[f(0.25)]))
        .unwrap()
        .get_scalar(0);
    let expected_product = (expected_gradient * width) * f(0.25);
    assert!(product.is_finite());
    assert!(
        (product - expected_product).abs()
            <= tolerance * expected_product.abs()
    );

    let lower = F::min_positive_value() * F::epsilon();
    let upper = F::max_value();
    let transform = ParameterTransform::new(
        basin::Constrained::new(
            LinearSlope {
                origin: lower,
                slope: F::one(),
                marker: std::marker::PhantomData,
            },
            ConstraintSet::new().with_bounds(BoxBounds::new(
                Some(make(&[lower])),
                Some(make(&[upper])),
            )),
        ),
        vec![ParameterScale::Logarithmic],
    )
    .unwrap();
    for unit in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let physical = transform.decode(&make(&[f(unit)])).unwrap();
        assert!(physical.get_scalar(0).is_finite());
        assert!(
            physical.get_scalar(0) >= lower && physical.get_scalar(0) <= upper
        );
        let encoded = transform.encode(&physical).unwrap().get_scalar(0);
        assert!((encoded - f(unit)).abs() <= f(16.0) * F::epsilon());
        if unit == 0.0 {
            assert_eq!(physical.get_scalar(0), lower);
        }
        if unit == 1.0 {
            assert_eq!(physical.get_scalar(0), upper);
        }
    }
}
#[test]
fn transforms_vec_f64() {
    exercise::<Vec<f64>, f64>(|s| s.to_vec());
}
#[test]
fn transforms_vec_f32() {
    exercise::<Vec<f32>, f32>(|s| s.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn transforms_nalgebra() {
    exercise::<backends::nalgebra::DVector<f64>, f64>(|s| {
        backends::nalgebra::DVector::from_column_slice(s)
    });
    exercise::<backends::nalgebra::DVector<f32>, f32>(|s| {
        backends::nalgebra::DVector::from_column_slice(s)
    });
}
#[cfg(feature = "ndarray_all")]
#[test]
fn transforms_ndarray() {
    exercise::<backends::ndarray::Array1<f64>, f64>(|s| {
        backends::ndarray::Array1::from_vec(s.to_vec())
    });
    exercise::<backends::ndarray::Array1<f32>, f32>(|s| {
        backends::ndarray::Array1::from_vec(s.to_vec())
    });
}
#[cfg(feature = "faer_all")]
#[test]
fn transforms_faer() {
    exercise::<backends::faer::Col<f64>, f64>(|s| {
        backends::faer::Col::from_fn(s.len(), |i| s[i])
    });
    exercise::<backends::faer::Col<f32>, f32>(|s| {
        backends::faer::Col::from_fn(s.len(), |i| s[i])
    });
}

// Each callback has a distinct counter so fallback dispatch cannot hide a
// dropped fused or in-place override.
#[derive(Default)]
struct Callbacks {
    cost: Cell<usize>,
    gradient: Cell<usize>,
    gradient_into: Cell<usize>,
    fused: Cell<usize>,
    product_into: Cell<usize>,
    fail: bool,
    malformed: bool,
    forbid_callbacks: bool,
}

impl CostFunction for Callbacks {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        assert!(!self.forbid_callbacks);
        self.cost.set(self.cost.get() + 1);
        if self.fail {
            Err("physical cost")
        } else {
            Ok(x[0] * x[0])
        }
    }
}

impl Gradient for Callbacks {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        assert!(!self.forbid_callbacks);
        self.gradient.set(self.gradient.get() + 1);
        if self.fail {
            Err("physical gradient")
        } else {
            Ok(vec![2.0 * x[0]])
        }
    }

    fn gradient_into(
        &self,
        x: &Vec<f64>,
        out: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        assert!(!self.forbid_callbacks);
        self.gradient_into.set(self.gradient_into.get() + 1);
        if self.fail {
            return Err("physical gradient into");
        }
        *out = if self.malformed {
            vec![]
        } else {
            vec![2.0 * x[0]]
        };
        Ok(())
    }

    fn cost_and_gradient_into(
        &self,
        x: &Vec<f64>,
        out: &mut Vec<f64>,
    ) -> Result<f64, Self::Error> {
        assert!(!self.forbid_callbacks);
        self.fused.set(self.fused.get() + 1);
        if self.fail {
            return Err("physical fused");
        }
        *out = if self.malformed {
            vec![]
        } else {
            vec![2.0 * x[0]]
        };
        Ok(x[0] * x[0])
    }
}

impl HessianProduct for Callbacks {
    fn hessian_product(
        &self,
        x: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Self::Error> {
        let mut out = Vec::new();
        self.hessian_product_into(x, v, &mut out)?;
        Ok(out)
    }

    fn hessian_product_into(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
        out: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        assert!(!self.forbid_callbacks);
        self.product_into.set(self.product_into.get() + 1);
        if self.fail {
            return Err("physical product");
        }
        *out = if self.malformed {
            vec![]
        } else {
            vec![2.0 * v[0]]
        };
        Ok(())
    }
}

type TransformedCallbacks = ParameterTransform<
    basin::Constrained<Callbacks, ConstraintSet<BoxBounds<Vec<f64>>>>,
    Vec<f64>,
>;

fn transform(
    callbacks: Callbacks,
    scale: ParameterScale,
) -> TransformedCallbacks {
    ParameterTransform::new(
        basin::Constrained::new(
            callbacks,
            ConstraintSet::new()
                .with_bounds(BoxBounds::new(Some(vec![1.0]), Some(vec![4.0]))),
        ),
        vec![scale],
    )
    .unwrap()
}

#[test]
fn invalid_geometry_is_rejected_without_evaluation() {
    use basin::ParameterTransformError as E;
    for (lower, upper, scales, expected) in [
        (
            Some(vec![1.0]),
            None,
            vec![ParameterScale::Linear],
            E::MissingBounds,
        ),
        (
            None,
            Some(vec![4.0]),
            vec![ParameterScale::Linear],
            E::MissingBounds,
        ),
        (
            Some(vec![1.0]),
            Some(vec![4.0, 5.0]),
            vec![ParameterScale::Linear],
            E::Dimension,
        ),
        (Some(vec![1.0]), Some(vec![4.0]), vec![], E::Dimension),
        (
            Some(vec![1.0]),
            Some(vec![1.0]),
            vec![ParameterScale::Linear],
            E::Bounds,
        ),
        (
            Some(vec![4.0]),
            Some(vec![1.0]),
            vec![ParameterScale::Linear],
            E::Bounds,
        ),
        (
            Some(vec![f64::NAN]),
            Some(vec![4.0]),
            vec![ParameterScale::Linear],
            E::Bounds,
        ),
        (
            Some(vec![1.0]),
            Some(vec![f64::INFINITY]),
            vec![ParameterScale::Linear],
            E::Bounds,
        ),
        (
            Some(vec![-f64::MAX]),
            Some(vec![f64::MAX]),
            vec![ParameterScale::Linear],
            E::Bounds,
        ),
        (
            Some(vec![0.0]),
            Some(vec![4.0]),
            vec![ParameterScale::Logarithmic],
            E::NonPositiveLogarithm,
        ),
        (
            Some(vec![-1.0]),
            Some(vec![4.0]),
            vec![ParameterScale::Logarithmic],
            E::NonPositiveLogarithm,
        ),
    ] {
        let problem = basin::Constrained::new(
            Callbacks {
                forbid_callbacks: true,
                ..Callbacks::default()
            },
            ConstraintSet::new().with_bounds(BoxBounds::new(lower, upper)),
        );
        assert_eq!(
            ParameterTransform::new(problem, scales).err(),
            Some(expected)
        );
    }
}

#[test]
fn invalid_inputs_do_not_reach_physical_callbacks() {
    use basin::{
        ParameterEvaluationError::Invalid, ParameterTransformError as E,
    };
    let p = transform(Callbacks::default(), ParameterScale::Logarithmic);
    for (x, expected) in [
        (vec![], E::Dimension),
        (vec![0.5, 0.5], E::Dimension),
        (vec![-0.1], E::OutsideDomain),
        (vec![1.1], E::OutsideDomain),
        (vec![f64::NAN], E::OutsideDomain),
        (vec![f64::INFINITY], E::OutsideDomain),
    ] {
        assert_eq!(p.cost(&x), Err(Invalid(expected.clone())));
        assert_eq!(p.gradient(&x), Err(Invalid(expected.clone())));
        assert_eq!(p.cost_and_gradient(&x), Err(Invalid(expected.clone())));
        assert_eq!(p.hessian_product(&x, &vec![1.0]), Err(Invalid(expected)));
    }
    assert!(p.encode(&vec![0.0]).is_err());
    assert!(p.encode(&vec![f64::NAN]).is_err());
    assert!(p.hessian_product(&vec![0.5], &vec![]).is_err());
    assert!(p.hessian_product(&vec![0.5], &vec![f64::NAN]).is_err());
    let c = p.get_ref().inner();
    assert_eq!(
        [
            c.cost.get(),
            c.gradient.get(),
            c.gradient_into.get(),
            c.fused.get(),
            c.product_into.get()
        ],
        [0; 5]
    );
}

#[test]
fn fused_and_in_place_callbacks_preserve_chain_rules_and_counts() {
    for scale in [ParameterScale::Linear, ParameterScale::Logarithmic] {
        let p = transform(Callbacks::default(), scale);
        let x = vec![0.5];
        let physical = p.decode(&x).unwrap()[0];
        let jacobian = if scale == ParameterScale::Linear {
            3.0
        } else {
            physical * 4.0_f64.ln()
        };
        let mut out = vec![];
        p.gradient_into(&x, &mut out).unwrap();
        assert!((out[0] - 2.0 * physical * jacobian).abs() < 1e-12);
        let mut counted = basin::Problem::new(p);
        let cost = counted.cost_and_gradient_into(&x, &mut out).unwrap();
        assert_eq!(cost, physical * physical);
        assert!((out[0] - 2.0 * physical * jacobian).abs() < 1e-12);
        counted
            .hessian_product_into(&x, &vec![0.25], &mut out)
            .unwrap();
        let expected = if scale == ParameterScale::Linear {
            4.5
        } else {
            physical * physical * 4.0_f64.ln().powi(2)
        };
        assert!((out[0] - expected).abs() < 1e-12);
        let c = counted.inner().get_ref().inner();
        assert_eq!(c.cost.get(), 0);
        assert_eq!(c.gradient_into.get(), 1);
        assert_eq!(c.fused.get(), 1);
        assert_eq!(c.product_into.get(), 1);
        assert_eq!(
            c.gradient.get(),
            usize::from(scale == ParameterScale::Logarithmic)
        );
        assert_eq!(counted.counts().cost_evals, 1);
        assert_eq!(counted.counts().gradient_evals, 1);
        assert_eq!(counted.counts().hessian_product_evals, 1);
    }
}

#[test]
fn physical_errors_and_malformed_derivatives_remain_distinct() {
    use basin::{
        ParameterEvaluationError::{Evaluation, Invalid},
        ParameterTransformError as E,
    };
    let p = transform(
        Callbacks {
            fail: true,
            ..Callbacks::default()
        },
        ParameterScale::Linear,
    );
    assert_eq!(p.cost(&vec![0.5]), Err(Evaluation("physical cost")));
    assert_eq!(
        p.gradient(&vec![0.5]),
        Err(Evaluation("physical gradient into"))
    );
    assert_eq!(
        p.cost_and_gradient(&vec![0.5]),
        Err(Evaluation("physical fused"))
    );
    assert_eq!(
        p.hessian_product(&vec![0.5], &vec![1.0]),
        Err(Evaluation("physical product"))
    );
    let p = transform(
        Callbacks {
            malformed: true,
            ..Callbacks::default()
        },
        ParameterScale::Linear,
    );
    assert_eq!(p.gradient(&vec![0.5]), Err(Invalid(E::Dimension)));
    assert_eq!(p.cost_and_gradient(&vec![0.5]), Err(Invalid(E::Dimension)));
    assert_eq!(
        p.hessian_product(&vec![0.5], &vec![1.0]),
        Err(Invalid(E::Dimension))
    );
}
