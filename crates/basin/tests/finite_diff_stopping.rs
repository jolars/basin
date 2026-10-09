//! Stopping thresholds must account for the error in a supplied gradient.

use std::convert::Infallible;

use basin::core::numdiff::Method;
use basin::solver::lbfgs::AsFloatSliceMut;
use basin::{
    BoundedFiniteDiff, BoxConstraints, CostFunction, Dot, Executor, FiniteDiff,
    FirstOrderState, Gradient, Lbfgs, Scalar, ScaledAdd, Termination,
    VectorIndex, VectorLen,
};

struct Quadratic<V, F: Scalar = f64> {
    lower: V,
    upper: V,
    multiplier: F,
}

impl<V: VectorIndex<F>, F: Scalar> CostFunction for Quadratic<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        let a = x.get_scalar(0) - F::one();
        let b = x.get_scalar(1) + F::from_f64(2.0).unwrap();
        Ok(self.multiplier * F::from_f64(0.5).unwrap() * (a * a + b * b))
    }
}

impl<V: VectorIndex<F>, F: Scalar> BoxConstraints for Quadratic<V, F> {
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}

#[test]
fn forward_bias_explains_strict_failure_and_explicit_tolerance_avoids_it() {
    let problem = || Quadratic {
        lower: vec![-8.0; 2],
        upper: vec![8.0; 2],
        multiplier: 1.0,
    };
    let forward = FiniteDiff::new(problem()).gradient_method(Method::Forward);
    let gradient = forward.gradient(&vec![1.0, -2.0]).unwrap();
    assert_eq!(gradient, vec![2.0_f64.powi(-27), 2.0_f64.powi(-26)]);
    assert!(gradient.iter().all(|g| *g > 1e-10));

    let strict = Executor::new(
        forward,
        Lbfgs::new().with_absolute_projected_gradient_tolerance(1e-10),
        FirstOrderState::new(vec![4.0, 3.0]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!(matches!(strict.report.termination, Termination::Failed(_)));
    assert_eq!(strict.param(), &vec![1.0, -2.0]);

    let configured = Executor::new(
        FiniteDiff::new(problem()).gradient_method(Method::Forward),
        Lbfgs::new().with_absolute_projected_gradient_tolerance(1e-7),
        FirstOrderState::new(vec![4.0, 3.0]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!(matches!(
        configured.report.termination,
        Termination::Converged(_)
    ));
    assert_eq!(configured.param(), &vec![1.0, -2.0]);
    assert_eq!(configured.cost(), 0.0);
    assert!(configured.counts.cost_evals < strict.counts.cost_evals);

    let central = Executor::new(
        FiniteDiff::new(problem()),
        Lbfgs::new().with_absolute_projected_gradient_tolerance(1e-10),
        FirstOrderState::new(vec![4.0, 3.0]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!(matches!(
        central.report.termination,
        Termination::Converged(_)
    ));
    assert_eq!(central.param(), &vec![1.0, -2.0]);
}

fn check<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar + Send + Sync,
    V: Clone
        + VectorIndex<F>
        + VectorLen
        + AsFloatSliceMut<F>
        + Dot<F>
        + ScaledAdd<F>
        + Send
        + Sync,
{
    let num = |x| F::from_f64(x).unwrap();
    let tolerance = if F::epsilon() > num(1e-10) {
        num(1e-3)
    } else {
        num(1e-7)
    };
    // Positive objective scaling multiplies both the gradient bias and its
    // threshold. These fixture settings make no scale-independent guarantee.
    for multiplier in [num(1e-8), F::one(), num(1e8)] {
        let lower = make(&[num(-8.0); 2]);
        let upper = make(&[num(8.0); 2]);
        let problem = Quadratic {
            lower: lower.clone(),
            upper: upper.clone(),
            multiplier,
        };
        let forward = BoundedFiniteDiff::new(problem, lower, upper)
            .gradient_method(Method::Forward);
        let witness = make(&[F::one(), num(-2.0)]);
        let gradient = forward.gradient(&witness).unwrap();
        let bias = gradient
            .get_scalar(0)
            .abs()
            .max(gradient.get_scalar(1).abs());
        assert!(bias > multiplier * num(1e-10));
        assert!(bias < multiplier * tolerance);
    }
    let lower = make(&[num(-8.0); 2]);
    let upper = make(&[num(8.0); 2]);
    let problem = Quadratic {
        lower: lower.clone(),
        upper: upper.clone(),
        multiplier: F::one(),
    };
    let forward = BoundedFiniteDiff::new(problem, lower, upper)
        .gradient_method(Method::Forward);
    let result = Executor::new(
        forward,
        Lbfgs::new().with_absolute_projected_gradient_tolerance(tolerance),
        FirstOrderState::new(make(&[num(4.0), num(3.0)])),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!(
        matches!(result.report.termination, Termination::Converged(_)),
        "{:?}",
        result.report
    );
    let point = result.param();
    let a = point.get_scalar(0) - F::one();
    let b = point.get_scalar(1) + num(2.0);
    assert!(a.abs().max(b.abs()) <= num(4.0) * tolerance);
    assert!(result.cost() <= num(16.0) * tolerance * tolerance);
}

macro_rules! backend {
    ($module:ident, $make:expr) => {
        mod $module {
            #[test]
            fn f64() {
                super::check::<_, f64>($make);
            }
            #[test]
            fn f32() {
                super::check::<_, f32>($make);
            }
        }
    };
}

backend!(vec, |x| x.to_vec());
#[cfg(feature = "nalgebra_v0_32")]
backend!(nalgebra_0_32, ::nalgebra_0_32::DVector::from_column_slice);
#[cfg(feature = "nalgebra_v0_33")]
backend!(nalgebra_0_33, ::nalgebra_0_33::DVector::from_column_slice);
#[cfg(feature = "nalgebra_v0_34")]
backend!(nalgebra_0_34, ::nalgebra_0_34::DVector::from_column_slice);
#[cfg(feature = "nalgebra_v0_35")]
backend!(nalgebra_0_35, ::nalgebra::DVector::from_column_slice);
#[cfg(feature = "ndarray_v0_15")]
backend!(ndarray_0_15, |x| ::ndarray_0_15::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "ndarray_v0_16")]
backend!(ndarray_0_16, |x| ::ndarray_0_16::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "ndarray_v0_17")]
backend!(ndarray_0_17, |x| ::ndarray::Array1::from_vec(x.to_vec()));
#[cfg(feature = "faer_v0_22")]
backend!(faer_0_22, |x| ::faer_0_22::Col::from_fn(x.len(), |i| x[i]));
#[cfg(feature = "faer_v0_23")]
backend!(faer_0_23, |x| ::faer_0_23::Col::from_fn(x.len(), |i| x[i]));
#[cfg(feature = "faer_v0_24")]
backend!(faer_0_24, |x| ::faer::Col::from_fn(x.len(), |i| x[i]));
