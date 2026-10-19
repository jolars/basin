use std::convert::Infallible;

use basin::{
    ConvergenceEvidence, CostFunction, Dot, Executor, Gradient,
    GradientDescent, NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd,
    Termination, TerminationCode, VectorLen,
};

struct Linear<V, F> {
    gradient: V,
    offset: F,
}

impl<V, F> basin::ProblemConstraints for Linear<V, F> {
    type Constraints = basin::NoConstraints;

    fn constraint_set(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}

impl<V, F> CostFunction for Linear<V, F>
where
    V: Dot<F>,
    F: Scalar,
{
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        Ok(self.gradient.dot(x) + self.offset)
    }
}

impl<V, F> Gradient for Linear<V, F>
where
    V: Clone + Dot<F>,
    F: Scalar,
{
    type Gradient = V;

    fn gradient(&self, _: &V) -> Result<V, Infallible> {
        Ok(self.gradient.clone())
    }
}

fn assert_factored_gradient<F: Scalar>(termination: &Termination<F>) {
    let Termination::Converged(convergence) = termination else {
        panic!("expected gradient convergence");
    };
    let ConvergenceEvidence::FactoredUpperBound { value, bound, .. } =
        &convergence.criteria()[0].evidence
    else {
        panic!("expected gradient evidence outside the squared scalar range");
    };
    assert!(value.significand.is_finite() && value.significand > F::zero());
    assert!(bound.significand.is_finite() && bound.significand > F::zero());
    assert!(
        (value.exponent, value.significand)
            <= (bound.exponent, bound.significand)
    );
}

fn check_extremes<V, F>(make: impl Fn(&[F]) -> V, small: F, large: F)
where
    F: Scalar,
    V: Clone
        + Dot<F>
        + ScaledAdd<F>
        + NegInPlace
        + ScaleInPlace<F>
        + VectorLen
        + NormSquared<F>,
{
    let two = F::from_f64(2.).unwrap();
    let half = F::from_f64(0.5).unwrap();
    for magnitude in [small, large] {
        let vector = make(&[magnitude, magnitude]);
        let expected = magnitude * two.sqrt();
        assert!(
            (vector.norm_l2() / expected - F::one()).abs()
                <= F::from_f64(4.).unwrap() * F::epsilon()
        );
        assert!(
            vector.norm_squared() == F::zero()
                || vector.norm_squared().is_infinite()
        );

        for tolerance in [F::zero(), half * magnitude] {
            let solver = GradientDescent::new(F::one())
                .with_absolute_gradient_tolerance(tolerance);
            let result = Executor::from_start(
                Linear {
                    gradient: make(&[magnitude]),
                    offset: F::zero(),
                },
                solver,
                make(&[F::zero()]),
            )
            .max_iter(2)
            .run()
            .unwrap();
            assert_eq!(result.report.code(), TerminationCode::MaxIter);
        }
        for tolerance in [F::zero(), half, F::one()] {
            let solver = GradientDescent::new(F::one())
                .with_relative_gradient_tolerance(tolerance);
            let result = Executor::from_start(
                Linear {
                    gradient: make(&[magnitude]),
                    offset: F::zero(),
                },
                solver,
                make(&[F::zero()]),
            )
            .max_iter(2)
            .run()
            .unwrap();
            if tolerance == F::one() {
                assert_eq!(
                    result.report.code(),
                    TerminationCode::RelativeGradientTolerance
                );
                assert_factored_gradient(&result.report.termination);
            } else {
                assert_eq!(result.report.code(), TerminationCode::MaxIter);
            }
        }
        let solver = GradientDescent::new(F::one())
            .with_absolute_gradient_tolerance(two * magnitude);
        let result = Executor::from_start(
            Linear {
                gradient: make(&[magnitude]),
                offset: F::zero(),
            },
            solver,
            make(&[F::zero()]),
        )
        .max_iter(2)
        .run()
        .unwrap();
        assert_eq!(result.report.code(), TerminationCode::GradientTolerance);
        assert_factored_gradient(&result.report.termination);
    }

    for (absolute, relative) in [
        (Some(F::zero()), None),
        (Some(half * small), None),
        (None, Some(F::zero())),
        (None, Some(half)),
    ] {
        let solver = GradientDescent::new(small)
            .with_absolute_step_tolerance(absolute)
            .with_relative_step_tolerance(relative);
        let result = Executor::from_start(
            Linear {
                gradient: make(&[F::one()]),
                offset: F::zero(),
            },
            solver,
            make(&[F::zero()]),
        )
        .max_iter(2)
        .run()
        .unwrap();
        assert_eq!(result.report.code(), TerminationCode::MaxIter);
        assert_eq!(result.iter(), 2);
        assert_eq!(result.param().norm_l2(), two * small);
    }

    for (absolute, relative, expected) in [
        (Some(two * small), None, TerminationCode::ParamTolerance),
        (
            None,
            Some(F::one()),
            TerminationCode::RelativeParamTolerance,
        ),
    ] {
        let solver = GradientDescent::new(small)
            .with_absolute_step_tolerance(absolute)
            .with_relative_step_tolerance(relative);
        let result = Executor::from_start(
            Linear {
                gradient: make(&[F::one()]),
                offset: F::zero(),
            },
            solver,
            make(&[F::zero()]),
        )
        .max_iter(2)
        .run()
        .unwrap();
        assert_eq!(result.report.code(), expected);
        let Termination::Converged(convergence) = &result.report.termination
        else {
            panic!("expected step convergence");
        };
        let ConvergenceEvidence::UpperBound { value, .. } =
            convergence.criteria()[0].evidence
        else {
            panic!("the small step norm is representable");
        };
        assert_eq!(value, small);
    }

    for values in [
        &[F::infinity(), F::nan()][..],
        &[F::nan(), F::infinity()][..],
    ] {
        assert!(make(values).norm_l2().is_nan());
    }
    assert!(make(&[F::infinity()]).norm_l2().is_infinite());
    assert_eq!(make(&[]).norm_l2(), F::zero());
    assert!(make(&[F::min_positive_value() / two]).norm_l2() > F::zero());
}

#[test]
fn vec_f32() {
    check_extremes(|values| values.to_vec(), 1e-30_f32, 1e30_f32);
}

#[test]
fn vec_f64() {
    check_extremes(|values| values.to_vec(), 1e-200_f64, 1e200_f64);
}

macro_rules! nalgebra_tests {
    ($feature:literal, $backend:ident, $f32_test:ident, $f64_test:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $f32_test() {
            check_extremes(
                $backend::DVector::from_column_slice,
                1e-30_f32,
                1e30_f32,
            );
        }
        #[cfg(feature = $feature)]
        #[test]
        fn $f64_test() {
            check_extremes(
                $backend::DVector::from_column_slice,
                1e-200_f64,
                1e200_f64,
            );
        }
    };
}

macro_rules! ndarray_tests {
    ($feature:literal, $backend:ident, $f32_test:ident, $f64_test:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $f32_test() {
            check_extremes(
                |values| $backend::Array1::from_vec(values.to_vec()),
                1e-30_f32,
                1e30_f32,
            );
        }
        #[cfg(feature = $feature)]
        #[test]
        fn $f64_test() {
            check_extremes(
                |values| $backend::Array1::from_vec(values.to_vec()),
                1e-200_f64,
                1e200_f64,
            );
        }
    };
}

macro_rules! faer_tests {
    ($feature:literal, $backend:ident, $f32_test:ident, $f64_test:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $f32_test() {
            check_extremes(
                |values| $backend::Col::from_fn(values.len(), |i| values[i]),
                1e-30_f32,
                1e30_f32,
            );
        }
        #[cfg(feature = $feature)]
        #[test]
        fn $f64_test() {
            check_extremes(
                |values| $backend::Col::from_fn(values.len(), |i| values[i]),
                1e-200_f64,
                1e200_f64,
            );
        }
    };
}

nalgebra_tests!(
    "nalgebra_v0_32",
    nalgebra_0_32,
    nalgebra_032_f32,
    nalgebra_032_f64
);
nalgebra_tests!(
    "nalgebra_v0_33",
    nalgebra_0_33,
    nalgebra_033_f32,
    nalgebra_033_f64
);
nalgebra_tests!(
    "nalgebra_v0_34",
    nalgebra_0_34,
    nalgebra_034_f32,
    nalgebra_034_f64
);
nalgebra_tests!(
    "nalgebra_v0_35",
    nalgebra,
    nalgebra_035_f32,
    nalgebra_035_f64
);
ndarray_tests!(
    "ndarray_v0_15",
    ndarray_0_15,
    ndarray_015_f32,
    ndarray_015_f64
);
ndarray_tests!(
    "ndarray_v0_16",
    ndarray_0_16,
    ndarray_016_f32,
    ndarray_016_f64
);
ndarray_tests!("ndarray_v0_17", ndarray, ndarray_017_f32, ndarray_017_f64);
faer_tests!("faer_v0_22", faer_0_22, faer_022_f32, faer_022_f64);
faer_tests!("faer_v0_23", faer_0_23, faer_023_f32, faer_023_f64);
faer_tests!("faer_v0_24", faer, faer_024_f32, faer_024_f64);
