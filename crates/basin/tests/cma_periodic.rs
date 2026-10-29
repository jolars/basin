//! Periodic eigen refresh retains numerical and evaluation contracts.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

macro_rules! check {
    ($name:ident, $f:ty, $v:ty, $m:ty, $from:expr) => {
        #[test]
        fn $name() {
            struct Quadratic;
            impl basin::CostFunction for Quadratic {
                type Param = $v;
                type Output = $f;
                type Error = std::convert::Infallible;
                fn cost(&self, x: &$v) -> Result<$f, Self::Error> {
                    Ok((0..4)
                        .map(|i| x[i] * x[i] * (10.0 as $f).powi(i as i32))
                        .sum())
                }
            }
            let run = |period| {
                basin::Executor::new(
                    Quadratic,
                    basin::CmaEs::<$v, $m, $f>::new(42, 0.5)
                        .with_lambda(8)
                        .with_eigen_update_period(period)
                        .with_absolute_distribution_size_tolerance(1e-8),
                    basin::PopulationProgress::from_point(($from)(&[1.0; 4])),
                )
                .max_iter(400)
                .run_with_solver()
                .unwrap()
            };
            let a = run(3);
            let b = run(3);
            assert_eq!(a.state, b.state);
            assert_eq!(a.counts, b.counts);
            assert_eq!(a.counts.cost_evals, (a.iter() + 1) * 9);
            assert_eq!(a.report.code(), basin::TerminationCode::CmaEsTolerance);
            assert!(a.cost().is_finite() && a.cost() < 1e-8, "{}", a.cost());
            use basin::CostFunction;
            assert_eq!(a.cost(), Quadratic.cost(a.param()).unwrap());
        }
    };
}

check!(
    vec_f32,
    f32,
    Vec<f32>,
    basin::DenseMatrix<f32>,
    |x: &[f32]| x.to_vec()
);
check!(
    vec_f64,
    f64,
    Vec<f64>,
    basin::DenseMatrix<f64>,
    |x: &[f64]| x.to_vec()
);

#[cfg(feature = "nalgebra_all")]
mod nalgebra {
    use super::backend_aliases::nalgebra::{DMatrix, DVector};
    check!(
        f32,
        f32,
        DVector<f32>,
        DMatrix<f32>,
        DVector::from_column_slice
    );
    check!(
        f64,
        f64,
        DVector<f64>,
        DMatrix<f64>,
        DVector::from_column_slice
    );
}

#[cfg(feature = "ndarray_all")]
mod ndarray {
    use super::backend_aliases::ndarray::{Array1, Array2};
    check!(f32, f32, Array1<f32>, Array2<f32>, |x: &[f32]| {
        Array1::from_vec(x.to_vec())
    });
    check!(f64, f64, Array1<f64>, Array2<f64>, |x: &[f64]| {
        Array1::from_vec(x.to_vec())
    });
}

#[cfg(feature = "faer_all")]
mod faer {
    use super::backend_aliases::faer::{Col, Mat};
    check!(f32, f32, Col<f32>, Mat<f32>, |x: &[f32]| Col::from_fn(
        x.len(),
        |i| x[i]
    ));
    check!(f64, f64, Col<f64>, Mat<f64>, |x: &[f64]| Col::from_fn(
        x.len(),
        |i| x[i]
    ));
}

#[test]
#[should_panic(
    expected = "eigen update period must be at least one generation"
)]
fn zero_period_is_rejected() {
    basin::CmaEs::<Vec<f64>, basin::DenseMatrix>::new(0, 1.0)
        .with_eigen_update_period(0);
}
