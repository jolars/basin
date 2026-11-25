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
            assert!(
                a.current_record().unwrap().cost.unwrap().is_finite()
                    && a.current_record().unwrap().cost.unwrap() < 1e-8,
                "{}",
                a.current_record().unwrap().cost.unwrap()
            );
            use basin::CostFunction;
            assert_eq!(
                a.current_record().unwrap().cost.unwrap(),
                Quadratic.cost(a.param()).unwrap()
            );

            let run_bounded = |period| {
                let bounds = basin::BoxBounds::new(
                    Some(($from)(&[-2.0; 4])),
                    Some(($from)(&[2.0; 4])),
                );
                let problem = basin::Constrained::new(
                    Quadratic,
                    basin::ConstraintSet::new().with_bounds(bounds),
                );
                basin::Executor::new(
                    problem,
                    basin::BoundedCmaEs::<$v, $m, $f>::new(42, 0.5)
                        .with_lambda(8)
                        .with_eigen_update_period(period),
                    basin::PopulationProgress::from_point(($from)(&[1.0; 4])),
                )
                .max_iter(12)
                .run()
                .unwrap()
            };
            let bounded = run_bounded(3);
            assert_eq!(bounded.state, run_bounded(3).state);
            assert_ne!(bounded.state, run_bounded(1).state);
            assert_eq!(bounded.counts.cost_evals, 13 * 9);
            assert_eq!(
                bounded.current_record().unwrap().cost.unwrap(),
                Quadratic.cost(bounded.param()).unwrap(),
            );
            for candidate in bounded.state.candidates() {
                assert!((0..4).all(|i| (-2.0..=2.0).contains(&candidate[i])));
            }
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

struct BoxQuadratic;

impl basin::CostFunction for BoxQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok((x[0] - 1.0).powi(2) + 4.0 * x[1].powi(2))
    }
}

fn boxed_quadratic() -> impl basin::CostFunction<
    Param = Vec<f64>,
    Output = f64,
    Error = std::convert::Infallible,
> + basin::Constraints<
    Set = basin::ConstraintSet<basin::BoxBounds<Vec<f64>>>,
> {
    basin::Constrained::new(
        BoxQuadratic,
        basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
            Some(vec![-3.0; 2]),
            Some(vec![0.0, 3.0]),
        )),
    )
}

type Bounded = basin::BoundedCmaEs<Vec<f64>, basin::DenseMatrix>;

fn bounded() -> Bounded {
    Bounded::new(42, 0.5).with_lambda(8)
}

fn start() -> basin::PopulationProgress<Vec<f64>> {
    basin::PopulationProgress::from_point(vec![-1.0, 2.0])
}

#[test]
fn bounded_default_matches_explicit_period_one() {
    let run = |solver| {
        basin::Executor::new(boxed_quadratic(), solver, start())
            .max_iter(12)
            .run()
            .unwrap()
    };
    let default = run(bounded());
    let explicit = run(bounded().with_eigen_update_period(1));
    assert_eq!(default.state, explicit.state);
    assert_eq!(default.counts, explicit.counts);

    let wrapper = || {
        basin::BoundedCmaInject::with_inner_solver(
            bounded(),
            basin::NelderMead::new().projected(),
        )
        .with_inner_max_iter(2)
    };
    let run = |solver| {
        basin::Executor::new(boxed_quadratic(), solver, start())
            .max_iter(12)
            .run()
            .unwrap()
    };
    let default = run(wrapper());
    let explicit = run(wrapper().with_eigen_update_period(1));
    assert_eq!(default.state, explicit.state);
    assert_eq!(default.counts, explicit.counts);
    assert!(default.counts.cost_evals > 13 * 9);
}

#[test]
#[should_panic(
    expected = "eigen update period must be at least one generation"
)]
fn bounded_zero_period_is_rejected() {
    bounded().with_eigen_update_period(0);
}

#[test]
fn bounded_refresh_survives_checkpoint_and_restarts_on_fresh_run() {
    let run = |solver, limit| {
        basin::Executor::new(boxed_quadratic(), solver, start())
            .max_iter(limit)
            .run_with_solver()
            .unwrap()
    };
    let initial = run(bounded().with_eigen_update_period(3), 0);
    let paused = run(bounded().with_eigen_update_period(3), 2);
    let initial_model = initial.solver.distribution().unwrap();
    let paused_model = paused.solver.distribution().unwrap();
    assert_eq!(
        initial_model.principal_axis_stds,
        paused_model.principal_axis_stds
    );
    assert_ne!(initial_model.covariance, paused_model.covariance);
    let reference = run(bounded().with_eigen_update_period(3), 7);
    assert_ne!(
        initial_model.principal_axis_stds,
        reference.solver.distribution().unwrap().principal_axis_stds,
    );
    let checkpoint = paused.into_checkpoint();
    #[cfg(feature = "serde")]
    let checkpoint = {
        let bytes = postcard::to_allocvec(&checkpoint).unwrap();
        postcard::from_bytes(&bytes).unwrap()
    };
    let resumed =
        basin::Executor::resume_from_checkpoint(boxed_quadratic(), checkpoint)
            .max_iter(7)
            .run_with_solver()
            .unwrap();
    assert_eq!(reference.state, resumed.state);
    assert_eq!(reference.counts, resumed.counts);
    let fresh = run(resumed.solver, 2);
    let expected = run(bounded().with_eigen_update_period(3), 2);
    assert_eq!(fresh.state, expected.state);
    assert_eq!(fresh.counts, expected.counts);
}

#[cfg(feature = "serde")]
#[test]
fn bounded_maps_missing_the_period_default_to_one() {
    let mut serialized = serde_json::to_value(bounded()).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("eigen_update_period");
    let restored: Bounded = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        serde_json::to_value(restored).unwrap()["eigen_update_period"],
        1
    );
}

#[test]
fn bounded_injection_retains_refresh_schedule_across_checkpoint() {
    let run = |limit| {
        let solver = basin::BoundedCmaInject::with_inner_solver(
            bounded(),
            basin::NelderMead::new().projected(),
        )
        .with_inner_max_iter(2)
        .with_eigen_update_period(3);
        basin::Executor::new(boxed_quadratic(), solver, start())
            .max_iter(limit)
            .run_with_solver()
            .unwrap()
    };
    let baseline = run(7);
    let checkpoint = run(2).into_checkpoint();
    let resumed =
        basin::Executor::resume_from_checkpoint(boxed_quadratic(), checkpoint)
            .max_iter(7)
            .run_with_solver()
            .unwrap();
    assert_eq!(baseline.state, resumed.state);
    assert_eq!(baseline.counts, resumed.counts);
    assert_eq!(
        baseline
            .solver
            .cma()
            .distribution()
            .unwrap()
            .principal_axis_stds,
        resumed
            .solver
            .cma()
            .distribution()
            .unwrap()
            .principal_axis_stds,
    );
}
