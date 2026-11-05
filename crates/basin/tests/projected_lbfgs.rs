#[path = "../benches/support/lbfgsb_dimension.rs"]
mod support;

use basin::{
    Dot, Scalar, ScaledAdd, VectorLen, solver::lbfgs::AsFloatSliceMut,
};
use support::{Case, Quadratic};

fn check<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: AsFloatSliceMut<F> + Clone + Dot<F> + ScaledAdd<F> + VectorLen,
{
    use basin::line_search::projected_armijo::ProjectedArmijo;
    use basin::solver::lbfgs::model_strategy::ProjectedSearch;
    use basin::{Executor, FirstOrderState};
    for case in Case::ALL {
        for search in [
            ProjectedSearch::FeasibleRay,
            ProjectedSearch::Chord,
            ProjectedSearch::Armijo(ProjectedArmijo::new()),
            ProjectedSearch::Armijo(
                ProjectedArmijo::new().with_fused_evaluations(),
            ),
        ] {
            for retain in [true, false] {
                let p = Quadratic::new(case, make);
                let result = Executor::new(
                    &p,
                    p.solver()
                        .with_compact_model()
                        .with_projected_search(search.clone())
                        .with_curvature_retention(retain),
                    FirstOrderState::new(p.start()),
                )
                .max_iter(1000)
                .run()
                .unwrap();
                p.verify(&result);
                assert_eq!(
                    result.report.code(),
                    basin::TerminationCode::ProjectedGradientTolerance,
                    "{} {search:?} retention={retain}",
                    case.name()
                );
            }
        }
    }
}

#[test]
fn vec_analytic_solutions() {
    check::<_, f64>(|x| x.to_vec());
    check::<_, f32>(|x| x.to_vec());
}
macro_rules! backend {
    ($name:ident,$feature:literal,$make:expr) => {
        #[cfg(feature=$feature)]
        #[test]
        fn $name() {
            check::<_, f64>($make);
            check::<_, f32>($make);
        }
    };
}
backend!(
    nalgebra_032,
    "nalgebra_v0_32",
    nalgebra_0_32::DVector::from_column_slice
);
backend!(
    nalgebra_033,
    "nalgebra_v0_33",
    nalgebra_0_33::DVector::from_column_slice
);
backend!(
    nalgebra_034,
    "nalgebra_v0_34",
    nalgebra_0_34::DVector::from_column_slice
);
backend!(
    nalgebra_035,
    "nalgebra_v0_35",
    nalgebra::DVector::from_column_slice
);
backend!(ndarray_015, "ndarray_v0_15", |x| {
    ndarray_0_15::Array1::from_vec(x.to_vec())
});
backend!(ndarray_016, "ndarray_v0_16", |x| {
    ndarray_0_16::Array1::from_vec(x.to_vec())
});
backend!(ndarray_017, "ndarray_v0_17", |x| ndarray::Array1::from_vec(
    x.to_vec()
));
backend!(faer_022, "faer_v0_22", |x| faer_0_22::Col::from_fn(
    x.len(),
    |i| x[i]
));
backend!(faer_023, "faer_v0_23", |x| faer_0_23::Col::from_fn(
    x.len(),
    |i| x[i]
));
backend!(faer_024, "faer_v0_24", |x| faer::Col::from_fn(
    x.len(),
    |i| x[i]
));

#[test]
fn exact_continuation_preserves_every_search_and_retention_choice() {
    use basin::Executor;
    use basin::line_search::projected_armijo::ProjectedArmijo;
    use basin::solver::lbfgs::model_strategy::ProjectedSearch;
    #[cfg(feature = "serde")]
    use basin::{
        FirstOrderState, solver::lbfgs::model_strategy::ProjectedLbfgs,
    };
    let p = Quadratic::new(Case::Rosenbrock2, |x: &[f64]| x.to_vec());
    for search in [
        ProjectedSearch::FeasibleRay,
        ProjectedSearch::Chord,
        ProjectedSearch::Armijo(
            ProjectedArmijo::new().with_fused_evaluations(),
        ),
    ] {
        for retain in [true, false] {
            for split in [0, 1, 7, 15] {
                let configure = || {
                    p.solver()
                        .with_compact_model()
                        .with_projected_search(search.clone())
                        .with_curvature_retention(retain)
                };
                let mut stepper =
                    Executor::from_start(&p, configure(), p.start())
                        .max_iter(20)
                        .into_stepper()
                        .unwrap();
                for _ in 0..split {
                    stepper.step().unwrap();
                }
                let checkpoint = stepper.into_checkpoint().unwrap();
                #[cfg(feature = "serde")]
                let checkpoint = {
                    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
                    postcard::from_bytes::<
                        basin::ExactCheckpoint<
                            ProjectedLbfgs<Vec<f64>>,
                            FirstOrderState<Vec<f64>>,
                        >,
                    >(&bytes)
                    .unwrap()
                };
                let resumed = Executor::resume_from_checkpoint(&p, checkpoint)
                    .max_iter(20)
                    .run()
                    .unwrap();
                let fresh = Executor::from_start(&p, configure(), p.start())
                    .max_iter(20)
                    .run()
                    .unwrap();
                assert_eq!(resumed.param(), fresh.param());
                assert_eq!(resumed.cost(), fresh.cost());
                assert_eq!(resumed.state.counts(), fresh.state.counts());
                assert_eq!(resumed.report, fresh.report);
            }
        }
    }
}

#[test]
fn fresh_reuse_resets_search_masks_and_dimension() {
    use basin::solver::lbfgs::model_strategy::ProjectedSearch;
    let p = Quadratic::new(Case::Rosenbrock2, |x: &[f64]| x.to_vec());
    let q = Quadratic::new(Case::Active50Two, |x: &[f64]| x.to_vec());
    for search in [
        ProjectedSearch::FeasibleRay,
        ProjectedSearch::Armijo(
            basin::line_search::projected_armijo::ProjectedArmijo::new(),
        ),
    ] {
        let configure = || {
            p.solver()
                .with_compact_model()
                .with_projected_search(search.clone())
                .with_curvature_retention(false)
        };
        let first = basin::Executor::from_start(&p, configure(), p.start())
            .max_iter(7)
            .run_with_solver()
            .unwrap();
        let reused = basin::Executor::from_start(&q, first.solver, q.start())
            .max_iter(1000)
            .run()
            .unwrap();
        let fresh = basin::Executor::from_start(&q, configure(), q.start())
            .max_iter(1000)
            .run()
            .unwrap();
        assert_eq!(reused.param(), fresh.param());
        assert_eq!(reused.report, fresh.report);
        assert_eq!(reused.state.counts(), fresh.state.counts());
    }
}
