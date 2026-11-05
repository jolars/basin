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
    for case in Case::ALL {
        let p = Quadratic::new(case, make);
        let reference = p.solve(false);
        let alternate = p.solve(true);
        p.verify(&reference);
        p.verify(&alternate);
        assert_eq!(
            reference.report.code(),
            basin::TerminationCode::ProjectedGradientTolerance,
            "{} reference",
            case.name()
        );
        assert_eq!(
            alternate.report.code(),
            basin::TerminationCode::ProjectedGradientTolerance,
            "{} alternate",
            case.name()
        );
        assert_eq!(
            reference.state.counts().cost_evals,
            reference.state.counts().gradient_evals
        );
        assert_eq!(
            alternate.state.counts().cost_evals,
            alternate.state.counts().gradient_evals
        );
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
fn exact_continuation_rebuilds_only_scratch() {
    use basin::{Executor, FirstOrderState};
    let p = Quadratic::new(Case::Rosenbrock2, |x: &[f64]| x.to_vec());
    for split in [0, 1, 7, 15] {
        let mut stepper = Executor::new(
            &p,
            p.solver().with_dimension_aware_subspace(),
            FirstOrderState::new(p.start()),
        )
        .max_iter(20)
        .into_stepper()
        .unwrap();
        for _ in 0..split {
            stepper.step().unwrap();
        }
        let checkpoint = stepper.into_checkpoint().unwrap();
        #[cfg(feature = "serde")]
        let checkpoint = {
            let encoded = postcard::to_allocvec(&checkpoint).unwrap();
            postcard::from_bytes::<
                basin::ExactCheckpoint<
                    basin::solver::lbfgs::dimension_aware::DimensionAwareLbfgsb<
                        Vec<f64>,
                    >,
                    FirstOrderState<Vec<f64>>,
                >,
            >(&encoded)
            .unwrap()
        };
        let resumed = Executor::resume_from_checkpoint(&p, checkpoint)
            .max_iter(20)
            .run()
            .unwrap();
        let reference = Executor::new(
            &p,
            p.solver().with_dimension_aware_subspace(),
            FirstOrderState::new(p.start()),
        )
        .max_iter(20)
        .run()
        .unwrap();
        assert_eq!(resumed.param(), reference.param());
        assert_eq!(resumed.cost(), reference.cost());
        assert_eq!(resumed.state.counts(), reference.state.counts());
        assert_eq!(resumed.report, reference.report);
    }
}

#[test]
fn fresh_reuse_after_dimension_change_matches_new_configuration() {
    use basin::{Executor, FirstOrderState};
    let p = Quadratic::new(Case::Rosenbrock2, |x: &[f64]| x.to_vec());
    let first = Executor::new(
        &p,
        p.solver().with_dimension_aware_subspace(),
        FirstOrderState::new(p.start()),
    )
    .max_iter(12)
    .run_with_solver()
    .unwrap();
    let q = Quadratic::new(Case::Active50Two, |x: &[f64]| x.to_vec());
    // Keep configuration matched: both use a history capacity of ten.
    let reused =
        Executor::new(&q, first.solver, FirstOrderState::new(q.start()))
            .max_iter(500)
            .run()
            .unwrap();
    let fresh = Executor::new(
        &q,
        p.solver().with_dimension_aware_subspace(),
        FirstOrderState::new(q.start()),
    )
    .max_iter(500)
    .run()
    .unwrap();
    assert_eq!(reused.param(), fresh.param());
    assert_eq!(reused.state.counts(), fresh.state.counts());
    assert_eq!(reused.report, fresh.report);
}

struct Logged<'a> {
    problem: &'a Quadratic<Vec<f64>>,
    points: std::sync::Arc<std::sync::Mutex<Vec<Vec<f64>>>>,
    fail_at: Option<usize>,
}
impl basin::Constraints for Logged<'_> {
    type Set = basin::ConstraintSet<basin::BoxBounds<Vec<f64>>>;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        self.problem.constraints()
    }
}
impl basin::CostFunction for Logged<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        for (j, &value) in x.iter().enumerate() {
            assert!(
                value >= self.problem.lower[j]
                    && value <= self.problem.upper[j]
            );
        }
        let mut points = self.points.lock().unwrap();
        if Some(points.len()) == self.fail_at {
            return Err("callback failed");
        }
        points.push(x.clone());
        self.problem.cost(x).map_err(|e| match e {})
    }
}
impl basin::Gradient for Logged<'_> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.problem.gradient(x).map_err(|e| match e {})
    }
}

#[test]
fn callback_paths_and_counts_agree_without_extra_probes() {
    use basin::{Executor, FirstOrderState};
    for case in Case::ALL {
        let p = Quadratic::new(case, |x: &[f64]| x.to_vec());
        let run = |alternate| {
            let points = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let objective = Logged {
                problem: &p,
                points: points.clone(),
                fail_at: None,
            };
            let initial = FirstOrderState::new(p.start());
            let result = if alternate {
                Executor::new(
                    objective,
                    p.solver().with_dimension_aware_subspace(),
                    initial,
                )
                .max_iter(500)
                .run()
                .unwrap()
            } else {
                Executor::new(objective, p.solver(), initial)
                    .max_iter(500)
                    .run()
                    .unwrap()
            };
            let path = points.lock().unwrap().clone();
            assert_eq!(result.state.counts().cost_evals, path.len() as u64);
            (result, path)
        };
        let (reference, old) = run(false);
        let (alternate, new) = run(true);
        p.verify(&reference);
        p.verify(&alternate);
        assert_eq!(reference.report.code(), alternate.report.code());
        assert_eq!(old.len(), new.len(), "{} callback count", case.name());
        for (a, b) in old.iter().zip(&new) {
            for (a, b) in a.iter().zip(b) {
                assert!(
                    (a - b).abs() < 1e-5 * (1.0 + a.abs()),
                    "{} callback trajectory",
                    case.name()
                );
            }
        }
    }
}

#[test]
fn original_callback_errors_propagate() {
    use basin::{Executor, FirstOrderState};
    let p = Quadratic::new(Case::Rosenbrock2, |x: &[f64]| x.to_vec());
    let objective = Logged {
        problem: &p,
        points: Default::default(),
        fail_at: Some(5),
    };
    let result = Executor::new(
        objective,
        p.solver().with_dimension_aware_subspace(),
        FirstOrderState::new(p.start()),
    )
    .max_iter(500)
    .run();
    assert!(matches!(
        result,
        Err(basin::ConstraintSolveError::Evaluation("callback failed"))
    ));
}
