use basin::{
    BoxConstraints, CostFunction, DenseMatrix, Executor, Jacobian,
    LevenbergMarquardt, LevenbergMarquardtQr, LmDamping,
    NativeConvergenceDiagnostics, NativeConvergenceTest as Test, PointState,
    Residual, RobustLeastSquares, Solver, SquaredLoss, TerminationCode,
    TrustRegionReflective,
};
use std::convert::Infallible;

#[derive(Clone)]
struct Fit {
    target: f64,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Fit {
    fn new(target: f64) -> Self {
        Self {
            target,
            lower: vec![-10.],
            upper: vec![10.],
        }
    }
}

impl Residual for Fit {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0] - self.target])
    }
}

impl Jacobian for Fit {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[1.]))
    }
}

impl CostFunction for Fit {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Infallible> {
        panic!("least-squares solvers evaluate residuals")
    }
}

impl BoxConstraints for Fit {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn check_lifecycle<So>(make: impl Fn() -> So)
where
    So: Solver<Fit, PointState<Vec<f64>>, Error = Infallible>
        + NativeConvergenceDiagnostics,
{
    let uninterrupted =
        Executor::new(Fit::new(3.), make(), PointState::new(vec![1.]))
            .max_iter(100)
            .run_with_solver()
            .unwrap();
    assert_eq!(
        uninterrupted.report.code(),
        TerminationCode::SolverConverged
    );
    assert!(!uninterrupted.native_convergence_tests().is_empty());

    let mut stepper =
        Executor::new(Fit::new(3.), make(), PointState::new(vec![1.]))
            .max_iter(100)
            .into_stepper()
            .unwrap();
    stepper.step().unwrap();
    let resumed = Executor::resume_from_checkpoint(
        Fit::new(3.),
        stepper.into_checkpoint().unwrap(),
    )
    .max_iter(100)
    .run_with_solver()
    .unwrap();
    assert_eq!(resumed.param(), uninterrupted.param());
    assert_eq!(resumed.counts, uninterrupted.counts);
    assert_eq!(resumed.iter(), uninterrupted.iter());
    assert_eq!(
        resumed.native_convergence_tests(),
        uninterrupted.native_convergence_tests()
    );

    let iter = resumed.iter();
    let limited = Executor::resume_from_checkpoint(
        Fit::new(3.),
        resumed.into_checkpoint(),
    )
    .max_iter(iter)
    .run_with_solver()
    .unwrap();
    assert_eq!(limited.report.code(), TerminationCode::MaxIter);
    assert!(limited.native_convergence_tests().is_empty());

    let fresh =
        Executor::new(Fit::new(-2.), limited.solver, PointState::new(vec![1.]))
            .max_iter(0)
            .run_with_solver()
            .unwrap();
    assert!(fresh.solver.native_convergence_tests().is_empty());
    let continued = Executor::resume_from_checkpoint(
        Fit::new(-2.),
        fresh.into_checkpoint(),
    )
    .max_iter(100)
    .run_with_solver()
    .unwrap();
    let reference =
        Executor::new(Fit::new(-2.), make(), PointState::new(vec![1.]))
            .max_iter(100)
            .run_with_solver()
            .unwrap();
    assert_eq!(continued.param(), reference.param());
    assert_eq!(continued.counts, reference.counts);
    assert_eq!(
        continued.native_convergence_tests(),
        reference.native_convergence_tests()
    );
}

#[test]
fn diagnostics_follow_fresh_and_exact_runs() {
    check_lifecycle(LevenbergMarquardt::new);
    check_lifecycle(LevenbergMarquardtQr::new);
    check_lifecycle(TrustRegionReflective::new);
    check_lifecycle(|| {
        LevenbergMarquardtQr::new().with_absolute_cost_change_tolerance(None)
    });
    check_lifecycle(|| {
        TrustRegionReflective::new().with_relative_step_tolerance(None)
    });
}

#[test]
fn gradient_tests_are_reported_together_before_trial_tests() {
    let solver = LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new()
        .with_absolute_gradient_tolerance(0.)
        .with_gradient_orthogonality_tolerance(0.)
        .with_relative_step_tolerance(1.);
    let result = Executor::from_start(Fit::new(1.), solver, vec![1.])
        .run_with_solver()
        .unwrap();
    assert_eq!(
        result.native_convergence_tests(),
        &[Test::AbsoluteGradient, Test::GradientOrthogonality]
    );
    assert_eq!(result.counts.residual_evals, 1);
    assert_eq!(result.counts.jacobian_evals, 1);
    assert_eq!(result.iter(), 0);
}

#[test]
fn orthogonality_is_distinct_from_absolute_gradient_and_robust_normalization() {
    let solver = || {
        LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new()
            .with_absolute_gradient_tolerance(None)
            .with_gradient_orthogonality_tolerance(1.)
    };
    let result = Executor::from_start(Fit::new(3.), solver(), vec![1.])
        .run_with_solver()
        .unwrap();
    assert_eq!(
        result.native_convergence_tests(),
        &[Test::GradientOrthogonality]
    );
    let robust = Executor::from_start(
        RobustLeastSquares::new(Fit::new(3.), SquaredLoss),
        solver(),
        vec![1.],
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(
        robust.native_convergence_tests(),
        &[Test::RobustGradientOrthogonality]
    );
    assert_eq!(result.counts, robust.counts);
    assert_eq!(result.param(), robust.param());

    let robust_trf = Executor::from_start(
        RobustLeastSquares::new(Fit::new(1.), SquaredLoss),
        TrustRegionReflective::new(),
        vec![1.],
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(
        robust_trf.native_convergence_tests(),
        &[Test::AbsoluteScaledGradient]
    );
}

#[test]
fn continuing_a_native_stop_replaces_its_record() {
    let result = Executor::from_start(
        Fit::new(1.),
        LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new(),
        vec![1.],
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(result.native_convergence_tests(), &[Test::AbsoluteGradient]);
    let (solver, state, counts) = result.into_checkpoint().into_parts();
    let checkpoint = basin::ExactCheckpoint::from_parts(
        solver.with_absolute_gradient_tolerance(None),
        state,
        counts,
    );
    let resumed = Executor::resume_from_checkpoint(Fit::new(1.), checkpoint)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.report.code(), TerminationCode::NumericalNoProgress);
    assert!(resumed.native_convergence_tests().is_empty());
    assert!(resumed.solver.native_convergence_tests().is_empty());
}

#[test]
fn trial_tests_report_all_passing_tests_without_changing_work() {
    for (model, step, radius, expected) in [
        (Some(2.), None, None, vec![Test::RelativeModelReduction]),
        (None, Some(1.), None, vec![Test::RelativeTrialStep]),
        (None, None, Some(2.), vec![Test::RelativeTrustRadius]),
        (
            Some(2.),
            Some(1.),
            Some(2.),
            vec![
                Test::RelativeModelReduction,
                Test::RelativeTrialStep,
                Test::RelativeTrustRadius,
            ],
        ),
    ] {
        let solver = LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new()
            .with_damping(LmDamping::TrustRegion)
            .with_absolute_gradient_tolerance(None)
            .with_relative_model_reduction_tolerance(model)
            .with_relative_step_tolerance(step)
            .with_relative_trust_radius_tolerance(radius);
        let result = Executor::from_start(Fit::new(3.), solver, vec![1.])
            .max_iter(1)
            .run_with_solver()
            .unwrap();
        assert_eq!(result.report.code(), TerminationCode::SolverConverged);
        assert_eq!(result.native_convergence_tests(), expected);
        assert_eq!(result.counts.residual_evals, 2);
        assert_eq!(result.counts.jacobian_evals, 1);
        assert_eq!(result.iter(), 1);
    }
}

#[test]
fn trf_distinguishes_stationarity_from_fixed_parameters() {
    for fixed in [false, true] {
        let mut fit = Fit::new(1.);
        if fixed {
            fit.lower = vec![1.];
            fit.upper = vec![1.];
        }
        let solver = TrustRegionReflective::new()
            .with_absolute_scaled_gradient_tolerance(if fixed {
                None
            } else {
                Some(0.)
            });
        let result = Executor::from_start(fit, solver, vec![1.])
            .run_with_solver()
            .unwrap();
        assert_eq!(
            result.native_convergence_tests(),
            if fixed {
                &[Test::NoFreeParameters]
            } else {
                &[Test::AbsoluteScaledGradient]
            }
        );
        assert_eq!(result.counts.residual_evals, 1);
        assert_eq!(result.counts.jacobian_evals, u64::from(!fixed));
        assert_eq!(result.iter(), 0);
    }
}

#[test]
fn safeguards_failures_and_observed_checks_are_not_native_convergence() {
    let result = Executor::from_start(
        Fit::new(1.),
        LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new()
            .with_absolute_gradient_tolerance(None),
        vec![1.],
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::NumericalNoProgress);
    assert!(result.native_convergence_tests().is_empty());
    assert!(result.solver.native_convergence_tests().is_empty());

    let failed = Executor::from_start(
        Fit::new(f64::NAN),
        TrustRegionReflective::new(),
        vec![1.],
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(failed.report.code(), TerminationCode::SolverFailed);
    assert!(failed.native_convergence_tests().is_empty());

    let observed = Executor::from_start(
        Fit::new(3.),
        LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new()
            .with_absolute_gradient_tolerance(None)
            .with_absolute_step_tolerance(10.),
        vec![1.],
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(observed.report.code(), TerminationCode::ParamTolerance);
    assert!(observed.native_convergence_tests().is_empty());
}
