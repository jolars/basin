//! Verify observations against publication, physical callbacks, and lifecycle.
use basin::solver::least_squares_diagnostics::LeastSquaresDiagnostics;
use basin::{
    Executor, LevenbergMarquardt, LmDamping, PointState, Trf,
    TrustRegionReflective,
};
use competitor_bench::convergence::least_squares::{
    AnalyticModel, Instrumented, measure,
};
use competitor_bench::convergence::ledger::WorkLedger;
use competitor_bench::convergence::runner::RunOutcome;
#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix, DVector};
use std::time::Duration;

macro_rules! checks {
    ($name:ident, $f:ty, $solver:expr) => {
        #[test]
        fn $name() {
            let run = |enabled, cap, model| {
                let ledger = WorkLedger::new(cap);
                let problem =
                    Instrumented::<$f>::analytic(model, ledger.clone());
                measure(
                    Executor::new(
                        problem,
                        $solver.with_trial_diagnostics(enabled),
                        PointState::new(DVector::<$f>::from_element(1, 0.1)),
                    )
                    .max_iter(200),
                    &ledger,
                    Duration::from_secs(10),
                )
            };
            let plain = run(false, 400, AnalyticModel::Nonzero);
            let traced = run(true, 400, AnalyticModel::Nonzero);
            assert!(plain.native.is_empty());
            assert!(!traced.native.is_empty());
            assert_eq!(plain.run.counts, traced.run.counts);
            assert_eq!(
                format!("{:?}", plain.run.outcome),
                format!("{:?}", traced.run.outcome)
            );
            assert_eq!(plain.run.ledger.work(), traced.run.ledger.work());
            assert_eq!(
                plain.run.recommendations.len(),
                traced.run.recommendations.len()
            );
            for (a, b) in plain
                .run
                .recommendations
                .iter()
                .zip(&traced.run.recommendations)
            {
                assert_eq!(a.point, b.point);
                assert_eq!(a.solver_cost, b.solver_cost);
                assert_eq!(a.counts, b.counts);
            }
            assert!(
                traced
                    .native
                    .windows(2)
                    .all(|w| w[0].observation.sequence
                        < w[1].observation.sequence)
            );
            assert!(traced.native.iter().any(|r| {
                r.observation.checks.iter().any(|c| c.passed == Some(false))
            }));
            for cap in 0..=2 {
                let measured = run(true, cap, AnalyticModel::Linear);
                assert_eq!(measured.run.ledger.work(), cap);
                assert_eq!(measured.run.ledger.denied, 1);
                assert!(measured.run.returned().is_none());
                assert!(matches!(
                    measured.run.outcome,
                    RunOutcome::InitializationError(_)
                        | RunOutcome::StepError(_)
                ));
                assert_eq!(measured.run.counts.is_none(), cap == 0);
            }
            let nonfinite = run(true, 400, AnalyticModel::NonFiniteTrial);
            assert!(nonfinite.native.iter().any(|r| {
                r.observation.trial_cost.is_some_and(|f| !f.is_finite())
                    && r.observation.accepted == Some(false)
                    && !r.published
            }));
        }
    };
}
macro_rules! precision {
    ($f:ty, $n:ident, $t:ident, $qn:ident, $qt:ident, $legacy:ident, $full:ident) => {
        checks!(
            $n,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default()
        );
        checks!(
            $t,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default()
                .with_damping(LmDamping::TrustRegion)
        );
        checks!(
            $qn,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default()
                .with_pivoted_qr()
        );
        checks!(
            $qt,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default()
                .with_pivoted_qr()
                .with_damping(LmDamping::TrustRegion)
        );
        checks!($legacy, $f, Trf::<DVector<$f>, DMatrix<$f>, $f>::default());
        checks!($full, $f, TrustRegionReflective::<$f>::new());
    };
}
precision!(f64, lm64, trust64, qr64, qr_trust64, legacy64, full64);
precision!(f32, lm32, trust32, qr32, qr_trust32, legacy32, full32);

#[test]
fn accepted_trial_before_jacobian_denial_is_not_published() {
    let ledger = WorkLedger::new(2);
    let problem =
        Instrumented::<f64>::analytic(AnalyticModel::Linear, ledger.clone());
    let measured = measure(
        Executor::new(
            problem,
            TrustRegionReflective::new().with_trial_diagnostics(true),
            PointState::new(DVector::from_element(1, 0.1)),
        ),
        &ledger,
        Duration::from_secs(10),
    );
    assert!(
        measured
            .native
            .iter()
            .any(|r| r.observation.accepted == Some(true) && !r.published)
    );
    assert_eq!(measured.run.last_published().unwrap().point, vec![0.1]);
    assert!(measured.run.returned().is_none());
}

#[test]
fn checkpoint_keeps_sequence_and_fresh_initialization_resets_it() {
    let ledger = WorkLedger::new(400);
    let problem = Instrumented::<f64>::analytic(AnalyticModel::Nonzero, ledger);
    let solver = LevenbergMarquardt::<DVector<f64>, DMatrix<f64>>::new()
        .with_trial_diagnostics(true);
    let mut stepper = Executor::new(
        problem.clone(),
        solver,
        PointState::new(DVector::from_element(1, 0.1)),
    )
    .into_stepper()
    .unwrap();
    stepper.step().unwrap();
    let sequence = stepper
        .solver()
        .least_squares_observations()
        .last()
        .unwrap()
        .sequence;
    let mut continued = Executor::resume_from_checkpoint(
        problem.clone(),
        stepper.into_checkpoint().unwrap(),
    )
    .into_stepper()
    .unwrap();
    assert_eq!(
        continued
            .solver()
            .least_squares_observations()
            .last()
            .unwrap()
            .sequence,
        sequence
    );
    continued.step().unwrap();
    assert!(
        continued.solver().least_squares_observations()[0].sequence > sequence
    );
    let (solver, _, _) = continued.into_checkpoint().unwrap().into_parts();
    let mut fresh = Executor::new(
        problem,
        solver,
        PointState::new(DVector::from_element(1, 0.1)),
    )
    .into_stepper()
    .unwrap();
    assert!(fresh.solver().least_squares_observations().is_empty());
    fresh.step().unwrap();
    assert_eq!(fresh.solver().least_squares_observations()[0].sequence, 1);
}

#[test]
fn split_initialization_charges_two_physical_calls_and_preserves_logical_counts()
 {
    let run = |fused| {
        let ledger = WorkLedger::new(400);
        let mut problem = Instrumented::<f64>::analytic(
            AnalyticModel::Linear,
            ledger.clone(),
        );
        problem.fused = fused;
        measure(
            Executor::new(
                problem,
                TrustRegionReflective::new().with_trial_diagnostics(true),
                PointState::new(DVector::from_element(1, 0.1)),
            ),
            &ledger,
            Duration::from_secs(10),
        )
    };
    let fused = run(true);
    let split = run(false);
    assert_eq!(fused.run.counts, split.run.counts);
    assert_eq!(
        fused.run.returned().unwrap().point,
        split.run.returned().unwrap().point
    );
    assert_eq!(split.run.ledger.work(), fused.run.ledger.work() + 1);
    assert_eq!(
        split.run.ledger.count(
            competitor_bench::convergence::ledger::OracleKind::ResidualJacobian
        ),
        0
    );
}
