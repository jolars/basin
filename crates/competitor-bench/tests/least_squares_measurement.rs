//! Verify observations against publication, physical callbacks, and lifecycle.
use basin::solver::least_squares_diagnostics::LeastSquaresDiagnostics;
use basin::{
    Executor, LevenbergMarquardt, LmDamping, PointState, Trf,
    TrustRegionReflective,
};
use competitor_bench::convergence::least_squares::{
    AnalyticModel, Instrumented, RobustStoppingPolicy, measure,
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

macro_rules! stopping_ablation {
    ($name:ident, $f:ty, $solver_name:ident, $solver:expr) => {
        #[test]
        fn $name() {
            let run = |policy: RobustStoppingPolicy| {
                let ledger = WorkLedger::new(4000);
                let $solver_name = policy.configure(
                    LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default(
                    ),
                    if <$f>::EPSILON > 1e-10 { 1e-4 } else { 1e-8 },
                );
                measure(
                    Executor::from_start(
                        basin::RobustLeastSquares::new(
                            Instrumented::analytic(
                                AnalyticModel::Linear,
                                ledger.clone(),
                            ),
                            basin::CauchyLoss,
                        ),
                        ($solver).with_trial_diagnostics(true),
                        DVector::<$f>::from_element(1, -1.),
                    )
                    .max_iter(10_000),
                    &ledger,
                    Duration::from_secs(10),
                )
            };
            let control = run(RobustStoppingPolicy::DefaultGradient);
            assert!(
                (control.run.returned().unwrap().point[0] - 1.).abs() < 1e-6
            );
            for policy in RobustStoppingPolicy::ALL {
                let probe = run(policy);
                for (a, b) in
                    probe.run.ledger.calls.iter().zip(&control.run.ledger.calls)
                {
                    assert_eq!(a.kind, b.kind);
                    assert_eq!(a.point, b.point);
                    assert_eq!(a.outcome, b.outcome);
                }
                assert!(probe.run.returned().is_some());
                if policy == RobustStoppingPolicy::NormalizedGradient {
                    assert!(
                        (probe.run.returned().unwrap().point[0] - 1.).abs()
                            < 1e-6
                    );
                }
            }
        }
    };
}
stopping_ablation!(stopping_normal64, f64, s, s);
stopping_ablation!(stopping_normal32, f32, s, s);
stopping_ablation!(
    stopping_trust64,
    f64,
    s,
    s.with_damping(LmDamping::TrustRegion)
);
stopping_ablation!(
    stopping_trust32,
    f32,
    s,
    s.with_damping(LmDamping::TrustRegion)
);
stopping_ablation!(stopping_qr64, f64, s, s.with_pivoted_qr());
stopping_ablation!(stopping_qr32, f32, s, s.with_pivoted_qr());
stopping_ablation!(
    stopping_qr_trust64,
    f64,
    s,
    s.with_pivoted_qr().with_damping(LmDamping::TrustRegion)
);
stopping_ablation!(
    stopping_qr_trust32,
    f32,
    s,
    s.with_pivoted_qr().with_damping(LmDamping::TrustRegion)
);

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

macro_rules! bounded_checks {
    ($name:ident, $f:ty) => {
        #[test]
        fn $name() {
            for (lower, upper, start, target) in [
                (
                    vec![0., -0.5],
                    vec![0.5, 0.],
                    vec![0.1, -0.1],
                    vec![0.5, -0.5],
                ),
                (
                    vec![0.25, -2.],
                    vec![0.25, 0.],
                    vec![0.25, -0.1],
                    vec![0.25, -1.],
                ),
                (
                    vec![0.25, -0.25],
                    vec![0.25, -0.25],
                    vec![0.25, -0.25],
                    vec![0.25, -0.25],
                ),
            ] {
                let run = |enabled| {
                    let ledger = WorkLedger::new(4000);
                    let problem = Instrumented::<$f>::analytic(
                        AnalyticModel::BoxLinear,
                        ledger.clone(),
                    )
                    .with_bounds(lower.clone(), upper.clone());
                    measure(
                        Executor::new(
                            problem,
                            TrustRegionReflective::<$f>::new()
                                .with_trial_diagnostics(enabled),
                            PointState::new(DVector::from_vec(start.clone())),
                        )
                        .max_iter(1000),
                        &ledger,
                        Duration::from_secs(10),
                    )
                };
                let plain = run(false);
                let traced = run(true);
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
                    for (i, &v) in b.point.iter().enumerate() {
                        assert!(v >= lower[i] as f64 && v <= upper[i] as f64);
                    }
                }
                let returned = traced.run.returned().unwrap();
                for (v, t) in returned.point.iter().zip(&target) {
                    assert!(
                        (v - *t as f64).abs()
                            <= if stringify!($f) == "f32" {
                                1e-3
                            } else {
                                1e-6
                            }
                    );
                }
                if lower == upper {
                    assert_eq!(traced.run.ledger.work(), 1);
                    assert_eq!(traced.run.counts.unwrap().jacobian_evals, 0);
                    assert!(traced.native.iter().any(|r| {
                        r.observation.checks.iter().any(|c| {
                            c.name == "no_free_parameters"
                                && c.passed == Some(true)
                        })
                    }));
                }
            }
        }
    };
}
bounded_checks!(bounded64, f64);
bounded_checks!(bounded32, f32);

macro_rules! robust_checks {
    ($name:ident, $f:ty, $solver:expr, $bounded:expr) => {
        #[test]
        fn $name() {
            use competitor_bench::convergence::least_squares::{
                ROBUST_EXTENDED_FIXTURES, ROBUST_FIXTURES,
            };
            for fixture in ROBUST_FIXTURES
                .iter()
                .chain(ROBUST_EXTENDED_FIXTURES.iter())
            {
                if fixture.bounds.is_some() && !$bounded {
                    continue;
                }
                let run = |enabled, cap| {
                    let ledger = WorkLedger::new(cap);
                    let mut raw = Instrumented::<$f>::analytic(
                        fixture.model,
                        ledger.clone(),
                    );
                    if let Some((lo, hi)) = fixture.bounds {
                        raw = raw.with_bounds(vec![lo as $f], vec![hi as $f]);
                    }
                    let problem =
                        basin::RobustLeastSquares::new(raw, fixture.loss)
                            .with_scale(fixture.scale as $f);
                    measure(
                        Executor::new(
                            problem,
                            $solver.with_trial_diagnostics(enabled),
                            PointState::new(DVector::from_vec(
                                fixture
                                    .start
                                    .iter()
                                    .map(|v| *v as $f)
                                    .collect(),
                            )),
                        )
                        .max_iter(10000),
                        &ledger,
                        Duration::from_secs(10),
                    )
                };
                let plain = run(false, 4000);
                let traced = run(true, 4000);
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
                for cap in 0..=2 {
                    let limited = run(true, cap);
                    assert_eq!(limited.run.ledger.work(), cap);
                    assert!(limited.run.returned().is_none());
                    assert_eq!(limited.run.ledger.denied, 1);
                }
            }
        }
    };
}
macro_rules! robust_precision {
    ($f:ty, $n:ident, $t:ident, $qn:ident, $qt:ident, $legacy:ident, $full:ident) => {
        robust_checks!(
            $n,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default(),
            false
        );
        robust_checks!(
            $t,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default()
                .with_damping(LmDamping::TrustRegion),
            false
        );
        robust_checks!(
            $qn,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default()
                .with_pivoted_qr(),
            false
        );
        robust_checks!(
            $qt,
            $f,
            LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default()
                .with_pivoted_qr()
                .with_damping(LmDamping::TrustRegion),
            false
        );
        robust_checks!(
            $legacy,
            $f,
            Trf::<DVector<$f>, DMatrix<$f>, $f>::default(),
            true
        );
        robust_checks!($full, $f, TrustRegionReflective::<$f>::new(), true);
    };
}
robust_precision!(
    f64,
    robust_lm64,
    robust_trust64,
    robust_qr64,
    robust_qt64,
    robust_legacy64,
    robust_full64
);
robust_precision!(
    f32,
    robust_lm32,
    robust_trust32,
    robust_qr32,
    robust_qt32,
    robust_legacy32,
    robust_full32
);

macro_rules! extended_model_checks {
    ($name:ident, $f:ty) => {
        #[test]
        fn $name() {
            use basin::{Jacobian, Residual};
            use competitor_bench::convergence::least_squares::ROBUST_EXTENDED_FIXTURES;
            for fixture in ROBUST_EXTENDED_FIXTURES {
                let problem = Instrumented::<$f>::analytic(fixture.model, WorkLedger::new(100));
                let reference = DVector::from_vec(fixture.reference.iter().map(|v| *v as $f).collect());
                let residual = problem.residual(&reference).unwrap();
                let jacobian = problem.jacobian(&reference).unwrap();
                let (rows, rank) = match fixture.model {
                    AnalyticModel::RobustLinear4 => (12, 4),
                    AnalyticModel::RobustRankDeficient4 => (8, 2),
                    _ => (1, 1),
                };
                assert_eq!(residual.len(), rows);
                assert_eq!(jacobian.shape(), (rows, reference.len()));
                let singular = jacobian.clone().svd(false, false).singular_values;
                assert_eq!(singular.iter().filter(|v| **v > 1e-5).count(), rank);
                if rank == 2 {
                    let shifted = &reference + DVector::from_vec(vec![1.5, -1.5, 3.5, -3.5]);
                    assert_eq!(problem.residual(&shifted).unwrap(), residual);
                    assert_eq!(residual[6], 2.);
                    assert_eq!(residual[7], -2.);
                } else {
                    assert!(residual.iter().all(|v| *v == 0.));
                }
                let start = DVector::from_vec(fixture.start.iter().map(|v| *v as $f).collect());
                let direction = DVector::from_element(start.len(), 0.125);
                let difference = problem.residual(&(&start + &direction)).unwrap() - problem.residual(&start).unwrap();
                let expected = jacobian * direction;
                assert!((&difference - expected).norm() < 1e-5);
            }
        }
    };
}
extended_model_checks!(extended_models64, f64);
extended_model_checks!(extended_models32, f32);
