//! Analytic checks for CDP-1 quality, physical work, and publication semantics.

use std::time::Duration;

use basin::core::numdiff::Method;
use basin::{
    BoundedFiniteDiff, CostFunction, Executor, FiniteDiff, FirstOrderState,
    Gradient, Jacobian, Lbfgs, NelderMead, PointState, Problem, Residual,
    Solver, SolverStep, Stall, State, Termination, TerminationStage,
};
use competitor_bench::convergence::{
    fixtures::Quadratic,
    ledger::{OracleError, OracleKind, WorkLedger},
    quality::{
        Eligibility, Interval, RootCertificate, SmoothCertificate, TargetStatus,
    },
    runner::{PublicationStage, RunOutcome, measure},
};

fn certificate() -> SmoothCertificate {
    SmoothCertificate {
        reference: Interval::exact(0.0),
        objective_scale: 1.0,
        coordinate_scales: vec![1.0],
        solution_set: vec![vec![0.0]],
        require_parameter: true,
        bounds: None,
    }
}

#[test]
fn joint_quality_does_not_confuse_small_gradient_with_accuracy() {
    let c = certificate();
    let poor = c.assess(&[2.0], Interval::exact(1.0), &[1e-12]);
    assert_eq!(
        poor.target(1e-6, Eligibility::Eligible),
        TargetStatus::Missed
    );
    let optimum = c.assess(&[0.0], Interval::exact(0.0), &[0.0]);
    assert_eq!(
        optimum.target(1e-12, Eligibility::Eligible),
        TargetStatus::Passed
    );
    assert_eq!(
        optimum.target(1e-12, Eligibility::ReferencePending),
        TargetStatus::ReferencePending
    );
    assert_eq!(
        optimum.target(1e-12, Eligibility::PrecisionIneligible),
        TargetStatus::PrecisionIneligible
    );
    let below = c.assess(&[0.0], Interval::exact(-1.0), &[0.0]);
    assert_eq!(
        below.target(1e-6, Eligibility::Eligible),
        TargetStatus::InvalidReference
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            c.assess(&[bad], Interval::exact(0.0), &[0.0])
                .target(1e-6, Eligibility::Eligible),
            TargetStatus::NonFinite
        );
        assert_eq!(
            c.assess(&[0.0], Interval::exact(bad), &[0.0])
                .target(1e-6, Eligibility::Eligible),
            TargetStatus::NonFinite
        );
        assert_eq!(
            c.assess(&[0.0], Interval::exact(0.0), &[bad])
                .target(1e-6, Eligibility::Eligible),
            TargetStatus::NonFinite
        );
    }
}

#[test]
fn declared_scales_preserve_quality_under_coordinate_and_objective_changes() {
    let base = certificate();
    let expected = base.assess(&[0.25], Interval::exact(0.03125), &[0.25]);
    let scaled = SmoothCertificate {
        objective_scale: 1e8,
        coordinate_scales: vec![1e4],
        ..base
    };
    let actual =
        scaled.assess(&[2500.0], Interval::exact(3_125_000.0), &[2500.0]);
    assert_eq!(actual.objective_error, expected.objective_error);
    assert_eq!(actual.stationarity, expected.stationarity);
    assert_eq!(actual.parameter_error, expected.parameter_error);
    let p = Quadratic::<f64>::new(WorkLedger::new(10));
    let mut shifted = p.clone();
    shifted.offset = 1e20;
    let x = vec![1.0 + 1e-3, -2.0];
    assert_eq!(shifted.cost(&x).unwrap(), shifted.offset);
    assert!(shifted.verify(&x).0.lower > 0.0);
    assert_eq!(p.verify(&x).0.upper, shifted.verify(&x).0.upper);
}

#[test]
fn box_stationarity_and_feasibility_handle_active_and_fixed_coordinates() {
    let c = SmoothCertificate {
        bounds: Some((vec![0.0], vec![1.0])),
        ..certificate()
    };
    assert_eq!(
        c.assess(&[f64::NAN], Interval::exact(0.0), &[0.0])
            .target(1e-6, Eligibility::Eligible),
        TargetStatus::NonFinite
    );
    assert_eq!(
        c.assess(&[0.0], Interval::exact(0.0), &[2.0]).stationarity,
        0.0
    );
    assert_eq!(
        c.assess(&[0.0], Interval::exact(0.0), &[-2.0]).stationarity,
        1.0
    );
    let fixed = SmoothCertificate {
        bounds: Some((vec![0.0], vec![0.0])),
        ..c
    };
    assert_eq!(
        fixed
            .assess(&[0.0], Interval::exact(0.0), &[100.0])
            .stationarity,
        0.0
    );
    assert_eq!(
        fixed
            .assess(&[0.1], Interval::exact(0.0), &[0.0])
            .feasibility,
        0.1
    );
    assert_eq!(
        fixed
            .assess(&[0.1], Interval::exact(0.0), &[0.0])
            .target(1e-3, Eligibility::Eligible),
        TargetStatus::Missed
    );
}

#[test]
fn solution_set_distance_respects_symmetry_and_optional_identifiability() {
    let c = SmoothCertificate {
        solution_set: vec![vec![-1.0], vec![1.0]],
        ..certificate()
    };
    assert_eq!(
        c.assess(&[-1.0], Interval::exact(0.0), &[0.0])
            .parameter_error,
        Some(0.0)
    );
    let nonidentifiable = SmoothCertificate {
        solution_set: vec![],
        require_parameter: false,
        ..c
    };
    assert_eq!(
        nonidentifiable
            .assess(&[4.0], Interval::exact(0.0), &[0.0])
            .target(1e-6, Eligibility::Eligible),
        TargetStatus::Passed
    );
}

#[test]
fn flat_roots_need_position_and_exact_exits_still_need_valid_enclosures() {
    let c = RootCertificate {
        roots: vec![1.0],
        position_scale: 1.0,
        residual_scale: 1.0,
    };
    assert_eq!(
        c.target(1.01, 1e-10, None, false, 1e-6, Eligibility::Eligible),
        TargetStatus::Missed
    );
    assert_eq!(
        c.target(
            1.0,
            0.0,
            Some((-2.0, 4.0)),
            true,
            1e-6,
            Eligibility::Eligible
        ),
        TargetStatus::Passed
    );
    assert_eq!(
        c.target(
            1.0,
            0.0,
            Some((-2.0, 4.0)),
            false,
            1e-6,
            Eligibility::Eligible
        ),
        TargetStatus::Missed
    );
    for bracket in [(2.0, 3.0), (4.0, -2.0), (f64::NAN, 4.0)] {
        assert_eq!(
            c.target(
                1.0,
                0.0,
                Some(bracket),
                true,
                1e-6,
                Eligibility::Eligible
            ),
            TargetStatus::Missed
        );
    }
}

#[test]
fn fused_and_split_callbacks_reconcile_with_authoritative_categories() {
    for fused in [false, true] {
        let ledger = WorkLedger::new(10);
        let mut p = Quadratic::<f64>::new(ledger.clone());
        p.fused = fused;
        let mut problem = Problem::new(p);
        let (cost, gradient) =
            problem.cost_and_gradient(&vec![4.0, 3.0]).unwrap();
        assert_eq!(cost, 17.0);
        assert_eq!(gradient, vec![3.0, 5.0]);
        assert_eq!(problem.counts().cost_evals, 1);
        assert_eq!(problem.counts().gradient_evals, 1);
        assert_eq!(ledger.work(), if fused { 1 } else { 2 });
    }
}

#[test]
fn finite_differences_charge_each_leaf_and_abort_without_overshoot() {
    for (method, expected) in [(Method::Central, 4), (Method::Forward, 3)] {
        let ledger = WorkLedger::new(100);
        let mut problem = Problem::new(
            FiniteDiff::new(Quadratic::<f64>::new(ledger.clone()))
                .gradient_method(method),
        );
        let g = problem.gradient(&vec![4.0, 3.0]).unwrap();
        assert!((g[0] - 3.0).abs() < 1e-6 && (g[1] - 5.0).abs() < 1e-6);
        assert_eq!(problem.counts().gradient_evals, 1);
        assert_eq!(problem.counts().cost_evals, 0);
        assert_eq!(ledger.work(), expected);
        assert_eq!(ledger.snapshot().count(OracleKind::Cost), expected);
    }
    let ledger = WorkLedger::new(2);
    let mut problem =
        Problem::new(FiniteDiff::new(Quadratic::<f64>::new(ledger.clone())));
    assert_eq!(
        problem.gradient(&vec![4.0, 3.0]),
        Err(OracleError::Budget { cap: 2 })
    );
    assert_eq!(ledger.work(), 2);
    assert!(ledger.snapshot().denied >= 1);
    assert_eq!(problem.counts().gradient_evals, 1);
}

#[test]
fn native_f32_differentiation_and_exact_witness_are_not_cast_f64_solves() {
    let ledger = WorkLedger::new(100);
    let fixture = Quadratic::<f32>::new(ledger.clone());
    assert_eq!(fixture.cost(&vec![1.0, -2.0]).unwrap(), 0.0_f32);
    assert_eq!(
        fixture.gradient(&vec![1.0, -2.0]).unwrap(),
        vec![0.0_f32; 2]
    );
    let fd =
        BoundedFiniteDiff::new(fixture, vec![-8.0_f32; 2], vec![8.0_f32; 2]);
    let mut problem = Problem::new(fd);
    let gradient = problem.gradient(&vec![4.0, 3.0]).unwrap();
    assert!(
        (gradient[0] - 3.0).abs() < 1e-3 && (gradient[1] - 5.0).abs() < 1e-3
    );
    assert_eq!(problem.counts().gradient_evals, 1);
    // BoundedFiniteDiff evaluates the base once, then both sides of each column.
    assert_eq!(ledger.work(), 7);
}

#[test]
fn batches_charge_points_and_record_submitted_work_separately() {
    let ledger = WorkLedger::new(2);
    let mut problem = Problem::new(Quadratic::<f64>::new(ledger.clone()));
    let points = vec![vec![4.0, 3.0], vec![1.0, -2.0], vec![0.0, 0.0]];
    assert!(matches!(
        problem.cost_batch(&points),
        Err(OracleError::Budget { cap: 2 })
    ));
    assert_eq!(problem.counts().cost_evals, 3);
    assert_eq!(ledger.work(), 2);
    assert_eq!(ledger.snapshot().denied, 1);
}

#[test]
fn inner_initialization_uses_the_remaining_outer_budget() {
    let ledger = WorkLedger::new(3);
    let mut outer = Problem::new(Quadratic::<f64>::new(ledger.clone()));
    outer.cost(&vec![4.0, 3.0]).unwrap();
    let inner = ledger.in_scope("inner");
    let measured = measure(
        Executor::from_start(
            Quadratic::<f64>::new(inner),
            NelderMead::new(),
            vec![4.0, 3.0],
        ),
        &ledger,
        Duration::from_secs(60),
    );
    assert!(matches!(
        measured.outcome,
        RunOutcome::InitializationError(OracleError::Budget { cap: 3 })
    ));
    assert_eq!(ledger.work(), 3);
    assert_eq!(
        ledger
            .snapshot()
            .calls
            .iter()
            .filter(|c| c.scope == "inner")
            .count(),
        2
    );
}

#[test]
fn failed_calls_cache_hits_and_nested_wrappers_share_one_cap() {
    let ledger = WorkLedger::new(3);
    let inner = ledger.in_scope("inner");
    let mut outer = Problem::new(Quadratic::<f64>::new(ledger.clone()));
    let mut nested = Problem::new(Quadratic::<f64>::new(inner.clone()));
    outer.cost(&vec![4.0, 3.0]).unwrap();
    nested.cost_and_gradient(&vec![1.0, -2.0]).unwrap();
    outer.counts_mut().add(nested.counts());
    inner.cache_hit();
    let failure: Result<(), _> =
        inner.evaluate(OracleKind::Cost, &[0.0], || {
            Err(OracleError::Callback("domain"))
        });
    assert_eq!(failure, Err(OracleError::Callback("domain")));
    assert_eq!(outer.counts().cost_evals, 2);
    assert_eq!(outer.counts().gradient_evals, 1);
    assert_eq!(ledger.work(), 3);
    assert_eq!(ledger.snapshot().cache_hits, 1);
    assert_eq!(ledger.snapshot().calls[1].scope, "inner");
    assert_eq!(
        inner.evaluate(OracleKind::Gradient, &[], || Ok(((), None))),
        Err(OracleError::Budget { cap: 3 })
    );
}

struct FusedResidual(WorkLedger);
impl Residual for FusedResidual {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = OracleError;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, OracleError> {
        self.0
            .evaluate(OracleKind::Residual, x, || Ok((vec![x[0]; 20], None)))
    }
}
impl Jacobian for FusedResidual {
    type Jacobian = Vec<Vec<f64>>;
    fn jacobian(&self, x: &Vec<f64>) -> Result<Self::Jacobian, OracleError> {
        self.0.evaluate(OracleKind::Jacobian, x, || {
            Ok((vec![vec![1.0]; 20], None))
        })
    }
    fn residual_and_jacobian(
        &self,
        x: &Vec<f64>,
    ) -> Result<(Vec<f64>, Self::Jacobian), OracleError> {
        self.0.evaluate(OracleKind::ResidualJacobian, x, || {
            Ok(((vec![x[0]; 20], vec![vec![1.0]; 20]), None))
        })
    }
}

#[test]
fn residual_vectors_and_fused_jacobians_are_not_charged_per_observation() {
    let ledger = WorkLedger::new(5);
    let mut problem = Problem::new(FusedResidual(ledger.clone()));
    problem.residual_and_jacobian(&vec![2.0]).unwrap();
    assert_eq!(ledger.work(), 1);
    assert_eq!(problem.counts().residual_evals, 1);
    assert_eq!(problem.counts().jacobian_evals, 1);
}

#[test]
fn initialization_exhaustion_never_invents_a_returned_point() {
    for cap in [0, 1, 2] {
        let ledger = WorkLedger::new(cap);
        let p = Quadratic::<f64>::new(ledger.clone());
        let measured = measure(
            Executor::from_start(p, NelderMead::new(), vec![4.0, 3.0])
                .max_iter(10),
            &ledger,
            Duration::from_secs(60),
        );
        assert!(matches!(
            measured.outcome,
            RunOutcome::InitializationError(OracleError::Budget { .. })
        ));
        assert!(
            measured.returned().is_none()
                && measured.last_published().is_none()
        );
        assert!(measured.counts.is_none());
        assert_eq!(measured.ledger.work(), cap);
    }
}

#[test]
fn interrupted_steps_retain_history_but_do_not_claim_a_final_checkpoint() {
    let ledger = WorkLedger::new(4);
    let p = Quadratic::<f64>::new(ledger.clone());
    let measured = measure(
        Executor::from_start(p, NelderMead::new(), vec![4.0, 3.0]).max_iter(10),
        &ledger,
        Duration::from_secs(60),
    );
    assert!(matches!(
        measured.outcome,
        RunOutcome::StepError(OracleError::Budget { cap: 4 })
    ));
    assert!(measured.returned().is_none());
    assert_eq!(measured.last_published().unwrap().work, 3);
    assert!(measured.at_budget(2).is_none());
    assert_eq!(measured.at_budget(4).unwrap().work, 3);
    assert_eq!(measured.counts.unwrap().cost_evals, 5);
    assert_eq!(measured.ledger.work(), 4);
}

struct RejectTrial;
impl Solver<Quadratic, PointState<Vec<f64>>> for RejectTrial {
    type Error = OracleError;
    fn init(
        &mut self,
        p: &mut Problem<Quadratic>,
        mut state: PointState<Vec<f64>>,
    ) -> Result<PointState<Vec<f64>>, OracleError> {
        let cost = p.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        Ok(state)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<Quadratic>,
        state: PointState<Vec<f64>>,
    ) -> Result<SolverStep<PointState<Vec<f64>>>, OracleError> {
        p.cost(&vec![1.0, -2.0])?;
        Ok(SolverStep::stopped(
            state,
            Termination::Stalled(Stall::Numerical {
                message: "analytic rejected-trial fixture".into(),
                measurements: vec![],
            }),
        ))
    }
}

#[test]
fn rejected_trial_quality_cannot_replace_the_returned_recommendation() {
    let ledger = WorkLedger::new(10);
    let p = Quadratic::new(ledger.clone());
    let c = p.certificate(&[4.0, 3.0]);
    let measured = measure(
        Executor::new(p.clone(), RejectTrial, PointState::new(vec![4.0, 3.0])),
        &ledger,
        Duration::from_secs(60),
    );
    let returned = measured.returned().unwrap();
    assert_eq!(returned.point, vec![4.0, 3.0]);
    assert_eq!(
        returned.stage,
        PublicationStage::Stop(TerminationStage::Step { completed: false })
    );
    assert_eq!(returned.iteration, 0);
    assert_eq!(
        measured.ledger.calls.last().unwrap().sampled_cost,
        Some(0.0)
    );
    let (f, g) = p.verify(&returned.point);
    assert_eq!(
        c.assess(&returned.point, f, &g)
            .target(1e-6, Eligibility::Eligible),
        TargetStatus::Missed
    );
    assert_eq!(ledger.work(), 2);
}

#[test]
fn actual_lbfgsb_reports_stop_evidence_and_verification_adds_no_solve_work() {
    let ledger = WorkLedger::new(6000);
    let p = Quadratic::<f64>::new(ledger.clone());
    let c = p.certificate(&[4.0, 3.0]);
    let measured = measure(
        Executor::new(
            p.clone(),
            Lbfgs::new(),
            FirstOrderState::new(vec![4.0, 3.0]),
        )
        .max_iter(10000),
        &ledger,
        Duration::from_secs(60),
    );
    let RunOutcome::Stopped(report) = &measured.outcome else {
        panic!("{:?}", measured.outcome)
    };
    let Termination::Converged(criteria) = &report.termination else {
        panic!("{report:?}")
    };
    assert!(!criteria.criteria().is_empty());
    let returned = measured.returned().unwrap();
    let work = ledger.work();
    let (f, g) = p.verify(&returned.point);
    assert_eq!(
        c.assess(&returned.point, f, &g)
            .target(1e-6, Eligibility::Eligible),
        TargetStatus::Passed
    );
    assert_eq!(ledger.work(), work);
    let physical = ledger.snapshot();
    let counts = measured.counts.unwrap();
    assert_eq!(
        counts.cost_evals,
        physical.count(OracleKind::Cost)
            + physical.count(OracleKind::CostGradient)
    );
    assert_eq!(
        counts.gradient_evals,
        physical.count(OracleKind::Gradient)
            + physical.count(OracleKind::CostGradient)
    );
}
