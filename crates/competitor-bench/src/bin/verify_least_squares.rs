//! Fixed-policy development pilot; independent verification runs afterward.
use basin::{
    ConvergenceEvidence, ConvergenceTest, Executor, LevenbergMarquardt,
    LmDamping, PointState, RobustLeastSquares, Scalar, Termination, Trf,
    TrustRegionReflective,
};
use competitor_bench::convergence::{
    least_squares::{
        AnalyticModel, Instrumented, Measured, ROBUST_FIXTURES,
        RobustStoppingPolicy, measure,
    },
    ledger::WorkLedger,
    nist::datasets,
    runner::RunOutcome,
};
#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix, DVector};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    time::Duration,
};

type Out = BufWriter<std::fs::File>;
struct Files {
    runs: Out,
    points: Out,
    leaves: Out,
    native: Out,
    checks: Out,
}
fn row(out: &mut Out, fields: &[String]) -> std::io::Result<()> {
    for (i, field) in fields.iter().enumerate() {
        if i != 0 {
            write!(out, ",")?;
        }
        write!(out, "\"{}\"", field.replace('"', "\"\""))?;
    }
    writeln!(out)
}
fn scalar<F: Scalar>(v: F) -> String {
    v.to_f64().unwrap().to_string()
}
fn opt<F: Scalar>(v: Option<F>) -> String {
    v.map(scalar).unwrap_or_default()
}
fn vector<F: Scalar>(v: &[F]) -> String {
    v.iter().map(|v| scalar(*v)).collect::<Vec<_>>().join(";")
}
fn predicate(test: &ConvergenceTest) -> &'static str {
    match test {
        ConvergenceTest::AbsoluteGradientInfinity => "absolute_gradient",
        ConvergenceTest::GradientOrthogonality => "orthogonality",
        ConvergenceTest::RobustGradientOrthogonality => "robust_orthogonality",
        ConvergenceTest::RelativeModelReduction => "model_reduction",
        ConvergenceTest::RelativeTrialStep => "trial_step",
        ConvergenceTest::RelativeTrustRadius => "trust_radius",
        ConvergenceTest::AbsoluteScaledGradient => "scaled_gradient",
        ConvergenceTest::NoFreeParameters => "no_free_parameters",
        ConvergenceTest::Custom { key, .. }
            if key == "trf.legacy_scaled_gradient" =>
        {
            "scaled_gradient"
        }
        _ => "unexpected",
    }
}
struct Labels<'a> {
    id: &'a str,
    dataset: &'a str,
    precision: &'a str,
    start: usize,
    route: &'a str,
    policy: &'a str,
}
impl Files {
    fn new(dir: &Path) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(dir)?;
        let open = |name, header| -> Result<Out, Box<dyn Error>> {
            let mut out = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(dir.join(name))?,
            );
            writeln!(out, "{header}")?;
            Ok(out)
        };
        Ok(Self {
            runs: open(
                "runs.csv",
                "id,dataset,precision,start,route,policy,cap,outcome,stage,criteria,work,denied,cost_evals,residual_evals,jacobian_evals,elapsed_seconds,last_publication,returned,detail",
            )?,
            points: open(
                "publications.csv",
                "id,publication,stage,iteration,work,cost,point",
            )?,
            leaves: open("leaves.csv", "id,work,kind,outcome,point")?,
            native: open(
                "native.csv",
                "id,sequence,starting_iteration,work_before,residual_work,published,trial,base_cost,trial_cost,actual,predicted,ratio,accepted,damping,radius_before,radius_after,step,gradient,diagonal,curvature,coordinate_scale,free,model_solves",
            )?,
            checks: open(
                "checks.csv",
                "id,sequence,name,tolerance,passed,evidence,value,value_exponent,bound,bound_exponent,reference,reference_exponent,actual,predicted,reference_cost,ratio",
            )?,
        })
    }
    fn emit<F: Scalar>(
        &mut self,
        labels: Labels<'_>,
        measured: Measured<F>,
    ) -> Result<(), Box<dyn Error>> {
        let Labels {
            id,
            dataset,
            precision,
            start,
            route,
            policy,
        } = labels;
        let run = measured.run;
        let (outcome, stage, criteria, detail) = match &run.outcome {
            RunOutcome::Stopped(report) => {
                let outcome = match report.termination {
                    Termination::Converged(_) => "converged",
                    Termination::Limit(_) => "limit",
                    Termination::Stalled(_) => "stalled",
                    Termination::Failed(_) => "failed",
                    _ => "other",
                };
                let criteria =
                    if let Termination::Converged(c) = &report.termination {
                        c.criteria()
                            .iter()
                            .map(|c| predicate(&c.test))
                            .collect::<Vec<_>>()
                            .join(";")
                    } else {
                        String::new()
                    };
                (
                    outcome,
                    format!("{:?}", report.stage),
                    criteria,
                    format!("{:?}", report.termination),
                )
            }
            RunOutcome::InitializationError(e) => (
                "initialization_error",
                String::new(),
                String::new(),
                e.to_string(),
            ),
            RunOutcome::StepError(e) => (
                "callback_error",
                String::new(),
                String::new(),
                e.to_string(),
            ),
            RunOutcome::WallLimit => {
                ("wall_limit", String::new(), String::new(), String::new())
            }
        };
        let returned = run.returned().is_some();
        row(
            &mut self.runs,
            &[
                id.into(),
                dataset.into(),
                precision.into(),
                start.to_string(),
                route.into(),
                policy.into(),
                run.ledger.cap.to_string(),
                outcome.into(),
                stage,
                criteria,
                run.ledger.work().to_string(),
                run.ledger.denied.to_string(),
                run.counts
                    .map(|c| c.cost_evals.to_string())
                    .unwrap_or_default(),
                run.counts
                    .map(|c| c.residual_evals.to_string())
                    .unwrap_or_default(),
                run.counts
                    .map(|c| c.jacobian_evals.to_string())
                    .unwrap_or_default(),
                run.elapsed.as_secs_f64().to_string(),
                run.recommendations
                    .len()
                    .checked_sub(1)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                returned.to_string(),
                detail,
            ],
        )?;
        for (i, p) in run.recommendations.iter().enumerate() {
            row(
                &mut self.points,
                &[
                    id.into(),
                    i.to_string(),
                    format!("{:?}", p.stage),
                    p.iteration.to_string(),
                    p.work.to_string(),
                    p.solver_cost.to_string(),
                    vector(&p.point),
                ],
            )?;
        }
        for call in &run.ledger.calls {
            row(
                &mut self.leaves,
                &[
                    id.into(),
                    call.work.to_string(),
                    format!("{:?}", call.kind),
                    format!("{:?}", call.outcome),
                    vector(&call.point),
                ],
            )?;
        }
        for record in measured.native {
            let o = record.observation;
            row(
                &mut self.native,
                &[
                    id.into(),
                    o.sequence.to_string(),
                    record.starting_iteration.to_string(),
                    record.work_before.to_string(),
                    record
                        .residual_work
                        .map(|v| v.to_string())
                        .unwrap_or_default(),
                    record.published.to_string(),
                    o.trial.to_string(),
                    scalar(o.base_cost),
                    opt(o.trial_cost),
                    opt(o.actual_reduction),
                    opt(o.predicted_reduction),
                    opt(o.gain_ratio),
                    o.accepted.map(|v| v.to_string()).unwrap_or_default(),
                    opt(o.damping),
                    opt(o.radius_before),
                    opt(o.radius_after),
                    vector(&o.step),
                    vector(&o.gradient),
                    vector(&o.diagonal),
                    vector(&o.curvature),
                    vector(&o.coordinate_scale),
                    o.free
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(";"),
                    o.model_solves.to_string(),
                ],
            )?;
            for check in o.checks {
                let mut fields = vec![
                    id.into(),
                    o.sequence.to_string(),
                    if dataset.starts_with("robust_")
                        && check.name == "orthogonality"
                    {
                        "robust_orthogonality".into()
                    } else {
                        check.name.into()
                    },
                    opt(check.tolerance),
                    check.passed.map(|p| p.to_string()).unwrap_or_default(),
                ];
                let mut evidence = vec![String::new(); 11];
                match check.evidence {
                    Some(ConvergenceEvidence::UpperBound {
                        value,
                        bound,
                        reference,
                        ..
                    }) => {
                        evidence[0] = "upper_bound".into();
                        evidence[1] = scalar(value);
                        evidence[3] = scalar(bound);
                        evidence[5] = opt(reference);
                    }
                    Some(ConvergenceEvidence::FactoredUpperBound {
                        value,
                        bound,
                        reference,
                        ..
                    }) => {
                        evidence[0] = "factored_upper_bound".into();
                        evidence[1] = scalar(value.significand);
                        evidence[2] = value.exponent.to_string();
                        evidence[3] = scalar(bound.significand);
                        evidence[4] = bound.exponent.to_string();
                        if let Some(r) = reference {
                            evidence[5] = scalar(r.significand);
                            evidence[6] = r.exponent.to_string();
                        }
                    }
                    Some(ConvergenceEvidence::ModelReduction {
                        actual,
                        predicted,
                        reference_cost,
                        gain_ratio,
                        ..
                    }) => {
                        evidence[0] = "model_reduction".into();
                        evidence[7] = scalar(actual);
                        evidence[8] = scalar(predicted);
                        evidence[9] = scalar(reference_cost);
                        evidence[10] = scalar(gain_ratio);
                    }
                    Some(ConvergenceEvidence::NoFreeParameters) => {
                        evidence[0] = "no_free_parameters".into()
                    }
                    None => {}
                    _ => return Err("unexpected evidence variant".into()),
                }
                fields.extend(evidence);
                row(&mut self.checks, &fields)?;
            }
        }
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.runs.flush()?;
        self.points.flush()?;
        self.leaves.flush()?;
        self.native.flush()?;
        self.checks.flush()
    }
}

macro_rules! precision {
    ($function:ident, $f:ty) => {
        fn $function(files: &mut Files, dataset: &str, start: usize, x: Vec<$f>, problem: Instrumented<$f>, cap: u64, policy: &str) -> Result<(), Box<dyn Error>> {
            let precision = stringify!($f);
            macro_rules! route {
                ($name:literal, $solver:expr) => {{
                    let ledger = WorkLedger::new(cap);
                    let mut p = problem.clone(); p.ledger = ledger.clone();
                    let solver = $solver.with_trial_diagnostics(true);
                    let state = PointState::new(DVector::from_vec(x.clone()));
                    let result = if policy.starts_with("robust_") {
                        let fixture = ROBUST_FIXTURES.iter().find(|f| f.name == dataset).unwrap();
                        let p = RobustLeastSquares::new(p, fixture.loss).with_scale(fixture.scale as $f);
                        measure(Executor::new(p, solver, state).max_iter(10_000), &ledger, Duration::from_secs(600))
                    } else {
                        measure(Executor::new(p, solver, state).max_iter(10_000), &ledger, Duration::from_secs(600))
                    };
                    let id = format!("{dataset}-{precision}-{start}-{}-{policy}-{cap}", $name);
                    files.emit(Labels { id: &id, dataset, precision, start, route: $name, policy }, result)?;
                }};
            }
            if dataset == "robust_huber_bound" {
                route!("trf_legacy", Trf::<DVector<$f>, DMatrix<$f>, $f>::default());
                route!("trf_full", TrustRegionReflective::<$f>::new());
                return Ok(());
            }
            if policy == "bounded_default" {
                route!("trf_full", TrustRegionReflective::<$f>::new());
                return Ok(());
            }
            let lm = |damping| {
                let mut s = LevenbergMarquardt::<DVector<$f>, DMatrix<$f>, $f>::default().with_damping(damping);
                let tol: $f = if precision == "f32" { 1e-4 } else { 1e-8 };
                if let Some(probe) = RobustStoppingPolicy::from_name(policy) {
                    s = probe.configure(s, tol);
                } else if policy == "relative_probe" {
                    s = s.with_absolute_gradient_tolerance(0.0).with_gradient_orthogonality_tolerance(tol)
                        .with_relative_model_reduction_tolerance(tol).with_relative_step_tolerance(tol)
                        .with_relative_trust_radius_tolerance(tol);
                }
                s
            };
            route!("lm_normal_nielsen", lm(LmDamping::Nielsen));
            route!("lm_normal_trust", lm(LmDamping::TrustRegion));
            route!("lm_qr_nielsen", lm(LmDamping::Nielsen).with_pivoted_qr());
            route!("lm_qr_trust", lm(LmDamping::TrustRegion).with_pivoted_qr());
            if policy != "relative_probe" && RobustStoppingPolicy::from_name(policy).is_none() {
                route!("trf_legacy", Trf::<DVector<$f>, DMatrix<$f>, $f>::default());
                route!("trf_full", TrustRegionReflective::<$f>::new());
            }
            Ok(())
        }
    };
}
precision!(run64, f64);
precision!(run32, f32);

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5
        || args[1] != "--phase"
        || args[3] != "--output"
        || !matches!(
            args[2].as_str(),
            "analytic" | "bounded" | "robust" | "robust-stopping" | "nist"
        )
    {
        return Err("usage: verify_least_squares --phase analytic|bounded|robust|robust-stopping|nist --output <new-directory>".into());
    }
    let mut files = Files::new(Path::new(&args[4]))?;
    if args[2] == "analytic" {
        for (name, model) in [
            ("linear", AnalyticModel::Linear),
            ("nonzero", AnalyticModel::Nonzero),
            ("nonfinite", AnalyticModel::NonFiniteTrial),
        ] {
            run64(
                &mut files,
                name,
                1,
                vec![0.1],
                Instrumented::analytic(model, WorkLedger::new(4000)),
                4000,
                "default",
            )?;
            run32(
                &mut files,
                name,
                1,
                vec![0.1],
                Instrumented::analytic(model, WorkLedger::new(4000)),
                4000,
                "default",
            )?;
        }
        for cap in 0..=2 {
            run64(
                &mut files,
                "linear_budget",
                1,
                vec![0.1],
                Instrumented::analytic(
                    AnalyticModel::Linear,
                    WorkLedger::new(cap),
                ),
                cap,
                "default",
            )?;
            run32(
                &mut files,
                "linear_budget",
                1,
                vec![0.1],
                Instrumented::analytic(
                    AnalyticModel::Linear,
                    WorkLedger::new(cap),
                ),
                cap,
                "default",
            )?;
        }
        run64(
            &mut files,
            "nonzero",
            1,
            vec![0.1],
            Instrumented::analytic(
                AnalyticModel::Nonzero,
                WorkLedger::new(4000),
            ),
            4000,
            "relative_probe",
        )?;
        run32(
            &mut files,
            "nonzero",
            1,
            vec![0.1],
            Instrumented::analytic(
                AnalyticModel::Nonzero,
                WorkLedger::new(4000),
            ),
            4000,
            "relative_probe",
        )?;
    } else if args[2] == "bounded" {
        for (name, lower, upper, x) in [
            ("box_active", vec![0., -0.5], vec![0.5, 0.], vec![0.1, -0.1]),
            (
                "box_mixed_fixed",
                vec![0.25, -2.],
                vec![0.25, 0.],
                vec![0.25, -0.1],
            ),
            (
                "box_all_fixed",
                vec![0.25, -0.25],
                vec![0.25, -0.25],
                vec![0.25, -0.25],
            ),
            (
                "box_stationary",
                vec![-1., -2.],
                vec![1., 0.],
                vec![0., -1.],
            ),
        ] {
            let model = if name == "box_stationary" {
                AnalyticModel::BoxStationary
            } else {
                AnalyticModel::BoxLinear
            };
            for cap in [0, 1, 2, 4000] {
                run64(
                    &mut files,
                    name,
                    1,
                    x.clone(),
                    Instrumented::analytic(model, WorkLedger::new(cap))
                        .with_bounds(lower.clone(), upper.clone()),
                    cap,
                    "bounded_default",
                )?;
                run32(
                    &mut files,
                    name,
                    1,
                    x.iter().map(|v| *v as f32).collect(),
                    Instrumented::analytic(model, WorkLedger::new(cap))
                        .with_bounds(
                            lower.iter().map(|v| *v as f32).collect(),
                            upper.iter().map(|v| *v as f32).collect(),
                        ),
                    cap,
                    "bounded_default",
                )?;
            }
        }
    } else if args[2] == "robust-stopping" {
        for fixture in ROBUST_FIXTURES.iter().filter(|f| f.bounds.is_none()) {
            for policy in RobustStoppingPolicy::ALL {
                let cap = 4000;
                run64(
                    &mut files,
                    fixture.name,
                    1,
                    vec![fixture.start],
                    Instrumented::analytic(fixture.model, WorkLedger::new(cap)),
                    cap,
                    policy.name(),
                )?;
                run32(
                    &mut files,
                    fixture.name,
                    1,
                    vec![fixture.start as f32],
                    Instrumented::analytic(fixture.model, WorkLedger::new(cap)),
                    cap,
                    policy.name(),
                )?;
            }
        }
    } else if args[2] == "robust" {
        for fixture in ROBUST_FIXTURES {
            let policies: &[&str] = if fixture.bounds.is_some() {
                &["robust_default"]
            } else {
                &["robust_default", "robust_relative_probe"]
            };
            for &policy in policies {
                let caps: &[u64] = if policy == "robust_default" {
                    &[0, 1, 2, 4000]
                } else {
                    &[4000]
                };
                for &cap in caps {
                    let mut p64 = Instrumented::analytic(
                        fixture.model,
                        WorkLedger::new(cap),
                    );
                    let mut p32 = Instrumented::analytic(
                        fixture.model,
                        WorkLedger::new(cap),
                    );
                    if let Some((lo, hi)) = fixture.bounds {
                        p64 = p64.with_bounds(vec![lo], vec![hi]);
                        p32 = p32.with_bounds(vec![lo as f32], vec![hi as f32]);
                    }
                    run64(
                        &mut files,
                        fixture.name,
                        1,
                        vec![fixture.start],
                        p64,
                        cap,
                        policy,
                    )?;
                    run32(
                        &mut files,
                        fixture.name,
                        1,
                        vec![fixture.start as f32],
                        p32,
                        cap,
                        policy,
                    )?;
                }
            }
        }
    } else {
        for dataset in datasets()?
            .into_iter()
            .filter(|d| d.partition == "development")
        {
            for start in 0..2 {
                let cap = 2000 * (dataset.reference.len() as u64 + 1);
                run64(
                    &mut files,
                    dataset.id,
                    start + 1,
                    dataset.start::<f64>(start),
                    Instrumented::nist(dataset.clone(), WorkLedger::new(cap)),
                    cap,
                    "default",
                )?;
                run32(
                    &mut files,
                    dataset.id,
                    start + 1,
                    dataset.start::<f32>(start),
                    Instrumented::nist(dataset.clone(), WorkLedger::new(cap)),
                    cap,
                    "default",
                )?;
            }
            eprintln!("measured {}", dataset.id);
        }
    }
    files.flush()?;
    Ok(())
}
