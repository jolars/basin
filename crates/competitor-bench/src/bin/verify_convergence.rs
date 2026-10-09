//! First CDP-1 analytic measurement validation, not default calibration.
//! Run with `--output-dir target/convergence-defaults/<run-id>` in release mode.
//! The CSV files distinguish leaf trials, published recommendations, and returns.

use std::{
    fs::{self, File},
    io::{self, Write},
    path::Path,
    time::Duration,
};

use basin::core::numdiff::Method;
use basin::{
    Executor, FiniteDiff, FirstOrderState, Gradient, Lbfgs, NelderMead, Scalar,
};
use competitor_bench::convergence::{
    fixtures::Quadratic,
    ledger::WorkLedger,
    quality::{Eligibility, TargetStatus},
    runner::{Measurement, measure},
};

fn quoted(value: impl std::fmt::Debug) -> String {
    format!("\"{}\"", format!("{value:?}").replace('"', "\"\""))
}

fn preflight<F: Scalar>() {
    let p = Quadratic::<F>::new(WorkLedger::new(2));
    let witness = vec![F::one(), F::from_f64(-2.0).unwrap()];
    let (cost, gradient) = p.cost_and_gradient(&witness).unwrap();
    assert_eq!(cost, F::zero());
    assert!(gradient.iter().all(|g| *g == F::zero()));
    let c =
        p.certificate(&[F::from_f64(4.0).unwrap(), F::from_f64(3.0).unwrap()]);
    let (f, g) = p.verify(&[1.0, -2.0]);
    // The dyadic optimum has zero arithmetic/reference uncertainty in both
    // native precisions. Other model certificates cannot reuse this claim.
    assert_eq!(f.lower, 0.0);
    assert_eq!(f.upper, 0.0);
    assert_eq!(
        c.assess(&[1.0, -2.0], f, &g)
            .target(1e-12, Eligibility::Eligible),
        TargetStatus::Passed
    );
}

fn emit<F: Scalar>(
    directory: &Path,
    id: &str,
    precision: &str,
    fixture: &Quadratic<F>,
    start: &[F],
    measured: &Measurement<F>,
    summary: &mut File,
) -> io::Result<()> {
    let c = fixture.certificate(start);
    let q = if precision == "f32" { 1e-3 } else { 1e-6 };
    let mut recommendations =
        File::create(directory.join(format!("{id}-recommendations.csv")))?;
    writeln!(
        recommendations,
        "schema,work,iteration,stage,point,solver_cost,objective_error,stationarity,feasibility,parameter_error,target_status,cost_evals,gradient_evals,residual_evals,jacobian_evals,hessian_evals,hessian_product_evals"
    )?;
    let mut verification_calls = 0;
    let mut first_attainment = None;
    let mut last_quality = None;
    for r in &measured.recommendations {
        let (value, gradient) = fixture.verify(&r.point);
        verification_calls += 1;
        let quality = c.assess(&r.point, value, &gradient);
        let status = quality.target(q, Eligibility::Eligible);
        if first_attainment.is_none() && status == TargetStatus::Passed {
            first_attainment = Some(r.work);
        }
        let counts = &r.counts;
        writeln!(
            recommendations,
            "1,{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{:?},{},{},{},{},{},{}",
            r.work,
            r.iteration,
            quoted(r.stage),
            quoted(&r.point),
            r.solver_cost,
            quality.objective_error,
            quality.stationarity,
            quality.feasibility,
            quoted(quality.parameter_error),
            status,
            counts.cost_evals,
            counts.gradient_evals,
            counts.residual_evals,
            counts.jacobian_evals,
            counts.hessian_evals,
            counts.hessian_product_evals
        )?;
        last_quality = Some(status);
    }
    let mut leaves = File::create(directory.join(format!("{id}-leaves.csv")))?;
    writeln!(
        leaves,
        "schema,work,scope,kind,point,sampled_solver_cost,outcome,sampled_target_status"
    )?;
    for leaf in &measured.ledger.calls {
        let sampled_status = leaf.sampled_cost.map(|_| {
            let (value, gradient) = fixture.verify(&leaf.point);
            verification_calls += 1;
            c.assess(&leaf.point, value, &gradient)
                .target(q, Eligibility::Eligible)
        });
        writeln!(
            leaves,
            "1,{},{},{:?},{},{},{},{}",
            leaf.work,
            leaf.scope,
            leaf.kind,
            quoted(&leaf.point),
            quoted(leaf.sampled_cost),
            quoted(&leaf.outcome),
            quoted(sampled_status)
        )?;
    }
    let returned_status = measured.returned().and(last_quality);
    writeln!(
        summary,
        "1,{id},{precision},{},{},{},{},{},{},{},{},{},{},{}",
        measured.ledger.cap,
        measured.ledger.work(),
        measured.ledger.denied,
        quoted(&measured.outcome),
        quoted(measured.counts),
        quoted(returned_status),
        quoted(last_quality),
        quoted(first_attainment),
        verification_calls,
        measured.elapsed.as_nanos(),
        quoted(measured.last_published().map(|r| &r.point))
    )?;
    println!(
        "{id}: W={}, returned={returned_status:?}, last_published={last_quality:?}, outcome={:?}",
        measured.ledger.work(),
        measured.outcome
    );
    Ok(())
}

fn analytic_f64(directory: &Path, summary: &mut File) -> io::Result<()> {
    let start = vec![4.0, 3.0];
    let ledger = WorkLedger::new(6000);
    let p = Quadratic::<f64>::new(ledger.clone());
    let nm = NelderMead::new()
        .with_absolute_simplex_size_tolerance(1e-6)
        .with_absolute_simplex_cost_tolerance(1e-6);
    let m = measure(
        Executor::from_start(p.clone(), nm, start.clone()).max_iter(10000),
        &ledger,
        Duration::from_secs(60),
    );
    emit(directory, "nm-f64-paired", "f64", &p, &start, &m, summary)?;

    let ledger = WorkLedger::new(6000);
    let p = Quadratic::<f64>::new(ledger.clone());
    let m = measure(
        Executor::new(
            p.clone(),
            Lbfgs::new(),
            FirstOrderState::new(start.clone()),
        )
        .max_iter(10000),
        &ledger,
        Duration::from_secs(60),
    );
    emit(
        directory,
        "lbfgsb-f64-default",
        "f64",
        &p,
        &start,
        &m,
        summary,
    )?;

    for (name, method) in
        [("central", Method::Central), ("forward", Method::Forward)]
    {
        let ledger = WorkLedger::new(6000);
        let p = Quadratic::<f64>::new(ledger.clone());
        let fd = FiniteDiff::new(p.clone()).gradient_method(method);
        let m = measure(
            Executor::new(
                fd,
                Lbfgs::new(),
                FirstOrderState::new(start.clone()),
            )
            .max_iter(10000),
            &ledger,
            Duration::from_secs(60),
        );
        emit(
            directory,
            &format!("lbfgsb-f64-{name}"),
            "f64",
            &p,
            &start,
            &m,
            summary,
        )?;
    }
    for (id, cap) in [
        ("nm-f64-default", 6000),
        ("nm-f64-init-cap", 2),
        ("nm-f64-step-cap", 4),
    ] {
        let ledger = WorkLedger::new(cap);
        let p = Quadratic::<f64>::new(ledger.clone());
        let m = measure(
            Executor::from_start(p.clone(), NelderMead::new(), start.clone())
                .max_iter(10000),
            &ledger,
            Duration::from_secs(60),
        );
        emit(directory, id, "f64", &p, &start, &m, summary)?;
    }
    Ok(())
}

fn analytic_f32(directory: &Path, summary: &mut File) -> io::Result<()> {
    let start = vec![4.0_f32, 3.0];
    let ledger = WorkLedger::new(6000);
    let p = Quadratic::<f32>::new(ledger.clone());
    let nm = NelderMead::new()
        .with_absolute_simplex_size_tolerance(1e-3_f32)
        .with_absolute_simplex_cost_tolerance(1e-3_f32);
    let m = measure(
        Executor::from_start(p.clone(), nm, start.clone()).max_iter(10000),
        &ledger,
        Duration::from_secs(60),
    );
    emit(directory, "nm-f32-paired", "f32", &p, &start, &m, summary)?;

    let ledger = WorkLedger::new(6000);
    let p = Quadratic::<f32>::new(ledger.clone());
    let m = measure(
        Executor::new(
            p.clone(),
            Lbfgs::new(),
            FirstOrderState::new(start.clone()),
        )
        .max_iter(10000),
        &ledger,
        Duration::from_secs(60),
    );
    emit(
        directory,
        "lbfgsb-f32-default",
        "f32",
        &p,
        &start,
        &m,
        summary,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("--output-dir") {
        return Err(
            "usage: verify_convergence --output-dir <new-directory>".into()
        );
    }
    let directory = args.next().ok_or("missing output directory")?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let directory = Path::new(&directory);
    if directory.exists() {
        return Err(
            "output directory already exists; retain earlier attempts".into()
        );
    }
    fs::create_dir_all(directory)?;
    preflight::<f64>();
    preflight::<f32>();
    let mut summary = File::create(directory.join("summary.csv"))?;
    writeln!(
        summary,
        "schema,case,precision,cap,physical_work,denied,outcome,logical_counts,returned_target_status,last_published_target_status,first_attainment_work,verification_calls,instrumented_elapsed_ns,last_published_point"
    )?;
    analytic_f64(directory, &mut summary)?;
    analytic_f32(directory, &mut summary)?;
    Ok(())
}
