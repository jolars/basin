//! Outer-boundary records. Internal rejected trials are not reconstructed.

use super::Work;
use basin::{
    CountsMirror, EvalCounts, Executor, Scalar, Solver, State, StepOutcome,
};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct Quality {
    pub param: Vec<f64>,
    pub cost: f64,
    pub difference: f64,
    pub gradient_inf: f64,
    pub violation: f64,
    pub parameter_error: f64,
}

#[derive(Clone, Debug)]
pub struct Row {
    pub stage: String,
    pub iteration: u64,
    pub termination: String,
    pub report: String,
    pub native_cost: f64,
    pub quality: Quality,
    pub counts: EvalCounts,
    pub work: Work,
    pub seconds: f64,
    pub max_segment_seconds: f64,
    pub time_limit_exceeded: bool,
    pub pass_limit_exceeded: bool,
    pub work_recorded: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub iterations: u64,
    pub passes: Option<u64>,
    pub time: Option<Duration>,
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub limits: Limits,
    /// Final-only observation avoids reevaluating every published point.
    pub final_only: bool,
}

/// The pass budget is checked at published boundaries. A final step can
/// overshoot it; the recorded actual work is never clipped to the budget.
pub fn run<P, S, So, D>(
    problem: P,
    solver: So,
    state: S,
    work: Arc<Mutex<Work>>,
    max_iter: u64,
    max_passes: u64,
    diagnose: D,
) -> Vec<Row>
where
    S: State + CountsMirror,
    So: Solver<P, S, Error = Infallible>,
    D: Fn(&S::Param) -> Quality,
{
    run_with_settings(
        problem,
        solver,
        state,
        work,
        Settings {
            limits: Limits {
                iterations: max_iter,
                passes: Some(max_passes),
                time: None,
            },
            final_only: false,
        },
        diagnose,
    )
}

/// Initialization and each step consume time. Observation and reference
/// evaluation occur outside that clock, including on a stopping boundary.
/// Native stopping evidence is retained if a stopping step exceeds a budget.
pub fn run_with_settings<P, S, So, D>(
    problem: P,
    solver: So,
    state: S,
    work: Arc<Mutex<Work>>,
    settings: Settings,
    diagnose: D,
) -> Vec<Row>
where
    S: State + CountsMirror,
    So: Solver<P, S, Error = Infallible>,
    D: Fn(&S::Param) -> Quality,
{
    let epoch = Instant::now();
    run_with_clock(problem, solver, state, work, settings, diagnose, || {
        epoch.elapsed()
    })
}

fn run_with_clock<P, S, So, D, C>(
    problem: P,
    solver: So,
    state: S,
    work: Arc<Mutex<Work>>,
    settings: Settings,
    diagnose: D,
    now: C,
) -> Vec<Row>
where
    S: State + CountsMirror,
    So: Solver<P, S, Error = Infallible>,
    D: Fn(&S::Param) -> Quality,
    C: Fn() -> Duration,
{
    let began = now();
    let mut stepper = Executor::new(problem, solver, state)
        .max_iter(settings.limits.iterations)
        .into_stepper()
        .unwrap();
    let mut elapsed = now() - began;
    let mut largest_segment = elapsed;
    let mut rows = vec![];
    let mut stage = "initialized".to_owned();
    let mut stop = None;
    loop {
        let report = stop.as_ref().or_else(|| stepper.finished());
        let time_limit_exceeded =
            settings.limits.time.is_some_and(|limit| elapsed >= limit);
        let pass_limit_exceeded = settings
            .limits
            .passes
            .is_some_and(|limit| work.lock().unwrap().passes() >= limit);
        let (stage_label, termination, evidence) = if let Some(report) = report
        {
            (
                format!("{:?}", report.stage),
                format!("{:?}", report.code()),
                format!("{:?}", report.termination),
            )
        } else if time_limit_exceeded {
            (stage.clone(), "HarnessTimeBudget".into(), String::new())
        } else if pass_limit_exceeded {
            (stage.clone(), "HarnessPassBudget".into(), String::new())
        } else {
            (stage.clone(), String::new(), String::new())
        };
        let done = !termination.is_empty();
        if !settings.final_only || done {
            rows.push(Row {
                stage: stage_label,
                iteration: stepper.state().iter(),
                termination,
                report: evidence,
                native_cost: to_f64(stepper.state().cost()),
                quality: diagnose(stepper.state().param()),
                counts: *stepper.counts(),
                work: work.lock().unwrap().clone(),
                seconds: elapsed.as_secs_f64(),
                max_segment_seconds: largest_segment.as_secs_f64(),
                time_limit_exceeded,
                pass_limit_exceeded,
                work_recorded: true,
            });
        }
        if done {
            break;
        }
        let began = now();
        let outcome = stepper.step().unwrap();
        let segment = now() - began;
        elapsed += segment;
        largest_segment = largest_segment.max(segment);
        stage = "published_boundary".into();
        if let StepOutcome::Stopped(report) = outcome {
            stop = Some(report);
        }
    }
    rows
}

/// Whole-executor baseline for measuring recorder overhead on matching work.
/// Pass and time caps are intentionally absent: compare only runs with equal
/// returned points, authoritative counts, and stopping reports.
pub fn run_plain<P, S, So, D>(
    problem: P,
    solver: So,
    state: S,
    iterations: u64,
    diagnose: D,
) -> Row
where
    S: State + CountsMirror,
    So: Solver<P, S, Error = Infallible>,
    D: Fn(&S::Param) -> Quality,
{
    let began = Instant::now();
    let result = Executor::new(problem, solver, state)
        .max_iter(iterations)
        .run()
        .unwrap();
    let seconds = began.elapsed().as_secs_f64();
    Row {
        stage: format!("{:?}", result.report.stage),
        iteration: result.state.iter(),
        termination: format!("{:?}", result.report.code()),
        report: format!("{:?}", result.report.termination),
        native_cost: to_f64(result.state.cost()),
        quality: diagnose(result.state.param()),
        counts: result.counts,
        work: Work::default(),
        seconds,
        max_segment_seconds: f64::NAN,
        time_limit_exceeded: false,
        pass_limit_exceeded: false,
        work_recorded: false,
    }
}

fn to_f64<F: Scalar>(x: F) -> f64 {
    x.to_f64().unwrap()
}

pub fn csv(fields: impl IntoIterator<Item = String>) -> String {
    fields
        .into_iter()
        .map(|s| format!("\"{}\"", s.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(",")
}

pub const HEADER: &str = "suite,case,family,partition,precision,backend,start,solver,seed,stage,iteration,termination,native_cost,checked_cost,cost_difference,gradient_inf,violation,parameter_error,best_sampled_cost,cost_evals,gradient_evals,residual_evals,jacobian_evals,value_passes,derivative_passes,sample_derivatives,solver_seconds,returned_param,best_sampled_param,report";

pub fn print_rows(prefix: &[String], rows: &[Row], final_only: bool) {
    for row in rows
        .iter()
        .filter(|row| !final_only || !row.termination.is_empty())
    {
        println!("{}", csv(fields(prefix, row)));
    }
}

pub fn fields(prefix: &[String], row: &Row) -> Vec<String> {
    let mut fields = prefix.to_vec();
    fields.extend([
        row.stage.clone(),
        row.iteration.to_string(),
        row.termination.clone(),
        row.native_cost.to_string(),
        row.quality.cost.to_string(),
        row.quality.difference.to_string(),
        row.quality.gradient_inf.to_string(),
        row.quality.violation.to_string(),
        row.quality.parameter_error.to_string(),
        row.work.best_cost.to_string(),
        row.counts.cost_evals.to_string(),
        row.counts.gradient_evals.to_string(),
        row.counts.residual_evals.to_string(),
        row.counts.jacobian_evals.to_string(),
        row.work.value_passes.to_string(),
        row.work.derivative_passes.to_string(),
        row.work.sample_derivatives.to_string(),
        row.seconds.to_string(),
        format!("{:?}", row.quality.param),
        format!("{:?}", row.work.best_param),
        row.report.clone(),
    ]);
    fields
}

pub fn vector_quality<F: Scalar>(case: &super::Case, p: &[F]) -> Quality {
    let param: Vec<_> = p.iter().map(|x| x.to_f64().unwrap()).collect();
    let cost = case.cost(&param);
    let gradient_inf = norm_inf(case.gradient(&param));
    let parameter_error = if case.identifiable {
        norm_inf(
            param
                .iter()
                .zip(&case.reference)
                .map(|(a, b)| (a - b) / b.abs().max(1e-12)),
        )
    } else {
        f64::NAN
    };
    Quality {
        violation: if param.iter().all(|x| x.is_finite()) {
            0.0
        } else {
            f64::NAN
        },
        param,
        cost,
        difference: cost - case.reference_cost,
        gradient_inf,
        parameter_error,
    }
}

pub fn norm_inf(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b.abs())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convergence::{Fixture, cases};
    use basin::{LevenbergMarquardt, PointState};

    #[derive(Clone, Default)]
    struct Clock(std::rc::Rc<std::cell::Cell<Duration>>);
    impl Clock {
        fn advance(&self, millis: u64) {
            self.0.set(self.0.get() + Duration::from_millis(millis));
        }
        fn now(&self) -> Duration {
            self.0.get()
        }
    }
    struct TimedSolver {
        clock: Clock,
        fail: bool,
    }
    impl Solver<Fixture<f64>, PointState<Vec<f64>>> for TimedSolver {
        type Error = Infallible;
        fn init(
            &mut self,
            p: &mut basin::Problem<Fixture<f64>>,
            mut s: PointState<Vec<f64>>,
        ) -> Result<PointState<Vec<f64>>, Infallible> {
            let cost = p.cost(s.param())?;
            s.replace(s.param().clone(), cost);
            self.clock.advance(3);
            Ok(s)
        }
        fn next_iter(
            &mut self,
            p: &mut basin::Problem<Fixture<f64>>,
            mut s: PointState<Vec<f64>>,
        ) -> Result<basin::SolverStep<PointState<Vec<f64>>>, Infallible>
        {
            let cost = p.cost(s.param())?;
            s.replace(s.param().clone(), cost);
            self.clock.advance(4);
            Ok(if self.fail {
                basin::SolverStep::stopped(
                    s,
                    basin::Termination::numerical_failure("scripted"),
                )
            } else {
                basin::SolverStep::completed(s)
            })
        }
    }

    #[test]
    fn time_budget_charges_initialization_and_steps_but_not_diagnostics() {
        let case = cases().remove(0);
        let clock = Clock::default();
        let p = Fixture::<f64>::new(case.clone());
        let rows = run_with_clock(
            p.clone(),
            TimedSolver {
                clock: clock.clone(),
                fail: false,
            },
            PointState::new(case.starts[0].clone()),
            p.work.clone(),
            Settings {
                limits: Limits {
                    iterations: 20,
                    passes: None,
                    time: Some(Duration::from_millis(9)),
                },
                final_only: false,
            },
            |x| {
                clock.advance(3_600_000);
                vector_quality(&case, x)
            },
            || clock.now(),
        );
        let last = rows.last().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(last.iteration, 2);
        assert_eq!(last.seconds, 0.011);
        assert_eq!(last.max_segment_seconds, 0.004);
        assert_eq!(last.termination, "HarnessTimeBudget");
        assert!(last.time_limit_exceeded);
        assert!(!last.pass_limit_exceeded);
        assert_eq!(last.counts.cost_evals, 3);
    }

    #[test]
    fn stopping_step_keeps_native_evidence_and_both_budget_flags() {
        let case = cases().remove(0);
        let clock = Clock::default();
        let p = Fixture::<f64>::new(case.clone());
        let rows = run_with_clock(
            p.clone(),
            TimedSolver {
                clock: clock.clone(),
                fail: true,
            },
            PointState::new(case.starts[0].clone()),
            p.work.clone(),
            Settings {
                limits: Limits {
                    iterations: 20,
                    passes: Some(2),
                    time: Some(Duration::from_millis(5)),
                },
                final_only: true,
            },
            |x| vector_quality(&case, x),
            || clock.now(),
        );
        assert_eq!(rows.len(), 1);
        let last = &rows[0];
        assert_eq!(last.seconds, 0.007);
        assert_eq!(last.termination, "SolverFailed");
        assert!(last.time_limit_exceeded && last.pass_limit_exceeded);
        assert!(last.report.contains("scripted"));
    }

    #[test]
    fn uninstrumented_executor_matches_recorded_work_and_point() {
        let case = cases().remove(0);
        let p = Fixture::<f64>::new(case.clone());
        let rows = run_with_settings(
            p.clone(),
            LevenbergMarquardt::new(),
            PointState::new(case.starts[0].clone()),
            p.work.clone(),
            Settings {
                limits: Limits {
                    iterations: 100,
                    passes: None,
                    time: None,
                },
                final_only: true,
            },
            |x| vector_quality(&case, x),
        );
        let p = Fixture::<f64>::new(case.clone()).without_recording();
        let work = p.work.clone();
        let plain = run_plain(
            p,
            LevenbergMarquardt::new(),
            PointState::new(case.starts[0].clone()),
            100,
            |x| vector_quality(&case, x),
        );
        let last = rows.last().unwrap();
        assert_eq!(last.quality.param, plain.quality.param);
        assert_eq!(last.counts, plain.counts);
        assert_eq!(last.report, plain.report);
        assert!(last.work_recorded && !plain.work_recorded);
        assert_eq!(work.lock().unwrap().passes(), 0);
    }
    #[test]
    fn trace_includes_initialization_and_matches_executor_counts() {
        let case = cases()
            .into_iter()
            .find(|c| c.name == "quadratic_n2_k1_rot0_rank2_noise0")
            .unwrap();
        let fixture = Fixture::<f64>::new(case.clone());
        let rows = run(
            fixture.clone(),
            LevenbergMarquardt::new(),
            PointState::new(case.starts[0].clone()),
            fixture.work.clone(),
            50,
            2000,
            |p| vector_quality(&case, p),
        );
        assert_eq!(rows[0].stage, "initialized");
        assert!(rows[0].work.passes() > 0);
        let baseline = Executor::new(
            Fixture::<f64>::new(case.clone()),
            LevenbergMarquardt::new(),
            PointState::new(case.starts[0].clone()),
        )
        .max_iter(50)
        .run()
        .unwrap();
        let last = rows.last().unwrap();
        assert_eq!(last.counts, baseline.counts);
        assert_eq!(last.quality.param, *baseline.param());
        assert_eq!(last.termination, format!("{:?}", baseline.report.code()));
        assert!(last.quality.cost < 1e-16);
        let fixture = Fixture::<f64>::new(case.clone());
        let limited = run(
            fixture.clone(),
            LevenbergMarquardt::new(),
            PointState::new(case.starts[0].clone()),
            fixture.work.clone(),
            50,
            1,
            |p| vector_quality(&case, p),
        );
        assert_eq!(limited.last().unwrap().termination, "HarnessPassBudget");
        assert_eq!(limited.len(), 1);
    }

    #[test]
    fn failed_partial_step_keeps_trial_work_and_returned_point_separate() {
        use basin::{Problem, SolverStep, Termination};
        struct RejectTrial(Vec<f64>);
        impl Solver<Fixture<f64>, PointState<Vec<f64>>> for RejectTrial {
            type Error = Infallible;
            fn init(
                &mut self,
                problem: &mut Problem<Fixture<f64>>,
                mut state: PointState<Vec<f64>>,
            ) -> Result<PointState<Vec<f64>>, Infallible> {
                let cost = problem.cost(state.param())?;
                state.replace(state.param().clone(), cost);
                Ok(state)
            }
            fn next_iter(
                &mut self,
                problem: &mut Problem<Fixture<f64>>,
                state: PointState<Vec<f64>>,
            ) -> Result<SolverStep<PointState<Vec<f64>>>, Infallible>
            {
                problem.cost(&self.0)?;
                Ok(SolverStep::stopped(
                    state,
                    Termination::numerical_failure("scripted rejected trial"),
                ))
            }
        }
        let case = cases()
            .into_iter()
            .find(|c| c.name == "quadratic_n2_k1_rot0_rank2_noise0")
            .unwrap();
        let fixture = Fixture::<f64>::new(case.clone());
        let rows = run(
            fixture.clone(),
            RejectTrial(case.reference.clone()),
            PointState::new(case.starts[0].clone()),
            fixture.work.clone(),
            20,
            2000,
            |p| vector_quality(&case, p),
        );
        let last = rows.last().unwrap();
        assert_eq!(last.stage, "Step { completed: false }");
        assert_eq!(last.termination, "SolverFailed");
        assert_eq!(last.iteration, 0);
        assert_eq!(last.counts.cost_evals, 2);
        assert_eq!(last.work.value_passes, 2);
        assert_eq!(last.quality.param, case.starts[0]);
        assert_eq!(last.work.best_param, case.reference);
        assert!(last.work.best_cost < last.quality.cost);
    }
}
