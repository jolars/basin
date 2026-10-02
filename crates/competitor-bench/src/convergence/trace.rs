//! Outer-boundary records. Internal rejected trials are not reconstructed.

use super::Work;
use basin::{
    CountsMirror, EvalCounts, Executor, Scalar, Solver, State, StepOutcome,
};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Instant,
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
    let began = Instant::now();
    let mut stepper = Executor::new(problem, solver, state)
        .max_iter(max_iter)
        .into_stepper()
        .unwrap();
    let mut seconds = began.elapsed().as_secs_f64();
    let snapshot =
        |stage: String, termination: String, report: String, seconds| Row {
            stage,
            iteration: 0,
            termination,
            report,
            native_cost: f64::NAN,
            quality: Quality {
                param: vec![],
                cost: f64::NAN,
                difference: f64::NAN,
                gradient_inf: f64::NAN,
                violation: f64::NAN,
                parameter_error: f64::NAN,
            },
            counts: EvalCounts::default(),
            work: work.lock().unwrap().clone(),
            seconds,
        };
    let mut rows = vec![];
    let mut stage = "initialized".to_owned();
    let mut stop = None;
    loop {
        let report = stop.as_ref().or_else(|| stepper.finished());
        let mut row = if let Some(report) = report {
            snapshot(
                format!("{:?}", report.stage),
                format!("{:?}", report.code()),
                format!("{:?}", report.termination),
                seconds,
            )
        } else if work.lock().unwrap().passes() >= max_passes {
            snapshot(
                stage.clone(),
                "HarnessPassBudget".into(),
                String::new(),
                seconds,
            )
        } else {
            snapshot(stage.clone(), String::new(), String::new(), seconds)
        };
        row.iteration = stepper.state().iter();
        row.native_cost = to_f64(stepper.state().cost());
        row.quality = diagnose(stepper.state().param());
        row.counts = *stepper.counts();
        let done = !row.termination.is_empty();
        rows.push(row);
        if done {
            break;
        }
        let began = Instant::now();
        let outcome = stepper.step().unwrap();
        seconds += began.elapsed().as_secs_f64();
        stage = "published_boundary".into();
        if let StepOutcome::Stopped(report) = outcome {
            stop = Some(report);
        }
    }
    rows
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
        println!("{}", csv(fields));
    }
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
