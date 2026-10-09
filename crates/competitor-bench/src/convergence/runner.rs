//! Observe authoritative boundaries without reconstructing native trial tests.

use std::time::{Duration, Instant};

use basin::{
    CountsMirror, EvalCounts, Executor, Scalar, Solver, State, StepOutcome,
    TerminationReport, TerminationStage,
};

use super::ledger::{LedgerSnapshot, OracleError, WorkLedger};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicationStage {
    Initialization,
    Boundary,
    Stop(TerminationStage),
}

#[derive(Clone, Debug)]
pub struct Recommendation {
    pub point: Vec<f64>,
    pub solver_cost: f64,
    pub iteration: u64,
    pub work: u64,
    pub counts: EvalCounts,
    pub stage: PublicationStage,
}

#[derive(Clone, Debug)]
pub enum RunOutcome<F: Scalar = f64> {
    Stopped(TerminationReport<F>),
    InitializationError(OracleError),
    StepError(OracleError),
    WallLimit,
}

#[derive(Clone, Debug)]
pub struct Measurement<F: Scalar = f64> {
    pub recommendations: Vec<Recommendation>,
    pub outcome: RunOutcome<F>,
    /// Initialization errors consume the executor and expose no public counters.
    /// Physical history remains available; missing logical counts stay missing.
    pub counts: Option<EvalCounts>,
    pub ledger: LedgerSnapshot,
    pub elapsed: Duration,
}

impl<F: Scalar> Measurement<F> {
    /// A callback error consumes active state. This is the last recommendation
    /// published before that transition, not a recoverable final checkpoint.
    pub fn last_published(&self) -> Option<&Recommendation> {
        self.recommendations.last()
    }

    pub fn returned(&self) -> Option<&Recommendation> {
        matches!(self.outcome, RunOutcome::Stopped(_))
            .then(|| self.last_published())
            .flatten()
    }

    pub fn at_budget(&self, budget: u64) -> Option<&Recommendation> {
        self.recommendations.iter().rev().find(|r| r.work <= budget)
    }
}

fn recommendation<F: Scalar, S: State<Param = Vec<F>, Float = F>>(
    state: &S,
    counts: &EvalCounts,
    ledger: &WorkLedger,
    stage: PublicationStage,
) -> Recommendation {
    Recommendation {
        point: state.param().iter().map(|x| x.to_f64().unwrap()).collect(),
        solver_cost: state.cost().to_f64().unwrap(),
        iteration: state.iter(),
        work: ledger.work(),
        counts: *counts,
        stage,
    }
}

/// Native stopping settings and safeguards belong to the supplied solver.
/// Typed budget errors enforce the aggregate cap even within an active step.
pub fn measure<P, S, So, F>(
    executor: Executor<P, S, So>,
    ledger: &WorkLedger,
    wall_cap: Duration,
) -> Measurement<F>
where
    F: Scalar,
    S: State<Param = Vec<F>, Float = F> + CountsMirror,
    So: Solver<P, S, Error = OracleError>,
{
    let start = Instant::now();
    let mut stepper = match executor.into_stepper() {
        Ok(stepper) => stepper,
        Err(error) => {
            return Measurement {
                recommendations: Vec::new(),
                outcome: RunOutcome::InitializationError(error),
                counts: None,
                ledger: ledger.snapshot(),
                elapsed: start.elapsed(),
            };
        }
    };
    let mut recommendations = vec![recommendation(
        stepper.state(),
        stepper.counts(),
        ledger,
        PublicationStage::Initialization,
    )];
    let outcome = loop {
        if start.elapsed() >= wall_cap {
            break RunOutcome::WallLimit;
        }
        match stepper.step() {
            Ok(StepOutcome::Continue) => recommendations.push(recommendation(
                stepper.state(),
                stepper.counts(),
                ledger,
                PublicationStage::Boundary,
            )),
            Ok(StepOutcome::Stopped(report)) => {
                recommendations.push(recommendation(
                    stepper.state(),
                    stepper.counts(),
                    ledger,
                    PublicationStage::Stop(report.stage),
                ));
                break RunOutcome::Stopped(report);
            }
            Err(error) => break RunOutcome::StepError(error),
        }
    };
    Measurement {
        recommendations,
        outcome,
        counts: Some(*stepper.counts()),
        ledger: ledger.snapshot(),
        elapsed: start.elapsed(),
    }
}
