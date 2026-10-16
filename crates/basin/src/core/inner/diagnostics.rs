//! Optional owned explanations of inner work consumed by a composed solver.

use crate::{
    EvalCounts, OptimizationResult, Problem, Scalar, State, TerminationReport,
};

/// The outer boundary that started an inner run or chain segment.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Context<F: Scalar = f64> {
    /// Outer completed iteration count at entry.
    pub outer_iteration: u64,
    /// Selected population member, when the outer has a population.
    pub member: Option<usize>,
    /// Whether this segment continues a retained solver and state.
    pub resumed: bool,
    /// Scale supplied to a fresh inner seeder, when applicable.
    pub initial_scale: Option<F>,
}

/// Availability of an inner result, independently of its stopping decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Status {
    /// The inner call has started but has not returned.
    Pending,
    /// A coherent result and its stopping report were returned.
    Returned,
    /// A typed callback error interrupted the inner without a result.
    CallbackError,
}

/// Owned stopping evidence before an outer solver consumes an inner result.
///
/// Counts are snapshots of the shared problem wrapper, so their difference is
/// inner work in its original categories. The report describes only the inner
/// run. It never establishes outer convergence. Callback errors retain counts
/// but promise no recoverable state or report.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Record<F: Scalar = f64> {
    /// Position in this fresh outer run's diagnostic sequence.
    pub sequence: u64,
    /// Outer selection and chain context at entry.
    pub context: Context<F>,
    /// Shared wrapper counts before dispatch.
    pub counts_before: EvalCounts,
    /// Shared wrapper counts after dispatch, including interrupted callbacks.
    pub counts_after: Option<EvalCounts>,
    /// Availability of the returned result.
    pub status: Status,
    /// Actual inner report, available only when the inner returned a result.
    pub report: Option<TerminationReport<F>>,
}

pub(crate) fn run<P, S, F, E>(
    trace: &mut Option<Vec<Record<F>>>,
    problem: &mut Problem<P>,
    context: Context<F>,
    dispatch: impl FnOnce(&mut Problem<P>) -> Result<OptimizationResult<S>, E>,
) -> Result<OptimizationResult<S>, E>
where
    F: Scalar,
    S: State<Float = F>,
{
    let index = trace.as_mut().map(|records| {
        let index = records.len();
        records.push(Record {
            sequence: index as u64,
            context,
            counts_before: *problem.counts(),
            counts_after: None,
            status: Status::Pending,
            report: None,
        });
        index
    });
    let result = dispatch(problem);
    if let Some(index) = index {
        let record = &mut trace.as_mut().unwrap()[index];
        record.counts_after = Some(*problem.counts());
        if let Ok(result) = &result {
            record.status = Status::Returned;
            record.report = Some(result.report.clone());
        } else {
            record.status = Status::CallbackError;
        }
    }
    result
}
