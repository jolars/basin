//! Opt-in native model and trial observations for least-squares solvers.
//!
//! Observations belong to the solver, and getters perform no problem callbacks.
//! Fresh initialization resets the sequence. Exact continuation retains it.
//! Each native decision clears the previous batch, so consume observations
//! between steps. An execution-control stop can leave the previous batch intact;
//! sequence numbers distinguish new observations from retained ones.

use crate::core::math::ComponentZip;
use crate::{ConvergenceEvidence, Scalar};

/// One enabled, disabled, or unavailable native convergence comparison.
#[derive(Clone, Debug)]
pub struct LeastSquaresCheck<F: Scalar = f64> {
    /// Stable predicate name within the producing solver.
    pub name: &'static str,
    /// `None` means the criterion is disabled at this observation stage,
    /// except for the structural `no_free_parameters` check.
    pub tolerance: Option<F>,
    /// The solver's actual comparison result; absent for disabled criteria.
    pub passed: Option<bool>,
    /// Native operands, including failing comparisons when available.
    pub evidence: Option<ConvergenceEvidence<F>>,
}

/// A native model check or attempted residual trial, before publication.
///
/// `accepted` records the numerical acceptance decision. A later callback error
/// can prevent publication, so it does not by itself identify a returned point.
/// Missing trial cost means the residual callback did not complete. Native
/// steps can differ from subtraction of rounded callback coordinates.
#[derive(Clone, Debug)]
pub struct LeastSquaresObservation<F: Scalar = f64> {
    /// Monotone sequence within one fresh solve.
    pub sequence: u64,
    /// Whether a trial residual callback was attempted.
    pub trial: bool,
    /// Objective at the retained base iterate.
    pub base_cost: F,
    /// Native objective computed from a completed trial residual.
    pub trial_cost: Option<F>,
    /// Actual model decrease, including the legacy TRF curvature correction.
    pub actual_reduction: Option<F>,
    /// Native predicted model decrease.
    pub predicted_reduction: Option<F>,
    /// Native safeguarded acceptance ratio.
    pub gain_ratio: Option<F>,
    /// Native acceptance decision, absent before a completed trial callback.
    pub accepted: Option<bool>,
    /// Damping used to compute this step, if applicable.
    pub damping: Option<F>,
    /// Radius before the trial, if applicable.
    pub radius_before: Option<F>,
    /// Radius after the acceptance update, if applicable.
    pub radius_after: Option<F>,
    /// Actual native step: LM's `h`, legacy TRF's `alpha*h`, or full TRF's
    /// rounded step in scaled free-coordinate units.
    pub step: Vec<F>,
    /// Model gradient in the same units as `step`.
    pub gradient: Vec<F>,
    /// LM's monotone Marquardt diagonal, or legacy TRF's squared scaling.
    pub diagonal: Vec<F>,
    /// Full TRF's free-coordinate scaling; empty for LM and legacy TRF.
    pub coordinate_scale: Vec<F>,
    /// Full TRF's free-coordinate indices; empty for LM and legacy TRF.
    pub free: Vec<usize>,
    /// Coleman-Li curvature diagonal, if applicable.
    pub curvature: Vec<F>,
    /// Number of damped model solves used to obtain this trial. Full TRF
    /// counts subproblem calls, not iterations inside its secular search.
    pub model_solves: u32,
    /// Comparisons at this native observation stage.
    pub checks: Vec<LeastSquaresCheck<F>>,
}

impl<F: Scalar> LeastSquaresObservation<F> {
    pub(crate) fn model(cost: F) -> Self {
        Self {
            sequence: 0,
            trial: false,
            base_cost: cost,
            trial_cost: None,
            actual_reduction: None,
            predicted_reduction: None,
            gain_ratio: None,
            accepted: None,
            damping: None,
            radius_before: None,
            radius_after: None,
            step: Vec::new(),
            gradient: Vec::new(),
            diagonal: Vec::new(),
            coordinate_scale: Vec::new(),
            free: Vec::new(),
            curvature: Vec::new(),
            model_solves: 0,
            checks: Vec::new(),
        }
    }
}

/// Read the most recent native least-squares decision batch without callbacks.
pub trait LeastSquaresDiagnostics<F: Scalar = f64> {
    /// Empty when recording is disabled or no native decision has run.
    fn least_squares_observations(&self) -> &[LeastSquaresObservation<F>];
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Trace<F: Scalar> {
    next: u64,
    pub(crate) observations: Vec<LeastSquaresObservation<F>>,
}

impl<F: Scalar> Trace<F> {
    pub(crate) fn clear(&mut self) {
        self.observations.clear();
    }
    pub(crate) fn reset(&mut self) {
        self.clear();
        self.next = 0;
    }
    pub(crate) fn push(
        &mut self,
        mut observation: LeastSquaresObservation<F>,
    ) -> usize {
        self.next += 1;
        observation.sequence = self.next;
        self.observations.push(observation);
        self.observations.len() - 1
    }
}

pub(crate) fn vector<V: ComponentZip<F>, F: Scalar>(value: &V) -> Vec<F> {
    let mut result = Vec::new();
    value.all_zip(value, |x, _| {
        result.push(x);
        true
    });
    result
}
