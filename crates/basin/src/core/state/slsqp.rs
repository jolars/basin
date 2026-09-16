//! Accepted SLSQP progress. Models, constraint definitions, and workspaces
//! belong to the problem and solver; this state contains evaluated progress.

use super::{
    CountsMirror, EvaluatedGradientState, EvaluatedState, GradientState,
    IncumbentRef, IncumbentState, RawEvaluationState, State,
};
use crate::solver::slsqp::SlsqpFailure;
use crate::{EvalCounts, Scalar};

/// An accepted SLSQP iterate and its matching objective gradient.
///
/// `best_*` retains the latest accepted iterate, whose objective may increase
/// while constraints become feasible. This state therefore does not implement
/// [`super::ObjectiveIncumbentState`]. Diagnostics use the latest QP multiplier
/// estimates and are not independent certificates of nonlinear optimality.
/// Before initialization, checked records and diagnostics return `None`.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// with `F = f64` or `f32`. Exact continuation retains this state together
/// with the solver through [`crate::ExactCheckpoint`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlsqpState<V, F: Scalar = f64> {
    pub(crate) param: V,
    pub(crate) record: Option<(F, V)>,
    pub(crate) violation: Option<F>,
    pub(crate) stationarity: Option<F>,
    pub(crate) complementarity: Option<F>,
    pub(crate) failure: Option<SlsqpFailure>,
    pub(crate) counts: EvalCounts,
    pub(crate) iter: u64,
    pub(crate) pending_publication: bool,
    pub(crate) incumbent: Option<(V, F, u64, EvalCounts)>,
}
impl<V, F: Scalar> SlsqpState<V, F> {
    /// Construct an unevaluated starting point.
    pub fn new(param: V) -> Self {
        Self {
            param,
            record: None,
            violation: None,
            stationarity: None,
            complementarity: None,
            failure: None,
            counts: EvalCounts::default(),
            iter: 0,
            pending_publication: false,
            incumbent: None,
        }
    }
    /// Matching evaluated point, cost, and objective gradient.
    pub fn current(&self) -> Option<(&V, F, &V)> {
        self.record.as_ref().map(|(f, g)| (&self.param, *f, g))
    }
    /// Sum of absolute equality residuals and positive inequality violations.
    /// Includes box-bound violations.
    pub fn constraint_violation(&self) -> Option<F> {
        self.violation
    }
    /// Infinity norm of `x - project_box(x - ∇L)`, using latest QP multipliers.
    pub fn stationarity(&self) -> Option<F> {
        self.stationarity
    }
    /// Maximum absolute inequality multiplier times its constraint residual.
    /// Bound multipliers are not included; stationarity accounts for bounds.
    pub fn complementarity(&self) -> Option<F> {
        self.complementarity
    }
    /// Numerical failure detail when the solver reports `SolverFailed`.
    pub fn failure(&self) -> Option<SlsqpFailure> {
        self.failure
    }
}
impl<V: Clone, F: Scalar> State for SlsqpState<V, F> {
    type Param = V;
    type Float = F;
    fn iter(&self) -> u64 {
        self.iter
    }
    fn increment_iter(&mut self) {
        self.iter += 1;
    }
    fn param(&self) -> &V {
        &self.param
    }
    fn cost(&self) -> F {
        self.record
            .as_ref()
            .expect("SLSQP state has not been evaluated")
            .0
    }
    fn cost_evals(&self) -> u64 {
        self.counts.cost_evals + self.counts.residual_evals
    }
    fn best_param(&self) -> &V {
        &self
            .incumbent
            .as_ref()
            .expect("no accepted SLSQP iterate")
            .0
    }
    fn best_cost(&self) -> F {
        self.incumbent.as_ref().map_or(F::infinity(), |v| v.1)
    }
    fn best_iter(&self) -> u64 {
        self.incumbent.as_ref().map_or(0, |v| v.2)
    }
    fn best_cost_evals(&self) -> u64 {
        self.incumbent
            .as_ref()
            .map_or(0, |v| v.3.cost_evals + v.3.residual_evals)
    }
    fn update_best(&mut self) {
        if self.pending_publication {
            if let Some((cost, _)) = &self.record {
                if let Some((param, value, iter, counts)) = &mut self.incumbent
                {
                    param.clone_from(&self.param);
                    *value = *cost;
                    *iter = self.iter;
                    *counts = self.counts;
                } else {
                    self.incumbent = Some((
                        self.param.clone(),
                        *cost,
                        self.iter,
                        self.counts,
                    ));
                }
                self.pending_publication = false;
            }
        }
    }
    fn reset_best(&mut self) {
        self.incumbent = None;
        self.pending_publication = self.record.is_some();
    }
}
impl<V: Clone, F: Scalar> GradientState for SlsqpState<V, F> {
    fn gradient(&self) -> Option<&V> {
        self.record.as_ref().map(|v| &v.1)
    }
    fn gradient_evals(&self) -> u64 {
        self.counts.gradient_evals + self.counts.jacobian_evals
    }
    fn best_gradient_evals(&self) -> u64 {
        self.incumbent
            .as_ref()
            .map_or(0, |v| v.3.gradient_evals + v.3.jacobian_evals)
    }
}
impl<V: Clone, F: Scalar> CountsMirror for SlsqpState<V, F> {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.counts = *counts;
    }
}
impl<V: Clone, F: Scalar> RawEvaluationState for SlsqpState<V, F> {
    fn raw_counts(&self) -> &EvalCounts {
        &self.counts
    }
}
impl<V: Clone, F: Scalar> EvaluatedState for SlsqpState<V, F> {
    fn current_record(&self) -> Option<(&V, F)> {
        self.current().map(|(x, f, _)| (x, f))
    }
}
impl<V: Clone, F: Scalar> EvaluatedGradientState for SlsqpState<V, F> {
    fn current_gradient_record(&self) -> Option<(&V, F, &V)> {
        self.current()
    }
}
impl<V: Clone, F: Scalar> IncumbentState for SlsqpState<V, F> {
    fn incumbent_record(&self) -> Option<IncumbentRef<'_, V, F>> {
        self.incumbent.as_ref().map(|v| IncumbentRef {
            param: &v.0,
            cost: v.1,
            iter: v.2,
            counts: &v.3,
        })
    }
}
