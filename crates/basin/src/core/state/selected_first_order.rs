//! Shared constrained progress with a matching objective gradient.

use super::{
    CountsMirror, EvaluatedGradientState, EvaluatedState,
    GradientDimensionMismatch, GradientState, IncumbentRef, IncumbentState,
    RawEvaluationState, SelectedState, State,
};
use crate::{EvalCounts, Scalar, VectorLen};

/// Solver-selected constrained progress with a matching objective gradient.
///
/// Construction provides an unevaluated seed. [`replace`](Self::replace)
/// checks point and gradient dimensions before publishing a complete current
/// record, while [`select_current`](Self::select_current) explicitly chooses
/// an incumbent. Selection can increase the objective to improve feasibility,
/// so this state does not implement [`super::ObjectiveIncumbentState`].
/// The solver defines the violation measure and eligibility rules.
///
/// Current records contain a point, objective, gradient, and violation. Best
/// records contain the selected point, objective, and violation; their raw
/// counts and iteration are stamped by the executor at publication. Selection
/// events follow [`SelectedState`], including preservation when current progress
/// changes before publication. Derivatives are advertised only for current
/// evaluated progress. All six evaluation categories retain their raw values.
///
/// Fresh initialization calls [`reset`](Self::reset) and reevaluates the seed.
/// Exact continuation retains the solver and state together. With `serde`,
/// serialization is available when `V` and `F` support it.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for `F = f64` (default) or `f32`. Replacement requires [`VectorLen`];
/// state capabilities require only `V: Clone`.
///
/// ```compile_fail
/// use basin::{RunControl, SelectedFirstOrderState};
/// let _ = RunControl::<SelectedFirstOrderState<Vec<f64>>>::new().target_objective(0.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SelectedFirstOrderState<V, F: Scalar = f64> {
    progress: SelectedState<V, F>,
    gradient: Option<V>,
}
impl<V, F: Scalar> SelectedFirstOrderState<V, F> {
    /// Construct an unevaluated seed with no selected incumbent.
    pub fn new(param: V) -> Self {
        Self {
            progress: SelectedState::new(param),
            gradient: None,
        }
    }
    /// Clear records and bookkeeping, retaining the parameter as a fresh seed.
    pub fn reset(&mut self) {
        self.progress.reset();
        self.gradient = None;
    }
    /// Matching point, objective, gradient, and violation, or `None` for a seed.
    pub fn current(&self) -> Option<(&V, F, &V, F)> {
        let (x, cost, violation) = self.progress.current()?;
        Some((x, cost, self.gradient.as_ref()?, violation))
    }
    /// Selected point, objective, and violation, or `None` before selection.
    pub fn best(&self) -> Option<(&V, F, F)> {
        self.progress.best()
    }
    /// All six raw evaluation categories at the current publication boundary.
    pub fn counts(&self) -> &EvalCounts {
        self.progress.counts()
    }
    /// Raw counts recorded when the selected incumbent was published.
    pub fn best_counts(&self) -> Option<&EvalCounts> {
        self.progress.best_counts()
    }
}
impl<V: VectorLen, F: Scalar> SelectedFirstOrderState<V, F> {
    /// Replace current progress atomically, preserving any pending selection.
    /// A dimension mismatch leaves every record and counter unchanged.
    pub fn replace(
        &mut self,
        param: V,
        cost: F,
        gradient: V,
        violation: F,
    ) -> Result<(), GradientDimensionMismatch> {
        if param.vec_len() != gradient.vec_len() {
            return Err(GradientDimensionMismatch {
                param_len: param.vec_len(),
                gradient_len: gradient.vec_len(),
            });
        }
        self.progress.replace(param, cost, violation);
        self.gradient = Some(gradient);
        Ok(())
    }
}
impl<V: Clone, F: Scalar> SelectedFirstOrderState<V, F> {
    /// Select the current evaluated record for the next publication boundary.
    /// Returns `false` for an unevaluated seed. Repeated observation does not
    /// refresh the selected record's metadata.
    pub fn select_current(&mut self) -> bool {
        self.progress.select_current()
    }
}
impl<V: Clone, F: Scalar> State for SelectedFirstOrderState<V, F> {
    type Param = V;
    type Float = F;
    fn iter(&self) -> u64 {
        self.progress.iter()
    }
    fn increment_iter(&mut self) {
        self.progress.increment_iter();
    }
    fn param(&self) -> &V {
        self.progress.param()
    }
    fn cost(&self) -> F {
        self.progress.cost()
    }
    fn cost_evals(&self) -> u64 {
        self.progress.cost_evals()
    }
    fn best_param(&self) -> &V {
        self.progress.best_param()
    }
    fn best_cost(&self) -> F {
        self.progress.best_cost()
    }
    fn best_iter(&self) -> u64 {
        self.progress.best_iter()
    }
    fn best_cost_evals(&self) -> u64 {
        self.progress.best_cost_evals()
    }
    fn update_best(&mut self) {
        self.progress.update_best();
    }
    fn reset_best(&mut self) {
        self.progress.reset_best();
    }
}
impl<V: Clone, F: Scalar> CountsMirror for SelectedFirstOrderState<V, F> {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.progress.mirror(counts);
    }
}
impl<V: Clone, F: Scalar> RawEvaluationState for SelectedFirstOrderState<V, F> {
    fn raw_counts(&self) -> &EvalCounts {
        self.counts()
    }
}
impl<V: Clone, F: Scalar> EvaluatedState for SelectedFirstOrderState<V, F> {
    fn current_record(&self) -> Option<(&V, F)> {
        self.progress.current_record()
    }
}
impl<V: Clone, F: Scalar> EvaluatedGradientState
    for SelectedFirstOrderState<V, F>
{
    fn current_gradient_record(&self) -> Option<(&V, F, &V)> {
        self.current()
            .map(|(x, cost, gradient, _)| (x, cost, gradient))
    }
}
impl<V: Clone, F: Scalar> GradientState for SelectedFirstOrderState<V, F> {
    fn gradient(&self) -> Option<&V> {
        self.gradient.as_ref()
    }
    fn gradient_evals(&self) -> u64 {
        self.counts().gradient_evals
    }
    fn best_gradient_evals(&self) -> u64 {
        self.best_counts().map_or(0, |c| c.gradient_evals)
    }
}
impl<V: Clone, F: Scalar> IncumbentState for SelectedFirstOrderState<V, F> {
    fn incumbent_record(&self) -> Option<IncumbentRef<'_, V, F>> {
        self.progress.incumbent_record()
    }
}
