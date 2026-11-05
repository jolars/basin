//! Projected quasi-Newton searches with retained free-coordinate curvature.
use super::*;
use crate::line_search::projected_armijo::ProjectedArmijo;
#[cfg(test)]
#[path = "projected_tests.rs"]
mod tests;

/// Path used by a [`ProjectedLbfgs`] search, independently of curvature retention.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ProjectedSearch<F: Scalar = f64> {
    /// Use the free-coordinate inverse direction, removing outward components
    /// at active bounds, and cap steps at the first bound reached along the ray.
    /// Uses the base solver's bounded line search, including its configuration.
    /// This is the path used by sisyfuzz's More-Thuente and Wolfe strategies.
    #[default]
    FeasibleRay,
    /// Project each trial and test Armijo decrease using its actual displacement.
    /// Uses this configuration instead of the base solver's straight-ray search.
    Armijo(ProjectedArmijo<F>),
    /// Project the endpoint first and search the straight feasible chord to it.
    /// Uses the base solver's bounded line search. This differs from both of
    /// sisyfuzz's projected search choices.
    Chord,
}

/// Projected L-BFGS with explicit search geometry and curvature retention.
///
/// Construct with [`ModelLbfgsb::with_projected_search`] after configuring the
/// base model. Raw curvature pairs survive free-set changes by default; unsafe
/// restrictions are skipped for that direction. Capacity eviction removes old
/// pairs; numerical recovery and unusable global curvature can clear history.
/// Pairs use the actual
/// accepted displacement and gradient difference, including projected trials.
///
/// This wrapper has its own relative curvature safeguard, default `1e-10`,
/// accepting `s'y > tolerance * ||s|| * ||y||` with finite positive inverse
/// curvature and scaling. An unsafe global pair clears the model, matching
/// sisyfuzz's projected solver policy. This is separate from the base solver's
/// reference curvature safeguard and does not inherit its configured threshold.
/// Restricted pairs are tested separately with a fixed relative threshold of
/// `1e-10` on the free coordinates; changing the global threshold does not change
/// that test. Base convergence tolerances remain configured
/// on the solver; disable relative cost reduction when comparing against
/// sisyfuzz's gradient-only projected stopping rule.
///
/// Setting [`Self::with_curvature_retention`] to `false` provides a controlled comparison
/// that clears raw pairs when the free mask changes. Both settings use the same
/// search and curvature acceptance; pairs are still accumulated while the mask
/// stays unchanged. The previous mask is serialized, so exact
/// continuation preserves this behavior. Fresh runs reset history and masks.
/// All evaluations, including failed probes, remain counted. Hard callback
/// errors propagate. No new line-search or solver defaults affect existing code.
/// Choosing [`ProjectedSearch::Chord`] uses the same path geometry as
/// [`ModelDirection::ProjectedChord`], with this wrapper's global pair safeguard
/// and actual-displacement updates instead of the legacy pair policy.
///
/// ```
/// use basin::{Lbfgsb, line_search::projected_armijo::ProjectedArmijo};
/// use basin::solver::lbfgs::model_strategy::ProjectedSearch;
/// let solver = Lbfgsb::<Vec<f64>>::new()
///     .with_relative_cost_reduction_tolerance(None)
///     .with_compact_model()
///     .with_projected_search(ProjectedSearch::Armijo(
///         ProjectedArmijo::new().with_fused_evaluations()));
/// assert!(solver.curvature_retention());
/// ```
///
/// # Backends
///
/// Contiguous `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>` and faer
/// `Col<F>`, for both `f32` and `f64`, with the same capabilities as [`ModelLbfgsb`].
/// Feasible-ray/chord searches require the base search's bounded-step support.
///
/// # References
///
/// Retained free-coordinate BFGS and projected Armijo/ray policies adapted from
/// sisyfuzz 0.1.2 `lbfgs.rs`, `quasi_newton.rs`, `line_search.rs` and
/// `more_thuente.rs` (Apache-2.0). BFGS references: see [`Lbfgs`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProjectedLbfgs<V, F: Scalar = f64, S = MoreThuente<F>> {
    model: ModelLbfgsb<V, F, S>,
    settings: ProjectedSettings<F>,
    retain: bool,
    previous_free: Option<Vec<bool>>,
}
impl<V, F: Scalar, S> ProjectedLbfgs<V, F, S> {
    pub(super) fn new(
        model: ModelLbfgsb<V, F, S>,
        search: ProjectedSearch<F>,
    ) -> Self {
        Self {
            model,
            settings: ProjectedSettings {
                search,
                curvature_tolerance: F::from_f64(1e-10).unwrap(),
            },
            retain: true,
            previous_free: None,
        }
    }
    /// Preserve raw pairs across free-set changes (default true). Configure
    /// before a fresh run; false is the reset-on-change comparison policy.
    pub fn with_curvature_retention(mut self, retain: bool) -> Self {
        self.retain = retain;
        self
    }
    /// Whether raw pairs survive changes of the free mask.
    pub fn curvature_retention(&self) -> bool {
        self.retain
    }
    /// Relative global pair acceptance threshold (default `1e-10`). Does not
    /// change the fixed `1e-10` threshold for restricted pairs on free coordinates.
    /// Zero requires strictly positive finite curvature and usable inverse scaling.
    ///
    /// # Panics
    ///
    /// Panics for negative or non-finite values.
    pub fn with_relative_curvature_tolerance(mut self, tolerance: F) -> Self {
        assert!(tolerance.is_finite() && tolerance >= F::zero());
        self.settings.curvature_tolerance = tolerance;
        self
    }
    /// Current number of retained raw curvature pairs.
    pub fn history_len(&self) -> usize {
        self.model.solver.history_len()
    }
}
impl<V: Clone, F: Scalar, S> crate::InitialState<V>
    for ProjectedLbfgs<V, F, S>
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        FirstOrderState::new(x.clone())
    }
}
impl<P, V, S, F> Solver<P, FirstOrderState<V, F>> for ProjectedLbfgs<V, F, S>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + BoxProblem<Param = V>,
    V: AsFloatSliceMut<F> + Clone + Dot<F> + ScaledAdd<F> + VectorLen,
    S: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = crate::ConstraintSolveError<P::Error>;
    fn init(
        &mut self,
        p: &mut Problem<P>,
        s: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        self.previous_free = None;
        self.model.init(p, s)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<P>,
        s: FirstOrderState<V, F>,
    ) -> Result<crate::SolverStep<FirstOrderState<V, F>>, Self::Error> {
        let x = s.param().as_float_slice();
        let g = crate::GradientState::gradient(&s)
            .expect("initialized gradient")
            .as_float_slice();
        let lower = p.inner().lower().as_float_slice();
        let upper = p.inner().upper().as_float_slice();
        let had_previous = self.previous_free.is_some();
        let free = self
            .previous_free
            .get_or_insert_with(|| vec![false; x.len()]);
        let mut changed = false;
        for i in 0..x.len() {
            let value = lower[i] != upper[i]
                && !((x[i] <= lower[i] && g[i] > F::zero())
                    || (x[i] >= upper[i] && g[i] < F::zero()));
            changed |= free[i] != value;
            free[i] = value;
        }
        if !self.retain && had_previous && changed {
            clear_history(
                self.model
                    .solver
                    .history
                    .as_mut()
                    .expect("initialized history"),
            );
        }
        self.model.work.policy = ModelDirection::ProjectedChord;
        self.model.work.raw_direction =
            !matches!(self.settings.search, ProjectedSearch::Chord);
        self.model.work.projected_trials =
            matches!(self.settings.search, ProjectedSearch::Armijo(_));
        self.model.solver.next_iter_bounded_search(
            p,
            s,
            None,
            Some(&mut self.model.work),
            Some(&mut self.settings),
        )
    }
    fn reset_convergence(&mut self) {
        <ModelLbfgsb<V,F,S> as Solver<P,FirstOrderState<V,F>>>::reset_convergence(&mut self.model);
    }
    fn should_check_iterate_change(&self) -> bool {
        <ModelLbfgsb<V,F,S> as Solver<P,FirstOrderState<V,F>>>::should_check_iterate_change(&self.model)
    }
    fn check_convergence(
        &mut self,
        p: &Problem<P>,
        s: &FirstOrderState<V, F>,
    ) -> Option<Termination<F>> {
        self.model.check_convergence(p, s)
    }
    fn terminate(&self, s: &FirstOrderState<V, F>) -> Option<Termination<F>> {
        <ModelLbfgsb<V, F, S> as Solver<P, FirstOrderState<V, F>>>::terminate(
            &self.model,
            s,
        )
    }
}
