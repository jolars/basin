//! Solver-owned convergence settings with compile-time backend capabilities.
//!
//! Solver setters return [`ConfiguredSolver`] when a check requires additional
//! vector operations or iterate history. Its fixed slots replace settings of
//! the same kind. No user-defined convergence policy is required.

// Keep the Basin 1.x compatibility bridge and shared check implementations local.
#![allow(deprecated)]

use super::constraint::BoxConstraints;
use super::inner::{InitialState, WarmStart};
use super::math::{ClampInPlace, NormInfinity, NormSquared, Scalar, ScaledAdd};
use super::problem::Problem;
use super::solver::Solver;
use super::state::{GradientState, SimplexState, State};
use super::termination::*;

/// A solver carrying fixed, optional convergence checks.
///
/// Construct this through the solver's `with_*_tolerance` setters. Setters
/// replace the corresponding check; distinct tests combine with OR. Simplex
/// size and cost-spread conditions form a single AND group. New checks are
/// disabled until configured. `None` disables a check; zero is an exact-zero
/// threshold. The original solver's algorithm controls and safeguards remain
/// active. Settings and history accompany composed inner solves.
///
/// Unsupported settings are absent from a solver's API:
///
/// ```compile_fail
/// use basin::NelderMead;
/// let solver = NelderMead::new().with_absolute_gradient_tolerance(1e-8);
/// ```
///
/// # Backends
///
/// Supports the underlying solver's backends when they implement the selected
/// checks' operations. Unconfigured slots add no vector capability requirements.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfiguredSolver<So, G = (), X = (), C = (), T = ()> {
    pub(crate) solver: So,
    gradient: G,
    step: X,
    cost: C,
    simplex: T,
    checked: Option<(u64, u64)>,
    reason: Option<TerminationReason>,
}

impl<So> ConfiguredSolver<So> {
    pub(crate) fn new(solver: So) -> Self {
        Self {
            solver,
            gradient: (),
            step: (),
            cost: (),
            simplex: (),
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    /// Inspect the algorithm and its native convergence settings.
    pub fn solver(&self) -> &So {
        &self.solver
    }

    pub(crate) fn map_solver<Other>(
        self,
        f: impl FnOnce(So) -> Other,
    ) -> ConfiguredSolver<Other, G, X, C, T> {
        ConfiguredSolver {
            solver: f(self.solver),
            gradient: self.gradient,
            step: self.step,
            cost: self.cost,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

pub(crate) fn optional_tolerance<F: Scalar>(
    value: impl Into<Option<F>>,
) -> Option<F> {
    let value = value.into();
    assert!(
        value.is_none_or(|v| v.is_finite() && v >= F::zero()),
        "tolerance must be finite and nonnegative"
    );
    value
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for () {}
}

/// Implementation detail of the fixed convergence slots.
#[doc(hidden)]
pub trait Check<P, S>: sealed::Sealed {
    fn reset(&mut self);
    fn check(
        &mut self,
        problem: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason>;
}

impl<P, S> Check<P, S> for () {
    fn reset(&mut self) {}
    fn check(&mut self, _: &Problem<P>, _: &S) -> Option<TerminationReason> {
        None
    }
}

/// Fixed gradient-norm settings; constructed by solver setters.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientChecks<F: Scalar = f64> {
    absolute: Option<GradientTolerance<F>>,
    relative: Option<RelativeGradientTolerance<F>>,
}

impl<F: Scalar> From<()> for GradientChecks<F> {
    fn from(_: ()) -> Self {
        Self {
            absolute: None,
            relative: None,
        }
    }
}
impl<F: Scalar> sealed::Sealed for GradientChecks<F> {}
impl<P, S, F> Check<P, S> for GradientChecks<F>
where
    F: Scalar,
    S: GradientState<Float = F>,
    S::Param: NormSquared<F>,
{
    fn reset(&mut self) {
        if let Some(c) = &mut self.relative {
            <RelativeGradientTolerance<F> as TerminationCriterion<S>>::reset(c);
        }
    }
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        if !state.gradient()?.norm_squared().is_finite() {
            return None;
        }
        self.absolute
            .as_mut()
            .and_then(|c| c.check(state))
            .or_else(|| self.relative.as_mut().and_then(|c| c.check(state)))
    }
}

/// Fixed projected-gradient setting; bounds come from the current problem.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProjectedGradientCheck<F: Scalar = f64> {
    tolerance: Option<F>,
}
impl<F: Scalar> From<()> for ProjectedGradientCheck<F> {
    fn from(_: ()) -> Self {
        Self { tolerance: None }
    }
}
impl<F: Scalar> sealed::Sealed for ProjectedGradientCheck<F> {}
impl<P, S, F> Check<P, S> for ProjectedGradientCheck<F>
where
    F: Scalar,
    S: GradientState<Float = F>,
    P: BoxConstraints<Param = S::Param>,
    S::Param: Clone + ScaledAdd<F> + ClampInPlace + NormInfinity<F>,
{
    fn reset(&mut self) {}
    fn check(
        &mut self,
        problem: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        let tol = self.tolerance?;
        let g = state.gradient()?;
        if !g.norm_infinity().is_finite() {
            return None;
        }
        let mut probe = state.param().clone();
        probe.scaled_add(-F::one(), g);
        probe.clamp_in_place(problem.inner().lower(), problem.inner().upper());
        probe.scaled_add(-F::one(), state.param());
        (probe.norm_infinity() <= tol)
            .then_some(TerminationReason::ProjectedGradientTolerance)
    }
}

/// Fixed iterate-change settings; constructed by solver setters.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepChecks<V, F: Scalar = f64> {
    absolute: Option<F>,
    relative: Option<F>,
    last: Option<V>,
}
impl<V, F: Scalar> From<()> for StepChecks<V, F> {
    fn from(_: ()) -> Self {
        Self {
            absolute: None,
            relative: None,
            last: None,
        }
    }
}
impl<V, F: Scalar> sealed::Sealed for StepChecks<V, F> {}
impl<P, S, V, F> Check<P, S> for StepChecks<V, F>
where
    F: Scalar,
    S: State<Param = V, Float = F>,
    V: Clone + ScaledAdd<F> + NormSquared<F>,
{
    fn reset(&mut self) {
        self.last = None;
    }
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        if self.absolute.is_none() && self.relative.is_none() {
            return None;
        }
        let current = state.param();
        let last = self.last.replace(current.clone())?;
        let mut difference = current.clone();
        difference.scaled_add(-F::one(), &last);
        let step = difference.norm_squared().sqrt();
        // A finite iterate's squared norm can overflow even when its step is zero.
        if !step.is_finite() {
            return None;
        }
        if self.absolute.is_some_and(|tolerance| step <= tolerance) {
            return Some(TerminationReason::ParamTolerance);
        }
        self.relative
            .is_some_and(|tolerance| {
                // Avoid zero times infinity when the iterate's squared norm overflows.
                if tolerance == F::zero() {
                    step == F::zero()
                } else {
                    let norm = current.norm_squared().sqrt();
                    let bound = if norm.is_finite() {
                        tolerance * norm
                    } else {
                        // Scale before taking the norm so a small tolerance can
                        // still yield a finite bound for very large parameters.
                        let mut scaled = current.clone();
                        scaled.scaled_add(-F::one(), current);
                        scaled.scaled_add(tolerance, current);
                        scaled.norm_squared().sqrt()
                    };
                    step <= bound
                }
            })
            .then_some(TerminationReason::RelativeParamTolerance)
    }
}

/// Fixed objective-change settings; constructed by solver setters.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CostChecks<F: Scalar = f64> {
    absolute: Option<CostTolerance<F>>,
    relative: Option<RelativeCostTolerance<F>>,
}
impl<F: Scalar> From<()> for CostChecks<F> {
    fn from(_: ()) -> Self {
        Self {
            absolute: None,
            relative: None,
        }
    }
}
impl<F: Scalar> sealed::Sealed for CostChecks<F> {}
impl<P, S, F> Check<P, S> for CostChecks<F>
where
    F: Scalar,
    S: State<Float = F>,
{
    fn reset(&mut self) {
        if let Some(c) = &mut self.absolute {
            <CostTolerance<F> as TerminationCriterion<S>>::reset(c);
        }
        if let Some(c) = &mut self.relative {
            <RelativeCostTolerance<F> as TerminationCriterion<S>>::reset(c);
        }
    }
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        if !state.cost().is_finite() {
            <Self as Check<P, S>>::reset(self);
            return None;
        }
        let absolute = self.absolute.as_mut().and_then(|c| c.check(state));
        let relative = self.relative.as_mut().and_then(|c| c.check(state));
        absolute.or(relative)
    }
}

/// Fixed simplex-collapse settings; configured size and cost tests use AND.
/// The cost-only form uses `Size = ()` to avoid requiring vector norms.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimplexChecks<F: Scalar = f64, Size = Option<F>> {
    size: Size,
    cost: Option<F>,
}
impl<F: Scalar, Size: Default> From<()> for SimplexChecks<F, Size> {
    fn from(_: ()) -> Self {
        Self {
            size: Size::default(),
            cost: None,
        }
    }
}
impl<F: Scalar> From<SimplexChecks<F, ()>> for SimplexChecks<F> {
    fn from(checks: SimplexChecks<F, ()>) -> Self {
        Self {
            size: None,
            cost: checks.cost,
        }
    }
}
impl<F: Scalar, Size> sealed::Sealed for SimplexChecks<F, Size> {}

fn simplex_cost_within_tolerance<F: Scalar>(costs: &[F], tolerance: F) -> bool {
    let Some(&best) = costs.first() else {
        return false;
    };
    costs
        .iter()
        .all(|&cost| cost.is_finite() && (cost - best).abs() <= tolerance)
}

impl<P, S, F> Check<P, S> for SimplexChecks<F, ()>
where
    F: Scalar,
    S: SimplexState<Float = F>,
{
    fn reset(&mut self) {}
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        simplex_cost_within_tolerance(state.costs(), self.cost?)
            .then_some(TerminationReason::SimplexTolerance)
    }
}

impl<P, S, F> Check<P, S> for SimplexChecks<F>
where
    F: Scalar,
    S: SimplexState<Float = F>,
    S::Param: Clone + ScaledAdd<F> + NormInfinity<F>,
{
    fn reset(&mut self) {}
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        if self.size.is_none() && self.cost.is_none() {
            return None;
        }
        if !simplex_cost_within_tolerance(
            state.costs(),
            self.cost.unwrap_or(F::infinity()),
        ) {
            return None;
        }
        if let Some(tolerance) = self.size {
            let best = state.vertices().first()?;
            for vertex in state.vertices() {
                let mut difference = vertex.clone();
                difference.scaled_add(-F::one(), best);
                let distance = difference.norm_infinity();
                if !distance.is_finite() || distance > tolerance {
                    return None;
                }
            }
        }
        Some(TerminationReason::SimplexTolerance)
    }
}

impl<P, S, So, G, X, C, T> Solver<P, S> for ConfiguredSolver<So, G, X, C, T>
where
    S: State,
    So: Solver<P, S>,
    G: Check<P, S>,
    X: Check<P, S>,
    C: Check<P, S>,
    T: Check<P, S>,
{
    type Error = So::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<S, Self::Error> {
        self.solver.init(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<(S, Option<TerminationReason>), Self::Error> {
        self.solver.next_iter(problem, state)
    }
    fn terminate(&self, state: &S) -> Option<TerminationReason> {
        self.solver.terminate(state)
    }
    fn reset_convergence(&mut self) {
        self.checked = None;
        self.reason = None;
        self.gradient.reset();
        self.step.reset();
        self.cost.reset();
        self.simplex.reset();
        self.solver.reset_convergence();
    }
    fn check_convergence(
        &mut self,
        problem: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        let boundary = (state.iter(), state.cost_evals());
        if self.checked == Some(boundary) {
            return self.reason;
        }
        self.checked = Some(boundary);
        self.reason = self
            .gradient
            .check(problem, state)
            .or_else(|| self.step.check(problem, state))
            .or_else(|| self.cost.check(problem, state))
            .or_else(|| self.simplex.check(problem, state))
            .or_else(|| self.solver.check_convergence(problem, state));
        self.reason
    }
}

impl<V, So: InitialState<V>, G, X, C, T> InitialState<V>
    for ConfiguredSolver<So, G, X, C, T>
{
    type State = So::State;
    fn seed(&self, x: &V) -> Self::State {
        self.solver.seed(x)
    }
}
impl<V, So: WarmStart<V>, G, X, C, T> WarmStart<V>
    for ConfiguredSolver<So, G, X, C, T>
{
}
impl<V, F: Scalar, So: crate::solver::MemeticInner<V, F>, G, X, C, T>
    crate::solver::MemeticInner<V, F> for ConfiguredSolver<So, G, X, C, T>
{
    fn seed_scaled(&self, x: &V, sigma: F) -> Self::State {
        self.solver.seed_scaled(x, sigma)
    }
}

impl<V, F, So, G, X, C, T> super::inner::ResumableInner<V, F>
    for ConfiguredSolver<So, G, X, C, T>
where
    F: Scalar,
    So: super::inner::ResumableInner<V, F>,
    G: Clone,
    X: Clone,
    C: Clone,
    T: Clone,
{
    type State = So::State;
    fn seed_chain(
        &self,
        x: &V,
        fx: F,
        scale: F,
        seed: u64,
    ) -> (Self, Self::State) {
        let (solver, state) = self.solver.seed_chain(x, fx, scale, seed);
        (
            Self {
                solver,
                gradient: self.gradient.clone(),
                step: self.step.clone(),
                cost: self.cost.clone(),
                simplex: self.simplex.clone(),
                checked: None,
                reason: None,
            },
            state,
        )
    }
    fn prepare_resume(&self, state: &mut Self::State) {
        self.solver.prepare_resume(state);
    }
    fn configure_segment(
        &mut self,
        state: &Self::State,
        control: &mut crate::RunControl<Self::State>,
    ) {
        self.solver.configure_segment(state, control);
    }
    fn segment_criteria(
        &self,
        state: &Self::State,
    ) -> Vec<Box<dyn TerminationCriterion<Self::State>>> {
        self.solver.segment_criteria(state)
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_absolute_gradient_tolerance<F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, GradientChecks<F>, X, C, T>
    where
        G: Into<GradientChecks<F>>,
    {
        let mut slot: GradientChecks<F> = self.gradient.into();
        slot.absolute = optional_tolerance(value).map(GradientTolerance);
        ConfiguredSolver {
            solver: self.solver,
            gradient: slot,
            step: self.step,
            cost: self.cost,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_relative_gradient_tolerance<F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, GradientChecks<F>, X, C, T>
    where
        G: Into<GradientChecks<F>>,
    {
        let mut slot: GradientChecks<F> = self.gradient.into();
        slot.relative =
            optional_tolerance(value).map(RelativeGradientTolerance::new);
        ConfiguredSolver {
            solver: self.solver,
            gradient: slot,
            step: self.step,
            cost: self.cost,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_absolute_step_tolerance<V, F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, G, StepChecks<V, F>, C, T>
    where
        X: Into<StepChecks<V, F>>,
    {
        let mut slot: StepChecks<V, F> = self.step.into();
        slot.absolute = optional_tolerance(value);
        slot.last = None;
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: slot,
            cost: self.cost,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_relative_step_tolerance<V, F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, G, StepChecks<V, F>, C, T>
    where
        X: Into<StepChecks<V, F>>,
    {
        let mut slot: StepChecks<V, F> = self.step.into();
        slot.relative = optional_tolerance(value);
        slot.last = None;
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: slot,
            cost: self.cost,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_absolute_cost_change_tolerance<F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, G, X, CostChecks<F>, T>
    where
        C: Into<CostChecks<F>>,
    {
        let mut slot: CostChecks<F> = self.cost.into();
        slot.absolute = optional_tolerance(value).map(CostTolerance::new);
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: self.step,
            cost: slot,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_relative_cost_change_tolerance<F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, G, X, CostChecks<F>, T>
    where
        C: Into<CostChecks<F>>,
    {
        let mut slot: CostChecks<F> = self.cost.into();
        slot.relative =
            optional_tolerance(value).map(RelativeCostTolerance::new);
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: self.step,
            cost: slot,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_absolute_simplex_size_tolerance<F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, G, X, C, SimplexChecks<F>>
    where
        T: Into<SimplexChecks<F>>,
    {
        let mut slot: SimplexChecks<F> = self.simplex.into();
        slot.size = optional_tolerance(value);
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: self.step,
            cost: self.cost,
            simplex: slot,
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C> ConfiguredSolver<So, G, X, C> {
    fn set_absolute_simplex_cost_tolerance<F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, G, X, C, SimplexChecks<F, ()>> {
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: self.step,
            cost: self.cost,
            simplex: SimplexChecks {
                size: (),
                cost: optional_tolerance(value),
            },
            checked: None,
            reason: None,
        }
    }
}

impl<So, G, X, C, F: Scalar, Size>
    ConfiguredSolver<So, G, X, C, SimplexChecks<F, Size>>
{
    fn set_absolute_simplex_cost_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.simplex.cost = optional_tolerance(value);
        self.checked = None;
        self.reason = None;
        self
    }
}

impl<So, G, X, C, T> ConfiguredSolver<So, G, X, C, T> {
    fn set_absolute_projected_gradient_tolerance<F: Scalar>(
        self,
        value: impl Into<Option<F>>,
    ) -> ConfiguredSolver<So, ProjectedGradientCheck<F>, X, C, T>
    where
        G: Into<ProjectedGradientCheck<F>>,
    {
        let mut slot: ProjectedGradientCheck<F> = self.gradient.into();
        slot.tolerance = optional_tolerance(value);
        ConfiguredSolver {
            solver: self.solver,
            gradient: slot,
            step: self.step,
            cost: self.cost,
            simplex: self.simplex,
            checked: None,
            reason: None,
        }
    }
}

mod builders;

mod forward;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BasicSimplexState, BasicState};

    #[test]
    fn step_history_recovers_after_nonfinite_observations() {
        let problem = Problem::new(());
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut checks = StepChecks {
                absolute: Some(0.0),
                relative: Some(0.0),
                last: None,
            };
            for x in [0.0, nonfinite, nonfinite, 0.0] {
                assert_eq!(
                    checks.check(&problem, &BasicState::new(vec![x])),
                    None
                );
            }
            assert_eq!(
                checks.check(&problem, &BasicState::new(vec![0.0])),
                Some(TerminationReason::ParamTolerance),
            );
        }
    }

    #[test]
    fn overflowing_step_norm_is_not_convergence() {
        let mut checks = StepChecks {
            absolute: Some(1e200),
            relative: Some(1.0),
            last: Some(vec![-1e200]),
        };
        assert_eq!(
            checks.check(&Problem::new(()), &BasicState::new(vec![1e200])),
            None,
        );
    }

    #[test]
    fn simplex_checks_reject_nonfinite_costs() {
        let problem = Problem::new(());
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for costs in [vec![0.0, nonfinite], vec![nonfinite, 0.0]] {
                let mut state =
                    BasicSimplexState::from_simplex(vec![vec![0.0]; 2]);
                state.costs = costs;
                let mut cost_only = SimplexChecks {
                    size: (),
                    cost: Some(0.0),
                };
                let mut combined = SimplexChecks {
                    size: Some(0.0),
                    cost: Some(0.0),
                };
                assert_eq!(cost_only.check(&problem, &state), None);
                assert_eq!(combined.check(&problem, &state), None);
            }
        }
    }
}
