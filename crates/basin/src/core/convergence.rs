//! Solver-owned convergence settings with compile-time backend capabilities.
//!
//! Solver setters return [`ConfiguredSolver`] when a check requires additional
//! vector operations or iterate history. Its fixed slots replace settings of
//! the same kind. No user-defined convergence policy is required.

pub mod native;
use crate::core::constraint::evaluation::BoxProblem;
pub use native::{NativeConvergenceDiagnostics, NativeConvergenceTest};

use super::inner::{InitialState, WarmStart};
use super::math::{
    ClampInPlace, NormInfinity, NormSquared, Scalar, ScaledAdd, VectorIndex,
    VectorLen,
};
use super::problem::Problem;
use super::solver::Solver;
use super::state::{GradientState, SimplexState, State};
use super::termination::{
    BinaryValue, ConvergenceCriterion, ConvergenceEvidence, ConvergenceTest,
    Termination, Threshold,
};
use crate::SolverStep;

/// A solver carrying fixed, optional convergence checks.
///
/// Construct this through the solver's `with_*_tolerance` setters. Setters
/// replace the corresponding check; distinct tests combine with OR. Nelder-Mead
/// owns its simplex size and cost-spread settings, which form an AND group.
/// New wrapper checks are disabled until configured. `None` disables a check;
/// zero is an exact-zero threshold. The original solver's algorithm controls
/// and safeguards remain active. Settings and history accompany composed inner
/// solves.
/// Trust-region solvers skip cost- and step-change checks after rejected
/// iterations, retaining the initial or last accepted iterate as the anchor.
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
}

impl<So> ConfiguredSolver<So> {
    pub(crate) fn new(solver: So) -> Self {
        Self {
            solver,
            gradient: (),
            step: (),
            cost: (),
            simplex: (),
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
        }
    }
}

impl<So: NativeConvergenceDiagnostics, G, X, C, T> NativeConvergenceDiagnostics
    for ConfiguredSolver<So, G, X, C, T>
{
    fn native_convergence_tests(&self) -> &[NativeConvergenceTest] {
        self.solver.native_convergence_tests()
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

pub(crate) fn stationarity_within_tolerance<F: Scalar>(
    norm: F,
    initial: F,
    absolute: Option<F>,
    relative: Option<F>,
) -> bool {
    if absolute.is_none() && relative.is_none() {
        return true;
    }
    norm.is_finite()
        && (absolute.is_some_and(|tol| norm <= tol)
            || relative.is_some_and(|tol| {
                initial.is_finite() && norm <= tol * initial
            }))
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for () {}
}

/// Implementation detail of the fixed convergence slots.
#[doc(hidden)]
pub trait Check<P, S: State>: sealed::Sealed {
    fn reset(&mut self);
    fn invalidate_cache(&mut self) {}
    fn check(
        &mut self,
        problem: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>>;
}

impl<P, S: State> Check<P, S> for () {
    fn reset(&mut self) {}
    fn check(
        &mut self,
        _: &Problem<P>,
        _: &S,
    ) -> Option<Termination<S::Float>> {
        None
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct RelativeGradientCheck<F: Scalar> {
    tol: F,
    initial_norm: Option<F>,
}

/// Fixed gradient-norm settings; constructed by solver setters.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientChecks<F: Scalar = f64> {
    absolute: Option<F>,
    relative: Option<RelativeGradientCheck<F>>,
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
            c.initial_norm = None;
        }
    }
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        let norm = state.gradient()?.norm_l2();
        if !norm.is_finite() {
            return None;
        }
        let absolute = self.absolute.and_then(|tol| {
            gradient_termination(
                ConvergenceTest::AbsoluteGradientSquared,
                norm,
                tol,
                None,
            )
        });
        let relative = self.relative.as_mut().and_then(|check| {
            let initial = *check.initial_norm.get_or_insert(norm);
            gradient_termination(
                ConvergenceTest::RelativeGradientSquared,
                norm,
                check.tol,
                Some(initial),
            )
        });
        absolute
            .into_iter()
            .chain(relative)
            .reduce(Termination::merge)
    }
}

fn gradient_termination<F: Scalar>(
    test: ConvergenceTest,
    norm: F,
    tolerance: F,
    initial: Option<F>,
) -> Option<Termination<F>> {
    let initial_norm = initial.unwrap_or_else(F::one);
    let value_factors = [norm, norm];
    let bound_factors = [tolerance, tolerance, initial_norm, initial_norm];
    if !BinaryValue::product_le(&value_factors, &bound_factors) {
        return None;
    }
    let value = norm * norm;
    let reference = initial.map(|norm| norm * norm);
    let bound = tolerance * tolerance * reference.unwrap_or_else(F::one);
    let represents = |value: F, factors: &[F]| {
        value.is_finite()
            && (value != F::zero() || factors.contains(&F::zero()))
    };
    // Equivalent product groupings can round in opposite directions.
    let termination = if represents(value, &value_factors)
        && represents(bound, &bound_factors)
        && reference.is_none_or(|value| represents(value, &[initial_norm]))
        && value <= bound
    {
        Termination::upper_bound(test, value, bound, tolerance, reference)
    } else {
        Termination::converged(ConvergenceCriterion::new(
            test,
            ConvergenceEvidence::FactoredUpperBound {
                value: BinaryValue::product(&value_factors),
                bound: BinaryValue::product(&bound_factors),
                tolerance,
                reference: initial
                    .map(|norm| BinaryValue::product(&[norm, norm])),
            },
        ))
    };
    Some(termination)
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
    P: BoxProblem<Param = S::Param>,
    S::Param: Clone + ScaledAdd<F> + ClampInPlace + NormInfinity<F>,
{
    fn reset(&mut self) {}
    fn check(
        &mut self,
        problem: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        let tol = self.tolerance?;
        let g = state.gradient()?;
        if !g.norm_infinity().is_finite() {
            return None;
        }
        let mut probe = state.param().clone();
        probe.scaled_add(-F::one(), g);
        probe.clamp_in_place(problem.inner().lower(), problem.inner().upper());
        probe.scaled_add(-F::one(), state.param());
        let value = probe.norm_infinity();
        (value <= tol).then(|| {
            Termination::upper_bound(
                ConvergenceTest::ProjectedGradient,
                value,
                tol,
                tol,
                None,
            )
        })
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
    cache: Option<CheckCache<F>>,
}
impl<V, F: Scalar> From<()> for StepChecks<V, F> {
    fn from(_: ()) -> Self {
        Self {
            absolute: None,
            relative: None,
            last: None,
            cache: None,
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
        self.cache = None;
    }
    fn invalidate_cache(&mut self) {
        self.cache = None;
    }
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        let boundary = (state.iter(), state.cost_evals());
        if let Some((previous, result)) = &self.cache {
            if *previous == boundary {
                return result.clone();
            }
        }
        let result = (|| {
            if self.absolute.is_none() && self.relative.is_none() {
                return None;
            }
            let current = state.param();
            let last = self.last.replace(current.clone())?;
            let mut difference = current.clone();
            difference.scaled_add(-F::one(), &last);
            let step = difference.norm_l2();
            if !step.is_finite() {
                return None;
            }
            let absolute =
                self.absolute.filter(|&tol| step <= tol).map(|tol| {
                    Termination::upper_bound(
                        ConvergenceTest::AbsoluteStep,
                        step,
                        tol,
                        tol,
                        None,
                    )
                });
            let relative = self.relative.and_then(|tol| {
                let norm = current.norm_l2();
                let bound = if tol == F::zero() {
                    F::zero()
                } else if norm.is_finite() {
                    tol * norm
                } else {
                    // Scale before squaring to avoid overflow in the reference norm.
                    let mut scaled = current.clone();
                    scaled.scaled_add(-F::one(), current);
                    scaled.scaled_add(tol, current);
                    scaled.norm_l2()
                };
                (step <= bound).then(|| {
                    Termination::upper_bound(
                        ConvergenceTest::RelativeStep,
                        step,
                        bound,
                        tol,
                        Some(norm),
                    )
                })
            });
            absolute
                .into_iter()
                .chain(relative)
                .reduce(Termination::merge)
        })();
        self.cache = Some((boundary, result.clone()));
        result
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CostChangeCheck<F: Scalar> {
    tol: F,
    last: Option<F>,
}

type CheckCache<F> = ((u64, u64), Option<Termination<F>>);

/// Fixed objective-change settings; constructed by solver setters.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CostChecks<F: Scalar = f64> {
    absolute: Option<CostChangeCheck<F>>,
    relative: Option<CostChangeCheck<F>>,
    cache: Option<CheckCache<F>>,
}
impl<F: Scalar> From<()> for CostChecks<F> {
    fn from(_: ()) -> Self {
        Self {
            absolute: None,
            relative: None,
            cache: None,
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
        self.cache = None;
        if let Some(c) = &mut self.absolute {
            c.last = None;
        }
        if let Some(c) = &mut self.relative {
            c.last = None;
        }
    }
    fn invalidate_cache(&mut self) {
        self.cache = None;
    }
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        let boundary = (state.iter(), state.cost_evals());
        if let Some((previous, result)) = &self.cache {
            if *previous == boundary {
                return result.clone();
            }
        }
        if !state.cost().is_finite() {
            <Self as Check<P, S>>::reset(self);
            return None;
        }
        let current = state.cost();
        let absolute = self.absolute.as_mut().and_then(|check| {
            let last = check.last.replace(current)?;
            let value = (last - current).abs();
            (value <= check.tol).then(|| {
                Termination::upper_bound(
                    ConvergenceTest::AbsoluteCostChange,
                    value,
                    check.tol,
                    check.tol,
                    Some(last),
                )
            })
        });
        let relative = self.relative.as_mut().and_then(|check| {
            let last = check.last.replace(current)?;
            let value = (last - current).abs();
            let bound = check.tol * last.abs();
            (value <= bound).then(|| {
                Termination::upper_bound(
                    ConvergenceTest::RelativeCostChange,
                    value,
                    bound,
                    check.tol,
                    Some(last),
                )
            })
        });
        let result = absolute
            .into_iter()
            .chain(relative)
            .reduce(Termination::merge);
        self.cache = Some((boundary, result.clone()));
        result
    }
}

/// Solver-owned simplex checks; size and cost spread use AND, other tests use OR.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimplexChecks<F: Scalar = f64> {
    pub(crate) size: Option<F>,
    pub(crate) relative_size: Option<F>,
    pub(crate) cost: Option<F>,
    pub(crate) relative_cost: Option<F>,
    initial_cost_spread: Option<F>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) cost_std_dev: Option<F>,
}
impl<F: Scalar> From<()> for SimplexChecks<F> {
    fn from(_: ()) -> Self {
        Self {
            size: None,
            relative_size: None,
            cost: None,
            relative_cost: None,
            initial_cost_spread: None,
            cost_std_dev: None,
        }
    }
}
impl<F: Scalar> SimplexChecks<F> {
    pub(crate) fn nelder_mead_defaults() -> Self {
        let absolute = F::from_f64(1e-8).unwrap();
        // Corpus runs expose premature f32 stops with sqrt(epsilon), while
        // tighter settings approach float-resolution stalls. Keep the f64
        // default and allow 32 epsilon for lower-precision arithmetic.
        let relative = F::from_f64(f64::EPSILON.sqrt())
            .unwrap()
            .max(F::from_f64(32.0).unwrap() * F::epsilon());
        Self {
            size: Some(absolute),
            relative_size: Some(relative),
            cost: Some(absolute),
            relative_cost: Some(relative),
            initial_cost_spread: None,
            cost_std_dev: None,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.initial_cost_spread = None;
    }

    pub(crate) fn capture_initial_cost_spread(&mut self, costs: &[F]) {
        if self.initial_cost_spread.is_none() {
            self.initial_cost_spread = simplex_cost_spread(costs);
        }
    }

    fn check_cost_std_dev(&self, costs: &[F]) -> Option<Termination<F>> {
        let tolerance = self.cost_std_dev?;
        if !tolerance.is_finite() || tolerance < F::zero() {
            return None;
        }
        let &first = costs.first()?;
        // A nonconstant sample can round to zero at subnormal scales.
        if tolerance == F::zero() && costs.iter().any(|&cost| cost != first) {
            return None;
        }
        let value = sample_std_dev(costs)?;
        (value <= tolerance).then(|| {
            Termination::converged(ConvergenceCriterion::upper_bound(
                ConvergenceTest::AbsoluteSimplexCostStdDev,
                value,
                tolerance,
                tolerance,
                None,
            ))
        })
    }
}
impl<F: Scalar> sealed::Sealed for SimplexChecks<F> {}

fn simplex_cost_spread<F: Scalar>(costs: &[F]) -> Option<F> {
    let &best = costs.first()?;
    costs.iter().try_fold(F::zero(), |spread, &cost| {
        let difference = (cost - best).abs();
        (cost.is_finite() && difference.is_finite())
            .then(|| spread.max(difference))
    })
}

fn sample_std_dev<F: Scalar>(values: &[F]) -> Option<F> {
    if values.len() < 2 {
        return None;
    }
    let mut minimum = F::infinity();
    let mut maximum = F::neg_infinity();
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if minimum == maximum {
        return Some(F::zero());
    }

    // Center before scaling to retain small differences at large offsets.
    // Halving first also handles finite samples whose range overflows.
    let half = F::from_f64(0.5)?;
    let range = maximum - minimum;
    let factor = if range.is_finite() { F::one() } else { half };
    let scale = if range.is_finite() {
        range
    } else {
        maximum * half - minimum * half
    };
    let normalized = |value: F| (value * factor - minimum * factor) / scale;
    let mean = values.iter().map(|&value| normalized(value)).sum::<F>()
        / F::from_usize(values.len())?;
    let variance = values
        .iter()
        .map(|&value| (normalized(value) - mean).powi(2))
        .sum::<F>()
        / F::from_usize(values.len() - 1)?;
    let std_dev = (scale * variance.sqrt()) / factor;
    std_dev.is_finite().then_some(std_dev)
}

fn simplex_threshold<F: Scalar>(
    value: F,
    absolute: Option<F>,
    relative: Option<F>,
    reference: F,
) -> Option<Threshold<F>> {
    let relative_bound = relative.map_or(F::zero(), |tol| tol * reference);
    if !value.is_finite()
        || !reference.is_finite()
        || !relative_bound.is_finite()
        || absolute
            .into_iter()
            .chain(relative)
            .any(|tol| !tol.is_finite() || tol < F::zero())
    {
        return None;
    }
    let tolerance = absolute.unwrap_or(F::zero()).max(relative_bound);
    (value <= tolerance).then_some(Threshold {
        value,
        tolerance,
        absolute_tolerance: absolute,
        relative_tolerance: relative,
        reference: relative.map(|_| reference),
    })
}

impl<F: Scalar> SimplexChecks<F> {
    fn check_collapse<S>(&mut self, state: &S) -> Option<Termination<F>>
    where
        S: SimplexState<Float = F>,
        S::Param: VectorLen + VectorIndex<F>,
    {
        let check_size = self.size.is_some() || self.relative_size.is_some();
        let check_cost = self.cost.is_some() || self.relative_cost.is_some();
        if !check_size && !check_cost {
            return None;
        }
        let spread = simplex_cost_spread(state.costs())?;
        let initial_spread = *self.initial_cost_spread.get_or_insert(spread);
        let cost_spread = if check_cost {
            Some(simplex_threshold(
                spread,
                self.cost,
                self.relative_cost,
                state.costs()[0].abs().max(initial_spread),
            )?)
        } else {
            None
        };
        let size = if check_size {
            let best = state.vertices().first()?;
            let mut norm = F::zero();
            let mut max_distance = F::zero();
            for vertex in state.vertices() {
                if vertex.vec_len() != best.vec_len() {
                    return None;
                }
                for j in 0..best.vec_len() {
                    let anchor = best.get_scalar(j);
                    let coordinate = vertex.get_scalar(j);
                    let distance = (coordinate - anchor).abs();
                    if !anchor.is_finite()
                        || !coordinate.is_finite()
                        || !distance.is_finite()
                    {
                        return None;
                    }
                    norm = norm.max(anchor.abs());
                    max_distance = max_distance.max(distance);
                }
            }
            Some(simplex_threshold(
                max_distance,
                self.size,
                self.relative_size,
                norm,
            )?)
        } else {
            None
        };
        Some(Termination::converged(ConvergenceCriterion {
            test: ConvergenceTest::Simplex,
            evidence: ConvergenceEvidence::Simplex { size, cost_spread },
        }))
    }
}

impl<P, S, F> Check<P, S> for SimplexChecks<F>
where
    F: Scalar,
    S: SimplexState<Float = F>,
    S::Param: VectorLen + VectorIndex<F>,
{
    fn reset(&mut self) {
        self.reset();
    }
    fn check(
        &mut self,
        _: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        if state.vertices().len() != state.costs().len() {
            return None;
        }
        [
            self.check_collapse(state),
            self.check_cost_std_dev(state.costs()),
        ]
        .into_iter()
        .flatten()
        .reduce(Termination::merge)
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
    ) -> Result<SolverStep<S>, Self::Error> {
        let step = self.solver.next_iter(problem, state)?;
        // A partial step can publish new state without advancing either cache key.
        self.step.invalidate_cache();
        self.cost.invalidate_cache();
        Ok(step)
    }
    fn terminate(&self, state: &S) -> Option<Termination<S::Float>> {
        self.solver.terminate(state)
    }
    fn should_check_iterate_change(&self) -> bool {
        self.solver.should_check_iterate_change()
    }
    fn reset_convergence(&mut self) {
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
    ) -> Option<Termination<S::Float>> {
        let check_change = self.solver.should_check_iterate_change();
        let checks = [
            self.gradient.check(problem, state),
            if check_change {
                self.step.check(problem, state)
            } else {
                None
            },
            if check_change {
                self.cost.check(problem, state)
            } else {
                None
            },
            self.simplex.check(problem, state),
            self.solver.check_convergence(problem, state),
        ];
        checks.into_iter().flatten().reduce(Termination::merge)
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
    fn seed_scaled(&mut self, x: &V, sigma: F) -> Self::State {
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
            },
            state,
        )
    }
    fn seeded_chain_is_initialized(&self) -> bool {
        self.solver.seeded_chain_is_initialized()
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
        slot.absolute = optional_tolerance(value);
        ConfiguredSolver {
            solver: self.solver,
            gradient: slot,
            step: self.step,
            cost: self.cost,
            simplex: self.simplex,
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
            optional_tolerance(value).map(|tol| RelativeGradientCheck {
                tol,
                initial_norm: None,
            });
        ConfiguredSolver {
            solver: self.solver,
            gradient: slot,
            step: self.step,
            cost: self.cost,
            simplex: self.simplex,
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
        slot.cache = None;
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: slot,
            cost: self.cost,
            simplex: self.simplex,
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
        slot.cache = None;
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: slot,
            cost: self.cost,
            simplex: self.simplex,
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
        slot.cache = None;
        slot.absolute = optional_tolerance(value)
            .map(|tol| CostChangeCheck { tol, last: None });
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: self.step,
            cost: slot,
            simplex: self.simplex,
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
        slot.cache = None;
        slot.relative = optional_tolerance(value)
            .map(|tol| CostChangeCheck { tol, last: None });
        ConfiguredSolver {
            solver: self.solver,
            gradient: self.gradient,
            step: self.step,
            cost: slot,
            simplex: self.simplex,
        }
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
        }
    }
}

mod builders;

mod forward;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Executor, PointState, SimplexProgress, TerminationCode,
        TerminationStage,
    };

    struct PartialWithoutEvaluation(bool);

    impl Solver<(), PointState<Vec<f64>>> for PartialWithoutEvaluation {
        type Error = std::convert::Infallible;

        fn init(
            &mut self,
            _: &mut Problem<()>,
            mut state: PointState<Vec<f64>>,
        ) -> Result<PointState<Vec<f64>>, Self::Error> {
            state.replace(vec![2.0], 0.0);
            Ok(state)
        }

        fn next_iter(
            &mut self,
            _: &mut Problem<()>,
            mut state: PointState<Vec<f64>>,
        ) -> Result<SolverStep<PointState<Vec<f64>>>, Self::Error> {
            state.replace(vec![if self.0 { 0.0 } else { 1.0 }], 0.0);
            let completed = self.0;
            self.0 = true;
            Ok(SolverStep {
                state,
                completed,
                termination: None,
            })
        }
    }

    #[test]
    fn partial_step_without_evaluation_checks_step_and_cost_change() {
        let solver = ConfiguredSolver::new(PartialWithoutEvaluation(false))
            .set_absolute_step_tolerance::<Vec<f64>, f64>(1.0)
            .set_absolute_cost_change_tolerance(0.0);
        let result = Executor::new((), solver, PointState::new(vec![2.0]))
            .max_iter(1)
            .run()
            .unwrap();
        assert_eq!(result.report.stage, TerminationStage::Boundary);
        assert_eq!(result.iter(), 0);
        assert_eq!(result.report.code(), TerminationCode::ParamTolerance);
        let Termination::Converged(convergence) = result.report.termination
        else {
            panic!("expected convergence after the partial step");
        };
        assert_eq!(convergence.criteria().len(), 2);
        assert_eq!(
            convergence.criteria()[0].test,
            ConvergenceTest::AbsoluteStep
        );
        assert_eq!(
            convergence.criteria()[1].test,
            ConvergenceTest::AbsoluteCostChange
        );
    }

    #[test]
    fn step_history_recovers_after_nonfinite_observations() {
        let problem = Problem::new(());
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut checks = StepChecks {
                absolute: Some(0.0),
                relative: Some(0.0),
                last: None,
                cache: None,
            };
            for x in [0.0, nonfinite, nonfinite, 0.0] {
                checks.cache = None;
                assert_eq!(
                    checks.check(&problem, &PointState::new(vec![x])),
                    None
                );
            }
            checks.cache = None;
            assert_eq!(
                checks
                    .check(&problem, &PointState::new(vec![0.0]))
                    .as_ref()
                    .map(|stop| stop.code()),
                Some(crate::TerminationCode::ParamTolerance),
            );
        }
    }

    #[test]
    fn overflowing_step_norm_is_not_convergence() {
        let mut checks = StepChecks {
            absolute: Some(1e200),
            relative: Some(1.0),
            last: Some(vec![-1e200]),
            cache: None,
        };
        assert_eq!(
            checks.check(&Problem::new(()), &PointState::new(vec![1e200])),
            None,
        );
    }

    #[test]
    fn sample_standard_deviation_requires_two_costs() {
        assert_eq!(sample_std_dev::<f64>(&[]), None);
        assert_eq!(sample_std_dev(&[1.0_f64]), None);
    }

    #[test]
    fn simplex_checks_reject_nonfinite_costs() {
        let problem = Problem::new(());
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for costs in [vec![0.0, nonfinite], vec![nonfinite, 0.0]] {
                let mut state =
                    SimplexProgress::from_simplex(vec![vec![0.0]; 2]);
                state.replace(vec![vec![0.0]; 2], costs).unwrap();
                let mut cost_only = SimplexChecks::from(());
                cost_only.cost = Some(0.0);
                let mut combined = cost_only.clone();
                combined.size = Some(0.0);
                assert_eq!(cost_only.check(&problem, &state), None);
                assert_eq!(combined.check(&problem, &state), None);
            }
        }
    }

    #[test]
    fn cost_change_history_resets_between_runs_and_after_nonfinite_costs() {
        use crate::PointState;

        let problem = Problem::new(());
        for relative in [false, true] {
            let mut checks = CostChecks::<f64>::from(());
            let check = Some(CostChangeCheck {
                tol: 0.0,
                last: None,
            });
            let expected = if relative {
                checks.relative = check;
                crate::TerminationCode::RelativeCostTolerance
            } else {
                checks.absolute = check;
                crate::TerminationCode::CostTolerance
            };
            let mut state = PointState::new(1.0);
            state.replace(1.0, 1.0);
            assert_eq!(checks.check(&problem, &state), None);
            state.increment_iter();
            assert_eq!(
                checks
                    .check(&problem, &state)
                    .as_ref()
                    .map(|stop| stop.code()),
                Some(expected)
            );
            <CostChecks as Check<(), PointState<f64>>>::reset(&mut checks);
            assert_eq!(checks.check(&problem, &state), None);
            for cost in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                state.increment_iter();
                state.replace(1.0, cost);
                assert_eq!(checks.check(&problem, &state), None);
                state.increment_iter();
                state.replace(1.0, 1.0);
                assert_eq!(checks.check(&problem, &state), None);
                state.increment_iter();
                assert_eq!(
                    checks
                        .check(&problem, &state)
                        .as_ref()
                        .map(|stop| stop.code()),
                    Some(expected)
                );
            }
        }
    }
}
