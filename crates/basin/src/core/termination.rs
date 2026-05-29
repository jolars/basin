//! Termination layer: the [`TerminationCriterion`] trait and the
//! framework-level criteria solvers can be terminated by. Each criterion
//! bounds on the minimum state shape it needs (tenet 3 in `AGENTS.md`),
//! so mismatches are compile errors rather than runtime no-ops.

use web_time::{Duration, Instant};

use crate::core::constraint::BoxConstraints;
use crate::core::math::{ClampInPlace, NormInfinity, NormSquared, ScaledAdd};
use crate::core::state::{GradientState, SimplexState, State};

/// Why the executor stopped. Returned on
/// [`OptimizationResult::reason`](crate::core::executor::OptimizationResult::reason)
/// and the various step / run hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// `state.iter() >= max_iter`.
    MaxIter,
    /// Cost-evaluation budget exhausted.
    MaxCostEvals,
    /// Gradient-evaluation budget exhausted.
    MaxGradientEvals,
    /// `‖∇f(x)‖ ≤ tol`.
    GradientTolerance,
    /// `‖∇f(x_k)‖ ≤ tol · ‖∇f(x_0)‖` — gradient norm relative to the
    /// initial gradient (scale-invariant first-order stationarity).
    RelativeGradientTolerance,
    /// `‖x − π_C(x − ∇f(x))‖_∞ ≤ tol` — projected-gradient stationarity
    /// for box-constrained problems. Collapses to the unconstrained
    /// gradient norm when no constraint is active.
    ProjectedGradientTolerance,
    /// `‖x_k − x_{k−1}‖ ≤ tol`.
    ParamTolerance,
    /// `‖x_k − x_{k−1}‖ ≤ tol · ‖x_k‖` — scale-invariant step test
    /// (MINPACK `xtol`).
    RelativeParamTolerance,
    /// `|f_k − f_{k−1}| ≤ tol`.
    CostTolerance,
    /// `|f_k − f_{k−1}| ≤ tol · |f_{k−1}|` — scale-invariant cost
    /// reduction test (MINPACK `ftol`).
    RelativeCostTolerance,
    /// `f(x_k) ≤ target` — user-supplied target cost reached
    /// (NLopt's `stopval` / SciPy's `f_min`).
    TargetCost,
    /// Best-so-far cost has not improved by more than `tol` in
    /// `patience` consecutive iterations — the early-stopping pattern.
    NoImprovement,
    /// Simplex collapsed below the configured tolerance.
    SimplexTolerance,
    /// Wall-clock time limit reached.
    MaxTime,
    /// Solver determined it has converged (e.g. fixed point reached).
    SolverConverged,
    /// Solver cannot make further progress (e.g. line search failure).
    SolverFailed,
}

impl TerminationReason {
    /// Whether this reason represents an unrecoverable failure that an
    /// outer solver should bubble (rather than consume and continue).
    ///
    /// Currently only [`SolverFailed`](Self::SolverFailed) qualifies —
    /// [`MaxIter`](Self::MaxIter), the `*Tolerance` reasons, and
    /// [`SolverConverged`](Self::SolverConverged) are all "clean stops"
    /// that an outer solver running an inner per outer iter should treat
    /// as "result is fine, move on". See `AGENTS.md` "Solver
    /// composition" for the failure-routing contract.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::SolverFailed)
    }
}

/// A pluggable termination check evaluated by the executor.
///
/// # Contract
///
/// - **Caller must:** register criteria with
///   [`Executor::terminate_on`](crate::core::executor::Executor::terminate_on)
///   before calling [`Executor::run`](crate::core::executor::Executor::run).
///   Insertion order matters: criteria are checked in the order they
///   were registered, and the **first to return `Some(_)` halts the run**.
///   The built-in [`MaxIter`] limit (settable via
///   [`Executor::max_iter`](crate::core::executor::Executor::max_iter))
///   is checked *before* user criteria each iteration.
/// - **Caller must:** rely on the bound on `S` to encode capability:
///   e.g. [`GradientTolerance`] requires `S: GradientState`, so handing
///   it to a derivative-free solver is a compile error, not a runtime
///   "N/A" (tenet 3 in `AGENTS.md`).
/// - **Implementor must:** treat [`check`](Self::check) as side-effect
///   free *with respect to the optimization*. Internal state for criteria
///   that need history (e.g. [`ParamTolerance`], [`CostTolerance`])
///   lives inside the criterion itself.
/// - Criteria are checked *before* each iteration including iter 0 so
///   an already-optimal initial point exits immediately. See the
///   [`executor`](crate::core::executor) module docs for the full
///   per-iteration ordering.
pub trait TerminationCriterion<S> {
    /// Inspect the current state and return `Some(reason)` to halt the
    /// run, or `None` to continue. Called once per iteration before the
    /// solver's `next_iter`.
    fn check(&mut self, state: &S) -> Option<TerminationReason>;
}

/// Stop after `state.iter() >= n` iterations.
pub struct MaxIter(pub u64);

impl<S: State> TerminationCriterion<S> for MaxIter {
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        if state.iter() >= self.0 {
            Some(TerminationReason::MaxIter)
        } else {
            None
        }
    }
}

/// Stop after `state.cost_evals() >= n` cost-function evaluations.
/// Lagarias et al. (1998) (T3) — the budget users actually care about
/// when one iteration can spend many evals (line search, Nelder-Mead
/// shrink).
pub struct MaxCostEvals(pub u64);

impl<S: State> TerminationCriterion<S> for MaxCostEvals {
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        if state.cost_evals() >= self.0 {
            Some(TerminationReason::MaxCostEvals)
        } else {
            None
        }
    }
}

/// Stop after `state.gradient_evals() >= n` gradient evaluations. Bound
/// on `S: GradientState` so it can't be paired with derivative-free
/// solvers — a compile error rather than a silently no-op criterion.
pub struct MaxGradientEvals(pub u64);

impl<S: GradientState> TerminationCriterion<S> for MaxGradientEvals {
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        if state.gradient_evals() >= self.0 {
            Some(TerminationReason::MaxGradientEvals)
        } else {
            None
        }
    }
}

/// Stop when `‖∇f(x)‖ ≤ tol`. Skipped silently when the state has no
/// gradient populated yet (e.g. iter 0 before `init` has run).
///
/// Requires `S: GradientState` — pairing with a derivative-free solver
/// is a compile error.
pub struct GradientTolerance(pub f64);

impl<S> TerminationCriterion<S> for GradientTolerance
where
    S: GradientState,
    S::Param: NormSquared,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let g = state.gradient()?;
        if g.norm_squared() <= self.0 * self.0 {
            Some(TerminationReason::GradientTolerance)
        } else {
            None
        }
    }
}

/// Stop when `‖∇f(x_k)‖ ≤ tol · ‖∇f(x_0)‖` — the gradient norm relative
/// to the gradient at the starting point. The scale-invariant analogue
/// of [`GradientTolerance`]: scaling the objective by a constant scales
/// every gradient by the same constant, so the ratio — and hence the
/// stopping point — is unchanged, letting one `tol` port across
/// objectives of different magnitude.
///
/// Unlike [`RelativeCostTolerance`] / [`RelativeParamTolerance`] (which
/// normalize by the *current* iterate's quantity), this normalizes by
/// the *initial* gradient. The gradient → 0 at a minimizer, so a
/// relative-to-current gradient test would be degenerate (`0/0`);
/// relative-to-initial is the standard first-order rule (`‖∇f_k‖ ≤
/// ε·‖∇f_0‖`). Pair with an absolute [`GradientTolerance`] when the
/// starting gradient may itself be tiny.
///
/// Captures `‖∇f(x_0)‖` on the first check at which a gradient is
/// populated. Requires `S: GradientState` — pairing with a
/// derivative-free solver is a compile error. Skipped silently while
/// the state has no gradient populated yet.
pub struct RelativeGradientTolerance {
    tol: f64,
    initial_norm_squared: Option<f64>,
}

impl RelativeGradientTolerance {
    /// New tolerance with the given relative gradient-norm bound.
    pub fn new(tol: f64) -> Self {
        Self {
            tol,
            initial_norm_squared: None,
        }
    }
}

impl<S> TerminationCriterion<S> for RelativeGradientTolerance
where
    S: GradientState,
    S::Param: NormSquared,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let g = state.gradient()?;
        let norm_squared = g.norm_squared();
        // Anchor on the first populated gradient (the initial iterate).
        let initial = *self.initial_norm_squared.get_or_insert(norm_squared);
        // ‖∇f_k‖ ≤ tol·‖∇f_0‖ ⟺ ‖∇f_k‖² ≤ tol²·‖∇f_0‖², avoiding a sqrt.
        if norm_squared <= self.tol * self.tol * initial {
            Some(TerminationReason::RelativeGradientTolerance)
        } else {
            None
        }
    }
}

/// Stop when `‖x − π_C(x − ∇f(x))‖_∞ ≤ tol`, the canonical first-order
/// optimality measure for box-constrained minimization. `π_C` is the
/// projection onto the box `[lower, upper]` carried by this criterion.
///
/// The metric is zero exactly at a KKT point of the box-constrained
/// problem: when no constraint is active it collapses to `‖∇f‖_∞`;
/// when a face is active it collapses to the ∞-norm of the gradient
/// components corresponding to *inactive* coordinates. This is why
/// [`GradientTolerance`] is the wrong metric for constrained problems —
/// `‖∇f‖` need not vanish at a constrained optimum (the gradient
/// points into an active face), but the projected-gradient measure
/// always does.
///
/// Construct from explicit bounds with [`new`](Self::new), or clone
/// them off a [`BoxConstraints`] problem with
/// [`from_problem`](Self::from_problem). The bounds are stored once at
/// construction; the criterion does not call back into the problem.
///
/// Requires `S: GradientState` and `S::Param` to implement
/// [`ScaledAdd<f64>`], [`ClampInPlace`], [`NormInfinity`], and `Clone`.
/// Skipped silently when the state has no gradient populated yet
/// (e.g. iter 0 before `init` has run).
pub struct ProjectedGradientTolerance<P> {
    lower: P,
    upper: P,
    tol: f64,
}

impl<P> ProjectedGradientTolerance<P> {
    /// New criterion with explicit bounds.
    pub fn new(lower: P, upper: P, tol: f64) -> Self {
        Self { lower, upper, tol }
    }

    /// New criterion that clones its bounds off a [`BoxConstraints`]
    /// problem.
    pub fn from_problem<Pr>(problem: &Pr, tol: f64) -> Self
    where
        Pr: BoxConstraints<Param = P>,
        P: Clone,
    {
        Self {
            lower: problem.lower().clone(),
            upper: problem.upper().clone(),
            tol,
        }
    }
}

impl<S, P> TerminationCriterion<S> for ProjectedGradientTolerance<P>
where
    S: GradientState + State<Param = P>,
    P: ScaledAdd<f64> + ClampInPlace + NormInfinity + Clone,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let g = state.gradient()?;
        let mut probe = state.param().clone(); // x
        probe.scaled_add(-1.0, g); // x − ∇f
        probe.clamp_in_place(&self.lower, &self.upper); // π(x − ∇f)
        probe.scaled_add(-1.0, state.param()); // π(x − ∇f) − x
        if probe.norm_infinity() <= self.tol {
            Some(TerminationReason::ProjectedGradientTolerance)
        } else {
            None
        }
    }
}

/// Stop when `‖x_k − x_{k−1}‖ ≤ tol`. Holds its own copy of the previous
/// iterate so it doesn't depend on state-side history.
pub struct ParamTolerance<P> {
    tol_squared: f64,
    last: Option<P>,
}

impl<P> ParamTolerance<P> {
    /// New tolerance with the given absolute step bound.
    pub fn new(tol: f64) -> Self {
        Self {
            tol_squared: tol * tol,
            last: None,
        }
    }
}

impl<S, P> TerminationCriterion<S> for ParamTolerance<P>
where
    S: State<Param = P>,
    P: ScaledAdd<f64> + NormSquared + Clone,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let curr = state.param();
        let triggered = if let Some(last) = &self.last {
            let mut diff = curr.clone();
            diff.scaled_add(-1.0, last);
            diff.norm_squared() <= self.tol_squared
        } else {
            false
        };
        self.last = Some(curr.clone());
        triggered.then_some(TerminationReason::ParamTolerance)
    }
}

/// Stop when `‖x_k − x_{k−1}‖ ≤ tol · ‖x_k‖` — the scale-invariant
/// analogue of [`ParamTolerance`], matching MINPACK's `xtol`. Holds its
/// own copy of the previous iterate.
///
/// Unlike the absolute [`ParamTolerance`], the bound scales with the
/// magnitude of the iterate, so a single `tol` is portable across
/// problems whose parameters live at very different scales. Near
/// `x = 0` the relative bound collapses (the right-hand side → 0), so
/// pair it with an absolute [`ParamTolerance`] when the optimum may sit
/// at the origin.
pub struct RelativeParamTolerance<P> {
    tol: f64,
    last: Option<P>,
}

impl<P> RelativeParamTolerance<P> {
    /// New tolerance with the given relative step bound.
    pub fn new(tol: f64) -> Self {
        Self { tol, last: None }
    }
}

impl<S, P> TerminationCriterion<S> for RelativeParamTolerance<P>
where
    S: State<Param = P>,
    P: ScaledAdd<f64> + NormSquared + Clone,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let curr = state.param();
        let triggered = if let Some(last) = &self.last {
            let mut diff = curr.clone();
            diff.scaled_add(-1.0, last);
            // ‖Δx‖ ≤ tol·‖x_k‖ ⟺ ‖Δx‖² ≤ tol²·‖x_k‖², avoiding a sqrt.
            diff.norm_squared() <= self.tol * self.tol * curr.norm_squared()
        } else {
            false
        };
        self.last = Some(curr.clone());
        triggered.then_some(TerminationReason::RelativeParamTolerance)
    }
}

/// Stop when `|f_k − f_{k−1}| ≤ tol`. Holds its own copy of the previous
/// cost.
pub struct CostTolerance {
    tol: f64,
    last: Option<f64>,
}

impl CostTolerance {
    /// New tolerance with the given absolute cost-change bound.
    pub fn new(tol: f64) -> Self {
        Self { tol, last: None }
    }
}

impl<S> TerminationCriterion<S> for CostTolerance
where
    S: State<Float = f64>,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let curr = state.cost();
        let triggered = self
            .last
            .is_some_and(|l| (l - curr).abs() <= self.tol && curr.is_finite());
        self.last = Some(curr);
        triggered.then_some(TerminationReason::CostTolerance)
    }
}

/// Stop when `|f_k − f_{k−1}| ≤ tol · |f_{k−1}|` — the scale-invariant
/// analogue of [`CostTolerance`], matching MINPACK's `ftol` (whose
/// `actred = 1 − (‖r_k‖/‖r_{k−1}‖)²` reduces to this relative-cost test
/// for `f = ½‖r‖²`). Holds its own copy of the previous cost.
///
/// The bound scales with the current cost level, so one `tol` is
/// portable across problems whose cost magnitudes differ by orders of
/// magnitude (e.g. least-squares residuals carrying different
/// normalizations). Near `f = 0` the relative bound collapses, so pair
/// it with an absolute [`CostTolerance`] when the optimum cost is zero.
pub struct RelativeCostTolerance {
    tol: f64,
    last: Option<f64>,
}

impl RelativeCostTolerance {
    /// New tolerance with the given relative cost-change bound.
    pub fn new(tol: f64) -> Self {
        Self { tol, last: None }
    }
}

impl<S> TerminationCriterion<S> for RelativeCostTolerance
where
    S: State<Float = f64>,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let curr = state.cost();
        let triggered = self
            .last
            .is_some_and(|l| curr.is_finite() && (l - curr).abs() <= self.tol * l.abs());
        self.last = Some(curr);
        triggered.then_some(TerminationReason::RelativeCostTolerance)
    }
}

/// Stop when `f(x_k) ≤ target` — a user-supplied target cost level.
/// This is NLopt's `stopval` and SciPy's `f_min`: an absolute *level*
/// stop, not a change-in-cost stop like [`CostTolerance`].
///
/// Most useful for global / stochastic solvers (random search, CMA-ES,
/// the steady-state GA) where "good enough" is a more natural stopping
/// rule than asymptotic convergence, and for benchmarking
/// ("how long until the solver hits cost ≤ ε?").
///
/// # Semantics under different state shapes
///
/// `state.cost()` means *best-so-far* for [`BasicSimplexState`] and
/// [`BasicPopulationState`] (it returns `costs[0]`, the best vertex /
/// individual), and *current iterate* for single-iterate states like
/// [`BasicState`] / [`QuasiNewtonState`]. So on a non-monotone solver
/// (e.g. CMA-ES sampling, a line search that allows transient
/// increases) this fires once the best ever seen drops to the target,
/// not on a transient dip of the current iterate alone.
///
/// [`BasicState`]: crate::core::state::BasicState
/// [`BasicSimplexState`]: crate::core::state::BasicSimplexState
/// [`BasicPopulationState`]: crate::core::state::BasicPopulationState
/// [`QuasiNewtonState`]: crate::core::state::QuasiNewtonState
pub struct TargetCost(pub f64);

impl<S> TerminationCriterion<S> for TargetCost
where
    S: State<Float = f64>,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        (state.cost() <= self.0).then_some(TerminationReason::TargetCost)
    }
}

/// Stop when the best cost seen so far has not improved by more than
/// `tol` in `patience` consecutive iterations — the early-stopping
/// pattern from ML, minus the validation set.
///
/// Improvement is counted strictly: an observation counts as
/// improvement iff `curr < best_so_far − tol`. So `tol` is the minimum
/// drop that resets the patience counter (Keras calls this
/// `min_delta`). `tol = 0.0` means "any strict decrease resets".
///
/// Most useful for stochastic / global / non-monotone solvers (random
/// search, CMA-ES, the steady-state GA, future basin-hopping) where
/// the one-step [`CostTolerance`] family fires spuriously on accidental
/// small `|Δf|`. By tracking the running minimum of `state.cost()`
/// across all checks, this criterion is robust against transient
/// increases (CMA-ES generation-to-generation variation, non-monotone
/// line search, GA replacement noise).
///
/// # Semantics under different state shapes
///
/// Tracks `min` of `state.cost()` across checks. For
/// [`BasicSimplexState`] / [`BasicPopulationState`] that's the best
/// vertex / individual at each generation; for [`BasicState`] /
/// [`QuasiNewtonState`] it's the current iterate's cost. Either way
/// the running minimum is monotone non-increasing, so "improvement"
/// has a single consistent meaning.
///
/// [`BasicState`]: crate::core::state::BasicState
/// [`BasicSimplexState`]: crate::core::state::BasicSimplexState
/// [`BasicPopulationState`]: crate::core::state::BasicPopulationState
/// [`QuasiNewtonState`]: crate::core::state::QuasiNewtonState
pub struct NoImprovement {
    patience: u64,
    tol: f64,
    best: Option<f64>,
    stalled: u64,
}

impl NoImprovement {
    /// New criterion that fires after `patience` consecutive checks
    /// without an improvement of more than `tol`.
    pub fn new(patience: u64, tol: f64) -> Self {
        Self {
            patience,
            tol,
            best: None,
            stalled: 0,
        }
    }
}

impl<S> TerminationCriterion<S> for NoImprovement
where
    S: State<Float = f64>,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let curr = state.cost();
        let improved = match self.best {
            None => true,
            Some(best) => curr.is_finite() && curr < best - self.tol,
        };
        if improved {
            self.best = Some(curr);
            self.stalled = 0;
            None
        } else {
            self.stalled += 1;
            (self.stalled >= self.patience).then_some(TerminationReason::NoImprovement)
        }
    }
}

/// Simplex-collapse test for simplex-based solvers (e.g. Nelder-Mead),
/// per Lagarias et al. (1998), eq. (T1):
///
/// stop when `max_i ‖x_i − x_1‖_∞ ≤ tol_x` **and**
/// `max_i |f_i − f_1| ≤ tol_f`, where `x_1` / `f_1` are the best vertex
/// and its cost.
///
/// Requires `S: SimplexState` — single-iterate solvers (gradient
/// descent, BFGS) cannot be paired with it (compile error).
pub struct SimplexTolerance {
    tol_x: f64,
    tol_f: f64,
}

impl SimplexTolerance {
    /// New tolerance with separate vertex and cost bounds.
    pub fn new(tol_x: f64, tol_f: f64) -> Self {
        Self { tol_x, tol_f }
    }
}

impl<S> TerminationCriterion<S> for SimplexTolerance
where
    S: SimplexState<Float = f64>,
    S::Param: Clone + ScaledAdd<f64> + NormInfinity,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let vertices = state.vertices();
        let costs = state.costs();
        let best = &vertices[0];
        let best_cost = costs[0];

        for x_i in &vertices[1..] {
            let mut diff = x_i.clone();
            diff.scaled_add(-1.0, best);
            if diff.norm_infinity() > self.tol_x {
                return None;
            }
        }
        for &f_i in &costs[1..] {
            if (f_i - best_cost).abs() > self.tol_f {
                return None;
            }
        }
        Some(TerminationReason::SimplexTolerance)
    }
}

/// Stop after wall-clock time `limit` has elapsed since the first `check`.
///
/// Uses `web-time::Instant` so it works on both native and
/// `wasm32-unknown-unknown` without feature gating.
pub struct MaxTime {
    limit: Duration,
    start: Option<Instant>,
}

impl MaxTime {
    /// New wall-clock limit. The clock starts on the first `check`.
    pub fn new(limit: Duration) -> Self {
        Self { limit, start: None }
    }
}

impl<S> TerminationCriterion<S> for MaxTime {
    fn check(&mut self, _state: &S) -> Option<TerminationReason> {
        let start = *self.start.get_or_insert_with(Instant::now);
        if start.elapsed() >= self.limit {
            Some(TerminationReason::MaxTime)
        } else {
            None
        }
    }
}
