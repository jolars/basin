//! Log-barrier (sequential unconstrained minimization) method for linear
//! inequality constraints `A x ≤ b`.

use crate::core::barrier::{LogBarrier, strict_feasibility};
use crate::core::constraint::LinearInequalityConstraints;
use crate::core::executor::run_loop_with_control;
use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    MatTransposeVec, MatVec, NegInPlace, NormInfinity, NormSquared, Scalar,
    ScaledAdd, VectorIndex, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, PointState, State};
use crate::core::termination::Termination;

/// Owned evidence from the most recent attempted barrier subproblem.
///
/// Available only when inner diagnostics are enabled. A hard callback error
/// retains counts and the seed but supplies no recoverable returned point.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BarrierInnerDiagnostics<F: Scalar = f64> {
    /// Whether this attempt minimized the auxiliary Phase I objective.
    pub phase_one: bool,
    /// Barrier weight that defined the attempted subproblem.
    pub mu: F,
    /// Scheduled gap at that weight, before any continuation update.
    pub gap: F,
    /// Seed passed to the fresh inner state.
    pub start: Vec<F>,
    /// Published inner point on a clean return.
    pub returned: Option<Vec<F>>,
    /// Authoritative inner outcome on a clean return.
    pub report: Option<crate::TerminationReport<F>>,
    /// Raw categories charged during the native inner run, including failed
    /// calls, but excluding independent outer stationarity verification.
    pub counts: crate::EvalCounts,
    /// Whether the attempted inner run aborted with a typed callback error.
    pub hard_error: bool,
}

/// Two-phase log-barrier method for `min f(x) s.t. A x ≤ b`, layering a
/// barrier on an unconstrained inner solver.
///
/// Phase I automatically finds a strictly feasible point when the supplied
/// start does not satisfy `A x₀ < b`. Phase II then minimizes the log-barrier
/// objective
/// `φ_μ(x) = f(x) − μ · Σ log(bᵢ − aᵢᵀ x)` (the [`LogBarrier`] adapter) with
/// the inner solver `So`, warm-started from the current iterate, then
/// shrinks `μ`. As `μ → 0` the central path converges to the constrained
/// optimum.
///
/// The method is generic over inner solvers that implement [`WarmStart`].
/// The inner state is seeded at the
/// current iterate via [`InitialState::seed`],
/// so each of [`GradientDescent`](crate::solver::GradientDescent)
/// ([`FirstOrderState`](crate::FirstOrderState)), [`Bfgs`](crate::solver::Bfgs)
/// ([`FirstOrderState`](crate::core::state::FirstOrderState)), and unbounded
/// [`Lbfgs`](crate::solver::lbfgs::Lbfgs)
/// ([`FirstOrderState`](crate::core::state::FirstOrderState)) is usable. A
/// least-squares solver
/// ([`LevenbergMarquardt`](crate::solver::LevenbergMarquardt)), because the
/// barrier objective is not a sum of squares and the [`LogBarrier`] adapter
/// exposes only `CostFunction + Gradient`, not `Residual + Jacobian`.
/// A derivative-free solver such as [`NelderMead`](crate::solver::NelderMead)
/// may find a strictly feasible point in Phase I, but
/// cannot provide the gradient convergence certificate used to report an empty
/// strict interior.
///
/// **During Phase II the inner solver must keep iterates feasible.**
/// Feasibility is enforced only by the barrier returning `+∞` outside the
/// feasible set, so the inner solver's step acceptance has to honor that wall:
/// pair the inner with an
/// **Armijo backtracking** line search
/// ([`Backtracking`](crate::line_search::Backtracking)), which shrinks any
/// step whose cost is `+∞`. A fixed step ([`Constant`](crate::line_search::Constant))
/// can overshoot the boundary, and strong-Wolfe searches
/// ([`MoreThuente`](crate::line_search::MoreThuente),
/// [`Wolfe`](crate::line_search::Wolfe)) can stall bracketing on the `+∞`
/// wall; for `GradientDescent`, momentum
/// ([`with_momentum`](crate::solver::GradientDescent::with_momentum)) adds
/// velocity outside the line search's control and can carry the iterate
/// straight through the barrier.
///
/// # Algorithm
///
/// Boyd & Vandenberghe, *Convex Optimization* §11.4.1 followed by §11.3
/// (Alg. 11.1), in the `μ`-shrinking parametrization:
///
/// ```text
/// if A x₀ < b does not hold:
///   solve min s subject to A x - b ≤ s       # Phase I
///   stop Phase I as soon as A x < b
///   if centered and m · μ ≤ phase_one_tol: fail without a strict point
/// μ ← mu0
/// repeat:
///   x ← argmin φ_μ                              # Phase II
///   if centered and m · μ ≤ tol: stop         # approximate gap
///   if centered: μ ← μ / reduction
///   otherwise retry the current μ
/// ```
///
/// For a convex problem and an exactly centered subproblem, `m · μ` bounds
/// objective suboptimality (`m` is the number of constraints). Numerical
/// centering is approximate, so a small scheduled gap alone is not that bound.
///
/// # Phase I
///
/// The auxiliary scalar is eliminated analytically, so the configured inner
/// solver still works with the original parameter type `V`: for violations
/// `rᵢ = aᵢᵀx - bᵢ`, the reduced objective chooses the unique
/// `s > max rᵢ` satisfying `μ Σᵢ 1/(s-rᵢ) = 1`. Its gradient is
/// `Aᵀ[μ/(s-r)]`. This needs only the same matvec operations as Phase II and
/// therefore preserves every backend. The inner run stops immediately when it
/// produces `A x < b`; Phase II then restarts the `μ` schedule at `mu0`.
///
/// If a centered Phase I subproblem reaches
/// [`with_absolute_phase_one_gap_tolerance`](Self::with_absolute_phase_one_gap_tolerance)
/// without finding a strict point, the constraints are reported as
/// [`SolverFailed`](crate::TerminationCode::SolverFailed). An inner solve that
/// exhausts its iteration budget is not a certificate: Phase I retries the
/// same `μ` from the returned candidate. Numerically, the centered certificate
/// means "no strict interior at the configured scale": exact zero margin
/// cannot be distinguished from an arbitrarily thin interior in finite
/// precision.
///
/// # Termination
///
/// Outer convergence requires `m · μ ≤ tol` (default `1e-8`) and sufficient
/// centering of the consumed Phase II subproblem. Centering requires
/// `‖∇φ_μ‖∞ ≤ 1e-8` or a reduction to `1e-3` times its norm at the starting
/// point for that weight. Configure the two alternatives with
/// [`with_absolute_stationarity_tolerance`](Self::with_absolute_stationarity_tolerance)
/// and [`with_relative_stationarity_tolerance`](Self::with_relative_stationarity_tolerance).
/// An unfinished inner solve supplies a partial candidate and retries the same
/// weight. Its first reference persists across segments and exact continuation.
/// These tests do not certify global optimality or parameter accuracy.
///
/// The check is independent of the inner solver and adds up to two counted
/// gradient callbacks per consumed Phase II result. Disable both stationarity
/// settings with `None` to recover gap-only continuation and stopping. Zero
/// requests exact stationarity. Optional cost- and step-change tests also
/// require centering and observe only accepted Phase II points. The outer
/// [`PointState`] has no gradient capability.
///
/// A strictly feasible start uses Phase II; other starts spend additional
/// iterations in Phase I. Keep an executor [`max_iter`](crate::Executor::max_iter)
/// budget. Centering can require several capped segments per weight, so the
/// number of outer iterations need not equal the number of weight reductions.
/// Choose attainable gap and centering thresholds for the inner solver,
/// derivative accuracy, precision, and problem units.
///
/// # Progress and lifecycle
///
/// The outer state reports the original objective inside the strict domain
/// `A x < b`, and `+∞` elsewhere. Phase I computes this rejection value
/// without calling the original objective or gradient. Its current record is
/// available, but it cannot establish an incumbent until a strictly feasible
/// point with an eligible objective is published. Incumbents retain strict
/// objective improvements among these feasible points.
///
/// Fresh initialization clears progress, reevaluates the seed's domain and
/// objective, and restarts the phase and barrier schedule. Each inner solve
/// starts fresh. Exact continuation retains the phase, schedule, inner solver,
/// progress, and all six raw evaluation categories, and skips initialization.
/// Inner and independent centering evaluations remain charged even when a
/// callback aborts. Native inner diagnostics keep only the inner run's counts.
///
/// With `serde`, solver-aware checkpoints can serialize the outer method
/// whenever the inner solver supports serialization. Old `BasicState`
/// checkpoint payloads are incompatible with this progress representation.
///
/// # Backends
///
/// Supports `Vec<f32>` and `Vec<f64>` with
/// [`DenseMatrix`](crate::core::math::DenseMatrix), nalgebra `DVector`/`DMatrix`,
/// ndarray `Array1`/`Array2`, and faer `Col`/`Mat`, with either scalar type.
/// The outer method needs [`MatVec`] and [`MatTransposeVec`], with no linear
/// solve. Its vector bounds are [`ScaledAdd`], [`NegInPlace`], [`VectorIndex`],
/// [`VectorLen`], [`NormSquared`], [`NormInfinity`], and [`Clone`]. A custom inner solver can
/// impose additional capabilities.
///
/// # Composition
///
/// Use [`with_inner_solver`](Self::with_inner_solver) to supply a solver with
/// its own convergence settings. Each outer iteration starts a fresh inner
/// solve through [`run_loop_with_control`],
/// against the current surrogate problem. Convergence history resets and
/// inner evaluation counts are added to their corresponding outer categories.
///
/// # Examples
///
/// `BarrierMethod` wraps a gradient inner solver (e.g. `BFGS` paired with
/// `Backtracking`) to handle `LinearInequalityConstraints`. See
/// [`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent)
/// for the simpler box-constrained pattern.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BarrierMethod<So, F: Scalar = f64> {
    inner_solver: So,
    inner_max_iter: u64,
    mu0: F,
    mu: F,
    reduction: F,
    tol: Option<F>,
    stationarity_tol: Option<F>,
    relative_stationarity_tol: Option<F>,
    stationarity_initial_norm: F,
    stationarity_norm: F,
    centering_mu: Option<F>,
    phase_one_tol: F,
    /// `m · μ` of the most recent inner solve; `+∞` until the first solve
    /// so [`terminate`](Solver::terminate) cannot fire at iter 0.
    gap: F,
    phase: BarrierPhase,
    accepted_iterate: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    inner_diagnostics: bool,
    last_inner: Option<BarrierInnerDiagnostics<F>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum BarrierPhase {
    PhaseOne,
    PhaseTwo,
    Failed,
}

/// Delegates to the configured inner solver but ends a Phase I inner run as
/// soon as the current parameter is strictly feasible for the original
/// constraints. This is important when the auxiliary LP is unbounded below:
/// Phase I only needs one interior point, not the LP optimum.
struct StopAtStrictFeasibility<'a, So> {
    inner: &'a mut So,
}

impl<'a, 'p, P, V, M, S, So, F> Solver<LogBarrier<'p, P, F>, S>
    for StopAtStrictFeasibility<'a, So>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + LinearInequalityConstraints<Param = V, Matrix = M>,
    M: MatVec<V>,
    V: ScaledAdd<F> + VectorIndex<F> + VectorLen,
    S: State<Param = V>,
    So: Solver<LogBarrier<'p, P, F>, S>,
{
    type Error = So::Error;

    fn init(
        &mut self,
        problem: &mut Problem<LogBarrier<'p, P, F>>,
        state: S,
    ) -> Result<S, Self::Error> {
        self.inner.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<LogBarrier<'p, P, F>>,
        state: S,
    ) -> Result<crate::SolverStep<S>, Self::Error> {
        let crate::SolverStep {
            state,
            termination: reason,
            completed,
        } = self.inner.next_iter(problem, state)?;
        let inner_failed = reason.as_ref().is_some_and(|reason| {
            !reason.can_continue_as_inner(crate::PartialResultPolicy::Consume)
        });
        if !inner_failed
            && problem.inner().strict_feasibility(state.param()) == Some(true)
        {
            Ok(crate::SolverStep {
                state,
                completed,
                termination: Some(Termination::custom(
                    "barrier.strict_feasibility",
                    "All inequality slacks are strictly positive.",
                    vec![],
                )),
            })
        } else {
            Ok(crate::SolverStep {
                state,
                termination: reason,
                completed,
            })
        }
    }

    fn terminate(&self, state: &S) -> Option<Termination<S::Float>> {
        self.inner.terminate(state)
    }

    fn reset_convergence(&mut self) {
        self.inner.reset_convergence();
    }
    fn check_convergence(
        &mut self,
        problem: &Problem<LogBarrier<'p, P, F>>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        self.inner.check_convergence(problem, state)
    }
}

impl<So, F: Scalar> BarrierMethod<So, F> {
    /// Build a barrier method around an unconstrained inner solver.
    ///
    /// Defaults: `mu0 = 1.0`, `reduction = 10.0`, `tol = 1e-8`,
    /// `phase_one_tol = 1e-8`, absolute stationarity tolerance `1e-8`,
    /// relative stationarity tolerance `1e-3`, and `inner_max_iter = 50`.
    ///
    /// The inner budget limits each centering segment. An unfinished segment
    /// supplies progress, and the next segment retries the same barrier weight
    /// until a stationarity test passes.
    /// The supplied inner solver owns its convergence settings.
    pub fn with_inner_solver(inner_solver: So) -> Self {
        Self {
            inner_solver,
            inner_max_iter: 50,
            mu0: F::one(),
            mu: F::one(),
            reduction: F::from_f64(10.0).unwrap(),
            tol: Some(F::from_f64(1e-8).unwrap()),
            stationarity_tol: Some(F::from_f64(1e-8).unwrap()),
            relative_stationarity_tol: Some(F::from_f64(1e-3).unwrap()),
            stationarity_initial_norm: F::infinity(),
            stationarity_norm: F::infinity(),
            centering_mu: None,
            phase_one_tol: F::from_f64(1e-8).unwrap(),
            gap: F::infinity(),
            phase: BarrierPhase::PhaseTwo,
            accepted_iterate: false,
            inner_diagnostics: false,
            last_inner: None,
        }
    }
}

impl<So, F: Scalar> BarrierMethod<So, F> {
    fn stationarity_satisfied(&self) -> bool {
        crate::core::convergence::stationarity_within_tolerance(
            self.stationarity_norm,
            self.stationarity_initial_norm,
            self.stationarity_tol,
            self.relative_stationarity_tol,
        )
    }

    /// Require `‖∇φ_μ(x)‖∞` at the returned Phase II point to be at most this
    /// tolerance before outer convergence (default `1e-8`).
    ///
    /// Uses the barrier weight that defined the consumed inner solve. This
    /// independent check also applies to derivative-free and budget-limited
    /// inners. Together with the relative guard, it adds up to two gradient
    /// evaluations per consumed Phase II result. `None` disables this absolute
    /// test; zero requests exact stationarity. Either the absolute or relative
    /// test must pass before reducing the weight or declaring outer convergence.
    /// The guard also applies to optional outer cost- and step-change tests.
    pub fn with_absolute_stationarity_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.stationarity_tol =
            crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Require the returned surrogate-gradient infinity norm to be at most
    /// this fraction of its first norm at the current weight (default `1e-3`).
    ///
    /// Both norms use the same surrogate parameters before the outer update.
    /// Retries at that weight retain the first reference, including across
    /// exact continuation.
    /// Absolute and relative stationarity tests combine with OR. At least one
    /// must pass before outer convergence or a barrier continuation update.
    /// `None` disables this relative test; zero requests exact stationarity.
    /// Disable both stationarity tests to recover gap-only continuation and stopping.
    pub fn with_relative_stationarity_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.relative_stationarity_tol =
            crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Retain the last attempted inner run's owned outcome and parameter data.
    ///
    /// Disabled by default. Enabling this copies diagnostic coordinates and
    /// reports without changing the subproblem, its limits, or its stopping rule.
    pub fn with_inner_diagnostics(mut self, enabled: bool) -> Self {
        self.inner_diagnostics = enabled;
        if !enabled {
            self.last_inner = None;
        }
        self
    }

    /// Last attempted inner run, or `None` before an enabled attempt.
    pub fn inner_diagnostics(&self) -> Option<&BarrierInnerDiagnostics<F>> {
        self.last_inner.as_ref()
    }

    /// Initial barrier parameter `μ` (default `1.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `mu0 > 0`; a non-positive `μ` is not a barrier.
    pub fn mu0(mut self, mu0: F) -> Self {
        assert!(mu0 > F::zero(), "mu0 must be > 0");
        self.mu0 = mu0;
        self
    }

    /// Per-outer-iteration shrink factor: `μ ← μ / reduction` (default
    /// `10.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `reduction > 1`; otherwise `μ` would not shrink and
    /// the duality gap would never close.
    pub fn with_reduction(mut self, reduction: F) -> Self {
        assert!(reduction > F::one(), "reduction must be > 1");
        self.reduction = reduction;
        self
    }

    /// Stop when the outer duality gap `m · μ` reaches this tolerance
    /// (default `1e-8`). `None` disables the check.
    pub fn with_absolute_duality_gap_tolerance(
        mut self,
        tol: impl Into<Option<F>>,
    ) -> Self {
        self.tol = crate::core::convergence::optional_tolerance(tol);
        self
    }

    /// Set the Phase I accuracy used to classify a constraint system with no
    /// strict interior (default `1e-8`). If a centered Phase I subproblem has
    /// not found `A x < b` once its auxiliary duality gap `m · μ` reaches this
    /// tolerance, the solver reports
    /// [`SolverFailed`](crate::TerminationCode::SolverFailed).
    ///
    /// Finite precision cannot distinguish an empty interior from an
    /// arbitrarily thin one. This tolerance sets the numerical scale for that
    /// decision and must be positive.
    pub fn with_absolute_phase_one_gap_tolerance(
        mut self,
        phase_one_tol: F,
    ) -> Self {
        assert!(phase_one_tol > F::zero(), "phase_one_tol must be > 0");
        self.phase_one_tol = phase_one_tol;
        self
    }

    /// Iteration budget for each inner barrier-subproblem solve (default
    /// `50`).
    ///
    /// A capped solve supplies a partial candidate. Phase II retries the same
    /// weight until a stationarity test passes, so a smaller segment budget can
    /// require more outer iterations. Choose this budget together with the outer
    /// iteration limit and attainable centering thresholds. Near active bounds,
    /// the barrier can become ill-conditioned for a first-order inner solver.
    ///
    /// # Panics
    ///
    /// Panics unless `inner_max_iter ≥ 1` (a zero budget would never move the
    /// iterate).
    pub fn with_inner_max_iter(mut self, inner_max_iter: u64) -> Self {
        assert!(inner_max_iter >= 1, "inner_max_iter must be ≥ 1");
        self.inner_max_iter = inner_max_iter;
        self
    }
}

impl<So, V, F> InitialState<V> for BarrierMethod<So, F>
where
    F: Scalar,
    V: Clone,
{
    type State = PointState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        PointState::new(x.clone())
    }
}

impl<P, V, M, So, F> Solver<P, PointState<V, F>> for BarrierMethod<So, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + LinearInequalityConstraints<Param = V, Matrix = M>,
    M: MatVec<V> + MatTransposeVec<V>,
    V: ScaledAdd<F>
        + NegInPlace
        + VectorIndex<F>
        + VectorLen
        + NormSquared<F>
        + NormInfinity<F>
        + Clone,
    So: WarmStart<V>
        + for<'a> Solver<
            LogBarrier<'a, P, F>,
            So::State,
            Error = <P as CostFunction>::Error,
        >,
    So::State: State<Param = V, Float = F> + CountsMirror,
{
    type Error = <P as CostFunction>::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        state.reset();
        self.accepted_iterate = false;
        self.last_inner = None;
        self.stationarity_norm = F::infinity();
        self.stationarity_initial_norm = F::infinity();
        self.centering_mu = None;
        self.mu = self.mu0;
        self.gap = F::infinity();

        self.phase = match strict_feasibility(problem.inner(), state.param()) {
            Some(true) => BarrierPhase::PhaseTwo,
            Some(false) => BarrierPhase::PhaseOne,
            None => BarrierPhase::Failed,
        };

        if self.phase == BarrierPhase::Failed {
            // Keep cost-based convergence dormant until `next_iter` can
            // report the solver failure through the normal soft-stop path.
            state.replace(state.param().clone(), F::infinity());
            return Ok(state);
        }

        if self.phase == BarrierPhase::PhaseOne {
            // Executor criteria run before `next_iter`, including at iter 0.
            // The infeasible point is not a candidate for the original
            // problem, so exposing its true objective or gradient could let a
            // target or stationarity test bypass Phase I entirely.
            state.replace(state.param().clone(), F::infinity());
            return Ok(state);
        }

        // A feasible start is a valid candidate for the original problem.
        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        self.accepted_iterate = true;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<crate::SolverStep<PointState<V, F>>, Self::Error> {
        self.accepted_iterate = false;
        self.stationarity_norm = F::infinity();
        if self.phase == BarrierPhase::Failed {
            return Ok(crate::SolverStep::from((
                state,
                Some(Termination::numerical_failure(
                    "The barrier method could not initialize a finite barrier model.",
                )),
            )));
        }

        if self.phase == BarrierPhase::PhaseOne {
            let mut barrier_wrapper =
                Problem::new(LogBarrier::phase_one(problem.inner(), self.mu));
            let mut control = crate::core::run_control::RunControl::new()
                .max_iter(self.inner_max_iter);
            let diagnostic_start = self.inner_diagnostics.then(|| {
                (0..state.param().vec_len())
                    .map(|i| state.param().get_scalar(i))
                    .collect::<Vec<_>>()
            });
            let inner_state = self.inner_solver.seed(state.param());
            let mut phase_one_solver = StopAtStrictFeasibility {
                inner: &mut self.inner_solver,
            };
            let result = run_loop_with_control(
                &mut barrier_wrapper,
                inner_state,
                &mut phase_one_solver,
                &mut control,
            );

            let inner_counts = *barrier_wrapper.counts();
            problem.counts_mut().add(&inner_counts);
            self.last_inner =
                diagnostic_start.map(|start| BarrierInnerDiagnostics {
                    phase_one: true,
                    mu: self.mu,
                    gap: F::from_usize(problem.inner().b().vec_len()).unwrap()
                        * self.mu,
                    start,
                    returned: result.as_ref().ok().map(|r| {
                        (0..r.state.param().vec_len())
                            .map(|i| r.state.param().get_scalar(i))
                            .collect()
                    }),
                    report: result.as_ref().ok().map(|r| r.report.clone()),
                    counts: inner_counts,
                    hard_error: result.is_err(),
                });
            let result = result?;

            if !result
                .report
                .termination
                .can_continue_as_inner(crate::PartialResultPolicy::Consume)
            {
                self.phase = BarrierPhase::Failed;
                return Ok(crate::SolverStep::from((
                    state,
                    result.report.into_outer_termination(
                        crate::PartialResultPolicy::Consume,
                    ),
                )));
            }

            // Only the inner solver's gradient convergence can certify a
            // centered Phase I subproblem; a budget stop cannot.
            let centered = matches!(
                &result.report.termination,
                Termination::Converged(convergence) if convergence.criteria().iter().any(|criterion|
                    matches!(criterion.test, crate::ConvergenceTest::AbsoluteGradientSquared | crate::ConvergenceTest::RelativeGradientSquared)
                )
            );
            let candidate = result.state.param();
            let feasibility = strict_feasibility(problem.inner(), candidate);
            let Some(is_strictly_feasible) = feasibility else {
                self.phase = BarrierPhase::Failed;
                return Ok(crate::SolverStep::from((
                    state,
                    Some(Termination::numerical_failure(
                        "Phase I produced non-finite constraint slacks.",
                    )),
                )));
            };

            if is_strictly_feasible {
                let cost = problem.cost(candidate)?;
                if !cost.is_finite() {
                    self.phase = BarrierPhase::Failed;
                    return Ok(crate::SolverStep::stopped(
                        state,
                        Termination::numerical_failure(
                            "Phase I returned a strict point with a non-finite original objective.",
                        ),
                    ));
                }
                state.replace(candidate.clone(), cost);
                // Phase II starts a fresh continuation schedule; Phase I's μ
                // controls feasibility accuracy, not objective optimality.
                self.phase = BarrierPhase::PhaseTwo;
                self.mu = self.mu0;
                self.gap = F::infinity();
                self.accepted_iterate = true;
                return Ok(crate::SolverStep::from((state, None)));
            }

            state.replace(candidate.clone(), F::infinity());

            // An unfinished solve supplies neither a Phase I optimum nor the
            // associated m·μ bound. Preserve its progress, but retry this μ
            // instead of turning a shrinking continuation parameter into a
            // false infeasibility certificate.
            if centered {
                let phase_one_gap =
                    F::from_usize(problem.inner().b().vec_len()).unwrap()
                        * self.mu;
                if phase_one_gap <= self.phase_one_tol {
                    self.phase = BarrierPhase::Failed;
                    return Ok(crate::SolverStep::from((
                        state,
                        Some(Termination::numerical_failure(
                            "A centered Phase I subproblem reached its gap threshold without strict feasibility.",
                        )),
                    )));
                }

                self.mu = self.mu / self.reduction;
            }

            return Ok(crate::SolverStep::from((state, None)));
        }

        // Minimize the barrier objective at the current μ on a *separate*
        // inner state seeded (warm-started) at the current iterate. A fresh
        // inner state (rather than threading the outer one) keeps the
        // inner solver's iteration counter from polluting the outer's.
        // Fresh criteria each call satisfies the statelessness contract.
        let mut barrier_wrapper =
            Problem::new(LogBarrier::new(problem.inner(), self.mu));
        let mut control = crate::core::run_control::RunControl::new()
            .max_iter(self.inner_max_iter);
        let diagnostic_start = self.inner_diagnostics.then(|| {
            (0..state.param().vec_len())
                .map(|i| state.param().get_scalar(i))
                .collect::<Vec<_>>()
        });
        let inner_state = self.inner_solver.seed(state.param());
        let result = run_loop_with_control(
            &mut barrier_wrapper,
            inner_state,
            &mut self.inner_solver,
            &mut control,
        );

        // Eval aggregation (adapter-problem composition): fold the inner
        // wrapper's per-call counts back into the outer's wrapper. Copy out
        // before borrowing the outer mutably so the LogBarrier's `&P` borrow
        // (still held by `barrier_wrapper`) doesn't collide with the
        // `counts_mut` reborrow.
        let inner_counts = *barrier_wrapper.counts();
        let stationarity = result
            .as_ref()
            .ok()
            .filter(|r| {
                (self.stationarity_tol.is_some()
                    || self.relative_stationarity_tol.is_some())
                    && r.report.termination.can_continue_as_inner(
                        crate::PartialResultPolicy::Consume,
                    )
                    && strict_feasibility(problem.inner(), r.state.param())
                        == Some(true)
            })
            .map(|r| {
                // Retried segments solve the same model, so their accuracy target
                // retains the first reference instead of tightening after each cap.
                let initial = if self.centering_mu == Some(self.mu) {
                    self.stationarity_initial_norm
                } else {
                    barrier_wrapper.gradient(state.param())?.norm_infinity()
                };
                let final_norm =
                    barrier_wrapper.gradient(r.state.param())?.norm_infinity();
                Ok::<_, Self::Error>((initial, final_norm))
            })
            .transpose();
        let total_counts = *barrier_wrapper.counts();
        problem.counts_mut().add(&total_counts);
        self.last_inner =
            diagnostic_start.map(|start| BarrierInnerDiagnostics {
                phase_one: false,
                mu: self.mu,
                gap: F::from_usize(problem.inner().b().vec_len()).unwrap()
                    * self.mu,
                start,
                returned: result.as_ref().ok().map(|r| {
                    (0..r.state.param().vec_len())
                        .map(|i| r.state.param().get_scalar(i))
                        .collect()
                }),
                report: result.as_ref().ok().map(|r| r.report.clone()),
                counts: inner_counts,
                hard_error: result.is_err(),
            });
        let result = result?;
        if let Some((initial, final_norm)) = stationarity? {
            self.stationarity_initial_norm = initial;
            self.stationarity_norm = final_norm;
            self.centering_mu = Some(self.mu);
        }

        if !result
            .report
            .termination
            .can_continue_as_inner(crate::PartialResultPolicy::Consume)
        {
            self.phase = BarrierPhase::Failed;
            return Ok(crate::SolverStep::from((
                state,
                result.report.into_outer_termination(
                    crate::PartialResultPolicy::Consume,
                ),
            )));
        }

        // Publish the original objective only after verifying the barrier domain.
        let candidate = result.state.param();
        if strict_feasibility(problem.inner(), candidate) != Some(true) {
            self.phase = BarrierPhase::Failed;
            return Ok(crate::SolverStep::from((
                state,
                Some(Termination::numerical_failure(
                    "The inner barrier solve returned a point outside the strict barrier domain.",
                )),
            )));
        }
        let cost = problem.cost(candidate)?;
        if !cost.is_finite() {
            self.phase = BarrierPhase::Failed;
            return Ok(crate::SolverStep::stopped(
                state,
                Termination::numerical_failure(
                    "The inner barrier solve returned a non-finite original objective.",
                ),
            ));
        }
        state.replace(candidate.clone(), cost);
        self.accepted_iterate = true;

        // Continue the same subproblem after an unfinished centering solve.
        // Shrinking μ alone cannot turn that partial result into a gap bound.
        self.gap =
            F::from_usize(problem.inner().b().vec_len()).unwrap() * self.mu;
        if self.stationarity_satisfied() {
            self.mu = self.mu / self.reduction;
        }
        Ok(crate::SolverStep::from((state, None)))
    }

    fn should_check_iterate_change(&self) -> bool {
        self.accepted_iterate && self.stationarity_satisfied()
    }

    fn terminate(&self, _state: &PointState<V, F>) -> Option<Termination<F>> {
        // Log-barrier duality-gap bound m·μ from the most recent solve.
        if self.phase == BarrierPhase::PhaseTwo
            && self.stationarity_satisfied()
            && self.tol.is_some_and(|tol| self.gap <= tol)
        {
            let mut measurements = vec![
                crate::Measurement {
                    name: "gap".into(),
                    value: self.gap,
                },
                crate::Measurement {
                    name: "tolerance".into(),
                    value: self.tol.unwrap(),
                },
            ];
            if let Some(tolerance) = self.stationarity_tol {
                measurements.extend([
                    crate::Measurement {
                        name: "stationarity_norm".into(),
                        value: self.stationarity_norm,
                    },
                    crate::Measurement {
                        name: "stationarity_tolerance".into(),
                        value: tolerance,
                    },
                ]);
            }
            if let Some(tolerance) = self.relative_stationarity_tol {
                if self.stationarity_tol.is_none() {
                    measurements.push(crate::Measurement {
                        name: "stationarity_norm".into(),
                        value: self.stationarity_norm,
                    });
                }
                measurements.extend([
                    crate::Measurement {
                        name: "relative_stationarity_tolerance".into(),
                        value: tolerance,
                    },
                    crate::Measurement {
                        name: "initial_stationarity_norm".into(),
                        value: self.stationarity_initial_norm,
                    },
                ]);
            }
            Some(Termination::custom(
                "barrier.duality_gap",
                concat!(
                    "In Phase II, gap <= tolerance AND (the finite barrier-gradient infinity norm ",
                    "meets an enabled absolute stationarity bound OR its relative bound times ",
                    "the finite initial norm at this weight). The two stationarity tests combine ",
                    "with OR; disabling both bypasses their guard."
                ),
                measurements,
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The builder validation is backend-independent, so a unit inner stand-in
    // (`()`) suffices; these never run the solver, only the builders.

    #[test]
    #[should_panic(expected = "mu0 must be > 0")]
    fn rejects_nonpositive_mu0() {
        let _ = BarrierMethod::with_inner_solver(()).mu0(0.0);
    }

    #[test]
    #[should_panic(expected = "reduction must be > 1")]
    fn rejects_reduction_not_greater_than_one() {
        let _ = BarrierMethod::with_inner_solver(()).with_reduction(1.0);
    }

    #[test]
    #[should_panic(expected = "phase_one_tol must be > 0")]
    fn rejects_nonpositive_phase_one_tol() {
        let _ = BarrierMethod::with_inner_solver(())
            .with_absolute_phase_one_gap_tolerance(0.0);
    }

    #[test]
    #[should_panic(expected = "inner_max_iter must be ≥ 1")]
    fn rejects_zero_inner_max_iter() {
        let _ = BarrierMethod::<_, f64>::with_inner_solver(())
            .with_inner_max_iter(0);
    }
}
