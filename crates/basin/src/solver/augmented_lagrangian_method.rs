//! Augmented-Lagrangian method for linear equality constraints `A x = b`.

use crate::ConstraintSolveError;
use crate::core::augmented_lagrangian::AugmentedLagrangian;
use crate::core::constraint::evaluation::{LinearEqualityGradient, invalid};
use crate::core::executor::run_loop_with_control;
use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    Dot, NormInfinity, NormSquared, Scalar, ScaleInPlace, ScaledAdd,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, SelectedState, State};
use crate::core::termination::Termination;

/// Owned evidence from the most recent attempted augmented-Lagrangian solve.
///
/// The multiplier and penalty define the surrogate before the outer update.
/// A hard callback error retains its seed and counts without a returned state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AugmentedLagrangianInnerDiagnostics<V, F: Scalar = f64> {
    /// Penalty that defined the attempted subproblem.
    pub rho: F,
    /// Multiplier vector before the outer multiplier or penalty update.
    pub lambda: Vec<F>,
    /// Seed passed to the fresh inner state.
    pub start: V,
    /// Published inner point on a clean return.
    pub returned: Option<V>,
    /// Authoritative inner outcome on a clean return.
    pub report: Option<crate::TerminationReport<F>>,
    /// Raw categories charged during the native inner run, including failed
    /// calls, but excluding independent outer stationarity verification.
    pub counts: crate::EvalCounts,
    /// Whether the attempted inner run aborted with a typed callback error.
    pub hard_error: bool,
}

/// Augmented-Lagrangian method for `min f(x) s.t. A x = b`, the
/// equality-constrained analogue of the log-barrier
/// [`BarrierMethod`](crate::solver::BarrierMethod), layering a quadratic
/// penalty plus multiplier estimates on an unconstrained inner solver.
///
/// Each outer iteration minimizes the augmented Lagrangian
/// `L_ρ(x, λ) = f(x) + λᵀ c(x) + (ρ/2)‖c(x)‖²` (with `c(x) = A x − b`, the
/// [`AugmentedLagrangian`] adapter) over `x` with the inner solver `So`,
/// warm-started from the current iterate, then updates the multiplier
/// estimate `λ ← λ + ρ c(x)`. When feasibility fails to improve enough
/// between outer iterations the penalty `ρ` is increased instead. As `λ`
/// approaches the true multipliers `λ*`, the unconstrained minimizer of
/// `L_ρ` approaches the constrained optimum.
///
/// The method is generic over inner solvers that implement [`WarmStart`].
/// Each inner state is seeded at the current iterate via
/// [`InitialState::seed`]. That covers
/// [`GradientDescent`](crate::solver::GradientDescent)
/// ([`FirstOrderState`](crate::FirstOrderState)),
/// [`Bfgs`](crate::solver::Bfgs)
/// ([`FirstOrderState`](crate::core::state::FirstOrderState)), and unbounded
/// [`Lbfgs`](crate::solver::lbfgs::Lbfgs)
/// ([`FirstOrderState`](crate::core::state::FirstOrderState)). A least-squares inner
/// ([`LevenbergMarquardt`](crate::solver::LevenbergMarquardt)) does not fit:
/// `L_ρ` is not a sum of squares and the [`AugmentedLagrangian`] adapter
/// exposes only `CostFunction + Gradient`. A derivative-free inner solver
/// such as [`NelderMead`](crate::solver::NelderMead) may also be supplied.
///
/// # Infeasible starts are fine
///
/// Unlike the Phase II [`LogBarrier`](crate::core::barrier::LogBarrier), the
/// augmented Lagrangian is finite and smooth *everywhere* (there is no `+∞`
/// feasibility wall), so the starting point need **not** satisfy `A x₀ = b`,
/// and the inner solver may use any line search (Armijo backtracking, Wolfe,
/// Moré–Thuente) or momentum. No Phase I feasibility solve is required.
///
/// # Algorithm
///
/// Nocedal & Wright, *Numerical Optimization* §17.3 (Alg. 17.4, the
/// LANCELOT-style outer loop), simplified:
///
/// ```text
/// λ ← 0 ∈ ℝᵐ ;  ρ ← rho0
/// repeat:
///   x ← argmin_x L_ρ(x, λ)            # inner solver, warm-started at x
///   c ← A x − b
///   if ‖c‖ ≤ tol and surrogate stationarity passes: stop
///   if ‖c‖ ≤ feasibility_decrease · ‖c_prev‖:
///       λ ← λ + ρ c                    # sufficient feasibility improvement
///   else:
///       ρ ← rho_increase · ρ           # else tighten penalty, keep λ
/// ```
///
/// # Termination
///
/// Outer convergence requires `‖A x − b‖₂ ≤ tol` (default `1e-8`) and
/// sufficient surrogate stationarity at the returned inner point. The latter
/// requires `‖∇L_ρ‖∞ ≤ 1e-8` or a reduction to `1e-3` times its norm at the
/// inner starting point. Configure these with
/// [`with_absolute_stationarity_tolerance`](Self::with_absolute_stationarity_tolerance)
/// and [`with_relative_stationarity_tolerance`](Self::with_relative_stationarity_tolerance).
/// Both gradients use the same pre-update multipliers and penalty. These are
/// local numerical tests, not guarantees of objective or parameter accuracy.
/// Keep an executor [`max_iter`](crate::Executor::max_iter) budget.
///
/// The checks also apply to derivative-free and budget-limited inners. An
/// arbitrary inner convergence label cannot replace the independent gradient
/// check. Disable both stationarity settings with `None` to recover the former
/// feasibility-only behavior; zero requests exact stationarity. Up to two
/// additional gradient callbacks per consumed result are counted as outer
/// verification work, separately from the native inner diagnostics.
///
/// The outer [`SelectedState`] reports the original objective and residual
/// norm without advertising an objective gradient. Optional outer cost- and
/// step-change tests require both feasibility and surrogate stationarity.
///
/// Incumbent selection first prefers feasible points, using the configured
/// tolerance, and then lower objectives among feasible points. Before finding
/// one, it prefers smaller residual norms, breaking exact ties by objective.
/// A better-feasibility incumbent can therefore have a higher objective. NaN
/// objectives, positive-infinite objectives, and non-finite residual norms
/// cannot establish an incumbent. Use `state.best()` for checked access.
/// Objective-only [`Executor::target_objective`](crate::Executor::target_objective)
/// and [`no_objective_improvement`](crate::Executor::no_objective_improvement)
/// controls are unavailable for this state; constrained application stops
/// must check feasibility as well as objective value.
///
/// # Initialization and continuation
///
/// Use `SelectedState::new(x)` or [`Executor::from_start`](crate::Executor::from_start).
/// Fresh initialization resets progress, penalty history, and multipliers,
/// then reevaluates the objective and constraint residual. Each inner solve
/// also starts fresh. Exact solver-aware checkpoints retain the outer model,
/// selected incumbent, and all six raw evaluation categories. State-only
/// snapshots start a new solve. Inner and stationarity-verification gradient
/// calls remain charged in `state.counts().gradient_evals`. Adapter work is
/// merged even on hard errors.
/// With `serde`, solver-aware serialization is available when the inner solver,
/// parameter, and scalar types support it.
///
/// # Backends
///
/// Requires the constraint matrix to implement [`crate::MatVecInto`] (`A x`) and
/// [`crate::MatTransposeVecInto`] (`Aᵀ v`), never a linear solve. All backends supply
/// those two ops: `Vec<F>` with [`DenseMatrix`](crate::DenseMatrix),
/// nalgebra `DVector<F>`/`DMatrix<F>`, ndarray `Array1<F>`/`Array2<F>`, and
/// faer `Col<F>`/`Mat<F>`, with `F = f64` (default) or `f32`. Vectors need
/// [`ScaledAdd<F>`], [`ScaleInPlace<F>`], [`Dot<F>`], [`NormSquared<F>`],
/// [`NormInfinity<F>`], and `Clone`. The chosen inner solver can impose
/// additional capabilities.
///
/// # Composition
///
/// Use [`with_inner_solver`](Self::with_inner_solver) to supply a solver with
/// its own convergence settings. Each outer iteration starts a fresh inner
/// solve through [`run_loop_with_control`],
/// against the current surrogate problem. Convergence history resets and
/// inner evaluation counts are folded into the outer wrapper.
///
/// # Examples
///
/// `AugmentedLagrangianMethod` wraps a gradient inner solver to handle
/// `Constraints`, and tolerates infeasible starts. See
/// [`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent)
/// for the simpler box-constrained pattern.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AugmentedLagrangianMethod<So, V, F: Scalar = f64> {
    inner_solver: So,
    inner_max_iter: u64,
    rho0: F,
    rho: F,
    rho_increase: F,
    feasibility_decrease: F,
    tol: Option<F>,
    stationarity_tol: Option<F>,
    relative_stationarity_tol: Option<F>,
    stationarity_initial_norm: F,
    stationarity_norm: F,
    /// Multiplier estimate `λ ∈ ℝᵐ`; populated in [`init`](Solver::init) with
    /// a zero vector shaped like `b`, then carried across outer iterations.
    lambda: Option<Vec<F>>,
    /// `‖c(x)‖` of the most recent inner solve; `+∞` until the first solve so
    /// [`terminate`](Solver::terminate) cannot fire at iter 0.
    c_norm: F,
    /// `‖c(x)‖` of the previous outer iteration, for the `ρ`-increase test.
    c_norm_prev: F,
    #[cfg_attr(feature = "serde", serde(default))]
    inner_diagnostics: bool,
    last_inner: Option<AugmentedLagrangianInnerDiagnostics<V, F>>,
}

impl<So, V, F: Scalar> AugmentedLagrangianMethod<So, V, F> {
    /// Build an augmented-Lagrangian method around an unconstrained inner
    /// solver.
    ///
    /// Defaults: `rho0 = 10.0`, `rho_increase = 10.0`,
    /// `feasibility_decrease = 0.25`, `tol = 1e-8`, and `inner_max_iter = 50`.
    /// The supplied inner solver owns its convergence settings.
    pub fn with_inner_solver(inner_solver: So) -> Self {
        Self {
            inner_solver,
            inner_max_iter: 50,
            rho0: F::from_f64(10.0).unwrap(),
            rho: F::from_f64(10.0).unwrap(),
            rho_increase: F::from_f64(10.0).unwrap(),
            feasibility_decrease: F::from_f64(0.25).unwrap(),
            tol: Some(F::from_f64(1e-8).unwrap()),
            stationarity_tol: Some(F::from_f64(1e-8).unwrap()),
            relative_stationarity_tol: Some(F::from_f64(1e-3).unwrap()),
            stationarity_initial_norm: F::infinity(),
            stationarity_norm: F::infinity(),
            lambda: None,
            c_norm: F::infinity(),
            c_norm_prev: F::infinity(),
            inner_diagnostics: false,
            last_inner: None,
        }
    }
}

impl<So, V, F: Scalar> AugmentedLagrangianMethod<So, V, F> {
    /// Retain the last attempted inner run and the surrogate's defining stage.
    ///
    /// Disabled by default. Enabling this copies diagnostic vectors and reports
    /// without changing inner capabilities, controls, or stopping behavior.
    pub fn with_inner_diagnostics(mut self, enabled: bool) -> Self {
        self.inner_diagnostics = enabled;
        if !enabled {
            self.last_inner = None;
        }
        self
    }

    /// Last attempted inner run, or `None` before an enabled attempt.
    pub fn inner_diagnostics(
        &self,
    ) -> Option<&AugmentedLagrangianInnerDiagnostics<V, F>> {
        self.last_inner.as_ref()
    }

    /// Initial penalty parameter `ρ` (default `10.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `rho0 > 0`; a non-positive penalty is not a penalty.
    pub fn rho0(mut self, rho0: F) -> Self {
        assert!(rho0 > F::zero(), "rho0 must be > 0");
        self.rho0 = rho0;
        self
    }

    /// Penalty growth factor: `ρ ← rho_increase · ρ` when feasibility fails
    /// to improve sufficiently (default `10.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `rho_increase > 1`; otherwise the penalty would not
    /// grow and a stalled iterate could never be pushed onto the feasible
    /// set.
    pub fn with_rho_increase(mut self, rho_increase: F) -> Self {
        assert!(rho_increase > F::one(), "rho_increase must be > 1");
        self.rho_increase = rho_increase;
        self
    }

    /// Required feasibility-decrease ratio `τ ∈ (0, 1)`: the multipliers are
    /// updated only when `‖c(x_k)‖ ≤ τ · ‖c(x_{k−1})‖`; otherwise the penalty
    /// is increased instead (default `0.25`).
    ///
    /// # Panics
    ///
    /// Panics unless `0 < feasibility_decrease < 1`.
    pub fn with_feasibility_decrease(
        mut self,
        feasibility_decrease: F,
    ) -> Self {
        assert!(
            feasibility_decrease > F::zero() && feasibility_decrease < F::one(),
            "feasibility_decrease must be in (0, 1)"
        );
        self.feasibility_decrease = feasibility_decrease;
        self
    }

    /// Stop when outer feasibility `‖A x − b‖` reaches this tolerance
    /// (default `1e-8`). `None` disables the check.
    pub fn with_absolute_feasibility_tolerance(
        mut self,
        tol: impl Into<Option<F>>,
    ) -> Self {
        self.tol = crate::core::convergence::optional_tolerance(tol);
        self
    }

    /// Require `‖∇L_ρ(x, λ)‖∞` at the returned inner point to be at most this
    /// tolerance before outer convergence (default `1e-8`).
    ///
    /// Uses the penalty and multipliers that defined the inner problem, before
    /// their outer update. This independent check also applies to derivative-free
    /// and budget-limited inners. Together with the relative guard, it adds up to
    /// two gradient evaluations per consumed result. `None` disables this
    /// absolute test; zero requests exact stationarity.
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
    /// this fraction of its norm at the inner starting point (default `1e-3`).
    ///
    /// Both norms use the same surrogate parameters before the outer update.
    /// Absolute and relative stationarity tests combine with OR. At least one
    /// must pass before outer convergence.
    /// `None` disables this relative test; zero requests exact stationarity.
    /// Disable both stationarity tests to recover feasibility-only stopping.
    pub fn with_relative_stationarity_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.relative_stationarity_tol =
            crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Iteration budget for each inner subproblem solve (default `50`).
    ///
    /// As with the barrier, a first-order inner solver on the (increasingly
    /// ill-conditioned) penalized objective typically exhausts this budget
    /// rather than reaching its own convergence tolerance; the
    /// outer multiplier updates still converge from loosely-minimized
    /// subproblems. Raise it for hard or higher-dimensional problems.
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

impl<So, V: Clone, F: Scalar> AugmentedLagrangianMethod<So, V, F> {
    fn stationarity_satisfied(&self) -> bool {
        crate::core::convergence::stationarity_within_tolerance(
            self.stationarity_norm,
            self.stationarity_initial_norm,
            self.stationarity_tol,
            self.relative_stationarity_tol,
        )
    }

    fn select_incumbent(&self, state: &mut SelectedState<V, F>) {
        let (_, cost, violation) =
            state.current().expect("evaluated outer point");
        if cost.is_nan() || cost == F::infinity() || !violation.is_finite() {
            return;
        }
        let threshold = self.tol.unwrap_or(F::zero());
        let feasible = violation <= threshold;
        let improved =
            state.best().is_none_or(|(_, best_cost, best_violation)| {
                let best_feasible = best_violation <= threshold;
                if feasible != best_feasible {
                    feasible
                } else if feasible {
                    cost < best_cost
                } else {
                    violation < best_violation
                        || (violation == best_violation && cost < best_cost)
                }
            });
        if improved {
            state.select_current();
        }
    }
}

impl<So, V, F> InitialState<V> for AugmentedLagrangianMethod<So, V, F>
where
    F: Scalar,
    V: Clone,
{
    type State = SelectedState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        SelectedState::new(x.clone())
    }
}

impl<P, V, So, F> Solver<P, SelectedState<V, F>>
    for AugmentedLagrangianMethod<So, V, F>
where
    F: Scalar,
    P: LinearEqualityGradient<V, F> + Gradient<Gradient = V>,
    V: crate::VectorLen,
    V: ScaledAdd<F>
        + Dot<F>
        + NormSquared<F>
        + NormInfinity<F>
        + ScaleInPlace<F>
        + Clone,
    So: WarmStart<V>
        + for<'a> Solver<
            AugmentedLagrangian<'a, P, Vec<F>, F>,
            So::State,
            Error = ConstraintSolveError<<P as CostFunction>::Error>,
        >,
    So::State: State<Param = V, Float = F> + CountsMirror,
{
    type Error = ConstraintSolveError<<P as CostFunction>::Error>;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SelectedState<V, F>,
    ) -> Result<SelectedState<V, F>, Self::Error> {
        state.reset();
        self.rho = self.rho0;
        self.c_norm = F::infinity();
        self.c_norm_prev = F::infinity();
        self.stationarity_norm = F::infinity();
        self.stationarity_initial_norm = F::infinity();
        self.last_inner = None;

        // λ ← 0 ∈ ℝᵐ. Clone `b` for the right shape (m entries), then zero it:
        // backend-generic with no "zeros_like" in the math layer. No
        // feasibility precondition: the augmented Lagrangian tolerates an
        // infeasible x₀.
        problem
            .inner()
            .validate_linear(state.param().vec_len())
            .map_err(invalid)?;
        let lambda = vec![F::zero(); problem.inner().linear_count()];
        self.lambda = Some(lambda);

        let cost = problem.cost(state.param())?;
        let residual = problem
            .inner()
            .linear_residual(state.param())
            .map_err(invalid)?;
        let violation = residual.norm_squared().sqrt();
        state.replace(state.param().clone(), cost, violation);
        self.select_incumbent(&mut state);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SelectedState<V, F>,
    ) -> Result<crate::SolverStep<SelectedState<V, F>>, Self::Error> {
        self.stationarity_norm = F::infinity();
        self.stationarity_initial_norm = F::infinity();
        // Minimize the augmented Lagrangian at the current (λ, ρ) on a
        // *separate* inner state seeded (warm-started) at the current
        // iterate. Fresh criteria each call satisfies the statelessness
        // contract.
        let lambda = self.lambda.as_ref().expect("init populates lambda");
        let mut al_wrapper = Problem::new(AugmentedLagrangian::new(
            problem.inner(),
            lambda,
            self.rho,
        ));
        let mut control = crate::core::run_control::RunControl::new()
            .max_iter(self.inner_max_iter);
        let diagnostic_start = self
            .inner_diagnostics
            .then(|| (state.param().clone(), lambda.clone()));
        let inner_state = self.inner_solver.seed(state.param());
        let result = run_loop_with_control(
            &mut al_wrapper,
            inner_state,
            &mut self.inner_solver,
            &mut control,
        );

        // Eval aggregation (adapter-problem composition): fold the inner
        // wrapper's per-call counts back into the outer's wrapper. Copy out
        // before borrowing the outer mutably so the AL adapter's `&P` borrow
        // (still held by `al_wrapper`) doesn't collide with the
        // `counts_mut` reborrow.
        let inner_counts = *al_wrapper.counts();
        let stationarity = result
            .as_ref()
            .ok()
            .filter(|r| {
                (self.stationarity_tol.is_some()
                    || self.relative_stationarity_tol.is_some())
                    && r.report.termination.can_continue_as_inner(
                        crate::PartialResultPolicy::Consume,
                    )
            })
            .map(|r| {
                let initial =
                    al_wrapper.gradient(state.param())?.norm_infinity();
                let final_norm =
                    al_wrapper.gradient(r.state.param())?.norm_infinity();
                Ok::<_, Self::Error>((initial, final_norm))
            })
            .transpose();
        let total_counts = *al_wrapper.counts();
        problem.counts_mut().add(&total_counts);
        self.last_inner = diagnostic_start.map(|(start, lambda)| {
            AugmentedLagrangianInnerDiagnostics {
                rho: self.rho,
                lambda,
                start,
                returned: result.as_ref().ok().map(|r| r.state.param().clone()),
                report: result.as_ref().ok().map(|r| r.report.clone()),
                counts: inner_counts,
                hard_error: result.is_err(),
            }
        });
        let result = result?;
        if let Some((initial, final_norm)) = stationarity? {
            self.stationarity_initial_norm = initial;
            self.stationarity_norm = final_norm;
        }

        if !result
            .report
            .termination
            .can_continue_as_inner(crate::PartialResultPolicy::Consume)
        {
            return Ok(crate::SolverStep::from((
                state,
                result.report.into_outer_termination(
                    crate::PartialResultPolicy::Consume,
                ),
            )));
        }

        // Publish a coherent original objective and feasibility pair only after
        // checking both independently of the surrogate's stopping report.
        let candidate = result.state.param();
        let cost = problem.cost(candidate)?;
        let c = problem
            .inner()
            .linear_residual(candidate)
            .map_err(invalid)?;
        let c_norm = c.norm_squared().sqrt();
        if !cost.is_finite() || !c_norm.is_finite() {
            return Ok(crate::SolverStep::stopped(
                state,
                Termination::numerical_failure(
                    "The augmented-Lagrangian solve returned a non-finite original objective or equality residual.",
                ),
            ));
        }
        self.c_norm_prev = self.c_norm;
        self.c_norm = c_norm;
        state.replace(candidate.clone(), cost, self.c_norm);
        self.select_incumbent(&mut state);

        // Multiplier update vs. penalty increase. The first solve
        // (c_norm_prev = +∞) always takes the update branch.
        if self.c_norm <= self.feasibility_decrease * self.c_norm_prev {
            // Sufficient feasibility improvement: first-order multiplier
            // update λ ← λ + ρ c.
            let lambda = self.lambda.as_mut().expect("init populates lambda");
            lambda.scaled_add(self.rho, &c);
        } else {
            // Feasibility stalled: tighten the penalty, keep λ.
            self.rho = self.rho * self.rho_increase;
        }

        Ok(crate::SolverStep::from((state, None)))
    }

    fn should_check_iterate_change(&self) -> bool {
        self.stationarity_satisfied()
            && self.c_norm.is_finite()
            && self.c_norm <= self.tol.unwrap_or(F::zero())
    }

    fn terminate(
        &self,
        _state: &SelectedState<V, F>,
    ) -> Option<Termination<F>> {
        if self.stationarity_satisfied()
            && self.tol.is_some_and(|tol| self.c_norm <= tol)
        {
            let mut measurements = vec![
                crate::Measurement {
                    name: "residual_norm".into(),
                    value: self.c_norm,
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
                "augmented_lagrangian.feasibility",
                concat!(
                    "residual_norm <= tolerance AND (the finite surrogate-gradient infinity norm ",
                    "meets an enabled absolute stationarity bound OR its relative bound times ",
                    "the finite initial norm). The two stationarity tests combine with OR; ",
                    "disabling both bypasses their guard."
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
    // (`()`) suffices. The struct's `V` generic is unconstrained by the
    // builders, so pin it to a dummy `Vec<f64>` via turbofish.
    type Builder = AugmentedLagrangianMethod<(), Vec<f64>>;

    #[test]
    #[should_panic(expected = "rho0 must be > 0")]
    fn rejects_nonpositive_rho0() {
        let _ = Builder::with_inner_solver(()).rho0(0.0);
    }

    #[test]
    #[should_panic(expected = "rho_increase must be > 1")]
    fn rejects_rho_increase_not_greater_than_one() {
        let _ = Builder::with_inner_solver(()).with_rho_increase(1.0);
    }

    #[test]
    #[should_panic(expected = "feasibility_decrease must be in (0, 1)")]
    fn rejects_feasibility_decrease_out_of_range() {
        let _ = Builder::with_inner_solver(()).with_feasibility_decrease(1.0);
    }

    #[test]
    #[should_panic(expected = "inner_max_iter must be ≥ 1")]
    fn rejects_zero_inner_max_iter() {
        let _ = Builder::with_inner_solver(()).with_inner_max_iter(0);
    }
}
