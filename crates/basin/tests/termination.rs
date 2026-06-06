use basin::{
    Backtracking, BasicSimplexState, BasicState, CostFunction, CostTolerance, Executor, Gradient,
    GradientDescent, GradientState, GradientTolerance, MaxCostEvals, MaxGradientEvals, MaxIter,
    MaxTime, NelderMead, NoImprovement, ParamTolerance, Problem, RelativeCostTolerance,
    RelativeGradientTolerance, RelativeParamTolerance, Solver, State, TargetCost,
    TerminationCriterion, TerminationReason,
};
use std::time::Duration;

/// f(x) = ½ ‖x‖² — convex quadratic with minimum at origin, gradient = x.
struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(0.5 * x.iter().map(|v| v * v).sum::<f64>())
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok(x.clone())
    }
}

#[test]
fn gradient_tolerance_fires_at_iter_zero_when_starting_at_optimum() {
    // Initial param is the optimum: gradient = 0. Should terminate before
    // doing any iterations.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![0.0, 0.0]),
    )
    .terminate_on(GradientTolerance(1e-8))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert_eq!(result.iter(), 0, "should not have done any iterations");
}

#[test]
fn gradient_tolerance_fires_after_convergence() {
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, -1.0, 0.5]),
    )
    .max_iter(1_000)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert!(result.iter() > 0 && result.iter() < 1_000);
    let g = result
        .state
        .gradient()
        .expect("gradient should be populated");
    let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(g_norm <= 1e-6);
}

#[test]
fn relative_gradient_tolerance_fires_after_convergence() {
    // On f = ½‖x‖², GradientDescent(0.5) gives ∇f_k = x_k = 0.5^k·x_0, so
    // ‖∇f_k‖/‖∇f_0‖ = 0.5^k. A 1e-3 relative bound fires once 0.5^k ≤
    // 1e-3, i.e. around iter 10.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(1_000)
    .terminate_on(RelativeGradientTolerance::new(1e-3))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RelativeGradientTolerance);
    assert!(
        result.iter() > 0 && result.iter() < 20,
        "expected convergence near iter 10, got {}",
        result.iter()
    );
}

#[test]
fn relative_gradient_tolerance_fires_at_iter_zero_when_starting_at_optimum() {
    // ∇f(x_0) = 0 ⇒ initial norm 0, and 0 ≤ tol²·0, so it fires at iter 0
    // (same edge as the absolute GradientTolerance).
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![0.0, 0.0]),
    )
    .terminate_on(RelativeGradientTolerance::new(1e-6))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RelativeGradientTolerance);
    assert_eq!(result.iter(), 0);
}

#[test]
fn relative_gradient_tolerance_is_scale_invariant() {
    // The headline property: normalizing by the initial gradient makes
    // the stopping iteration independent of the objective's scale, where
    // the absolute GradientTolerance would fire at different iterations.
    // Scaling the start by 10⁶ scales every gradient by 10⁶ but leaves
    // the ratio ‖∇f_k‖/‖∇f_0‖ = 0.5^k unchanged.
    let run_from = |x0: f64| {
        Executor::new(
            Quadratic,
            GradientDescent::new(0.5),
            BasicState::new(vec![x0, x0]),
        )
        .max_iter(1_000)
        .terminate_on(RelativeGradientTolerance::new(1e-3))
        .run()
        .unwrap()
    };

    let small = run_from(1.0);
    let large = run_from(1.0e6);

    assert_eq!(small.reason, TerminationReason::RelativeGradientTolerance);
    assert_eq!(large.reason, TerminationReason::RelativeGradientTolerance);
    assert_eq!(
        small.iter(),
        large.iter(),
        "relative gradient tolerance should be scale-invariant"
    );
}

#[test]
fn max_iter_field_default_is_one_thousand() {
    // No criteria configured: the default `max_iter = 1000` should fire.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.001), // tiny step → won't converge in 1000
        BasicState::new(vec![10.0, 10.0]),
    )
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 1_000);
}

#[test]
fn explicit_max_iter_criterion_works_alongside_default() {
    // `MaxIter(5)` via `terminate_on` fires before the default 1000.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.001),
        BasicState::new(vec![10.0, 10.0]),
    )
    .terminate_on(MaxIter(5))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 5);
}

#[test]
fn param_tolerance_fires_when_steps_become_small() {
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(1_000)
    .terminate_on(ParamTolerance::new(1e-8))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::ParamTolerance);
}

#[test]
fn cost_tolerance_fires_when_cost_stagnates() {
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(1_000)
    .terminate_on(CostTolerance::new(1e-12))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::CostTolerance);
}

#[test]
fn relative_param_tolerance_fires_when_relative_step_small() {
    // On f = ½‖x‖², GradientDescent(α) gives x_{k+1} = (1−α)x_k, so the
    // relative step ‖Δx‖/‖x_k‖ = α/(1−α) is constant. With α = 0.001
    // that's ≈ 1e-3, below a 1e-2 relative bound, so the criterion fires
    // early (where an *absolute* ParamTolerance would keep shrinking as
    // x → 0).
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.001),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(1_000)
    .terminate_on(RelativeParamTolerance::new(1e-2))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RelativeParamTolerance);
    assert!(result.iter() < 5, "fired late at iter {}", result.iter());
}

#[test]
fn relative_cost_tolerance_fires_when_relative_reduction_small() {
    // Relative cost reduction |Δf|/|f_{k−1}| = α(2−α) is constant on the
    // quadratic; α = 0.001 gives ≈ 2e-3 < 1e-2, so the criterion fires.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.001),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(1_000)
    .terminate_on(RelativeCostTolerance::new(1e-2))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RelativeCostTolerance);
    assert!(result.iter() < 5, "fired late at iter {}", result.iter());
}

#[test]
fn target_cost_fires_at_iter_zero_when_start_is_below_target() {
    // f(x_0) = 0.5·‖(0.5, 0.5)‖² = 0.25 ≤ 1.0, so the criterion fires
    // before any iteration runs.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![0.5, 0.5]),
    )
    .terminate_on(TargetCost(1.0))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::TargetCost);
    assert_eq!(result.iter(), 0);
}

#[test]
fn target_cost_fires_when_cost_drops_to_target() {
    // f_k = 0.5·(1-α)^(2k)·‖x_0‖²; with α = 0.5, ‖x_0‖² = 2, we hit
    // f ≤ 1e-3 after a small number of iterations and well before
    // max_iter.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(1_000)
    .terminate_on(TargetCost(1e-3))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::TargetCost);
    assert!(result.iter() > 0 && result.iter() < 1_000);
    assert!(result.state.cost() <= 1e-3);
}

#[test]
fn target_cost_does_not_fire_when_target_unreachable() {
    // Target below the global min (f ≥ 0) — should never fire, MaxIter
    // wins.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, 1.0]),
    )
    .terminate_on(MaxIter(10))
    .terminate_on(TargetCost(-1.0))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 10);
}

#[test]
fn no_improvement_fires_after_patience_stalled_iters() {
    // tol = 10.0 is larger than any drop on the quadratic from [1, 1]
    // (max drop is 0.75 at iter 1), so every iter past the first counts
    // as "no improvement". With patience = 3 the criterion fires once
    // stalled == 3, which is at state.iter() == 3.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(100)
    .terminate_on(NoImprovement::new(3, 10.0))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::NoImprovement);
    assert_eq!(result.iter(), 3);
}

#[test]
fn no_improvement_does_not_fire_under_monotone_decrease() {
    // Quadratic with GD(α=0.5) gives f_k = 0.25^k — strictly
    // decreasing. With tol = 0.0 every iteration is "an improvement",
    // so the patience counter never advances and MaxIter wins.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, 1.0]),
    )
    .terminate_on(MaxIter(20))
    .terminate_on(NoImprovement::new(5, 0.0))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 20);
}

#[test]
fn no_improvement_resets_counter_on_real_improvement() {
    // f_k = 0.25^k on Quadratic with GD(α=0.5). Drops are 0.75,
    // 0.1875, 0.0469, 0.0117, 0.0029, …. With tol = 0.1 the first two
    // drops count as improvements (resetting `stalled` to 0 each
    // time), then drops 3+ all fall below tol. So stalled increments
    // from iter 3 onward; with patience = 3 the criterion fires at
    // iter 5 — not iter 3, which is what it would do without the
    // reset-on-improvement behavior.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(100)
    .terminate_on(NoImprovement::new(3, 0.1))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::NoImprovement);
    assert_eq!(result.iter(), 5);
}

#[test]
fn relative_cost_tolerance_is_scale_invariant() {
    // The headline property: a single relative `tol` fires at the same
    // iteration regardless of the cost scale, where an absolute
    // `CostTolerance` would fire at wildly different points. Run the
    // same solver from a small start and a 10⁶-times-larger start; the
    // relative criterion must stop at the same iter.
    let run_from = |x0: f64| {
        Executor::new(
            Quadratic,
            GradientDescent::new(0.001),
            BasicState::new(vec![x0, x0]),
        )
        .max_iter(1_000)
        .terminate_on(RelativeCostTolerance::new(1e-2))
        .run()
        .unwrap()
    };

    let small = run_from(1.0);
    let large = run_from(1.0e6);

    assert_eq!(small.reason, TerminationReason::RelativeCostTolerance);
    assert_eq!(large.reason, TerminationReason::RelativeCostTolerance);
    assert_eq!(
        small.iter(),
        large.iter(),
        "relative cost tolerance should be scale-invariant"
    );
}

#[test]
fn first_criterion_to_fire_wins() {
    // ParamTolerance with a huge tolerance fires immediately on iter 1
    // (any movement < 100). MaxIter(1000) would otherwise fire later.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(1_000)
    .terminate_on(ParamTolerance::new(100.0))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::ParamTolerance);
    assert!(result.iter() < 5);
}

#[test]
fn max_time_eventually_fires() {
    // Use a tiny budget so the test is fast but deterministic.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.001),
        BasicState::new(vec![1e6, 1e6, 1e6]),
    )
    .max_iter(u64::MAX)
    .terminate_on(MaxTime::new(Duration::from_millis(50)))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxTime);
}

/// Solver that always reports converged via the `terminate` hook, used to
/// verify the per-solver hook still works after framework criteria are
/// checked.
struct AlwaysConverged;

impl Solver<Quadratic, BasicState<Vec<f64>>> for AlwaysConverged {
    type Error = std::convert::Infallible;
    fn next_iter(
        &mut self,
        _problem: &mut Problem<Quadratic>,
        state: BasicState<Vec<f64>>,
    ) -> Result<(BasicState<Vec<f64>>, Option<TerminationReason>), Self::Error> {
        Ok((state, None))
    }

    fn terminate(&self, _state: &BasicState<Vec<f64>>) -> Option<TerminationReason> {
        Some(TerminationReason::SolverConverged)
    }
}

#[test]
fn solver_terminate_hook_is_honored() {
    let result = Executor::new(Quadratic, AlwaysConverged, BasicState::new(vec![1.0, 2.0]))
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.iter(), 0);
}

/// Solver that completes one full iteration, then reports a mid-iter
/// failure on the next call via the tuple return value. Verifies that
/// `next_iter`'s second return slot halts the executor and that
/// `state.iter()` is *not* incremented for the bailed iteration.
struct FailsOnSecondCall {
    calls: u64,
}

impl Solver<Quadratic, BasicState<Vec<f64>>> for FailsOnSecondCall {
    type Error = std::convert::Infallible;
    fn next_iter(
        &mut self,
        _problem: &mut Problem<Quadratic>,
        state: BasicState<Vec<f64>>,
    ) -> Result<(BasicState<Vec<f64>>, Option<TerminationReason>), Self::Error> {
        Ok({
            self.calls += 1;
            if self.calls >= 2 {
                (state, Some(TerminationReason::SolverFailed))
            } else {
                (state, None)
            }
        })
    }
}

#[test]
fn solver_can_signal_termination_mid_iter() {
    let result = Executor::new(
        Quadratic,
        FailsOnSecondCall { calls: 0 },
        BasicState::new(vec![1.0, 2.0]),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverFailed);
    // First call completed (iter 0 → 1), second call bailed without
    // incrementing, so iter stays at 1.
    assert_eq!(result.iter(), 1);
}

/// Verify that a custom criterion plays correctly through `Box<dyn>`.
struct StopAt(u64);

impl<S: State> TerminationCriterion<S> for StopAt {
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        (state.iter() == self.0).then_some(TerminationReason::SolverConverged)
    }
}

#[test]
fn cost_evals_matches_iter_for_constant_step_gradient_descent() {
    // Constant step + cost+gradient per iter ⇒ exactly 1 cost eval per
    // iter, plus 1 in init. So cost_evals == iter + 1.
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.001),
        BasicState::new(vec![10.0, 10.0]),
    )
    .terminate_on(MaxIter(20))
    .run()
    .unwrap();

    assert_eq!(result.iter(), 20);
    assert_eq!(result.state.cost_evals(), 21);
}

#[test]
fn cost_evals_exceeds_iter_with_backtracking() {
    // Each backtracking call evaluates the cost at least once. A
    // sufficiently aggressive `alpha_init` forces several rejections on
    // most iterations, so cost_evals > iter + 1.
    let result = Executor::new(
        Quadratic,
        GradientDescent::with_line_search(Backtracking::new().alpha_init(8.0).rho(0.5)),
        BasicState::new(vec![1.0, 1.0]),
    )
    .terminate_on(MaxIter(10))
    .run()
    .unwrap();

    assert_eq!(result.iter(), 10);
    assert!(
        result.state.cost_evals() > result.iter() + 1,
        "expected line search to inflate cost_evals beyond iter+1: cost_evals={}, iter={}",
        result.state.cost_evals(),
        result.iter()
    );
}

#[test]
fn cost_evals_exceeds_iter_for_nelder_mead_shrinks() {
    // Init evaluates n+1 vertices and each iteration spends 1–2 extra
    // cost evals (more on shrink), so cost_evals ≥ iter + 3.
    let result = Executor::new(
        Quadratic,
        NelderMead::new(),
        BasicSimplexState::new(vec![2.0, -3.0]),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();

    assert!(result.state.cost_evals() >= result.iter() + 3);
}

#[test]
fn max_gradient_evals_fires_before_max_iter() {
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.001),
        BasicState::new(vec![10.0, 10.0]),
    )
    .max_iter(10_000)
    .terminate_on(MaxGradientEvals(5))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxGradientEvals);
    assert!(result.state.gradient_evals() >= 5);
}

#[test]
fn max_cost_evals_fires_before_max_iter() {
    let result = Executor::new(
        Quadratic,
        NelderMead::new(),
        BasicSimplexState::new(vec![5.0, -2.0, 4.0]),
    )
    .max_iter(10_000)
    .terminate_on(MaxCostEvals(25))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert!(
        result.state.cost_evals() >= 25,
        "cost_evals should have reached the budget: {}",
        result.state.cost_evals()
    );
}

#[test]
fn custom_termination_criterion() {
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![5.0, 5.0]),
    )
    .max_iter(1_000)
    .terminate_on(StopAt(7))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.iter(), 7);
}
