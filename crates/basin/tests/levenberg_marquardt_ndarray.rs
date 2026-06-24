#![cfg(feature = "ndarray")]

//! Levenberg-Marquardt over the `ndarray` backend (`Array1<f64>` /
//! `Array2<f64>`). Mirrors `tests/levenberg_marquardt_nalgebra.rs`: `Array2`'s
//! `GramMatrix`, `AddDiagonalVectorInPlace`, and `LinearSolveSpd` route through
//! the same pure-Rust Cholesky as the `DenseMatrix` backend, so the Marquardt
//! damping dynamics, rank-deficiency recovery, and MINPACK-style termination
//! are backend-independent — the assertions match the nalgebra mirror exactly.

use basin::problems::{ExponentialFit, PowellSingular, RosenbrockResiduals};
use basin::{Executor, LevenbergMarquardt, NllsState, RelativeCostTolerance, TerminationReason};
use ndarray::Array1;

#[test]
fn levenberg_marquardt_converges_on_rosenbrock_residuals() {
    // LM should converge on Rosenbrock-as-residuals from the classical
    // start. Unlike GN's exact two-step convergence (the linear model
    // is exact along y at fixed x), LM takes a few extra iterations
    // because the damping starts non-zero — but it still reaches the
    // optimum cleanly and emits SolverConverged.
    let problem = RosenbrockResiduals::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-15, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-7,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-7,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn levenberg_marquardt_recovers_on_rank_deficient_powell_singular() {
    // Load-bearing "why LM" test, mirror of GN's failure at the same
    // point. At x = (1, 2, 1, 1) Powell's quadratic-residual rows
    // r₂, r₃ have vanishing Jacobian rows (J has rank 2 < 4), so JᵀJ
    // is singular and pure GN fails Cholesky. LM's damping makes
    // (JᵀJ + μI) SPD by construction, so it should converge cleanly
    // — the canonical demonstration that LM strictly subsumes GN.
    let problem = PowellSingular::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![1.0, 2.0, 1.0, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.cost() < 1e-10,
        "cost = {} (LM should drive Powell to the origin)",
        result.cost()
    );
    // Powell's optimum is x* = 0; check each component drifted toward it.
    for (i, &xi) in result.param().iter().enumerate() {
        assert!(xi.abs() < 1e-2, "x[{}] = {} (expected near 0)", i, xi);
    }
}

#[test]
fn levenberg_marquardt_converges_on_powell_singular_classical_start() {
    // Classical start (3, −1, 0, 1). GN converges here in 12 iterations
    // (per the S3 session notes) because the rank deficiency only
    // bites at the optimum. LM with default Nielsen damping should
    // converge in a comparable iteration count.
    let problem = PowellSingular::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![3.0, -1.0, 0.0, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(100)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.cost() < 1e-10,
        "cost = {} (Powell from classical start should reach near-zero)",
        result.cost()
    );
}

#[test]
fn levenberg_marquardt_emits_solver_converged_via_first_order_optimality() {
    // Convergence path lands SolverConverged (not MaxIter): LM's
    // internal ‖Jᵀr‖_∞ ≤ tol_grad check fires once the iterate is at
    // the optimum. Mirror of the GN test for the same property.
    let problem = RosenbrockResiduals::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(100)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[test]
fn levenberg_marquardt_converges_fast_on_poorly_scaled_exponential_fit() {
    // Regression guard for Marquardt diagonal damping (issue #6). The
    // exponential model ŷ = a·exp(b·t) has wildly disparate Jacobian
    // column scales — ∂r/∂b ≈ a·t·exp(b·t) is ~10⁵× larger than
    // ∂r/∂a = exp(b·t) at amplitude a = 1e5. Marquardt scaling
    // (μ·diag(JᵀJ)) is invariant to that and reaches the global minimum
    // (1e5, −1) in a handful of iterations; the old isotropic μI damping
    // converges to the *same* point but needs ~27 iterations (≈4× the
    // count — the wall-time penalty the issue reported). The tight iter
    // bound below fails under isotropic damping, so it locks the fix in.
    let problem = ExponentialFit::<Array1<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Array1::from_vec(vec![5.0e4, -0.3]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] + 1.0).abs() < 1e-6,
        "b = {} (expected −1)",
        result.param()[1]
    );
    assert!(
        result.iter() <= 15,
        "Marquardt scaling should reach the optimum in ≤15 iters; took {} \
         (isotropic μI damping needs ~27)",
        result.iter()
    );
}

#[test]
fn levenberg_marquardt_pairs_with_relative_cost_tolerance() {
    // The scale-invariant termination side of issue #6: a relative cost
    // tolerance is portable across problem scales where the absolute
    // CostTolerance is not. Disable the solver's own ‖Jᵀr‖∞ check so the
    // framework criterion is what stops the run.
    let problem = ExponentialFit::<Array1<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Array1::from_vec(vec![5.0e4, -0.3]);

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new().with_tol_grad(0.0),
        NllsState::new(initial),
    )
    .max_iter(200)
    .terminate_on(RelativeCostTolerance::new(1e-10))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RelativeCostTolerance);
    assert!(result.cost() < 1e-3, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1e2,
        "a = {} (expected ≈1e5)",
        result.param()[0]
    );
}

#[test]
fn levenberg_marquardt_converges_via_relative_gradient_tolerance() {
    // The MINPACK `gtol` test (issue #6): the scale-invariant cosine
    // measure max_j |gⱼ|/(‖J·,ⱼ‖‖r‖). Disable the absolute ‖Jᵀr‖∞ check
    // so only the relative gradient test can stop the run, and confirm
    // it both fires (SolverConverged, not MaxIter) and lands on the
    // global optimum of the poorly-scaled exponential fit.
    let problem = ExponentialFit::<Array1<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Array1::from_vec(vec![5.0e4, -0.3]);

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new()
            .with_tol_grad(0.0)
            .with_tol_grad_rel(1e-10),
        NllsState::new(initial),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] + 1.0).abs() < 1e-6,
        "b = {} (expected −1)",
        result.param()[1]
    );
}

#[test]
fn levenberg_marquardt_converges_via_ftol() {
    // The MINPACK `ftol` test (issue #8): converge when both the actual
    // and the *predicted* per-iteration reduction in ½‖r‖² are tiny
    // relative to the cost. Disable both gradient tests so only `ftol`
    // (or MaxIter) can stop the run — SolverConverged then implies `ftol`
    // fired — and confirm it lands on the global optimum of the
    // poorly-scaled exponential fit rather than stopping short.
    let problem = ExponentialFit::<Array1<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Array1::from_vec(vec![5.0e4, -0.3]);

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new()
            .with_tol_grad(0.0)
            .with_tol_grad_rel(0.0)
            .with_tol_cost_rel(1e-10),
        NllsState::new(initial),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] + 1.0).abs() < 1e-6,
        "b = {} (expected −1)",
        result.param()[1]
    );
}

#[test]
fn levenberg_marquardt_converges_via_xtol() {
    // The MINPACK `xtol` test (issue #8): converge when the step is
    // negligible relative to the iterate, ‖h‖ ≤ xtol·‖x‖. Disable both
    // gradient tests and `ftol` so SolverConverged implies `xtol` fired.
    let problem = ExponentialFit::<Array1<f64>>::sampled(1.0e5, -1.0, 10, 0.4);
    let initial = Array1::from_vec(vec![5.0e4, -0.3]);

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new()
            .with_tol_grad(0.0)
            .with_tol_grad_rel(0.0)
            .with_tol_step_rel(1e-10),
        NllsState::new(initial),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0e5).abs() < 1.0,
        "a = {} (expected 1e5)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] + 1.0).abs() < 1e-6,
        "b = {} (expected −1)",
        result.param()[1]
    );
}

#[test]
fn relative_gradient_tolerance_is_invariant_to_residual_scaling() {
    // The point of the cosine measure: scaling the residuals by a
    // constant doesn't move the convergence point. Scaling both the
    // amplitude a and the data y by `c` multiplies every residual by `c`
    // (the model is linear in a), so the per-column cosine is identical
    // — the relative gtol must stop at the same iteration for both,
    // where the absolute ‖Jᵀr‖∞ would not (it scales with c²).
    let solve = |scale: f64| {
        let problem = ExponentialFit::<Array1<f64>>::sampled(1.0e3 * scale, -1.0, 10, 0.4);
        let initial = Array1::from_vec(vec![5.0e2 * scale, -0.3]);
        Executor::new(
            problem,
            LevenbergMarquardt::new()
                .with_tol_grad(0.0)
                .with_tol_grad_rel(1e-8),
            NllsState::new(initial),
        )
        .max_iter(200)
        .run()
        .unwrap()
    };

    let small = solve(1.0);
    let large = solve(1.0e3);

    assert_eq!(small.reason, TerminationReason::SolverConverged);
    assert_eq!(large.reason, TerminationReason::SolverConverged);
    assert_eq!(
        small.iter(),
        large.iter(),
        "MINPACK gtol cosine should be invariant to residual scaling"
    );
}

#[test]
fn levenberg_marquardt_caches_residual_and_jacobian_across_iterations() {
    // Regression test for the Madsen-Nielsen caching contract (Alg.
    // 3.16, line 13: J — and so A = JᵀJ, g = Jᵀr — recomputed only
    // after acceptance). At the top of each `next_iter`, LM reuses the
    // residual stashed by `init`/the previous iter, and on a rejected
    // step reuses the cached Gram and gradient at the unchanged iterate
    // — re-evaluating J or reforming A there is wasted work.
    //
    // Disable the internal `‖Jᵀr‖_∞ ≤ tol_grad` check so termination
    // is purely by MaxIter; the early-exit path otherwise evaluates J
    // an extra time on a not-yet-counted iter and muddies the count.
    //
    // For K completed iters on Rosenbrock-as-residuals from the
    // classical start, LM's μ-update accepts every step (no rejections),
    // so:
    //   - cost_evals = 1 (init) + K (one trial per iter)
    //   - jacobian_evals = K (init's J carries iter 1; each subsequent
    //     iter re-evaluates J because the previous accept cleared the
    //     Gram/gradient cache — the last iter's accept clears it but no
    //     follow-up iter consumes it under MaxIter exit).
    let problem = RosenbrockResiduals::<Array1<f64>>::new();
    let initial = Array1::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new().with_tol_grad(0.0),
        NllsState::new(initial),
    )
    .max_iter(3)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 3);
    assert_eq!(
        result.cost_evals(),
        4,
        "expected init (1) + one trial per iter (3) = 4 — uncached LM would also \
         re-evaluate the start-of-iter residual and produce 1 + 2·iters = 7"
    );
    assert!(
        result.state.jacobian_evals() <= 3,
        "jacobian_evals = {} should be ≤ iters (3): init's J carries iter 1, and \
         rejected steps reuse J at the unchanged iterate. Uncached LM produces \
         1 + iters = 4.",
        result.state.jacobian_evals()
    );
}
