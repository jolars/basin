#![cfg(feature = "nalgebra")]

use basin::problems::BoothBoxedResiduals;
use basin::{Executor, MaxIter, NllsState, TerminationReason, Trf};
use nalgebra::DVector;

#[test]
fn trf_with_slack_bounds_reaches_unconstrained_min() {
    // Bounds `[-5, 5]²` are wide enough that Booth's unconstrained min
    // `(1, 3)` is interior — no constraint binds. TRF should reach it
    // to ‖·‖ < 1e-5 in a handful of iterations.
    let problem = BoothBoxedResiduals::<DVector<f64>>::new(
        DVector::from_vec(vec![-5.0, -5.0]),
        DVector::from_vec(vec![5.0, 5.0]),
    );
    let initial = DVector::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-5,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 3.0).abs() < 1e-5,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn trf_with_tight_bounds_converges_to_box_corner() {
    // Edge-active test: Booth's unconstrained min `(1, 3)` lies outside
    // `[-1, 1]²`. The constrained optimum is the corner `(1, 1)` (the
    // box vertex closest to `(1, 3)`). TRF should converge to that
    // corner — load-bearing demonstration that the BCL scaled-gradient
    // metric vanishes at face-active KKT points.
    let problem = BoothBoxedResiduals::<DVector<f64>>::new(
        DVector::from_vec(vec![-1.0, -1.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    );
    let initial = DVector::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    // The strict-interior θ < 1 keeps the iterate just inside the
    // corner (e.g. ~(0.9999..., 0.9999...)), so check tolerances are
    // looser than the unconstrained case but tight enough to confirm
    // the right vertex.
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-3,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-3,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn trf_init_projects_infeasible_start_strictly_inside_box() {
    // Infeasible start `(10, 10)` outside `[-1, 1]²`. After `init` the
    // iterate must be *strictly* inside the box — not just clamped to
    // the closed face — because `D` is undefined where `v_i = 0`.
    // Asserted via `MaxIter(0)` so only `init` runs; the state we read
    // is what `init` produced.
    let problem = BoothBoxedResiduals::<DVector<f64>>::new(
        DVector::from_vec(vec![-1.0, -1.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    );
    let initial = DVector::from_vec(vec![10.0, 10.0]);

    let mut executor = Executor::new(problem, Trf::new(), NllsState::new(initial));
    executor = executor.terminate_on(MaxIter(0));
    let result = executor.run().unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    let x = result.param();
    // Strict interior: each component should be strictly less than the
    // upper bound and strictly greater than the lower bound.
    assert!(
        x[0] < 1.0,
        "x[0] = {} should be < 1.0 (strictly inside)",
        x[0]
    );
    assert!(
        x[1] < 1.0,
        "x[1] = {} should be < 1.0 (strictly inside)",
        x[1]
    );
    assert!(x[0] > -1.0, "x[0] = {} should be > -1.0", x[0]);
    assert!(x[1] > -1.0, "x[1] = {} should be > -1.0", x[1]);
}

#[test]
fn trf_emits_solver_converged_via_scaled_first_order_optimality() {
    // The default `tol_grad = 1e-8` triggers `SolverConverged` once the
    // BCL scaled-gradient `‖D·Jᵀr‖_∞` falls below the threshold. Check
    // both the convergence and the explicit reason — mirror of the LM
    // test for the same property.
    let problem = BoothBoxedResiduals::<DVector<f64>>::new(
        DVector::from_vec(vec![-1.0, -1.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    );
    let initial = DVector::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[test]
fn trf_caches_residual_and_jacobian_across_iterations() {
    // Regression test for TRF's caching contract — symmetric to LM
    // (BCL §4 reuses the Madsen-Nielsen accept/reject shape, so the
    // cache logic is identical: stash r at the new iterate on accept,
    // keep both r and J on reject, drop J on accept since J(x_trial)
    // wasn't computed for the gain-ratio test).
    //
    // Disable the internal `‖D·Jᵀr‖_∞ ≤ tol_grad` check so termination
    // is purely MaxIter — the early-exit path otherwise causes an
    // extra (uncounted-against-iter) J evaluation that muddies the
    // count. Run on slack-bounded Booth so no face binds and the
    // dynamics reduce to LM's.
    let problem = BoothBoxedResiduals::<DVector<f64>>::new(
        DVector::from_vec(vec![-5.0, -5.0]),
        DVector::from_vec(vec![5.0, 5.0]),
    );
    let initial = DVector::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(
        problem,
        Trf::new().with_tol_grad(0.0),
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
        "expected init (1) + one trial per iter (3) = 4 — uncached TRF would also \
         re-evaluate the start-of-iter residual and produce 1 + 2·iters = 7"
    );
    assert!(
        result.state.jacobian_evals() <= 3,
        "jacobian_evals = {} should be ≤ iters (3): init's J carries iter 1, and \
         rejected steps reuse J at the unchanged iterate. Uncached TRF produces \
         1 + iters = 4.",
        result.state.jacobian_evals()
    );
}
