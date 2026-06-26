//! POC integration test for the `InnerExecutor` composition adapter.
//!
//! Defines a private outer solver `PerVertexRefine<G>` that holds k
//! parallel iterates and refines each one with an inner GD per outer
//! iter. Exercises the composition contracts documented in
//! `CONTRIBUTING.md` "Solver composition":
//!
//!   1. Eval-counter aggregation. Inner cost evals flow through the
//!      shared `Problem<P>` wrapper automatically; the outer state's
//!      `CountsMirror` impl picks them up via the executor.
//!   2. Criteria reset per run (the `InnerExecutor`'s single criteria vec
//!      is reused on every inner run, and `run_loop` resets each criterion
//!      at the start of every run so stateful criteria don't carry state
//!      across calls).
//!   3. Failure routing (a failing inner bubbles `SolverFailed` via the
//!      outer's mid-iter return).
//!
//! Booth (2D, convex quadratic, optimum `(1, 3)`) is the test problem.
//! The outer state is a custom `MultiStartState` rather than
//! `BasicSimplexState` because `BasicSimplexState`'s vertex/cost fields
//! are `pub(crate)`—the integration test is outside the crate, so we
//! show that composition works through the public `State`, `Solver`, and
//! `CountsMirror` traits alone.

use basin::problems::Booth;
use basin::{
    Backtracking, BasicState, CostFunction, CountsMirror, EvalCounts, Executor, Gradient,
    GradientDescent, GradientTolerance, InnerExecutor, Problem, Solver, State,
    TerminationCriterion, TerminationReason,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Outer-solver state: `k` parallel iterates with parallel costs, kept
/// sorted by ascending cost so [`param`](State::param) returns the best.
struct MultiStartState {
    iterates: Vec<Vec<f64>>,
    costs: Vec<f64>,
    iter: u64,
    cost_evals: u64,
    best_cost: f64,
    best_iter: u64,
    best_cost_evals: u64,
}

impl MultiStartState {
    fn new(iterates: Vec<Vec<f64>>) -> Self {
        let n = iterates.len();
        Self {
            iterates,
            costs: vec![f64::INFINITY; n],
            iter: 0,
            cost_evals: 0,
            best_cost: f64::INFINITY,
            best_iter: 0,
            best_cost_evals: 0,
        }
    }
}

impl State for MultiStartState {
    type Param = Vec<f64>;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }
    fn increment_iter(&mut self) {
        self.iter += 1;
    }
    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }
    fn param(&self) -> &Vec<f64> {
        &self.iterates[0]
    }
    fn cost(&self) -> f64 {
        self.costs[0]
    }
    fn best_param(&self) -> &Vec<f64> {
        &self.iterates[0]
    }
    fn best_cost(&self) -> f64 {
        self.best_cost
    }
    fn best_iter(&self) -> u64 {
        self.best_iter
    }
    fn best_cost_evals(&self) -> u64 {
        self.best_cost_evals
    }
    fn update_best(&mut self) {
        let curr = self.costs[0];
        if curr < self.best_cost {
            self.best_cost = curr;
            self.best_iter = self.iter;
            self.best_cost_evals = self.cost_evals;
        }
    }
    fn reset_best(&mut self) {
        self.best_cost = f64::INFINITY;
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl CountsMirror for MultiStartState {
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free outer: fold every kind of work into `cost_evals`.
        self.cost_evals = delta.total_work();
    }
}

/// Sort iterates and costs jointly by ascending cost.
fn sort_by_cost(iterates: &mut [Vec<f64>], costs: &mut [f64]) {
    let n = iterates.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        costs[i]
            .partial_cmp(&costs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let new_iterates: Vec<Vec<f64>> = idx.iter().map(|&i| iterates[i].clone()).collect();
    let new_costs: Vec<f64> = idx.iter().map(|&i| costs[i]).collect();
    iterates.clone_from_slice(&new_iterates);
    costs.clone_from_slice(&new_costs);
}

/// POC outer solver. Per outer iteration, runs the wrapped inner solver
/// from each iterate and replaces the iterate with the inner's converged
/// param. This is intentionally simple (it isn't real Nelder-Mead) but
/// exercises the full composition contract.
struct PerVertexRefine<G> {
    inner: InnerExecutor<BasicState<Vec<f64>>, G>,
}

impl<G> PerVertexRefine<G> {
    fn new(inner: InnerExecutor<BasicState<Vec<f64>>, G>) -> Self {
        Self { inner }
    }
}

impl<P, G> Solver<P, MultiStartState> for PerVertexRefine<G>
where
    P: CostFunction<Param = Vec<f64>, Output = f64>
        + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>,
    G: Solver<P, BasicState<Vec<f64>>, Error = P::Error>,
{
    type Error = P::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MultiStartState,
    ) -> Result<MultiStartState, Self::Error> {
        for (v, c) in state.iterates.iter().zip(state.costs.iter_mut()) {
            *c = problem.cost(v)?;
        }
        sort_by_cost(&mut state.iterates, &mut state.costs);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MultiStartState,
    ) -> Result<(MultiStartState, Option<TerminationReason>), Self::Error> {
        let mut new_iterates: Vec<Vec<f64>> = Vec::with_capacity(state.iterates.len());
        let mut new_costs: Vec<f64> = Vec::with_capacity(state.iterates.len());
        let prev_iterates = std::mem::take(&mut state.iterates);

        for v in prev_iterates {
            // Same-problem composition: the inner shares the outer's
            // wrapper, so inner evals are accounted for automatically.
            let result = self.inner.run(problem, BasicState::new(v))?;

            // Failure routing (contract 3): bubble `SolverFailed` via
            // the outer's mid-iter return; consume everything else.
            if result.reason.is_failure() {
                // Restore a non-empty `iterates` so `state.param()` /
                // `state.cost()` don't panic on the outer reason read.
                state.iterates = new_iterates;
                state.costs = new_costs;
                if state.iterates.is_empty() {
                    state.iterates.push(vec![0.0; 2]);
                    state.costs.push(f64::INFINITY);
                }
                return Ok((state, Some(result.reason)));
            }

            new_costs.push(result.cost());
            new_iterates.push(result.param().clone());
        }

        state.iterates = new_iterates;
        state.costs = new_costs;
        sort_by_cost(&mut state.iterates, &mut state.costs);
        Ok((state, None))
    }
}

/// Inner solver that always reports `SolverFailed`. Used to verify the
/// outer's failure-routing path without needing a real line-search
/// failure on Booth (which converges cleanly under any reasonable GD
/// configuration).
struct AlwaysFails;

impl<P, S: State> Solver<P, S> for AlwaysFails {
    type Error = std::convert::Infallible;
    fn next_iter(
        &mut self,
        _problem: &mut Problem<P>,
        state: S,
    ) -> Result<(S, Option<TerminationReason>), Self::Error> {
        Ok((state, Some(TerminationReason::SolverFailed)))
    }
}

#[test]
fn inner_executor_polishes_starts_to_booth_optimum() {
    let problem = Booth::<Vec<f64>>::default();
    let starts = vec![vec![0.0, 0.0], vec![-1.0, 5.0], vec![3.0, 1.0]];
    let outer_state = MultiStartState::new(starts);

    let inner = InnerExecutor::new(GradientDescent::with_line_search(Backtracking::new()))
        .max_iter(50)
        .terminate_on(GradientTolerance(1e-8));
    let outer = PerVertexRefine::new(inner);

    let result = Executor::new(problem, outer, outer_state)
        .max_iter(3)
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "expected near-zero cost at Booth optimum (1, 3), got {}",
        result.cost()
    );
    let best = result.param();
    assert!(
        (best[0] - 1.0).abs() < 1e-3,
        "x[0] = {} (expected ≈ 1)",
        best[0]
    );
    assert!(
        (best[1] - 3.0).abs() < 1e-3,
        "x[1] = {} (expected ≈ 3)",
        best[1]
    );
}

#[test]
fn inner_executor_aggregates_cost_evals_into_outer() {
    let problem = Booth::<Vec<f64>>::default();
    let starts = vec![vec![0.0, 0.0], vec![-1.0, 5.0], vec![3.0, 1.0]];
    let outer_state = MultiStartState::new(starts);

    let inner = InnerExecutor::new(GradientDescent::with_line_search(Backtracking::new()))
        .max_iter(50)
        .terminate_on(GradientTolerance(1e-8));
    let outer = PerVertexRefine::new(inner);

    let result = Executor::new(problem, outer, outer_state)
        .max_iter(2)
        .run()
        .unwrap();

    // The outer's init evaluates one cost per starting iterate (3 evals).
    // Each inner GD run does at least an `init` fused cost+gradient
    // (1 cost, 1 gradient) plus iteration work (line search + final
    // cost). With 3 starts × 2 outer iters and the derivative-free
    // mirror folding all work into cost_evals, the lower bound holds
    // comfortably; loose because GD's exact eval count depends on the
    // backtracking line-search probe schedule.
    let evals = result.state.cost_evals();
    assert!(
        evals >= 3 + 6,
        "expected outer to aggregate inner work; got {} cost evals (≥ 9 minimum)",
        evals
    );
}

/// Stateful inner criterion that records how many times `reset` is
/// invoked. Used to prove `run_loop` resets criteria once per inner run,
/// so an `InnerExecutor`'s reused criteria vector sees fresh per-run state
/// (contract 2). `check` never fires; termination is left to the real
/// `GradientTolerance` alongside it.
struct CountResets(Rc<RefCell<u32>>);

impl<S> TerminationCriterion<S> for CountResets {
    fn check(&mut self, _state: &S) -> Option<TerminationReason> {
        None
    }
    fn reset(&mut self) {
        *self.0.borrow_mut() += 1;
    }
}

#[test]
fn inner_executor_resets_criteria_once_per_run() {
    let problem = Booth::<Vec<f64>>::default();
    let starts = vec![vec![0.0, 0.0], vec![-1.0, 5.0], vec![3.0, 1.0]];
    let outer_state = MultiStartState::new(starts);

    let resets = Rc::new(RefCell::new(0u32));
    let inner = InnerExecutor::new(GradientDescent::with_line_search(Backtracking::new()))
        .max_iter(50)
        .terminate_on(GradientTolerance(1e-8))
        .terminate_on(CountResets(Rc::clone(&resets)));
    let outer = PerVertexRefine::new(inner);

    Executor::new(problem, outer, outer_state)
        .max_iter(2)
        .run()
        .unwrap();

    // 3 starts × 2 outer iters = 6 inner `run()` calls; `run_loop` resets
    // each criterion once at the start of every run.
    assert_eq!(
        *resets.borrow(),
        6,
        "run_loop should reset the reused criteria vector once per inner run"
    );
}

#[test]
fn inner_executor_bubbles_inner_solver_failed_via_outer() {
    let problem = Booth::<Vec<f64>>::default();
    let starts = vec![vec![0.0, 0.0]];
    let outer_state = MultiStartState::new(starts);

    let inner = InnerExecutor::new(AlwaysFails);
    let outer = PerVertexRefine::new(inner);

    let result = Executor::new(problem, outer, outer_state)
        .max_iter(5)
        .run()
        .unwrap();

    assert_eq!(
        result.reason,
        TerminationReason::SolverFailed,
        "outer should bubble SolverFailed from the inner; got {:?}",
        result.reason
    );
    // Outer didn't complete a full iter; `iter()` reflects the last
    // *fully completed* iteration per the executor contract.
    assert_eq!(result.iter(), 0);
}
