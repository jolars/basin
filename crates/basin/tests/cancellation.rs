//! Integration tests for first-class executor cancellation.

use std::cell::RefCell;
use std::convert::Infallible;
use std::rc::Rc;

use basin::{
    BasicState, CancellationToken, CostFunction, CountsMirror, EvalCounts,
    Executor, Gradient, GradientDescent, Observe, ObserverMode, Problem,
    Solver, State, StepOutcome, TerminationReason,
};

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|xi| 2.0 * xi).collect())
    }
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn cloned_tokens_share_a_sticky_cancellation_flag() {
    assert_send_sync::<CancellationToken>();

    let token = CancellationToken::new();
    let clone = token.clone();
    assert!(!token.is_cancelled());
    assert!(!clone.is_cancelled());

    clone.cancel();
    clone.cancel();

    assert!(token.is_cancelled());
    assert!(clone.is_cancelled());
}

#[test]
fn pre_cancelled_executor_returns_an_initialized_state() {
    let token = CancellationToken::new();
    token.cancel();

    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![3.0, 4.0]),
    )
    .max_iter(0)
    .with_cancellation_token(token)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::Cancelled);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.cost(), 25.0);
    assert_eq!(result.best_param(), &vec![3.0, 4.0]);
    assert_eq!(result.best_cost(), 25.0);
    assert_eq!(result.cost_evals(), 1);
}

#[test]
fn a_later_token_replaces_the_previous_one() {
    let cancelled = CancellationToken::new();
    cancelled.cancel();

    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0]),
    )
    .max_iter(0)
    .with_cancellation_token(cancelled)
    .with_cancellation_token(CancellationToken::new())
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
}

struct ScriptedSolver;

struct ScriptedState {
    param: Vec<f64>,
    cost: f64,
    iter: u64,
    cost_evals: u64,
    best_param: Vec<f64>,
    best_cost: f64,
    best_iter: u64,
    best_cost_evals: u64,
}

impl ScriptedState {
    fn new(param: Vec<f64>) -> Self {
        Self {
            param,
            cost: f64::INFINITY,
            iter: 0,
            cost_evals: 0,
            best_param: Vec::new(),
            best_cost: f64::INFINITY,
            best_iter: 0,
            best_cost_evals: 0,
        }
    }
}

impl State for ScriptedState {
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
        &self.param
    }

    fn cost(&self) -> f64 {
        self.cost
    }

    fn best_param(&self) -> &Vec<f64> {
        &self.best_param
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
        if self.best_param.is_empty() || self.cost < self.best_cost {
            self.best_param = self.param.clone();
            self.best_cost = self.cost;
            self.best_iter = self.iter;
            self.best_cost_evals = self.cost_evals;
        }
    }

    fn reset_best(&mut self) {
        self.best_param.clear();
        self.best_cost = f64::INFINITY;
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl CountsMirror for ScriptedState {
    fn mirror(&mut self, delta: &EvalCounts) {
        self.cost_evals = delta.cost_evals;
    }
}

impl Solver<Quadratic, ScriptedState> for ScriptedSolver {
    type Error = Infallible;

    fn init(
        &mut self,
        problem: &mut Problem<Quadratic>,
        mut state: ScriptedState,
    ) -> Result<ScriptedState, Infallible> {
        state.cost = problem.cost(&state.param)?;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Quadratic>,
        mut state: ScriptedState,
    ) -> Result<(ScriptedState, Option<TerminationReason>), Infallible> {
        let param = if state.iter() == 0 {
            vec![0.0]
        } else {
            vec![10.0]
        };
        let cost = problem.cost(&param)?;
        state.param = param;
        state.cost = cost;
        Ok((state, None))
    }
}

struct CancelAfter {
    iter: u64,
    token: CancellationToken,
    final_reason: Rc<RefCell<Option<TerminationReason>>>,
}

impl<S: State> Observe<S> for CancelAfter {
    fn observe_iter(&mut self, state: &S) {
        if state.iter() == self.iter {
            self.token.cancel();
        }
    }

    fn observe_final(&mut self, _state: &S, reason: &TerminationReason) {
        *self.final_reason.borrow_mut() = Some(*reason);
    }
}

#[test]
fn stepper_cancels_between_iterations_without_rewinding_state() {
    let token = CancellationToken::new();
    let final_reason = Rc::new(RefCell::new(None));
    let observer = CancelAfter {
        iter: 2,
        token: token.clone(),
        final_reason: Rc::clone(&final_reason),
    };
    let mut stepper =
        Executor::new(Quadratic, ScriptedSolver, ScriptedState::new(vec![2.0]))
            .max_iter(100)
            .with_cancellation_token(token)
            .observe_with(observer, ObserverMode::Always)
            .into_stepper()
            .unwrap();

    assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    assert_eq!(stepper.iter(), 2);
    assert_eq!(stepper.state().param(), &vec![10.0]);
    assert_eq!(stepper.state().best_param(), &vec![0.0]);

    let stopped = StepOutcome::Stopped(TerminationReason::Cancelled);
    assert_eq!(stepper.step().unwrap(), stopped);
    assert_eq!(stepper.step().unwrap(), stopped);
    assert_eq!(stepper.finished(), Some(&TerminationReason::Cancelled));
    assert_eq!(stepper.iter(), 2);
    assert_eq!(stepper.state().param(), &vec![10.0]);
    assert_eq!(stepper.state().cost(), 100.0);
    assert_eq!(stepper.state().best_param(), &vec![0.0]);
    assert_eq!(stepper.state().best_cost(), 0.0);
    assert_eq!(*final_reason.borrow(), Some(TerminationReason::Cancelled));
}
