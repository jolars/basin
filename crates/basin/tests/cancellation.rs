//! Integration tests for first-class executor cancellation.

use std::cell::RefCell;
use std::convert::Infallible;
use std::rc::Rc;

use basin::{
    CancellationToken, CostFunction, CountsMirror, EvalCounts, Executor,
    FirstOrderState, Gradient, GradientDescent, Observe, ObserverMode, Problem,
    Solver, State, StepOutcome, TerminationCode,
};

struct Quadratic;

impl basin::Constraints for Quadratic {
    type Set = basin::NoConstraints;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new()
    }
}
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
        FirstOrderState::new(vec![3.0, 4.0]),
    )
    .max_iter(0)
    .with_cancellation_token(token)
    .run()
    .unwrap();

    assert_eq!(result.report.code(), TerminationCode::Cancelled);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.current_record().unwrap().cost.unwrap(), 25.0);
    assert_eq!(result.selected_record().unwrap().param, &vec![3.0, 4.0]);
    assert_eq!(result.selected_record().unwrap().cost.unwrap(), 25.0);
    assert_eq!(result.cost_evals(), 1);
}

#[test]
fn a_later_token_replaces_the_previous_one() {
    let cancelled = CancellationToken::new();
    cancelled.cancel();

    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        FirstOrderState::new(vec![1.0]),
    )
    .max_iter(0)
    .with_cancellation_token(cancelled)
    .with_cancellation_token(CancellationToken::new())
    .run()
    .unwrap();

    assert_eq!(result.report.code(), TerminationCode::MaxIter);
}

struct ScriptedSolver;

struct ScriptedState {
    param: Vec<f64>,
    cost: f64,
    iter: u64,
    counts: EvalCounts,
    best_param: Vec<f64>,
    best_cost: f64,
    best_iter: u64,
    best_counts: EvalCounts,
}

impl ScriptedState {
    fn new(param: Vec<f64>) -> Self {
        Self {
            param,
            cost: f64::INFINITY,
            iter: 0,
            counts: EvalCounts::default(),
            best_param: Vec::new(),
            best_cost: f64::INFINITY,
            best_iter: 0,
            best_counts: EvalCounts::default(),
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

    fn param(&self) -> &Vec<f64> {
        &self.param
    }

    fn current_record(&self) -> Option<basin::CurrentRef<'_, Vec<f64>>> {
        (self.counts.cost_evals > 0)
            .then(|| basin::CurrentRef::new(&self.param, Some(self.cost), None))
    }
    fn selected_record(&self) -> Option<basin::SelectedRef<'_, Vec<f64>>> {
        (!self.best_param.is_empty()).then(|| {
            basin::SelectedRef::new(
                basin::CurrentRef::new(
                    &self.best_param,
                    Some(self.best_cost),
                    None,
                ),
                self.best_iter,
                &self.best_counts,
            )
        })
    }
    fn update_best(&mut self) {
        if self.best_param.is_empty() || self.cost < self.best_cost {
            self.best_param = self.param.clone();
            self.best_cost = self.cost;
            self.best_iter = self.iter;
            self.best_counts = self.counts;
        }
    }

    fn reset_best(&mut self) {
        self.best_param.clear();
        self.best_cost = f64::INFINITY;
        self.best_iter = 0;
        self.best_counts = EvalCounts::default();
    }
}

impl CountsMirror for ScriptedState {
    fn cost_evals(&self) -> u64 {
        self.counts.cost_evals
    }
    fn mirror(&mut self, delta: &EvalCounts) {
        self.counts = *delta;
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
        state: &mut ScriptedState,
    ) -> Result<basin::SolverStep<f64>, Infallible> {
        let param = if state.iter() == 0 {
            vec![0.0]
        } else {
            vec![10.0]
        };
        let cost = problem.cost(&param)?;
        state.param = param;
        state.cost = cost;
        Ok(basin::SolverStep::completed())
    }
}

struct CancelAfter {
    iter: u64,
    token: CancellationToken,
    final_reason: Rc<RefCell<Option<TerminationCode>>>,
}

impl<S: State> Observe<S> for CancelAfter {
    fn observe_iter(&mut self, state: &S) {
        if state.iter() == self.iter {
            self.token.cancel();
        }
    }

    fn observe_final(
        &mut self,
        _state: &S,
        reason: &basin::TerminationReport<S::Float>,
    ) {
        *self.final_reason.borrow_mut() = Some(reason.code());
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
    assert_eq!(stepper.state().selected_record().unwrap().param, &vec![0.0]);

    let stopped = stepper.step().unwrap();
    assert_eq!(stopped.report().unwrap().code(), TerminationCode::Cancelled);
    assert_eq!(stepper.step().unwrap(), stopped);
    assert_eq!(stepper.step().unwrap(), stopped);
    assert_eq!(
        stepper.finished().map(|report| report.code()),
        Some(TerminationCode::Cancelled)
    );
    assert_eq!(stepper.iter(), 2);
    assert_eq!(stepper.state().param(), &vec![10.0]);
    assert_eq!(
        stepper.state().current_record().unwrap().cost.unwrap(),
        100.0
    );
    assert_eq!(stepper.state().selected_record().unwrap().param, &vec![0.0]);
    assert_eq!(
        stepper.state().selected_record().unwrap().cost.unwrap(),
        0.0
    );
    assert_eq!(*final_reason.borrow(), Some(TerminationCode::Cancelled));
}
