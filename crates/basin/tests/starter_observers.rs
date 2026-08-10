//! Integration tests for the zero-dep starter observers [`Report`] and
//! [`History`].

use std::cell::RefCell;
use std::rc::Rc;

use basin::{
    BasicState, CostFunction, Executor, Gradient, GradientDescent, History,
    Observe, ObserverMode, Report, State,
};

/// f(x) = ½ ‖x‖²: strictly convex, so GD decreases the cost every step.
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

    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok(x.clone())
    }
}

/// Shared handle so the recorded trajectory outlives the executor, which takes
/// ownership of the registered observer.
#[derive(Clone, Default)]
struct SharedHistory(Rc<RefCell<History>>);

impl<S: State<Float = f64>> Observe<S> for SharedHistory {
    fn observe_init(&mut self, state: &S) {
        self.0.borrow_mut().observe_init(state);
    }
    fn observe_iter(&mut self, state: &S) {
        self.0.borrow_mut().observe_iter(state);
    }
}

#[test]
fn history_records_full_trajectory_with_monotone_best() {
    let history = SharedHistory::default();

    Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, -2.0, 3.0]),
    )
    .max_iter(6)
    .observe_with(history.clone(), ObserverMode::Always)
    .run()
    .unwrap();

    let history = history.0.borrow();
    let records = history.records();

    // init snapshot at iter 0, then one per iteration.
    assert_eq!(records.len(), 1 + 6);
    assert_eq!(records[0].0, 0);
    for (k, (iter, _, _)) in records.iter().skip(1).enumerate() {
        assert_eq!(*iter, k as u64 + 1);
    }

    // best_cost is non-increasing along the run.
    for pair in records.windows(2) {
        assert!(pair[1].2 <= pair[0].2, "best_cost must not increase");
    }
    // On this strictly convex problem, cost and best coincide every step.
    for (_, cost, best) in records {
        assert_eq!(cost, best);
    }
}

#[test]
fn history_every_n_thins_iters() {
    let history = SharedHistory::default();

    Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(9)
    .observe_with(history.clone(), ObserverMode::Every(3))
    .run()
    .unwrap();

    let history = history.0.borrow();
    let iters: Vec<u64> =
        history.records().iter().map(|(i, _, _)| *i).collect();
    // init (0) always fires; iter fires on multiples of 3.
    assert_eq!(iters, vec![0, 3, 6, 9]);
}

#[test]
fn report_smoke_runs_without_panicking() {
    // `Report` prints to stderr; this just exercises all three hooks. Run with
    // `-- --nocapture` to see the lines.
    Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(4)
    .observe_with(Report::with_prefix("gd"), ObserverMode::Every(2))
    .run()
    .unwrap();
}
