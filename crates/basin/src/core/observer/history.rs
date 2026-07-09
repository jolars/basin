//! An observer that records the cost trajectory of a run.

use crate::core::observer::Observe;
use crate::core::state::State;

/// Record `(iter, cost, best_cost)` at init and on every observed iteration.
///
/// [`OptimizationResult`](crate::core::executor::OptimizationResult) keeps only
/// the *final* iterate and the best-so-far; `History` fills the gap when you
/// want the whole trajectory for plotting or convergence analysis. Read it back
/// via [`records`](Self::records) after the run.
///
/// The scalar type defaults to `f64` (the crate-wide default), but the observer
/// is generic over any [`State::Float`]. Bind cadence with an
/// [`ObserverMode`](super::ObserverMode); `Always` captures every iteration,
/// `Every(n)` thins it.
///
/// Because the [`Executor`](crate::core::executor::Executor) takes ownership of
/// registered observers, wrap `History` in `Rc<RefCell<_>>` (or read it off the
/// stepper) if you need to reach the records after `run()`:
///
/// ```
/// # use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent};
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use basin::{History, Observe, ObserverMode, State};
/// # struct Quadratic;
/// # impl CostFunction for Quadratic {
/// #     type Param = Vec<f64>;
/// #     type Output = f64;
/// #     type Error = std::convert::Infallible;
/// #     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
/// #         Ok(0.5 * x.iter().map(|v| v * v).sum::<f64>())
/// #     }
/// # }
/// # impl Gradient for Quadratic {
/// #     type Gradient = Vec<f64>;
/// #     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> { Ok(x.clone()) }
/// # }
/// // A thin shared-handle wrapper so the records outlive the executor.
/// #[derive(Clone, Default)]
/// struct Shared(Rc<RefCell<History>>);
/// impl<S: State<Float = f64>> Observe<S> for Shared {
///     fn observe_init(&mut self, s: &S) { self.0.borrow_mut().observe_init(s); }
///     fn observe_iter(&mut self, s: &S) { self.0.borrow_mut().observe_iter(s); }
/// }
///
/// let history = Shared::default();
/// Executor::new(Quadratic, GradientDescent::new(0.1), BasicState::new(vec![1.0, 1.0]))
///     .max_iter(5)
///     .observe_with(history.clone(), ObserverMode::Always)
///     .run()
///     .unwrap();
/// assert_eq!(history.0.borrow().records().len(), 1 + 5); // init + 5 iters
/// ```
#[derive(Clone, Debug)]
pub struct History<F = f64> {
    records: Vec<(u64, F, F)>,
}

impl<F> Default for History<F> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

impl<F> History<F> {
    /// An empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded `(iter, cost, best_cost)` triples, in fire order. The
    /// first entry is the init snapshot at `iter == 0`.
    pub fn records(&self) -> &[(u64, F, F)] {
        &self.records
    }

    /// Drop the recorded trajectory, returning the owned vector.
    pub fn into_records(self) -> Vec<(u64, F, F)> {
        self.records
    }
}

impl<S> Observe<S> for History<S::Float>
where
    S: State,
    S::Float: Copy,
{
    fn observe_init(&mut self, state: &S) {
        self.records
            .push((state.iter(), state.cost(), state.best_cost()));
    }

    fn observe_iter(&mut self, state: &S) {
        self.records
            .push((state.iter(), state.cost(), state.best_cost()));
    }
}
