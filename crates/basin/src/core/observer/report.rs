//! A minimal progress-logging observer.

use core::fmt::Display;

use crate::core::observer::Observe;
use crate::core::state::State;
use crate::core::termination::TerminationReason;

/// Print a one-line progress report on every fire.
///
/// Each line carries the iteration counter, the current
/// [`cost`](State::cost), and the best cost so far
/// ([`best_cost`](State::best_cost)); the final line adds the
/// [`TerminationReason`]. Output goes to **stderr** (`eprintln!`) so it never
/// interleaves with data a caller might be writing to stdout.
///
/// Binds on the minimum shape [`State`] (with a [`Display`] scalar), so it
/// works with any solver. Pair it with an [`ObserverMode`](super::ObserverMode)
/// to control cadence: [`Every(n)`](super::ObserverMode::Every) for a heartbeat,
/// [`NewBest`](super::ObserverMode::NewBest) to log only on improvement.
///
/// # Example
///
/// ```
/// # use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent};
/// use basin::{Report, ObserverMode};
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
/// let result = Executor::new(Quadratic, GradientDescent::new(0.1), BasicState::new(vec![1.0, 1.0]))
///     .max_iter(10)
///     .observe_with(Report::with_prefix("gd"), ObserverMode::Every(2))
///     .run()
///     .unwrap();
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct Report {
    /// Optional label prepended to each line (e.g. the solver name).
    prefix: &'static str,
}

impl Report {
    /// A report with no prefix.
    pub fn new() -> Self {
        Self { prefix: "" }
    }

    /// A report whose lines are prefixed with `prefix` followed by a space,
    /// handy for telling several concurrent runs apart.
    pub fn with_prefix(prefix: &'static str) -> Self {
        Self { prefix }
    }

    fn line<S>(&self, tag: &str, state: &S)
    where
        S: State,
        S::Float: Display,
    {
        if self.prefix.is_empty() {
            eprintln!(
                "[{tag}] iter {} cost {} best {}",
                state.iter(),
                state.cost(),
                state.best_cost(),
            );
        } else {
            eprintln!(
                "{} [{tag}] iter {} cost {} best {}",
                self.prefix,
                state.iter(),
                state.cost(),
                state.best_cost(),
            );
        }
    }
}

impl<S> Observe<S> for Report
where
    S: State,
    S::Float: Display,
{
    fn observe_init(&mut self, state: &S) {
        self.line("init", state);
    }

    fn observe_iter(&mut self, state: &S) {
        self.line("iter", state);
    }

    fn observe_final(&mut self, state: &S, reason: &TerminationReason) {
        self.line("done", state);
        if self.prefix.is_empty() {
            eprintln!("[done] stopped: {reason:?}");
        } else {
            eprintln!("{} [done] stopped: {reason:?}", self.prefix);
        }
    }
}
