//! Worked example: attaching observers to a run.
//!
//! Two observers, registered with different [`ObserverMode`]s, drive a
//! 3-D quadratic with [`GradientDescent`] and read out:
//!
//! - A **trajectory recorder** that captures `(iter, cost, ‖∇f‖)` on
//!   every iteration via `ObserverMode::Always`, kept reachable from the
//!   test body through a shared `Rc<RefCell<_>>`.
//! - A **progress logger** that prints a one-liner every 5 iterations
//!   via `ObserverMode::Every(5)`. `--nocapture` is what makes those
//!   lines visible in your terminal.
//!
//! Run: `cargo test --test example_observer -- --nocapture`.

use std::cell::RefCell;
use std::rc::Rc;

use basin::{
    BasicState, CostFunction, Executor, Gradient, GradientDescent, GradientState,
    GradientTolerance, Observe, ObserverMode, State, TerminationReason,
};

/// f(x) = ½ ‖x‖²: convex quadratic, min at origin, gradient = x. Cheap
/// to optimize, so the example stays focused on the observer mechanics
/// rather than solver tuning.
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

// -----------------------------------------------------------------
// Observer 1: trajectory recorder.
//
// Stores a `Vec` of `(iter, cost, gradient_norm)` records, owned via
// `Rc<RefCell<_>>` so the test body can read it out *after* the run
// hands ownership of the observer to the executor.
//
// The trait bound `S: GradientState` is what couples this observer to
// the gradient family; handing it to a derivative-free solver is a
// compile error, not a runtime no-op. That's tenet 3 in action.
// -----------------------------------------------------------------
struct TrajectoryRecorder {
    records: Rc<RefCell<Vec<(u64, f64, f64)>>>,
}

impl<S> Observe<S> for TrajectoryRecorder
where
    S: GradientState<Float = f64, Param = Vec<f64>>,
{
    fn observe_init(&mut self, state: &S) {
        // Seed the trajectory with the starting point so the recorded
        // arc is complete (init shows iter 0).
        self.records.borrow_mut().push((
            state.iter(),
            state.cost(),
            l2_norm(state.gradient().expect("gradient seeded by Solver::init")),
        ));
    }

    fn observe_iter(&mut self, state: &S) {
        self.records.borrow_mut().push((
            state.iter(),
            state.cost(),
            l2_norm(state.gradient().expect("gradient set by next_iter")),
        ));
    }

    // observe_final left as the default no-op; the trajectory is
    // already complete after the last observe_iter call.
}

// -----------------------------------------------------------------
// Observer 2: progress logger.
//
// Plain `&dyn State` is enough to read iter and cost, so this observer
// binds on the minimum shape and works with any solver. Only `Every(N)`
// gates iter callbacks; `observe_init` and `observe_final` always fire,
// so the user gets a banner at the start and a summary at the end
// regardless of mode.
// -----------------------------------------------------------------
struct ProgressLogger;

impl<S: State<Float = f64>> Observe<S> for ProgressLogger {
    fn observe_init(&mut self, state: &S) {
        println!(
            "  start    iter={:>4}  cost={:>14.6e}",
            state.iter(),
            state.cost()
        );
    }

    fn observe_iter(&mut self, state: &S) {
        println!(
            "  step     iter={:>4}  cost={:>14.6e}",
            state.iter(),
            state.cost()
        );
    }

    fn observe_final(&mut self, state: &S, reason: &TerminationReason) {
        println!(
            "  stopped  iter={:>4}  cost={:>14.6e}  reason={:?}",
            state.iter(),
            state.cost(),
            reason
        );
    }
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn example_observer_on_quadratic() {
    // -----------------------------------------------------------------
    // 1. Shared handle to the trajectory. `Rc<RefCell<_>>` is the
    //    standard pattern for getting data *out* of an observer the
    //    executor has taken ownership of.
    // -----------------------------------------------------------------
    let trajectory = Rc::new(RefCell::new(Vec::<(u64, f64, f64)>::new()));
    let recorder = TrajectoryRecorder {
        records: Rc::clone(&trajectory),
    };

    // -----------------------------------------------------------------
    // 2. Attach both observers via the builder. Each registration
    //    carries its own `ObserverMode`; order is the firing order
    //    inside each hook.
    // -----------------------------------------------------------------
    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.5),
        BasicState::new(vec![3.0, -4.0, 5.0]),
    )
    .max_iter(200)
    .terminate_on(GradientTolerance(1e-8))
    .observe_with(recorder, ObserverMode::Always)
    .observe_with(ProgressLogger, ObserverMode::Every(5))
    .run()
    .unwrap();

    // -----------------------------------------------------------------
    // 3. Gradient descent on ½‖x‖² with α = 0.5 gives x_{k+1} = 0.5·x_k,
    //    so ‖∇f‖ halves each step and the run exits cleanly at the
    //    gradient tolerance well inside the budget.
    // -----------------------------------------------------------------
    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert!(result.cost() < 1e-15);

    // -----------------------------------------------------------------
    // 4. Trajectory: one record at init (iter 0) plus one per completed
    //    iteration. Cost decreases monotonically on this problem under
    //    the chosen step size.
    // -----------------------------------------------------------------
    let traj = trajectory.borrow();
    assert!(!traj.is_empty());
    assert_eq!(traj[0].0, 0); // first record is observe_init at iter 0
    assert_eq!(traj.last().unwrap().0, result.iter());
    for pair in traj.windows(2) {
        assert!(
            pair[1].1 <= pair[0].1 + 1e-12,
            "cost should not increase: {} -> {}",
            pair[0].1,
            pair[1].1
        );
    }

    // Print a tiny summary so `--nocapture` runs are informative.
    println!(
        "\ntrajectory: {} records, final cost {:.3e}, final ‖∇f‖ {:.3e}",
        traj.len(),
        traj.last().unwrap().1,
        traj.last().unwrap().2,
    );
}
