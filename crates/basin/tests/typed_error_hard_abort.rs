//! End-to-end coverage for the typed hard-abort path: a problem whose
//! `cost` returns `Err(MyError)` halts `Executor::run` with the same
//! `Err`, never silently coerced. Also pins the niche-optimization
//! invariant for `Result<f64, Infallible>` so a future regression
//! reintroducing a default error type would surface immediately.

use std::cell::Cell;
use std::convert::Infallible;
use std::mem::size_of;

use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent, MaxIter};

/// `Result<f64, Infallible>` must be the same size as `f64`—the entire
/// rationale for choosing `Infallible` as the default error in the
/// migration is that the happy path stays zero-cost.
#[test]
fn result_f64_infallible_has_same_size_as_f64() {
    assert_eq!(size_of::<Result<f64, Infallible>>(), size_of::<f64>());
}

/// User-chosen typed hard-abort error. Carries enough info to round-trip
/// the assertion that the error bubbled up untouched.
#[derive(Debug, PartialEq, Eq)]
enum BailReason {
    BatchCancelled,
}

/// Problem that runs as the standard sphere for the first `bail_after`
/// `cost` calls and then returns `Err(BailReason::BatchCancelled)`. The
/// gradient never aborts, so the only way Executor sees `Err` is via
/// `cost`, exercising the most common HFT-style early-stop pattern.
struct BailingSphere {
    bail_after: u32,
    calls: Cell<u32>,
}

impl CostFunction for BailingSphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = BailReason;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, BailReason> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        if n >= self.bail_after {
            return Err(BailReason::BatchCancelled);
        }
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}

impl Gradient for BailingSphere {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, BailReason> {
        Ok(x.iter().map(|xi| 2.0 * xi).collect())
    }
}

#[test]
fn cost_err_bubbles_out_of_executor_run_with_same_error() {
    let problem = BailingSphere {
        // Allow a few normal evaluations (init seeds, plus a couple of
        // iterations) before the abort fires, confirming the executor
        // *was* running, not failing before the loop began.
        bail_after: 5,
        calls: Cell::new(0),
    };

    let result = Executor::new(
        problem,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.5, -2.0]),
    )
    .terminate_on(MaxIter(100))
    .run();

    match result {
        Ok(_) => panic!("expected Err(BailReason::BatchCancelled), got Ok"),
        Err(e) => assert_eq!(e, BailReason::BatchCancelled),
    }
}

#[test]
fn cost_err_at_init_bubbles_out_of_executor_run() {
    // bail_after = 0: the *first* cost evaluation (during `Solver::init`)
    // already errors. Confirms init's error path is wired through.
    let problem = BailingSphere {
        bail_after: 0,
        calls: Cell::new(0),
    };

    let result = Executor::new(
        problem,
        GradientDescent::new(0.1),
        BasicState::new(vec![0.0]),
    )
    .terminate_on(MaxIter(10))
    .run();

    match result {
        Ok(_) => panic!("expected Err(BailReason::BatchCancelled), got Ok"),
        Err(e) => assert_eq!(e, BailReason::BatchCancelled),
    }
}

/// Soft reject (`Ok(f64::INFINITY)`) must NOT abort the run—line search
/// retreats off the infeasible region just like before the typed-error
/// migration. This pins the soft/hard split.
#[test]
fn soft_reject_via_infinity_does_not_abort() {
    struct InfiniteBeyondUnit;

    impl CostFunction for InfiniteBeyondUnit {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            // f(x) = ‖x‖² on the unit ball, +∞ outside. Optimum at 0 is
            // strictly interior, so the soft reject only fires for
            // line-search probes that overshoot.
            let s: f64 = x.iter().map(|xi| xi * xi).sum();
            if s > 1.0 { Ok(f64::INFINITY) } else { Ok(s) }
        }
    }
    impl Gradient for InfiniteBeyondUnit {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            Ok(x.iter().map(|xi| 2.0 * xi).collect())
        }
    }

    let result = Executor::new(
        InfiniteBeyondUnit,
        GradientDescent::with_line_search(basin::Backtracking::new()),
        BasicState::new(vec![0.5, 0.5]),
    )
    .terminate_on(MaxIter(200))
    .run()
    .expect("soft-reject must never produce Err");

    // Should have walked toward the interior optimum.
    assert!(result.cost() < 0.5, "cost = {}", result.cost());
}
