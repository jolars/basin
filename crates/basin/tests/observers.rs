//! Integration tests for the [`Observe`] / [`ObserverMode`] wiring on
//! [`Executor`] and [`Stepper`].

use std::cell::RefCell;
use std::rc::Rc;

use basin::{
    BasicState, CostFunction, Executor, Gradient, GradientDescent, MaxIter, Observe, ObserverMode,
    State, StepOutcome, TerminationReason,
};

/// f(x) = ½ ‖x‖² — convex quadratic, gradient = x.
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

/// Test observer that records `(kind, iter)` for every fire into a shared
/// log, and additionally remembers the reason from `observe_final`.
#[derive(Default)]
struct Recorder {
    log: Rc<RefCell<Vec<(&'static str, u64)>>>,
    final_reason: Rc<RefCell<Option<TerminationReason>>>,
}

impl Recorder {
    fn with_log(log: Rc<RefCell<Vec<(&'static str, u64)>>>) -> Self {
        Self {
            log,
            final_reason: Rc::new(RefCell::new(None)),
        }
    }
}

impl<S: State> Observe<S> for Recorder {
    fn observe_init(&mut self, state: &S) {
        self.log.borrow_mut().push(("init", state.iter()));
    }
    fn observe_iter(&mut self, state: &S) {
        self.log.borrow_mut().push(("iter", state.iter()));
    }
    fn observe_final(&mut self, state: &S, reason: &TerminationReason) {
        self.log.borrow_mut().push(("final", state.iter()));
        *self.final_reason.borrow_mut() = Some(*reason);
    }
}

/// A tag-emitting observer for the ordering test — records *which*
/// observer fired (not what hook).
struct Tagger {
    tag: &'static str,
    log: Rc<RefCell<Vec<&'static str>>>,
}

impl<S: State> Observe<S> for Tagger {
    fn observe_init(&mut self, _state: &S) {
        self.log.borrow_mut().push(self.tag);
    }
    fn observe_iter(&mut self, _state: &S) {
        self.log.borrow_mut().push(self.tag);
    }
    fn observe_final(&mut self, _state: &S, _reason: &TerminationReason) {
        self.log.borrow_mut().push(self.tag);
    }
}

#[test]
fn init_then_iter_per_step_then_final_with_reason() {
    // Drive a fixed-budget run: MaxIter(5) on a problem that won't converge
    // earlier with this step size, so we expect exactly 5 iterations.
    let log = Rc::new(RefCell::new(Vec::new()));
    let reason_holder = Rc::new(RefCell::new(None));
    let recorder = Recorder {
        log: Rc::clone(&log),
        final_reason: Rc::clone(&reason_holder),
    };

    let result = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, -2.0, 3.0]),
    )
    .max_iter(5)
    .observe_with(recorder, ObserverMode::Always)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 5);

    let log = log.borrow();
    // init once at iter 0, iter once per iteration (1..=5), final once at iter 5.
    assert_eq!(log.len(), 1 + 5 + 1);
    assert_eq!(log[0], ("init", 0));
    for (k, entry) in log.iter().skip(1).take(5).enumerate() {
        assert_eq!(*entry, ("iter", k as u64 + 1));
    }
    assert_eq!(log[6], ("final", 5));

    assert_eq!(*reason_holder.borrow(), Some(TerminationReason::MaxIter));
}

#[test]
fn every_n_gates_iter_only() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let recorder = Recorder::with_log(Rc::clone(&log));

    let _ = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(10)
    .observe_with(recorder, ObserverMode::Every(3))
    .run()
    .unwrap();

    let log = log.borrow();
    let iter_hits: Vec<u64> = log
        .iter()
        .filter_map(|(k, i)| (*k == "iter").then_some(*i))
        .collect();
    // Iter counts that are multiples of 3 in 1..=10 are 3, 6, 9.
    assert_eq!(iter_hits, vec![3, 6, 9]);
    // init / final still fire regardless of mode.
    assert!(log.iter().any(|(k, _)| *k == "init"));
    assert!(log.iter().any(|(k, _)| *k == "final"));
}

#[test]
fn never_skips_iter_but_init_final_fire() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let recorder = Recorder::with_log(Rc::clone(&log));

    let _ = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(4)
    .observe_with(recorder, ObserverMode::Never)
    .run()
    .unwrap();

    let log = log.borrow();
    let kinds: Vec<&'static str> = log.iter().map(|(k, _)| *k).collect();
    // No "iter" entries; one "init" and one "final".
    assert!(kinds.iter().all(|k| *k != "iter"));
    assert_eq!(kinds.iter().filter(|k| ***k == *"init").count(), 1);
    assert_eq!(kinds.iter().filter(|k| ***k == *"final").count(), 1);
}

#[test]
fn multiple_observers_fire_in_registration_order() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let a = Tagger {
        tag: "a",
        log: Rc::clone(&log),
    };
    let b = Tagger {
        tag: "b",
        log: Rc::clone(&log),
    };

    let _ = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0]),
    )
    .max_iter(2)
    .observe_with(a, ObserverMode::Always)
    .observe_with(b, ObserverMode::Always)
    .run()
    .unwrap();

    // Per hook firing, "a" precedes "b". Layout: init(a,b), iter1(a,b),
    // iter2(a,b), final(a,b) = 8 entries alternating.
    let log = log.borrow();
    assert_eq!(log.len(), 8);
    for pair in log.chunks_exact(2) {
        assert_eq!(pair, &["a", "b"]);
    }
}

#[test]
fn observer_fires_via_stepper_step_loop() {
    // Manual one-step-at-a-time driving exercises the Stepper::step path
    // (run_to_end goes through this same path, but covering it directly
    // catches regressions in the sticky-stop branch).
    let log = Rc::new(RefCell::new(Vec::new()));
    let recorder = Recorder::with_log(Rc::clone(&log));

    let mut stepper = Executor::new(
        Quadratic,
        GradientDescent::new(0.1),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(3)
    .terminate_on(MaxIter(3))
    .observe_with(recorder, ObserverMode::Always)
    .into_stepper()
    .unwrap();

    // init has already fired inside into_stepper.
    {
        let log = log.borrow();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0], ("init", 0));
    }

    // Drive to completion via repeated step() calls, then a couple more
    // calls after Stopped to confirm observe_final fires *once*.
    while let StepOutcome::Continue = stepper.step().unwrap() {}
    let _ = stepper.step().unwrap();
    let _ = stepper.step().unwrap();

    let log = log.borrow();
    // init + 3 iters + 1 final = 5
    assert_eq!(log.len(), 5);
    assert_eq!(log[4].0, "final");
}
