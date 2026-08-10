//! Integration tests for the public `Stepper` API.

use basin::problems::Rosenbrock;
use basin::{
    BasicState, CostFunction, Executor, GradientDescent, GradientTolerance,
    State, StepOutcome, TerminationReason,
};

#[test]
fn stepper_run_to_end_matches_executor_run() {
    let problem_a = Rosenbrock::<Vec<f64>>::default();
    let problem_b = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];

    let direct = Executor::new(
        problem_a,
        GradientDescent::new(0.001),
        BasicState::new(initial.clone()),
    )
    .max_iter(500)
    .run()
    .unwrap();

    let via_stepper = Executor::new(
        problem_b,
        GradientDescent::new(0.001),
        BasicState::new(initial),
    )
    .max_iter(500)
    .into_stepper()
    .unwrap()
    .run_to_end()
    .unwrap();

    assert_eq!(direct.iter(), via_stepper.iter());
    assert_eq!(direct.reason, via_stepper.reason);
    assert!(
        (direct.cost() - via_stepper.cost()).abs() < 1e-12,
        "direct cost {} != stepper cost {}",
        direct.cost(),
        via_stepper.cost()
    );
}

#[test]
fn stepper_advances_one_iter_per_step() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let mut stepper = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(vec![-1.2, 1.0]),
    )
    .max_iter(10)
    .into_stepper()
    .unwrap();

    assert_eq!(stepper.iter(), 0, "stepper sits at iter 0 after init");
    for expected in 1..=5 {
        let outcome = stepper.step().unwrap();
        assert_eq!(outcome, StepOutcome::Continue);
        assert_eq!(stepper.iter(), expected);
    }
}

#[test]
fn stepper_stops_on_max_iter_with_correct_reason() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let mut stepper = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(vec![-1.2, 1.0]),
    )
    .max_iter(3)
    .into_stepper()
    .unwrap();

    for _ in 0..3 {
        assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    }
    assert_eq!(
        stepper.step().unwrap(),
        StepOutcome::Stopped(TerminationReason::MaxIter),
    );
    assert_eq!(stepper.iter(), 3);
}

#[test]
fn stepper_is_sticky_after_stop() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let mut stepper = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(vec![-1.2, 1.0]),
    )
    .max_iter(1)
    .into_stepper()
    .unwrap();

    stepper.step().unwrap();
    let first_stop = stepper.step().unwrap();
    let second_stop = stepper.step().unwrap();
    assert_eq!(first_stop, StepOutcome::Stopped(TerminationReason::MaxIter));
    assert_eq!(first_stop, second_stop);
}

#[test]
fn stepper_honors_gradient_tolerance() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let stepper = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(vec![1.0, 1.0]), // already at the optimum
    )
    .max_iter(100)
    .terminate_on(GradientTolerance(1e-6))
    .into_stepper()
    .unwrap();

    let result = stepper.run_to_end().unwrap();
    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert_eq!(result.iter(), 0, "should fire at iter 0");
}

#[test]
fn stepper_state_is_observable_between_steps() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];
    let initial_cost = problem.cost(&initial).unwrap();

    let mut stepper = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(initial),
    )
    .max_iter(50)
    .into_stepper()
    .unwrap();

    let mut trajectory: Vec<Vec<f64>> = vec![stepper.state().param().clone()];
    while stepper.step().unwrap() == StepOutcome::Continue {
        trajectory.push(stepper.state().param().clone());
    }

    assert_eq!(trajectory.len(), 51, "iter 0 + 50 advances");
    let final_cost = stepper.state().cost();
    assert!(
        final_cost < initial_cost,
        "cost should decrease: {} -> {}",
        initial_cost,
        final_cost
    );
}
