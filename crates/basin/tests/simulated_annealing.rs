use basin::core::rng::ChaCha8Rng;
use basin::{
    CostFunction, Executor, Neighbor, NoAcceptance, Reannealing,
    SimulatedAnnealing, StepOutcome, TemperatureSchedule, TerminationReason,
};
use rand::TryRng;
use std::convert::Infallible;

struct IdentityCost;

impl CostFunction for IdentityCost {
    type Param = i32;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &i32) -> Result<f64, Infallible> {
        Ok(f64::from(*x))
    }
}

#[derive(Debug, PartialEq)]
enum ApplicationError {
    Proposal,
}

struct FallibleCost;

impl CostFunction for FallibleCost {
    type Param = i32;
    type Output = f64;
    type Error = ApplicationError;

    fn cost(&self, x: &i32) -> Result<f64, ApplicationError> {
        Ok(f64::from(*x))
    }
}

#[derive(Clone, Debug)]
struct FailingNeighbor;

impl Neighbor<i32> for FailingNeighbor {
    type Error = ApplicationError;

    fn propose(
        &mut self,
        _current: &i32,
        _temperature: f64,
        _rng: &mut ChaCha8Rng,
    ) -> Result<i32, ApplicationError> {
        Err(ApplicationError::Proposal)
    }
}

#[derive(Clone, Debug)]
struct ConstantRng(u64);

impl TryRng for ConstantRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.0 as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.0)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.0.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}

#[test]
fn schedules_start_at_t0_and_hold_each_temperature_level() {
    let geometric =
        TemperatureSchedule::geometric(0.5).with_steps_per_temperature(2);
    assert_eq!(geometric.temperature(8.0, 0), 8.0);
    assert_eq!(geometric.temperature(8.0, 1), 8.0);
    assert_eq!(geometric.temperature(8.0, 2), 4.0);

    let reciprocal = TemperatureSchedule::reciprocal();
    assert_eq!(reciprocal.temperature(8.0, 0), 8.0);
    assert_eq!(reciprocal.temperature(8.0, 1), 4.0);

    let logarithmic = TemperatureSchedule::logarithmic();
    assert_eq!(logarithmic.temperature(8.0, 0), 8.0);
    assert!(logarithmic.temperature(8.0, 1) < 8.0);
}

#[test]
fn closure_neighbor_uses_infallible_error() {
    let mut neighbor = |x: &i32, _: f64, _: &mut ConstantRng| x - 1;
    let proposal = neighbor.propose(&1, 1.0, &mut ConstantRng(0));

    assert_eq!(proposal, Ok(0));
}

#[test]
fn classical_metropolis_accepts_downhill_and_probabilistic_uphill_moves() {
    let downhill = SimulatedAnnealing::new_with_rng(
        |_: &i32, _: f64, _: &mut ConstantRng| 0,
        1.0,
        TemperatureSchedule::geometric(0.9),
        ConstantRng(u64::MAX),
    );
    let downhill = Executor::from_start(IdentityCost, downhill, 1)
        .max_iter(1)
        .run()
        .unwrap();
    assert_eq!(*downhill.param(), 0);
    assert_eq!(downhill.state.accepted_moves(), 1);

    let accept_uphill = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| x + 1,
        1.0,
        TemperatureSchedule::geometric(0.9),
        ConstantRng(0),
    );
    let accepted = Executor::from_start(IdentityCost, accept_uphill, 0)
        .max_iter(1)
        .run()
        .unwrap();
    assert_eq!(*accepted.param(), 1);
    assert_eq!(*accepted.best_param(), 0);
    assert_eq!(accepted.best_cost(), 0.0);

    let reject_uphill = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| x + 1,
        1.0,
        TemperatureSchedule::geometric(0.9),
        ConstantRng(u64::MAX),
    );
    let rejected = Executor::from_start(IdentityCost, reject_uphill, 0)
        .max_iter(1)
        .run()
        .unwrap();
    assert_eq!(*rejected.param(), 0);
    assert_eq!(rejected.state.rejected_moves(), 1);
}

#[test]
fn fallible_neighbor_propagates_the_application_error() {
    let solver = SimulatedAnnealing::new(
        FailingNeighbor,
        1.0,
        TemperatureSchedule::geometric(0.9),
        42,
    );

    let result = Executor::from_start(FallibleCost, solver, 0)
        .max_iter(1)
        .run();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("the failed proposal unexpectedly completed"),
    };

    assert_eq!(error, ApplicationError::Proposal);
}

#[test]
fn seeded_runs_are_bit_reproducible() {
    let run = || {
        let solver = SimulatedAnnealing::new(
            |x: &i32, _: f64, rng: &mut ChaCha8Rng| {
                use basin::core::rng::Rng;

                if rng.next_u64() & 1 == 0 {
                    x - 1
                } else {
                    x + 1
                }
            },
            3.0,
            TemperatureSchedule::geometric(0.98),
            123,
        );
        Executor::from_start(IdentityCost, solver, 10)
            .max_iter(100)
            .run()
            .unwrap()
    };

    let first = run();
    let second = run();
    assert_eq!(first.param(), second.param());
    assert_eq!(first.best_param(), second.best_param());
    assert_eq!(first.state.accepted_moves(), second.state.accepted_moves());
}

#[test]
fn fixed_interval_reannealing_restarts_the_schedule_for_the_next_move() {
    let solver = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| x - 1,
        8.0,
        TemperatureSchedule::geometric(0.5),
        ConstantRng(u64::MAX),
    )
    .with_reannealing(Reannealing::fixed_interval(2));

    let mut stepper = Executor::from_start(IdentityCost, solver, 0)
        .max_iter(3)
        .into_stepper()
        .unwrap();
    assert_eq!(stepper.state().temperature(), 8.0);
    assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    assert_eq!(stepper.state().temperature(), 4.0);
    assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    assert_eq!(stepper.state().temperature(), 8.0);
    assert_eq!(stepper.state().reannealings(), 1);
}

#[test]
fn rejection_and_no_best_reannealing_use_distinct_progress() {
    let rejected = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| x + 1,
        8.0,
        TemperatureSchedule::geometric(0.5),
        ConstantRng(u64::MAX),
    )
    .with_reannealing(Reannealing::after_rejections(2));
    let rejected = Executor::from_start(IdentityCost, rejected, 0)
        .max_iter(2)
        .run()
        .unwrap();
    assert_eq!(rejected.state.reannealings(), 1);
    assert_eq!(rejected.state.temperature(), 8.0);

    let no_best = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| *x,
        8.0,
        TemperatureSchedule::geometric(0.5),
        ConstantRng(u64::MAX),
    )
    .with_reannealing(Reannealing::after_no_best(2));
    let no_best = Executor::from_start(IdentityCost, no_best, 0)
        .max_iter(2)
        .run()
        .unwrap();
    assert_eq!(no_best.state.accepted_moves(), 2);
    assert_eq!(no_best.state.reannealings(), 1);
}

#[test]
fn no_acceptance_is_a_shared_resume_safe_criterion() {
    let solver = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| x + 1,
        1.0,
        TemperatureSchedule::geometric(0.9),
        ConstantRng(u64::MAX),
    );
    let result = Executor::from_start(IdentityCost, solver, 0)
        .terminate_on(NoAcceptance::new(3))
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::NoAcceptedMove);
    assert_eq!(result.iter(), 3);
}

#[derive(Clone, Debug)]
struct SequenceNeighbor {
    values: Vec<i32>,
    index: usize,
}

impl<R> Neighbor<i32, f64, R> for SequenceNeighbor {
    type Error = Infallible;

    fn propose(
        &mut self,
        _current: &i32,
        _temperature: f64,
        _rng: &mut R,
    ) -> Result<i32, Infallible> {
        let value = self.values[self.index];
        self.index += 1;
        Ok(value)
    }
}

#[test]
fn composable_reannealing_uses_any_trigger_and_resets_all_progress() {
    let solver = SimulatedAnnealing::new_with_rng(
        SequenceNeighbor {
            values: vec![1, 1, 0, 0, 0, -1, -2, -3, -4, -5],
            index: 0,
        },
        8.0,
        TemperatureSchedule::geometric(0.5),
        ConstantRng(u64::MAX),
    )
    .with_reannealing_fixed(5)
    .with_reannealing_accepted(2)
    .with_reannealing_best(3);

    let mut stepper = Executor::from_start(IdentityCost, solver, 0)
        .max_iter(10)
        .into_stepper()
        .unwrap();
    let expected_reannealings = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3];

    for expected in expected_reannealings {
        assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
        assert_eq!(stepper.state().reannealings(), expected);
    }
    assert_eq!(stepper.state().temperature(), 8.0);
}

#[test]
fn coincident_reannealing_triggers_restart_the_schedule_once() {
    let solver = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| x + 1,
        8.0,
        TemperatureSchedule::geometric(0.5),
        ConstantRng(u64::MAX),
    )
    .with_reannealing_fixed(2)
    .with_reannealing_accepted(2)
    .with_reannealing_best(2);
    let result = Executor::from_start(IdentityCost, solver, 0)
        .max_iter(2)
        .run()
        .unwrap();

    assert_eq!(result.state.reannealings(), 1);
    assert_eq!(result.state.temperature(), 8.0);
}

#[test]
fn repeated_reannealing_builder_replaces_only_its_threshold() {
    let solver = SimulatedAnnealing::new_with_rng(
        |x: &i32, _: f64, _: &mut ConstantRng| x - 1,
        8.0,
        TemperatureSchedule::geometric(0.5),
        ConstantRng(u64::MAX),
    )
    .with_reannealing_fixed(1)
    .with_reannealing_accepted(10)
    .with_reannealing_fixed(2);
    let result = Executor::from_start(IdentityCost, solver, 0)
        .max_iter(2)
        .run()
        .unwrap();

    assert_eq!(result.state.reannealings(), 1);
}

struct NonFiniteCost;

impl CostFunction for NonFiniteCost {
    type Param = i32;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &i32) -> Result<f64, Infallible> {
        Ok(match x {
            1 => f64::NAN,
            2 => f64::INFINITY,
            3 => f64::NEG_INFINITY,
            _ => 0.0,
        })
    }
}

#[test]
fn non_finite_costs_follow_the_documented_policy() {
    let solver: SimulatedAnnealing<_, f64, ChaCha8Rng> =
        SimulatedAnnealing::new(
            SequenceNeighbor {
                values: vec![1, 2, 3],
                index: 0,
            },
            1.0,
            TemperatureSchedule::geometric(0.9),
            42,
        );
    let result = Executor::from_start(NonFiniteCost, solver, 0)
        .max_iter(10)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.iter(), 3);
    assert_eq!(result.cost(), f64::NEG_INFINITY);
    assert_eq!(result.cost_evals(), 4);
    assert_eq!(result.state.accepted_moves(), 1);
    assert_eq!(result.state.rejected_moves(), 2);

    let nan_start = SimulatedAnnealing::new(
        |x: &i32, _: f64, _: &mut ChaCha8Rng| *x,
        1.0,
        TemperatureSchedule::geometric(0.9),
        42,
    );
    let nan_start = Executor::from_start(NonFiniteCost, nan_start, 1)
        .run()
        .unwrap();
    assert_eq!(nan_start.reason, TerminationReason::SolverFailed);
    assert_eq!(nan_start.iter(), 0);

    let negative_infinity_start = SimulatedAnnealing::new(
        |x: &i32, _: f64, _: &mut ChaCha8Rng| *x,
        1.0,
        TemperatureSchedule::geometric(0.9),
        42,
    );
    let negative_infinity_start =
        Executor::from_start(NonFiniteCost, negative_infinity_start, 3)
            .run()
            .unwrap();
    assert_eq!(
        negative_infinity_start.reason,
        TerminationReason::SolverConverged
    );
    assert_eq!(negative_infinity_start.iter(), 0);

    let finite_from_infinity = SimulatedAnnealing::new(
        |_: &i32, _: f64, _: &mut ChaCha8Rng| 0,
        1.0,
        TemperatureSchedule::geometric(0.9),
        42,
    );
    let finite_from_infinity =
        Executor::from_start(NonFiniteCost, finite_from_infinity, 2)
            .max_iter(1)
            .run()
            .unwrap();
    assert_eq!(*finite_from_infinity.param(), 0);
}

#[test]
fn f32_scalar_and_non_numeric_parameter_round_trip() {
    #[derive(Clone, Debug, PartialEq)]
    enum Label {
        Left,
        Right,
    }

    struct LabelCost;
    impl CostFunction for LabelCost {
        type Param = Label;
        type Output = f32;
        type Error = Infallible;

        fn cost(&self, x: &Label) -> Result<f32, Infallible> {
            Ok(match x {
                Label::Left => 1.0,
                Label::Right => 0.0,
            })
        }
    }

    let solver = SimulatedAnnealing::new(
        |_: &Label, _: f32, _: &mut ChaCha8Rng| Label::Right,
        1.0_f32,
        TemperatureSchedule::geometric(0.9_f32),
        7,
    );
    let result = Executor::from_start(LabelCost, solver, Label::Left)
        .max_iter(1)
        .run()
        .unwrap();
    assert_eq!(result.param(), &Label::Right);
    assert_eq!(result.cost(), 0.0_f32);
}

#[test]
fn continuous_neighbor_reaches_the_analytic_sphere_minimum() {
    struct Sphere;
    impl CostFunction for Sphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(x.iter().map(|value| value * value).sum())
        }
    }

    let solver = SimulatedAnnealing::new(
        |x: &Vec<f64>, _: f64, _: &mut ChaCha8Rng| {
            x.iter().map(|value| value * 0.5).collect()
        },
        1.0,
        TemperatureSchedule::reciprocal(),
        12,
    );
    let result = Executor::from_start(Sphere, solver, vec![8.0, -4.0])
        .max_iter(30)
        .run()
        .unwrap();

    assert!(result.best_cost() < 1e-15);
}

#[test]
fn discrete_permutation_neighbor_improves_a_tour() {
    use basin::core::rng::RngExt;

    struct Tour;
    impl CostFunction for Tour {
        type Param = Vec<usize>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, tour: &Vec<usize>) -> Result<f64, Infallible> {
            const POINTS: [(f64, f64); 5] =
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
            Ok((0..tour.len())
                .map(|i| {
                    let a = POINTS[tour[i]];
                    let b = POINTS[tour[(i + 1) % tour.len()]];
                    (a.0 - b.0).hypot(a.1 - b.1)
                })
                .sum())
        }
    }

    let neighbor = |tour: &Vec<usize>, _: f64, rng: &mut ChaCha8Rng| {
        let mut candidate = tour.clone();
        let i = rng.random_range(0..candidate.len());
        let mut j = rng.random_range(0..candidate.len() - 1);
        if j >= i {
            j += 1;
        }
        candidate.swap(i, j);
        candidate
    };
    let start = vec![0, 2, 1, 3, 4];
    let initial_cost = Tour.cost(&start).unwrap();
    let solver = SimulatedAnnealing::new(
        neighbor,
        1.0,
        TemperatureSchedule::geometric(0.995).with_steps_per_temperature(4),
        88,
    );
    let result = Executor::from_start(Tour, solver, start)
        .max_iter(1_000)
        .run()
        .unwrap();

    assert!(result.best_cost() < initial_cost);
    assert_eq!(result.best_param().len(), 5);
}

#[test]
fn cooling_never_reaches_zero_after_underflow() {
    let schedule = TemperatureSchedule::geometric(0.1_f64);
    assert!(schedule.temperature(1.0, u64::MAX) > 0.0);
}

#[test]
#[should_panic(expected = "finite 0 < alpha < 1")]
fn geometric_schedule_rejects_invalid_alpha() {
    let _ = TemperatureSchedule::geometric(1.0_f64);
}

#[test]
#[should_panic(expected = "steps_per_temperature > 0")]
fn schedule_rejects_empty_temperature_level() {
    let _ =
        TemperatureSchedule::<f64>::reciprocal().with_steps_per_temperature(0);
}

#[test]
#[should_panic(expected = "finite initial temperature > 0")]
fn solver_rejects_invalid_initial_temperature() {
    let _ = SimulatedAnnealing::new(
        |x: &i32, _: f64, _: &mut ChaCha8Rng| *x,
        f64::NAN,
        TemperatureSchedule::reciprocal(),
        1,
    );
}

#[test]
#[should_panic(expected = "reannealing threshold must be > 0")]
fn reannealing_rejects_zero_threshold() {
    let _ = Reannealing::after_rejections(0);
}

#[test]
#[should_panic(expected = "reannealing threshold must be > 0")]
fn fixed_reannealing_builder_rejects_zero_threshold() {
    let _ = SimulatedAnnealing::new(
        |x: &i32, _: f64, _: &mut ChaCha8Rng| *x,
        1.0,
        TemperatureSchedule::reciprocal(),
        1,
    )
    .with_reannealing_fixed(0);
}

#[test]
#[should_panic(expected = "reannealing threshold must be > 0")]
fn accepted_reannealing_builder_rejects_zero_threshold() {
    let _ = SimulatedAnnealing::new(
        |x: &i32, _: f64, _: &mut ChaCha8Rng| *x,
        1.0,
        TemperatureSchedule::reciprocal(),
        1,
    )
    .with_reannealing_accepted(0);
}

#[test]
#[should_panic(expected = "reannealing threshold must be > 0")]
fn best_reannealing_builder_rejects_zero_threshold() {
    let _ = SimulatedAnnealing::new(
        |x: &i32, _: f64, _: &mut ChaCha8Rng| *x,
        1.0,
        TemperatureSchedule::reciprocal(),
        1,
    )
    .with_reannealing_best(0);
}
