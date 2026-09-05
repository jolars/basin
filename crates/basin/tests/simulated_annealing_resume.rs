#![cfg(feature = "serde")]

use basin::core::rng::ChaCha8Rng;
use basin::{
    CostFunction, Executor, Neighbor, NoAcceptance, NoImprovement,
    SimulatedAnnealing, SimulatedAnnealingState, TemperatureSchedule,
    TerminationReason,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

struct RuggedCost;

impl CostFunction for RuggedCost {
    type Param = i32;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &i32) -> Result<f64, Infallible> {
        let x = f64::from(*x);
        Ok((x - 3.0).powi(2) + (x * 2.3).sin())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StatefulNeighbor {
    calls: u64,
}

impl Neighbor<i32, f64, ChaCha8Rng> for StatefulNeighbor {
    type Error = Infallible;

    fn propose(
        &mut self,
        current: &i32,
        _temperature: f64,
        rng: &mut ChaCha8Rng,
    ) -> Result<i32, Infallible> {
        use basin::core::rng::Rng;

        self.calls += 1;
        if rng.next_u64() & 1 == 0 {
            Ok(current - 1)
        } else {
            Ok(current + 1)
        }
    }
}

fn solver() -> SimulatedAnnealing<StatefulNeighbor> {
    SimulatedAnnealing::new(
        StatefulNeighbor { calls: 0 },
        4.0,
        TemperatureSchedule::geometric(0.97).with_steps_per_temperature(3),
        0x5eed,
    )
    .with_reannealing_fixed(17)
    .with_reannealing_accepted(11)
    .with_reannealing_best(13)
}

#[test]
fn serialized_state_resumes_bit_for_bit() {
    let reference = Executor::from_start(RuggedCost, solver(), 12)
        .max_iter(80)
        .run()
        .unwrap();

    let checkpoint = Executor::from_start(RuggedCost, solver(), 12)
        .max_iter(31)
        .run()
        .unwrap()
        .into_state();
    let bytes =
        bincode::serde::encode_to_vec(&checkpoint, bincode::config::standard())
            .unwrap();
    let (restored, _): (SimulatedAnnealingState<i32, StatefulNeighbor>, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .unwrap();

    let solver_bytes =
        bincode::serde::encode_to_vec(solver(), bincode::config::standard())
            .unwrap();
    let (restored_solver, _): (SimulatedAnnealing<StatefulNeighbor>, usize) =
        bincode::serde::decode_from_slice(
            &solver_bytes,
            bincode::config::standard(),
        )
        .unwrap();

    let resumed = Executor::resume(RuggedCost, restored_solver, restored)
        .max_iter(80)
        .run()
        .unwrap();

    let reference_bytes = bincode::serde::encode_to_vec(
        &reference.state,
        bincode::config::standard(),
    )
    .unwrap();
    let resumed_bytes = bincode::serde::encode_to_vec(
        &resumed.state,
        bincode::config::standard(),
    )
    .unwrap();
    assert_eq!(resumed_bytes, reference_bytes);
    assert_eq!(resumed.cost_evals(), 81);
}

#[test]
fn zero_tolerance_best_stall_retains_absolute_history_on_resume() {
    let always_reject = || {
        SimulatedAnnealing::new(
            |x: &i32, _: f64, _: &mut ChaCha8Rng| x + 100,
            1e-6,
            TemperatureSchedule::geometric(0.9),
            9,
        )
    };
    let checkpoint = Executor::from_start(RuggedCost, always_reject(), 3)
        .max_iter(4)
        .run()
        .unwrap()
        .into_state();

    let result = Executor::resume(RuggedCost, always_reject(), checkpoint)
        .terminate_on(NoImprovement::new(7, 0.0))
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::NoImprovement);
    assert_eq!(result.iter(), 7);
}

#[test]
fn acceptance_stall_criterion_retains_absolute_history_on_resume() {
    let always_reject = || {
        SimulatedAnnealing::new(
            |x: &i32, _: f64, _: &mut ChaCha8Rng| x + 100,
            1e-6,
            TemperatureSchedule::geometric(0.9),
            9,
        )
    };
    let checkpoint = Executor::from_start(RuggedCost, always_reject(), 3)
        .max_iter(4)
        .run()
        .unwrap()
        .into_state();

    let result = Executor::resume(RuggedCost, always_reject(), checkpoint)
        .terminate_on(NoAcceptance::new(7))
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::NoAcceptedMove);
    assert_eq!(result.iter(), 7);
    assert_eq!(result.best_iter(), 0);
}
