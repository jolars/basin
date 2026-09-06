#![cfg(all(feature = "serde", not(target_arch = "wasm32")))]

use basin::core::rng::ChaCha8Rng;
use basin::{
    CostFunction, ExactCheckpoint, ExactCheckpointWriter, Executor, Neighbor,
    NoAcceptance, NoImprovement, ObserverMode, SimulatedAnnealing,
    SimulatedAnnealingState, State, TemperatureSchedule, TerminationReason,
    read_exact_checkpoint,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::PathBuf;

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

type TestSolver = SimulatedAnnealing<StatefulNeighbor>;
type TestState = SimulatedAnnealingState<i32, StatefulNeighbor>;

fn solver() -> TestSolver {
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

fn checkpoint_path(name: &str) -> PathBuf {
    std::env::temp_dir()
        .join(format!("basin-exact-sa-{}-{name}.ckpt", std::process::id()))
}

fn remove_checkpoint(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let _ = std::fs::remove_file(PathBuf::from(temporary));
}

#[test]
fn serialized_solver_and_state_resume_bit_for_bit() {
    let reference_path = checkpoint_path("reference");
    let split_path = checkpoint_path("split");
    let resumed_path = checkpoint_path("resumed");
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }

    let reference_writer = ExactCheckpointWriter::new(&reference_path);
    let reference_status = reference_writer.status();
    let reference = Executor::from_start(RuggedCost, solver(), 12)
        .max_iter(80)
        .checkpoint_with(reference_writer, ObserverMode::Never)
        .run()
        .unwrap();
    assert_eq!(reference_status.last_successful_iter(), Some(80));

    Executor::from_start(RuggedCost, solver(), 12)
        .max_iter(31)
        .checkpoint_with(
            ExactCheckpointWriter::new(&split_path),
            ObserverMode::every(7),
        )
        .run()
        .unwrap();
    let checkpoint: ExactCheckpoint<TestSolver, TestState> =
        read_exact_checkpoint(&split_path).unwrap();
    assert_eq!(checkpoint.state().iter(), 31);
    assert_eq!(checkpoint.counts().cost_evals, 32);

    let resumed_writer = ExactCheckpointWriter::new(&resumed_path);
    let resumed_status = resumed_writer.status();
    let resumed = Executor::resume_from_checkpoint(RuggedCost, checkpoint)
        .max_iter(80)
        .checkpoint_with(resumed_writer, ObserverMode::Never)
        .run()
        .unwrap();

    assert_eq!(resumed_status.last_successful_iter(), Some(80));
    assert_eq!(resumed.cost_evals(), 81);
    let reference_state = bincode::serde::encode_to_vec(
        &reference.state,
        bincode::config::standard(),
    )
    .unwrap();
    let resumed_state = bincode::serde::encode_to_vec(
        &resumed.state,
        bincode::config::standard(),
    )
    .unwrap();
    assert_eq!(resumed_state, reference_state);
    assert_eq!(
        std::fs::read(&resumed_path).unwrap(),
        std::fs::read(&reference_path).unwrap()
    );

    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }
}

#[test]
fn state_only_resume_api_remains_exact() {
    let reference = Executor::from_start(RuggedCost, solver(), 12)
        .max_iter(80)
        .run()
        .unwrap();
    let split = Executor::from_start(RuggedCost, solver(), 12)
        .max_iter(31)
        .run()
        .unwrap()
        .into_state();
    let bytes =
        bincode::serde::encode_to_vec(&split, bincode::config::standard())
            .unwrap();
    let (restored, _): (TestState, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .unwrap();

    let resumed = Executor::resume(RuggedCost, solver(), restored)
        .max_iter(80)
        .run()
        .unwrap();

    assert_eq!(resumed.cost_evals(), 81);
    assert_eq!(
        bincode::serde::encode_to_vec(
            &resumed.state,
            bincode::config::standard(),
        )
        .unwrap(),
        bincode::serde::encode_to_vec(
            &reference.state,
            bincode::config::standard(),
        )
        .unwrap(),
    );
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AlwaysRejectNeighbor;

impl Neighbor<i32, f64, ChaCha8Rng> for AlwaysRejectNeighbor {
    type Error = Infallible;

    fn propose(
        &mut self,
        current: &i32,
        _temperature: f64,
        _rng: &mut ChaCha8Rng,
    ) -> Result<i32, Infallible> {
        Ok(current + 100)
    }
}

type RejectSolver = SimulatedAnnealing<AlwaysRejectNeighbor>;
type RejectState =
    SimulatedAnnealingState<i32, AlwaysRejectNeighbor, f64, ChaCha8Rng>;

fn always_reject() -> RejectSolver {
    SimulatedAnnealing::new(
        AlwaysRejectNeighbor,
        1e-6,
        TemperatureSchedule::geometric(0.9),
        9,
    )
}

fn rejection_checkpoint(
    name: &str,
) -> ExactCheckpoint<RejectSolver, RejectState> {
    let path = checkpoint_path(name);
    remove_checkpoint(&path);
    Executor::from_start(RuggedCost, always_reject(), 3)
        .max_iter(4)
        .checkpoint_with(ExactCheckpointWriter::new(&path), ObserverMode::Never)
        .run()
        .unwrap();
    let checkpoint = read_exact_checkpoint(&path).unwrap();
    remove_checkpoint(&path);
    checkpoint
}

#[test]
fn zero_tolerance_best_stall_retains_absolute_history_on_resume() {
    let checkpoint = rejection_checkpoint("best-stall");
    let result = Executor::resume_from_checkpoint(RuggedCost, checkpoint)
        .terminate_on(NoImprovement::new(7, 0.0))
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::NoImprovement);
    assert_eq!(result.iter(), 7);
}

#[test]
fn acceptance_stall_criterion_retains_absolute_history_on_resume() {
    let checkpoint = rejection_checkpoint("acceptance-stall");
    let result = Executor::resume_from_checkpoint(RuggedCost, checkpoint)
        .terminate_on(NoAcceptance::new(7))
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::NoAcceptedMove);
    assert_eq!(result.iter(), 7);
    assert_eq!(result.best_iter(), 0);
}
