#![cfg(all(feature = "serde", not(target_arch = "wasm32")))]

use std::convert::Infallible;
use std::path::PathBuf;

use basin::{
    BoxConstraints, CostFunction, ExactCheckpoint, ExactCheckpointWriter,
    Executor, GlobalBestPso, GlobalBestPsoState, ObserverMode, State,
    read_exact_checkpoint,
};

#[derive(Clone)]
struct Landscape {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for Landscape {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(x.iter()
            .map(|xi| xi * xi - 10.0 * (std::f64::consts::TAU * xi).cos())
            .sum::<f64>()
            + 10.0 * x.len() as f64)
    }
}

impl BoxConstraints for Landscape {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

type TestSolver = GlobalBestPso;
type TestState = GlobalBestPsoState<Vec<f64>>;

fn problem() -> Landscape {
    Landscape {
        lower: vec![-5.12; 3],
        upper: vec![5.12; 3],
    }
}

fn solver() -> TestSolver {
    GlobalBestPso::new(0x5eed).with_swarm_size(14)
}

fn encoded<T: serde::Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap()
}

fn checkpoint_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "basin-exact-global-best-pso-{}-{name}.ckpt",
        std::process::id()
    ))
}

fn remove_checkpoint(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let _ = std::fs::remove_file(PathBuf::from(temporary));
}

#[test]
fn serialized_state_only_resume_is_bit_identical() {
    let reference = Executor::new(problem(), solver(), TestState::new())
        .max_iter(55)
        .run()
        .unwrap();
    let split = Executor::new(problem(), solver(), TestState::new())
        .max_iter(19)
        .run()
        .unwrap()
        .into_state();
    let bytes = encoded(&split);
    let (restored, consumed): (TestState, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .unwrap();
    assert_eq!(consumed, bytes.len());

    let resumed = Executor::resume(problem(), solver(), restored)
        .max_iter(55)
        .run()
        .unwrap();

    assert_eq!(resumed.state.iter(), 55);
    assert_eq!(resumed.cost_evals(), 14 * 56);
    assert_eq!(encoded(&resumed.state), encoded(&reference.state));
}

#[test]
fn solver_aware_checkpoint_resume_is_bit_identical() {
    let reference_path = checkpoint_path("reference");
    let split_path = checkpoint_path("split");
    let resumed_path = checkpoint_path("resumed");
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }

    let reference = Executor::new(problem(), solver(), TestState::new())
        .max_iter(55)
        .checkpoint_with(
            ExactCheckpointWriter::new(&reference_path),
            ObserverMode::Never,
        )
        .run()
        .unwrap();
    Executor::new(problem(), solver(), TestState::new())
        .max_iter(19)
        .checkpoint_with(
            ExactCheckpointWriter::new(&split_path),
            ObserverMode::Never,
        )
        .run()
        .unwrap();

    let checkpoint: ExactCheckpoint<TestSolver, TestState> =
        read_exact_checkpoint(&split_path).unwrap();
    assert_eq!(checkpoint.state().iter(), 19);
    let resumed = Executor::resume_from_checkpoint(problem(), checkpoint)
        .max_iter(55)
        .checkpoint_with(
            ExactCheckpointWriter::new(&resumed_path),
            ObserverMode::Never,
        )
        .run()
        .unwrap();

    assert_eq!(encoded(&resumed.state), encoded(&reference.state));
    assert_eq!(
        std::fs::read(&resumed_path).unwrap(),
        std::fs::read(&reference_path).unwrap()
    );

    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }
}
