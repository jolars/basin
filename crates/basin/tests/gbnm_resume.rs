#![cfg(all(feature = "serde", not(target_arch = "wasm32")))]

use std::convert::Infallible;
use std::path::PathBuf;

use basin::{
    BoxConstraints, CostFunction, ExactCheckpoint, ExactCheckpointWriter,
    Executor, Gbnm, GbnmState, ObserverMode, State, read_exact_checkpoint,
};

#[derive(Clone)]
struct FlatBox {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for FlatBox {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, _x: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok(1.0)
    }
}

impl BoxConstraints for FlatBox {
    fn lower(&self) -> &Self::Param {
        &self.lower
    }

    fn upper(&self) -> &Self::Param {
        &self.upper
    }
}

type TestSolver = Gbnm;
type TestState = GbnmState<Vec<f64>>;

fn problem() -> FlatBox {
    FlatBox {
        lower: vec![-5.0, -5.0],
        upper: vec![5.0, 5.0],
    }
}

fn solver() -> TestSolver {
    Gbnm::new(0x006b_6e6d)
}

fn checkpoint_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "basin-exact-gbnm-{}-{name}.ckpt",
        std::process::id()
    ))
}

fn remove_checkpoint(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let _ = std::fs::remove_file(PathBuf::from(temporary));
}

fn encoded<T: serde::Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap()
}

#[test]
fn solver_aware_checkpoint_resume_is_bit_identical() {
    let reference_path = checkpoint_path("reference");
    let split_path = checkpoint_path("split");
    let resumed_path = checkpoint_path("resumed");
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }

    let reference =
        Executor::new(problem(), solver(), GbnmState::new(vec![0.0, 0.0]))
            .max_iter(30)
            .checkpoint_with(
                ExactCheckpointWriter::new(&reference_path),
                ObserverMode::Never,
            )
            .run()
            .unwrap();
    Executor::new(problem(), solver(), GbnmState::new(vec![0.0, 0.0]))
        .max_iter(11)
        .checkpoint_with(
            ExactCheckpointWriter::new(&split_path),
            ObserverMode::Never,
        )
        .run()
        .unwrap();

    let checkpoint: ExactCheckpoint<TestSolver, TestState> =
        read_exact_checkpoint(&split_path).unwrap();
    assert_eq!(checkpoint.state().iter(), 11);
    let resumed = Executor::resume_from_checkpoint(problem(), checkpoint)
        .max_iter(30)
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
