#![cfg(all(feature = "serde", not(target_arch = "wasm32")))]

use std::convert::Infallible;
use std::path::{Path, PathBuf};

use basin::{
    BasicPopulationState, BasicSimplexState, BasicState, BasinHopping,
    BoxConstraints, CmaEs, CmaEsState, CostFunction, De, DenseMatrix,
    ExactCheckpoint, ExactCheckpointWriter, Executor, NelderMead, NllsState,
    ObserverMode, SimplexTolerance, Ssga, State, read_exact_checkpoint,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

#[derive(Clone)]
struct Landscape {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Landscape {
    fn new(n: usize) -> Self {
        Self {
            lower: vec![-5.12; n],
            upper: vec![5.12; n],
        }
    }
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

fn checkpoint_path(solver: &str, stage: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "basin-exact-{solver}-{}-{stage}.ckpt",
        std::process::id()
    ))
}

fn remove_checkpoint(path: &Path) {
    let _ = std::fs::remove_file(path);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let _ = std::fs::remove_file(PathBuf::from(temporary));
}

fn encoded<T: Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap()
}

fn assert_round_trip<T>(value: &T)
where
    T: Serialize + DeserializeOwned,
{
    let before = encoded(value);
    let (restored, consumed): (T, usize) =
        bincode::serde::decode_from_slice(&before, bincode::config::standard())
            .unwrap();
    assert_eq!(consumed, before.len());
    assert_eq!(encoded(&restored), before);
}

fn assert_identical_finish<S: Serialize>(
    reference: &S,
    resumed: &S,
    reference_path: &Path,
    resumed_path: &Path,
) {
    assert_eq!(encoded(resumed), encoded(reference));
    assert_eq!(
        std::fs::read(resumed_path).unwrap(),
        std::fs::read(reference_path).unwrap()
    );
}

#[test]
fn supporting_state_shapes_round_trip() {
    let simplex: BasicSimplexState<Vec<f64>> =
        BasicSimplexState::from_simplex(vec![
            vec![1.0, 2.0],
            vec![1.1, 2.0],
            vec![1.0, 2.1],
        ]);
    let nlls: NllsState<Vec<f64>> = NllsState::new(vec![1.0, 2.0]);

    assert_round_trip(&simplex);
    assert_round_trip(&nlls);
}

#[test]
fn de_resume_is_bit_identical() {
    type TestSolver = De;
    type TestState = BasicPopulationState<Vec<f64>>;

    let reference_path = checkpoint_path("de", "reference");
    let split_path = checkpoint_path("de", "split");
    let resumed_path = checkpoint_path("de", "resumed");
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }

    let reference = Executor::new(
        Landscape::new(3),
        De::new(0xde).with_pop_size(10),
        BasicPopulationState::with_size(1),
    )
    .max_iter(24)
    .checkpoint_with(
        ExactCheckpointWriter::new(&reference_path),
        ObserverMode::Never,
    )
    .run()
    .unwrap();

    Executor::new(
        Landscape::new(3),
        De::new(0xde).with_pop_size(10),
        BasicPopulationState::with_size(1),
    )
    .max_iter(9)
    .checkpoint_with(
        ExactCheckpointWriter::new(&split_path),
        ObserverMode::Never,
    )
    .run()
    .unwrap();
    let checkpoint: ExactCheckpoint<TestSolver, TestState> =
        read_exact_checkpoint(&split_path).unwrap();
    assert_eq!(checkpoint.state().iter(), 9);

    let resumed =
        Executor::resume_from_checkpoint(Landscape::new(3), checkpoint)
            .max_iter(24)
            .checkpoint_with(
                ExactCheckpointWriter::new(&resumed_path),
                ObserverMode::Never,
            )
            .run()
            .unwrap();

    assert_identical_finish(
        &reference.state,
        &resumed.state,
        &reference_path,
        &resumed_path,
    );
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }
}

#[test]
fn ssga_resume_is_bit_identical() {
    type TestSolver = Ssga;
    type TestState = BasicPopulationState<Vec<f64>>;

    let reference_path = checkpoint_path("ssga", "reference");
    let split_path = checkpoint_path("ssga", "split");
    let resumed_path = checkpoint_path("ssga", "resumed");
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }

    let make_solver = || {
        Ssga::new(0x55_6a)
            .with_pop_size(10)
            .with_offspring_per_step(3)
    };
    let reference = Executor::new(
        Landscape::new(3),
        make_solver(),
        BasicPopulationState::with_size(10),
    )
    .max_iter(28)
    .checkpoint_with(
        ExactCheckpointWriter::new(&reference_path),
        ObserverMode::Never,
    )
    .run()
    .unwrap();

    Executor::new(
        Landscape::new(3),
        make_solver(),
        BasicPopulationState::with_size(10),
    )
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

    let resumed =
        Executor::resume_from_checkpoint(Landscape::new(3), checkpoint)
            .max_iter(28)
            .checkpoint_with(
                ExactCheckpointWriter::new(&resumed_path),
                ObserverMode::Never,
            )
            .run()
            .unwrap();

    assert_identical_finish(
        &reference.state,
        &resumed.state,
        &reference_path,
        &resumed_path,
    );
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }
}

#[test]
fn cma_es_resume_is_bit_identical() {
    type TestSolver = CmaEs<Vec<f64>, DenseMatrix>;
    type TestState = CmaEsState<Vec<f64>, DenseMatrix>;

    let reference_path = checkpoint_path("cma-es", "reference");
    let split_path = checkpoint_path("cma-es", "split");
    let resumed_path = checkpoint_path("cma-es", "resumed");
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }

    let make_solver = || -> TestSolver { CmaEs::new(0xc0_ffee).with_lambda(8) };
    let make_state =
        || -> TestState { CmaEsState::new(vec![2.5, -1.5, 0.75], 0.8) };
    let reference =
        Executor::new(Landscape::new(3), make_solver(), make_state())
            .max_iter(22)
            .checkpoint_with(
                ExactCheckpointWriter::new(&reference_path),
                ObserverMode::Never,
            )
            .run()
            .unwrap();

    Executor::new(Landscape::new(3), make_solver(), make_state())
        .max_iter(8)
        .checkpoint_with(
            ExactCheckpointWriter::new(&split_path),
            ObserverMode::Never,
        )
        .run()
        .unwrap();
    let checkpoint: ExactCheckpoint<TestSolver, TestState> =
        read_exact_checkpoint(&split_path).unwrap();
    assert_eq!(checkpoint.state().iter(), 8);

    let resumed =
        Executor::resume_from_checkpoint(Landscape::new(3), checkpoint)
            .max_iter(22)
            .checkpoint_with(
                ExactCheckpointWriter::new(&resumed_path),
                ObserverMode::Never,
            )
            .run()
            .unwrap();

    assert_identical_finish(
        &reference.state,
        &resumed.state,
        &reference_path,
        &resumed_path,
    );
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }
}

#[test]
fn basin_hopping_resume_is_bit_identical() {
    type TestSolver = BasinHopping<NelderMead, Vec<f64>>;
    type TestState = BasicState<Vec<f64>>;

    let reference_path = checkpoint_path("basin-hopping", "reference");
    let split_path = checkpoint_path("basin-hopping", "split");
    let resumed_path = checkpoint_path("basin-hopping", "resumed");
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }

    let make_solver = || {
        BasinHopping::new(NelderMead::adaptive(), 0xba_51)
            .with_stepsize(1.3)
            .with_temperature(0.7)
            .with_inner_max_iter(12)
            .with_adaptive_interval(3)
    };
    let reference = Executor::new(
        Landscape::new(2),
        make_solver(),
        BasicState::new(vec![3.1, -2.7]),
    )
    .max_iter(18)
    .checkpoint_with(
        ExactCheckpointWriter::new(&reference_path),
        ObserverMode::Never,
    )
    .run()
    .unwrap();

    Executor::new(
        Landscape::new(2),
        make_solver(),
        BasicState::new(vec![3.1, -2.7]),
    )
    .max_iter(7)
    .checkpoint_with(
        ExactCheckpointWriter::new(&split_path),
        ObserverMode::Never,
    )
    .run()
    .unwrap();
    let checkpoint: ExactCheckpoint<TestSolver, TestState> =
        read_exact_checkpoint(&split_path).unwrap();
    assert_eq!(checkpoint.state().iter(), 7);

    let resumed =
        Executor::resume_from_checkpoint(Landscape::new(2), checkpoint)
            .max_iter(18)
            .checkpoint_with(
                ExactCheckpointWriter::new(&resumed_path),
                ObserverMode::Never,
            )
            .run()
            .unwrap();

    assert_identical_finish(
        &reference.state,
        &resumed.state,
        &reference_path,
        &resumed_path,
    );
    for path in [&reference_path, &split_path, &resumed_path] {
        remove_checkpoint(path);
    }
}

#[test]
fn basin_hopping_checkpoint_rejects_erased_inner_criteria() {
    let path = checkpoint_path("basin-hopping", "inner-criterion");
    remove_checkpoint(&path);
    let writer = ExactCheckpointWriter::new(&path);
    let status = writer.status();
    let solver = BasinHopping::new(NelderMead::adaptive(), 7)
        .with_inner_max_iter(12)
        .inner_terminate_on(SimplexTolerance::new(1e-8, 1e-8));

    Executor::new(Landscape::new(2), solver, BasicState::new(vec![2.0, -2.0]))
        .max_iter(2)
        .checkpoint_with(writer, ObserverMode::Never)
        .run()
        .unwrap();

    assert_eq!(status.last_successful_iter(), None);
    assert_eq!(status.failure_count(), 1);
    assert!(
        status
            .last_error()
            .unwrap()
            .message()
            .contains("cannot serialize boxed termination criteria")
    );
    assert!(!path.exists());
    remove_checkpoint(&path);
}
