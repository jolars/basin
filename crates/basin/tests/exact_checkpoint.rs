use basin::{
    CheckpointSink, CountsMirror, EvalCounts, ExactCheckpoint,
    ExactResumeState, Executor, ObserverMode, Problem, Solver, State,
};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
struct TestState {
    param: (),
    iter: u64,
    cost_evals: u64,
}

impl TestState {
    fn new() -> Self {
        Self {
            param: (),
            iter: 0,
            cost_evals: 0,
        }
    }
}

impl State for TestState {
    type Param = ();
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn param(&self) -> &Self::Param {
        &self.param
    }

    fn cost(&self) -> Self::Float {
        0.0
    }

    fn best_param(&self) -> &Self::Param {
        &self.param
    }

    fn best_cost(&self) -> Self::Float {
        0.0
    }

    fn best_iter(&self) -> u64 {
        0
    }

    fn best_cost_evals(&self) -> u64 {
        0
    }

    fn update_best(&mut self) {}

    fn reset_best(&mut self) {}
}

impl CountsMirror for TestState {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.cost_evals = counts.cost_evals;
    }
}

impl ExactResumeState for TestState {
    fn resume_counts(&self) -> EvalCounts {
        EvalCounts {
            cost_evals: self.cost_evals,
            ..EvalCounts::default()
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
struct CountingSolver {
    init_calls: u64,
    steps: u64,
}

impl CountingSolver {
    fn new() -> Self {
        Self {
            init_calls: 0,
            steps: 0,
        }
    }
}

impl Solver<(), TestState> for CountingSolver {
    type Error = Infallible;

    fn init(
        &mut self,
        _problem: &mut Problem<()>,
        state: TestState,
    ) -> Result<TestState, Self::Error> {
        self.init_calls += 1;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        _problem: &mut Problem<()>,
        state: TestState,
    ) -> Result<(TestState, Option<basin::TerminationReason>), Self::Error>
    {
        self.steps += 1;
        Ok((state, None))
    }
}

type Snapshot = ExactCheckpoint<CountingSolver, TestState>;

#[derive(Clone)]
struct MemorySink {
    snapshots: Arc<Mutex<Vec<Snapshot>>>,
}

impl CheckpointSink<CountingSolver, TestState> for MemorySink {
    fn save(
        &mut self,
        solver: &CountingSolver,
        state: &TestState,
        counts: &EvalCounts,
    ) {
        self.snapshots
            .lock()
            .unwrap()
            .push(ExactCheckpoint::from_parts(
                solver.clone(),
                state.clone(),
                *counts,
            ));
    }
}

fn memory_sink() -> (MemorySink, Arc<Mutex<Vec<Snapshot>>>) {
    let snapshots = Arc::new(Mutex::new(Vec::new()));
    (
        MemorySink {
            snapshots: snapshots.clone(),
        },
        snapshots,
    )
}

#[test]
fn resume_skips_init_even_at_iteration_zero() {
    let (writer, snapshots) = memory_sink();
    Executor::new((), CountingSolver::new(), TestState::new())
        .max_iter(0)
        .checkpoint_with(writer, ObserverMode::Never)
        .run()
        .unwrap();
    let checkpoint = snapshots.lock().unwrap().pop().unwrap();
    assert_eq!(checkpoint.solver().init_calls, 1);

    let (writer, snapshots) = memory_sink();
    Executor::resume_from_checkpoint((), checkpoint)
        .max_iter(0)
        .checkpoint_with(writer, ObserverMode::Never)
        .run()
        .unwrap();
    let checkpoint = snapshots.lock().unwrap().pop().unwrap();
    assert_eq!(checkpoint.solver().init_calls, 1);
}

#[test]
fn legacy_state_resume_still_calls_resume_idempotent_init() {
    let state = TestState {
        param: (),
        iter: 4,
        cost_evals: 7,
    };
    let (writer, snapshots) = memory_sink();

    Executor::resume((), CountingSolver::new(), state)
        .max_iter(4)
        .checkpoint_with(writer, ObserverMode::Never)
        .run()
        .unwrap();

    let checkpoint = snapshots.lock().unwrap().pop().unwrap();
    assert_eq!(checkpoint.solver().init_calls, 1);
    assert_eq!(checkpoint.counts().cost_evals, 7);
}

#[test]
fn checkpoint_mode_gates_iterations_but_not_final_save() {
    let (writer, snapshots) = memory_sink();
    Executor::new((), CountingSolver::new(), TestState::new())
        .max_iter(3)
        .checkpoint_with(writer, ObserverMode::every(2))
        .run()
        .unwrap();

    let snapshots = snapshots.lock().unwrap();
    let iterations: Vec<_> = snapshots
        .iter()
        .map(|checkpoint| checkpoint.state().iter())
        .collect();
    assert_eq!(iterations, vec![2, 3]);
    assert_eq!(snapshots[0].solver().steps, 2);
    assert_eq!(snapshots[1].solver().steps, 3);
}

#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
mod file {
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};

    use basin::{ExactCheckpointWriter, read_exact_checkpoint};

    use super::*;

    fn checkpoint_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "basin-exact-api-{}-{name}.ckpt",
            std::process::id()
        ))
    }

    fn remove_checkpoint(path: &Path) {
        let _ = std::fs::remove_file(path);
        let mut temporary = path.as_os_str().to_owned();
        temporary.push(".tmp");
        let _ = std::fs::remove_file(PathBuf::from(temporary));
    }

    fn write_valid_checkpoint(path: &Path) {
        remove_checkpoint(path);
        Executor::new((), CountingSolver::new(), TestState::new())
            .max_iter(2)
            .checkpoint_with(
                ExactCheckpointWriter::new(path),
                ObserverMode::Never,
            )
            .run()
            .unwrap();
    }

    #[test]
    fn file_round_trip_and_header_validation() {
        let path = checkpoint_path("validation");
        write_valid_checkpoint(&path);
        let original = std::fs::read(&path).unwrap();

        let checkpoint: Snapshot = read_exact_checkpoint(&path).unwrap();
        assert_eq!(checkpoint.solver().steps, 2);
        assert_eq!(checkpoint.state().iter(), 2);

        let wrong_type =
            read_exact_checkpoint::<u64, TestState>(&path).unwrap_err();
        assert_eq!(wrong_type.kind(), ErrorKind::InvalidData);
        assert!(wrong_type.to_string().contains("solver type"));

        let wrong_state =
            read_exact_checkpoint::<CountingSolver, u64>(&path).unwrap_err();
        assert_eq!(wrong_state.kind(), ErrorKind::InvalidData);
        assert!(wrong_state.to_string().contains("state type"));

        let mut invalid_magic = original.clone();
        invalid_magic[0] ^= 0xff;
        std::fs::write(&path, invalid_magic).unwrap();
        let error = read_exact_checkpoint::<CountingSolver, TestState>(&path)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains("magic"));

        let mut invalid_version = original.clone();
        invalid_version[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, invalid_version).unwrap();
        let error = read_exact_checkpoint::<CountingSolver, TestState>(&path)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains("format version"));

        let mut invalid_basin_version = original.clone();
        let version = env!("CARGO_PKG_VERSION").as_bytes();
        let version_start = invalid_basin_version
            .windows(version.len())
            .position(|window| window == version)
            .expect("Basin version is present in the exact-checkpoint header");
        invalid_basin_version[version_start..version_start + version.len()]
            .fill(b'x');
        std::fs::write(&path, invalid_basin_version).unwrap();
        let error = read_exact_checkpoint::<CountingSolver, TestState>(&path)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains("written by Basin"));

        std::fs::write(&path, &original[..10]).unwrap();
        let error = read_exact_checkpoint::<CountingSolver, TestState>(&path)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains("truncated"));

        std::fs::write(&path, &original[..original.len() - 1]).unwrap();
        let error = read_exact_checkpoint::<CountingSolver, TestState>(&path)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);

        let mut trailing = original;
        trailing.push(0);
        std::fs::write(&path, trailing).unwrap();
        let error = read_exact_checkpoint::<CountingSolver, TestState>(&path)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains("trailing data"));

        remove_checkpoint(&path);
    }

    #[test]
    fn write_failures_are_recorded_without_stopping_the_run() {
        let parent = std::env::temp_dir()
            .join(format!("basin-exact-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&parent);
        let path = parent.join("run.ckpt");
        let writer = ExactCheckpointWriter::new(&path);
        let status = writer.status();

        let result = Executor::new((), CountingSolver::new(), TestState::new())
            .max_iter(2)
            .checkpoint_with(writer, ObserverMode::Never)
            .run()
            .unwrap();

        assert_eq!(result.iter(), 2);
        assert_eq!(status.last_successful_iter(), None);
        assert_eq!(status.failure_count(), 1);
        let error = status.last_error().unwrap();
        assert_eq!(error.path(), path);
        assert_eq!(error.iteration(), 2);
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }
}
