//! A `serde`-gated observer that snapshots the state to disk.
//!
//! Available only with the `serde` feature and off `wasm32` (it does file
//! I/O). The companion [`read_checkpoint`] reloads a state snapshot to warm
//! start a later run. Exact continuation instead uses the solver-aware
//! [`ExactCheckpointWriter`](crate::ExactCheckpointWriter).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::core::observer::Observe;
use crate::core::state::State;
use crate::core::termination::TerminationReason;

const MAGIC: &[u8; 8] = b"BASINST\0";
const FORMAT_VERSION: u32 = 1;

/// Write the current state to a file with [`postcard`], overwriting the
/// previous snapshot, so the file always holds the latest checkpoint.
///
/// New files have a format marker and version preceding the postcard payload.
/// [`read_checkpoint`] also accepts legacy, unprefixed bincode files. Older
/// Basin releases cannot read the new format.
///
/// State-only checkpointing is an observer's job in Basin. It records an
/// iterate for a later warm start without promising an identical trajectory.
/// Register it with an [`ObserverMode`](super::ObserverMode) to pick the
/// cadence ([`Every(n)`](super::ObserverMode::Every) for every `n`th iteration,
/// [`NewBest`](super::ObserverMode::NewBest) to snapshot only on improvement).
/// The writer always snapshots on `observe_final` as well.
///
/// The state type must be [`Serialize`]; the shipped checkpointable states are
/// [`BasicState`](crate::core::state::BasicState),
/// [`QuasiNewtonState`](crate::core::state::QuasiNewtonState) (with `Vec<f64>`
/// or nalgebra backends—faer has no serde support), and
/// [`SimulatedAnnealingState`](crate::SimulatedAnnealingState) when its
/// parameter, neighbor, and RNG are serializable. Handing a non-serializable
/// state is a compile error.
///
/// # Resume
///
/// Read the file with [`read_checkpoint`], deserialize into the concrete state,
/// and hand it to [`Executor::new`](crate::Executor::new). The solver runs its
/// normal `init` path and begins a new run from the restored iterate.
///
/// ```no_run
/// # use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent};
/// use basin::{CheckpointWriter, ObserverMode, read_checkpoint};
/// # struct Quadratic;
/// # impl CostFunction for Quadratic {
/// #     type Param = Vec<f64>;
/// #     type Output = f64;
/// #     type Error = std::convert::Infallible;
/// #     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
/// #         Ok(0.5 * x.iter().map(|v| v * v).sum::<f64>())
/// #     }
/// # }
/// # impl Gradient for Quadratic {
/// #     type Gradient = Vec<f64>;
/// #     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> { Ok(x.clone()) }
/// # }
/// // First run: checkpoint every 10 iterations.
/// Executor::new(Quadratic, GradientDescent::new(0.1), BasicState::new(vec![5.0, 5.0]))
///     .max_iter(50)
///     .observe_with(CheckpointWriter::new("run.ckpt"), ObserverMode::Every(10))
///     .run()
///     .unwrap();
///
/// // Later: reload and continue from where it stopped.
/// let state: BasicState<Vec<f64>> = read_checkpoint("run.ckpt").unwrap();
/// Executor::new(Quadratic, GradientDescent::new(0.1), state)
///     .max_iter(50)
///     .run()
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct CheckpointWriter {
    path: PathBuf,
}

impl CheckpointWriter {
    /// Snapshot to `path`, overwriting it on each fire.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Serialize `state` and write it to the checkpoint path. Writes to a
    /// sibling `*.tmp` first and renames into place so a crash mid-write can't
    /// truncate the previous good checkpoint.
    fn write<S: Serialize>(&self, state: &S) -> io::Result<()> {
        let payload = postcard::to_allocvec(state).map_err(io::Error::other)?;
        let mut bytes =
            Vec::with_capacity(MAGIC.len() + size_of::<u32>() + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&payload);
        let tmp = self.path.with_extension("tmp");
        fs::write(&tmp, &bytes)?;
        fs::rename(&tmp, &self.path)
    }

    /// Attempt the write; on failure, log to stderr and carry on. The
    /// [`Observe`] contract is infallible—a failing checkpoint must not kill
    /// the optimization run.
    fn try_write<S: Serialize>(&self, state: &S) {
        if let Err(err) = self.write(state) {
            eprintln!(
                "CheckpointWriter: failed to write {}: {err}",
                self.path.display()
            );
        }
    }
}

impl<S> Observe<S> for CheckpointWriter
where
    S: State + Serialize,
{
    fn observe_iter(&mut self, state: &S) {
        self.try_write(state);
    }

    fn observe_final(&mut self, state: &S, _reason: &TerminationReason) {
        self.try_write(state);
    }
}

/// Load a checkpoint previously written by [`CheckpointWriter`] into a concrete
/// state, ready for [`Executor::new`](crate::Executor::new) as a warm start.
///
/// Reads both versioned postcard files and legacy, unprefixed bincode files.
/// Reading does not modify the file. Unsupported versions, malformed data, and
/// trailing bytes in postcard files return [`io::ErrorKind::InvalidData`].
pub fn read_checkpoint<S: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> io::Result<S> {
    let bytes = fs::read(path)?;
    if let Some(body) = bytes.strip_prefix(MAGIC) {
        let invalid_data =
            |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
        if body.len() < size_of::<u32>() {
            return Err(invalid_data("truncated state checkpoint prefix"));
        }
        let version = u32::from_le_bytes(
            body[..size_of::<u32>()]
                .try_into()
                .expect("version slice has fixed length"),
        );
        if version != FORMAT_VERSION {
            return Err(invalid_data(&format!(
                "unsupported state checkpoint format version {version}; expected {FORMAT_VERSION}"
            )));
        }
        let (state, remaining) =
            postcard::take_from_bytes(&body[size_of::<u32>()..])
                .map_err(|error| invalid_data(&error.to_string()))?;
        if !remaining.is_empty() {
            return Err(invalid_data("trailing data in state checkpoint"));
        }
        return Ok(state);
    }

    let (state, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .map_err(io::Error::other)?;
    Ok(state)
}
