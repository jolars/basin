//! An observer that snapshots the state to disk.
//!
//! Use [`read_state_snapshot`] to load a state snapshot for a later warm start.
//! For exact continuation, use the solver-aware
//! [`CheckpointWriter`](crate::CheckpointWriter).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::core::observer::Observe;
use crate::core::state::State;
use crate::core::termination::TerminationReport;

const MAGIC: &[u8; 8] = b"BASINST\0";
const FORMAT_VERSION: u32 = 2;

/// Write the current state to a file with [`postcard`], overwriting the
/// previous snapshot.
///
/// Available on native targets with `checkpoint-postcard`.
///
/// Files start with `BASINST\0` and a little-endian `u32` format version of 1,
/// followed by the postcard payload.
///
/// A state snapshot records an iterate for a later warm start.
/// Register it with an [`ObserverMode`](super::ObserverMode) to pick the
/// cadence ([`Every(n)`](super::ObserverMode::Every) for every `n`th iteration,
/// [`NewBest`](super::ObserverMode::NewBest) to snapshot only on improvement).
/// The writer always snapshots on `observe_final` as well.
///
/// The state type must be [`Serialize`]. Shared progress types such as
/// [`PointState`](crate::PointState), [`FirstOrderState`](crate::FirstOrderState),
/// and [`ProposalState`](crate::ProposalState) support serialization when their
/// parameter and scalar types do. Solver-owned models, neighbors, and RNGs are
/// absent from these snapshots; use [`CheckpointWriter`](crate::CheckpointWriter)
/// to retain them.
/// Handing a non-serializable state is a compile error.
///
/// # Start a fresh run from a snapshot
///
/// Use [`read_state_snapshot`] to load the concrete state, then pass it to
/// [`Executor::new`](crate::Executor::new). The solver initializes a fresh run
/// from the saved iterate.
///
/// ```no_run
/// # use basin::{FirstOrderState, CostFunction, Executor, Gradient, GradientDescent};
/// use basin::{ObserverMode, StateSnapshotWriter, read_state_snapshot};
/// # struct Quadratic;
/// impl CostFunction for Quadratic {
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
/// // Save a snapshot every 10 iterations.
/// Executor::new(Quadratic, GradientDescent::new(0.1), FirstOrderState::new(vec![5.0, 5.0]))
///     .max_iter(50)
///     .observe_with(StateSnapshotWriter::new("run.snapshot"), ObserverMode::Every(10))
///     .run()
///     .unwrap();
///
/// // Start a fresh run from the saved iterate.
/// let state: FirstOrderState<Vec<f64>> = read_state_snapshot("run.snapshot").unwrap();
/// Executor::new(Quadratic, GradientDescent::new(0.1), state)
///     .max_iter(50)
///     .run()
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct StateSnapshotWriter {
    path: PathBuf,
}

impl StateSnapshotWriter {
    /// Snapshot to `path`, overwriting it on each fire.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Write the serialized state to a temporary sibling before renaming it
    /// into place to protect the previous snapshot during the write.
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

    /// Report write failures to stderr and let the optimization run continue.
    fn try_write<S: Serialize>(&self, state: &S) {
        if let Err(err) = self.write(state) {
            eprintln!(
                "StateSnapshotWriter: failed to write {}: {err}",
                self.path.display()
            );
        }
    }
}

impl<S> Observe<S> for StateSnapshotWriter
where
    S: State + Serialize,
{
    fn observe_iter(&mut self, state: &S) {
        self.try_write(state);
    }

    fn observe_selection(
        &mut self,
        state: &S,
        _change: crate::SelectionChange,
    ) {
        self.try_write(state);
    }

    fn observe_final(
        &mut self,
        state: &S,
        _reason: &TerminationReport<S::Float>,
    ) {
        self.try_write(state);
    }
}

/// Load a state snapshot written by [`StateSnapshotWriter`].
///
/// Pass the concrete state to [`Executor::new`](crate::Executor::new) to warm
/// start a fresh run.
///
/// Reads postcard state format version 2 without modifying the file. Missing
/// format markers, unsupported versions, malformed data, and trailing bytes
/// return [`io::ErrorKind::InvalidData`].
///
/// The file must match the concrete state's serialized layout. See the
/// [migration guide](https://basin.rs/guide/migrating-to-v4/#existing-files)
/// for using files from older versions of Basin.
pub fn read_state_snapshot<S: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> io::Result<S> {
    let bytes = fs::read(path)?;
    let invalid_data =
        |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    let body = bytes
        .strip_prefix(MAGIC)
        .ok_or_else(|| invalid_data("invalid state snapshot magic"))?;
    if body.len() < size_of::<u32>() {
        return Err(invalid_data("truncated state snapshot prefix"));
    }
    let version = u32::from_le_bytes(
        body[..size_of::<u32>()]
            .try_into()
            .expect("version slice has fixed length"),
    );
    if version != FORMAT_VERSION {
        return Err(invalid_data(&format!(
            "unsupported state snapshot format version {version}; expected {FORMAT_VERSION}"
        )));
    }
    let (state, remaining) =
        postcard::take_from_bytes(&body[size_of::<u32>()..])
            .map_err(|error| invalid_data(&error.to_string()))?;
    if !remaining.is_empty() {
        return Err(invalid_data("trailing data in state snapshot"));
    }
    Ok(state)
}
