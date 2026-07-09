//! A `serde`-gated observer that snapshots the state to disk.
//!
//! Available only with the `serde` feature and off `wasm32` (it does file
//! I/O). The companion [`read_checkpoint`] reloads a snapshot to warm-start a
//! later run.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::core::observer::Observe;
use crate::core::state::State;
use crate::core::termination::TerminationReason;

/// Write the current state to a file with [`bincode`], overwriting the
/// previous snapshot, so the file always holds the latest checkpoint.
///
/// Checkpointing "save the iterate periodically so a new run can warm-start"
/// is exactly an observer's job in basin—no framework support is needed.
/// Register it with an [`ObserverMode`](super::ObserverMode) to pick the
/// cadence ([`Every(n)`](super::ObserverMode::Every) for every `n`th iteration,
/// [`NewBest`](super::ObserverMode::NewBest) to snapshot only on improvement).
/// The writer always snapshots on `observe_final` as well.
///
/// The state type must be [`Serialize`]; the shipped checkpointable states are
/// [`BasicState`](crate::core::state::BasicState) and
/// [`QuasiNewtonState`](crate::core::state::QuasiNewtonState) (with `Vec<f64>`
/// or nalgebra backends—faer has no serde support). Handing a non-serializable
/// state is a compile error.
///
/// # Resume
///
/// Resume is not framework surface: read the file with [`read_checkpoint`],
/// deserialize into the concrete state, and hand it to
/// [`Executor::new`](crate::core::executor::Executor::new). The solver re-runs
/// its `init`, so any lazily-rebuilt working storage is reconstructed.
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
        let bytes = bincode::serde::encode_to_vec(state, bincode::config::standard())
            .map_err(io::Error::other)?;
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
/// state, ready to hand back to
/// [`Executor::new`](crate::core::executor::Executor::new) for a warm start.
pub fn read_checkpoint<S: DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<S> {
    let bytes = fs::read(path)?;
    let (state, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
        .map_err(io::Error::other)?;
    Ok(state)
}
