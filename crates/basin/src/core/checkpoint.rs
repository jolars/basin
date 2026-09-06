//! Solver-aware checkpoints for exact continuation.
//!
//! An exact checkpoint captures the solver, state, and authoritative
//! evaluation counters at one coherent iteration boundary. This differs from
//! `CheckpointWriter`, which intentionally writes only a state for a later
//! warm start.

use crate::core::problem::EvalCounts;

/// An owned solver, state, and evaluation-counter snapshot.
///
/// Pass a restored checkpoint to
/// [`Executor::resume_from_checkpoint`](crate::Executor::resume_from_checkpoint)
/// to continue without calling [`Solver::init`](crate::Solver::init) again.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct ExactCheckpoint<So, S> {
    solver: So,
    state: S,
    counts: EvalCounts,
}

impl<So, S> ExactCheckpoint<So, S> {
    /// Build a checkpoint from its complete parts.
    ///
    /// Callers constructing checkpoints manually are responsible for ensuring
    /// that all three parts describe the same iteration boundary.
    pub fn from_parts(solver: So, state: S, counts: EvalCounts) -> Self {
        Self {
            solver,
            state,
            counts,
        }
    }

    /// Solver snapshot carried by this checkpoint.
    pub fn solver(&self) -> &So {
        &self.solver
    }

    /// Mutable solver access, for reattaching execution policy that is not
    /// part of a serialized checkpoint.
    pub fn solver_mut(&mut self) -> &mut So {
        &mut self.solver
    }

    /// State snapshot carried by this checkpoint.
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Authoritative evaluation counters at the snapshot boundary.
    pub fn counts(&self) -> &EvalCounts {
        &self.counts
    }

    /// Consume the checkpoint into `(solver, state, counts)`.
    pub fn into_parts(self) -> (So, S, EvalCounts) {
        (self.solver, self.state, self.counts)
    }
}

/// A destination for solver-aware checkpoints.
///
/// The callback is infallible so attaching a sink does not change a solver's
/// typed application error. Implementations must record or otherwise handle
/// persistence failures themselves.
pub trait CheckpointSink<So, S> {
    /// Save one coherent solver/state/counter snapshot.
    fn save(&mut self, solver: &So, state: &S, counts: &EvalCounts);
}

#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
mod file {
    use std::any::type_name;
    use std::ffi::OsString;
    use std::fmt;
    use std::fs::{self, File};
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, MutexGuard};

    use serde::Serialize;
    use serde::de::DeserializeOwned;

    use super::{CheckpointSink, EvalCounts, ExactCheckpoint};

    const MAGIC: &[u8; 8] = b"BASINEX\0";
    const FORMAT_VERSION: u32 = 1;
    const PREFIX_LEN: usize = MAGIC.len() + 2 * size_of::<u32>();
    const MAX_HEADER_LEN: usize = 64 * 1024;

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Header {
        basin_version: String,
        solver_type: String,
        state_type: String,
    }

    #[derive(serde::Serialize)]
    struct PayloadRef<'a, So, S> {
        solver: &'a So,
        state: &'a S,
        counts: EvalCounts,
    }

    #[derive(serde::Deserialize)]
    struct Payload<So, S> {
        solver: So,
        state: S,
        counts: EvalCounts,
    }

    fn invalid_data(message: impl Into<String>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, message.into())
    }

    fn temporary_path(path: &Path) -> PathBuf {
        let mut temporary: OsString = path.as_os_str().to_owned();
        temporary.push(".tmp");
        temporary.into()
    }

    fn encode_checkpoint<So, S>(
        solver: &So,
        state: &S,
        counts: &EvalCounts,
    ) -> io::Result<Vec<u8>>
    where
        So: Serialize,
        S: Serialize,
    {
        let config = bincode::config::standard();
        let header = Header {
            basin_version: env!("CARGO_PKG_VERSION").to_owned(),
            solver_type: type_name::<So>().to_owned(),
            state_type: type_name::<S>().to_owned(),
        };
        let header = bincode::serde::encode_to_vec(header, config)
            .map_err(io::Error::other)?;
        if header.len() > MAX_HEADER_LEN {
            return Err(invalid_data("exact checkpoint header is too large"));
        }
        let header_len = u32::try_from(header.len()).map_err(|_| {
            invalid_data("exact checkpoint header is too large")
        })?;
        let payload = PayloadRef {
            solver,
            state,
            counts: *counts,
        };
        let payload = bincode::serde::encode_to_vec(payload, config)
            .map_err(io::Error::other)?;

        let mut bytes =
            Vec::with_capacity(PREFIX_LEN + header.len() + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    /// A recorded exact-checkpoint write failure.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct CheckpointWriteError {
        path: PathBuf,
        iteration: u64,
        kind: io::ErrorKind,
        message: String,
    }

    impl CheckpointWriteError {
        /// Destination whose write failed.
        pub fn path(&self) -> &Path {
            &self.path
        }

        /// Iteration represented by the attempted checkpoint.
        pub fn iteration(&self) -> u64 {
            self.iteration
        }

        /// Standard I/O error category.
        pub fn kind(&self) -> io::ErrorKind {
            self.kind
        }

        /// Original error message.
        pub fn message(&self) -> &str {
            &self.message
        }
    }

    impl fmt::Display for CheckpointWriteError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "failed to write exact checkpoint {} at iteration {}: {}",
                self.path.display(),
                self.iteration,
                self.message
            )
        }
    }

    impl std::error::Error for CheckpointWriteError {}

    #[derive(Debug, Default)]
    struct StatusInner {
        last_successful_iter: Option<u64>,
        failure_count: u64,
        last_error: Option<CheckpointWriteError>,
    }

    /// Shared health information for an [`ExactCheckpointWriter`].
    ///
    /// Clone this handle before moving the writer into an executor to inspect
    /// checkpoint health from another thread or after the run completes.
    #[derive(Clone, Debug, Default)]
    pub struct CheckpointStatus {
        inner: Arc<Mutex<StatusInner>>,
    }

    impl CheckpointStatus {
        fn lock(&self) -> MutexGuard<'_, StatusInner> {
            self.inner.lock().unwrap_or_else(|err| err.into_inner())
        }

        /// Most recent iteration written successfully, if any.
        pub fn last_successful_iter(&self) -> Option<u64> {
            self.lock().last_successful_iter
        }

        /// Number of writes that have failed.
        pub fn failure_count(&self) -> u64 {
            self.lock().failure_count
        }

        /// Most recent write failure, if any.
        pub fn last_error(&self) -> Option<CheckpointWriteError> {
            self.lock().last_error.clone()
        }

        fn record_success(&self, iteration: u64) {
            self.lock().last_successful_iter = Some(iteration);
        }

        fn record_failure(&self, error: CheckpointWriteError) {
            let mut status = self.lock();
            status.failure_count = status.failure_count.saturating_add(1);
            status.last_error = Some(error);
        }
    }

    /// Solver-aware exact-checkpoint file writer.
    ///
    /// Attach with [`Executor::checkpoint_with`](crate::Executor::checkpoint_with).
    /// Writes use a temporary sibling and atomic rename. Failures are reported
    /// to stderr and through [`status`](Self::status), but do not stop the run.
    /// Read the result with [`read_exact_checkpoint`] and pass it to
    /// [`Executor::resume_from_checkpoint`](crate::Executor::resume_from_checkpoint).
    #[derive(Clone, Debug)]
    pub struct ExactCheckpointWriter {
        path: PathBuf,
        status: CheckpointStatus,
    }

    impl ExactCheckpointWriter {
        /// Write exact checkpoints to `path`.
        pub fn new(path: impl Into<PathBuf>) -> Self {
            Self {
                path: path.into(),
                status: CheckpointStatus::default(),
            }
        }

        /// Cloneable status handle for observing write health.
        pub fn status(&self) -> CheckpointStatus {
            self.status.clone()
        }

        fn write<So, S>(
            &self,
            solver: &So,
            state: &S,
            counts: &EvalCounts,
        ) -> io::Result<()>
        where
            So: Serialize,
            S: Serialize,
        {
            let bytes = encode_checkpoint(solver, state, counts)?;
            let temporary = temporary_path(&self.path);
            let write_result = (|| {
                let mut file = File::create(&temporary)?;
                file.write_all(&bytes)?;
                file.sync_all()?;
                fs::rename(&temporary, &self.path)?;
                #[cfg(unix)]
                {
                    let parent = self
                        .path
                        .parent()
                        .filter(|path| !path.as_os_str().is_empty())
                        .unwrap_or_else(|| Path::new("."));
                    File::open(parent)?.sync_all()?;
                }
                Ok(())
            })();
            if write_result.is_err() {
                let _ = fs::remove_file(temporary);
            }
            write_result
        }
    }

    impl<So, S> CheckpointSink<So, S> for ExactCheckpointWriter
    where
        So: Serialize,
        S: Serialize + crate::core::state::State,
    {
        fn save(&mut self, solver: &So, state: &S, counts: &EvalCounts) {
            let iteration = state.iter();
            match self.write(solver, state, counts) {
                Ok(()) => self.status.record_success(iteration),
                Err(error) => {
                    let error = CheckpointWriteError {
                        path: self.path.clone(),
                        iteration,
                        kind: error.kind(),
                        message: error.to_string(),
                    };
                    eprintln!("ExactCheckpointWriter: {error}");
                    self.status.record_failure(error);
                }
            }
        }
    }

    /// Read and validate an exact checkpoint.
    ///
    /// The format, exact Basin version, and concrete solver/state type names
    /// must match before the payload is deserialized.
    pub fn read_exact_checkpoint<So, S>(
        path: impl AsRef<Path>,
    ) -> io::Result<ExactCheckpoint<So, S>>
    where
        So: DeserializeOwned,
        S: DeserializeOwned,
    {
        let bytes = fs::read(path)?;
        if bytes.len() < PREFIX_LEN {
            return Err(invalid_data("truncated exact checkpoint prefix"));
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(invalid_data("invalid exact checkpoint magic"));
        }

        let version_start = MAGIC.len();
        let version_end = version_start + size_of::<u32>();
        let format_version = u32::from_le_bytes(
            bytes[version_start..version_end]
                .try_into()
                .expect("version slice has fixed length"),
        );
        if format_version != FORMAT_VERSION {
            return Err(invalid_data(format!(
                "unsupported exact checkpoint format version {format_version}; expected {FORMAT_VERSION}"
            )));
        }

        let header_len = u32::from_le_bytes(
            bytes[version_end..PREFIX_LEN]
                .try_into()
                .expect("header-length slice has fixed length"),
        ) as usize;
        if header_len > MAX_HEADER_LEN {
            return Err(invalid_data("exact checkpoint header is too large"));
        }
        let header_end = PREFIX_LEN
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| invalid_data("truncated exact checkpoint header"))?;
        let config = bincode::config::standard();
        let (header, consumed): (Header, usize) =
            bincode::serde::decode_from_slice(
                &bytes[PREFIX_LEN..header_end],
                config,
            )
            .map_err(|error| invalid_data(error.to_string()))?;
        if consumed != header_len {
            return Err(invalid_data(
                "trailing data in exact checkpoint header",
            ));
        }
        if header.basin_version != env!("CARGO_PKG_VERSION") {
            return Err(invalid_data(format!(
                "exact checkpoint was written by Basin {}; expected {}",
                header.basin_version,
                env!("CARGO_PKG_VERSION")
            )));
        }
        if header.solver_type != type_name::<So>() {
            return Err(invalid_data(format!(
                "exact checkpoint solver type is {}; expected {}",
                header.solver_type,
                type_name::<So>()
            )));
        }
        if header.state_type != type_name::<S>() {
            return Err(invalid_data(format!(
                "exact checkpoint state type is {}; expected {}",
                header.state_type,
                type_name::<S>()
            )));
        }

        let payload_bytes = &bytes[header_end..];
        if payload_bytes.is_empty() {
            return Err(invalid_data("missing exact checkpoint payload"));
        }
        let (payload, consumed): (Payload<So, S>, usize) =
            bincode::serde::decode_from_slice(payload_bytes, config)
                .map_err(|error| invalid_data(error.to_string()))?;
        if consumed != payload_bytes.len() {
            return Err(invalid_data(
                "trailing data in exact checkpoint payload",
            ));
        }
        Ok(ExactCheckpoint::from_parts(
            payload.solver,
            payload.state,
            payload.counts,
        ))
    }
}

#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
pub use file::{
    CheckpointStatus, CheckpointWriteError, ExactCheckpointWriter,
    read_exact_checkpoint,
};
