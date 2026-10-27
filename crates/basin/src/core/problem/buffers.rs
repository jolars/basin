//! Typed storage for opt-in evaluation reuse.

/// Disposable vector and matrix storage retained between evaluations.
///
/// Solvers that do not retain backend outputs by default use this storage
/// through their `with_evaluation_buffers` builder. The builder infers the
/// vector and matrix types from the problem. When a backend buffer cannot be
/// constructed from the parameter, an owned callback seeds it. Later requests
/// overwrite it through the matching in-place callback. Buffer contents never
/// represent a cached evaluation.
///
/// Constructors, callback bounds, and auto traits of the original unbuffered
/// types stay unchanged. Selecting typed storage can change the resulting
/// type's `Send`, `Sync`, `Clone`, or `Copy` implementations, according to its
/// backend types and storage owner.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EvaluationBuffers<V, M> {
    pub(crate) vector: Option<V>,
    pub(crate) matrix: Option<M>,
}

impl<V, M> Default for EvaluationBuffers<V, M> {
    fn default() -> Self {
        Self {
            vector: None,
            matrix: None,
        }
    }
}

// A unit workspace preserves the original type and ownership requirements.
// Typed storage is selected explicitly rather than adding backend bounds to
// existing constructors or user-implemented evaluation traits.
pub(crate) trait BufferStorage<V, M>: Default {
    fn with_buffers<R>(
        &mut self,
        f: impl FnOnce(&mut Option<V>, &mut Option<M>) -> R,
    ) -> R;
}

impl<V, M> BufferStorage<V, M> for () {
    fn with_buffers<R>(
        &mut self,
        f: impl FnOnce(&mut Option<V>, &mut Option<M>) -> R,
    ) -> R {
        f(&mut None, &mut None)
    }
}

impl<V, M> BufferStorage<V, M> for EvaluationBuffers<V, M> {
    fn with_buffers<R>(
        &mut self,
        f: impl FnOnce(&mut Option<V>, &mut Option<M>) -> R,
    ) -> R {
        f(&mut self.vector, &mut self.matrix)
    }
}

/// Shared evaluation storage selected by the robust objective adapter.
///
/// Clones share the same scratch space. A lock protects concurrent callbacks,
/// retaining `Send` and `Sync` when the backend types support them. The storage
/// contains no cached function values: every request invokes the problem.
#[derive(Debug)]
pub struct SharedEvaluationBuffers<V, M, F: crate::Scalar = f64> {
    inner: std::sync::Arc<std::sync::Mutex<SharedScratch<V, M, F>>>,
}

#[derive(Debug)]
struct SharedScratch<V, M, F> {
    buffers: EvaluationBuffers<V, M>,
    derivatives: Vec<F>,
}

impl<V, M, F: crate::Scalar> Clone for SharedEvaluationBuffers<V, M, F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl<V, M, F: crate::Scalar> Default for SharedEvaluationBuffers<V, M, F> {
    fn default() -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(SharedScratch {
                buffers: EvaluationBuffers::default(),
                derivatives: Vec::new(),
            })),
        }
    }
}
impl<V, M, F: crate::Scalar> SharedEvaluationBuffers<V, M, F> {
    pub(crate) fn with_buffers<R>(
        &self,
        f: impl FnOnce(&mut EvaluationBuffers<V, M>, &mut Vec<F>) -> R,
    ) -> R {
        let mut scratch = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let SharedScratch {
            buffers,
            derivatives,
        } = &mut *scratch;
        f(buffers, derivatives)
    }
}
