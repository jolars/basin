//! Typed storage for evaluation reuse.

/// Disposable vector and matrix storage retained between evaluations.
///
/// Callbacks overwrite these buffers on every evaluation. In-memory
/// checkpoints retain the storage. Serde initializes empty buffers on decode.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "")
)]
pub struct EvaluationBuffers<V, M> {
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) vector: Option<V>,
    #[cfg_attr(feature = "serde", serde(skip))]
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
