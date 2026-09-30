//! Shared observable simplex progress, independent of solver workspaces.

use super::progress::Incumbent;
use super::{
    CountsMirror, EvaluatedState, IncumbentRef, IncumbentState,
    IntoInitialSimplex, ObjectiveIncumbentState, RawEvaluationState,
    SimplexState, State,
};
use crate::core::math::{Scalar, VectorLen};
use crate::core::problem::EvalCounts;

/// Observable simplex vertices, matching costs, and historical progress.
///
/// Vertices live directly in this state. Solvers own their scratch vectors,
/// coefficients, and other evolving machinery. [`replace`](Self::replace)
/// validates a complete replacement before changing the state and sorts its
/// point/cost pairs by increasing cost, placing NaNs last. Equal costs keep
/// their relative order. The executor retains strict objective improvements
/// separately, so replacing the entire simplex cannot overwrite its incumbent.
/// NaN and positive infinity never establish an incumbent; negative infinity
/// can be retained without establishing unboundedness.
///
/// Construction supplies unevaluated vertices: [`current`](Self::current),
/// [`best`](Self::best), and [`evaluated_vertices`](Self::evaluated_vertices)
/// return `None`, and [`costs`](Self::costs) is empty. A simplex has at least
/// two vertices of equal nonzero parameter length, and no more than `n + 1`
/// vertices in `n` coordinates. Lower-dimensional simplexes support searches
/// in a subspace, such as GBNM with pinned coordinates. Affine independence is
/// not required, so projected and collapsed simplexes remain representable.
///
/// Fresh [`reset`](Self::reset) preserves the vertices as seeds and clears
/// evaluation records, counts, and incumbent metadata. Exact continuation
/// requires a solver-aware checkpoint. With `serde`, serialization is available
/// when the parameter and scalar types support it; it excludes solver scratch.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`, with
/// `F = f32` or `f64`. Construction and replacement require [`VectorLen`];
/// observation and bookkeeping require only `Clone`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimplexProgress<V, F: Scalar = f64> {
    pub(crate) vertices: Vec<V>,
    pub(crate) costs: Vec<F>,
    iter: u64,
    counts: EvalCounts,
    best: Option<Incumbent<V, F>>,
}

/// A proposed simplex has inconsistent point or cost dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimplexShapeError {
    /// A simplex requires at least two vertices.
    TooFewVertices {
        /// Number of supplied vertices.
        actual: usize,
    },
    /// A simplex in `dimension` coordinates has at most `dimension + 1` vertices.
    TooManyVertices {
        /// Number of supplied vertices.
        actual: usize,
        /// Coordinate count of the first vertex.
        dimension: usize,
    },
    /// All vertices must have the same coordinate count.
    DimensionMismatch {
        /// Index of the inconsistent vertex.
        index: usize,
        /// Coordinate count of the first vertex.
        expected: usize,
        /// Coordinate count of the inconsistent vertex.
        actual: usize,
    },
    /// Each vertex requires one cost.
    CostCountMismatch {
        /// Number of supplied vertices.
        vertices: usize,
        /// Number of supplied costs.
        costs: usize,
    },
}
impl std::fmt::Display for SimplexShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewVertices { actual } => write!(
                f,
                "simplex requires at least two vertices, got {actual}"
            ),
            Self::TooManyVertices { actual, dimension } => write!(
                f,
                "{actual} simplex vertices exceed dimension {dimension} plus one"
            ),
            Self::DimensionMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "simplex vertex {index} has length {actual}, expected {expected}"
            ),
            Self::CostCountMismatch { vertices, costs } => write!(
                f,
                "{vertices} simplex vertices require {vertices} costs, got {costs}"
            ),
        }
    }
}
impl std::error::Error for SimplexShapeError {}

impl<V: VectorLen, F: Scalar> SimplexProgress<V, F> {
    fn validate(vertices: &[V]) -> Result<(), SimplexShapeError> {
        if vertices.len() < 2 {
            return Err(SimplexShapeError::TooFewVertices {
                actual: vertices.len(),
            });
        }
        let dimension = vertices[0].vec_len();
        if vertices.len() - 1 > dimension {
            return Err(SimplexShapeError::TooManyVertices {
                actual: vertices.len(),
                dimension,
            });
        }
        for (index, vertex) in vertices.iter().enumerate().skip(1) {
            if vertex.vec_len() != dimension {
                return Err(SimplexShapeError::DimensionMismatch {
                    index,
                    expected: dimension,
                    actual: vertex.vec_len(),
                });
            }
        }
        Ok(())
    }

    /// Construct unevaluated progress from explicit vertices.
    ///
    /// # Panics
    ///
    /// Panics if the vertex count or coordinate lengths violate this type's
    /// documented shape contract.
    pub fn from_simplex(vertices: Vec<V>) -> Self {
        Self::validate(&vertices).expect("invalid initial simplex");
        Self {
            vertices,
            costs: Vec::new(),
            iter: 0,
            counts: EvalCounts::default(),
            best: None,
        }
    }

    /// Construct the default FMINSEARCH-style simplex around a starting point.
    /// Uses a 5% relative step and an absolute `0.00025` for zero coordinates.
    /// Panics for an empty parameter vector.
    pub fn new<X: IntoInitialSimplex<V, F>>(point: X) -> Self {
        Self::with_step(point, F::from_f64(0.05).unwrap())
    }

    /// Construct a simplex with an explicit relative step. Zero coordinates
    /// retain the absolute `0.00025` step. Panics for an empty parameter vector.
    pub fn with_step<X: IntoInitialSimplex<V, F>>(
        point: X,
        relative_step: F,
    ) -> Self {
        Self::from_simplex(point.into_initial_simplex(relative_step))
    }

    /// Replace all evaluated vertices together, preserving the incumbent and
    /// counters until publication. Structural errors retain the entire old
    /// state. Valid replacements may have a different parameter dimension.
    pub fn replace(
        &mut self,
        vertices: Vec<V>,
        costs: Vec<F>,
    ) -> Result<(), SimplexShapeError> {
        Self::validate(&vertices)?;
        if vertices.len() != costs.len() {
            return Err(SimplexShapeError::CostCountMismatch {
                vertices: vertices.len(),
                costs: costs.len(),
            });
        }
        self.vertices = vertices;
        self.costs = costs;
        self.sort();
        Ok(())
    }
}

impl<V, F: Scalar> SimplexProgress<V, F> {
    /// Borrow the authoritative simplex vertices, including unevaluated seeds.
    pub fn vertices(&self) -> &[V] {
        &self.vertices
    }
    /// Borrow matching costs, or an empty slice before evaluation.
    pub fn costs(&self) -> &[F] {
        &self.costs
    }
    /// Borrow complete vertex/cost arrays, or `None` before evaluation.
    pub fn evaluated_vertices(&self) -> Option<(&[V], &[F])> {
        (!self.vertices.is_empty() && self.vertices.len() == self.costs.len())
            .then_some((&self.vertices, &self.costs))
    }
    /// Borrow the lowest-cost current vertex, or `None` before evaluation.
    pub fn current(&self) -> Option<(&V, F)> {
        let (vertices, costs) = self.evaluated_vertices()?;
        Some((&vertices[0], costs[0]))
    }
    /// Borrow the historical objective incumbent, if one exists.
    pub fn best(&self) -> Option<(&V, F)> {
        self.best.as_ref().map(|best| (&best.param, best.cost))
    }
    /// All raw categories at the latest publication boundary.
    pub fn counts(&self) -> &EvalCounts {
        &self.counts
    }
    /// All raw categories at the incumbent's selection boundary, if any.
    pub fn best_counts(&self) -> Option<&EvalCounts> {
        self.best.as_ref().map(|best| &best.counts)
    }
    /// Preserve the simplex as a seed and clear all evaluated progress.
    pub fn reset(&mut self) {
        self.costs.clear();
        self.iter = 0;
        self.counts = EvalCounts::default();
        self.best = None;
    }
    /// Move the simplex storage out for in-place solver work, retaining only
    /// historical bookkeeping. Restore a complete record with `replace` before
    /// publication. Current readers report no evaluated record in the meantime.
    pub fn take_vertices(&mut self) -> (Vec<V>, Vec<F>) {
        (
            std::mem::take(&mut self.vertices),
            std::mem::take(&mut self.costs),
        )
    }
    pub(crate) fn sort(&mut self) {
        for i in 1..self.costs.len() {
            let mut j = i;
            while j > 0
                && !self.costs[j].is_nan()
                && (self.costs[j - 1].is_nan()
                    || self.costs[j] < self.costs[j - 1])
            {
                self.vertices.swap(j, j - 1);
                self.costs.swap(j, j - 1);
                j -= 1;
            }
        }
    }
}
impl<V: Clone, F: Scalar> State for SimplexProgress<V, F> {
    type Param = V;
    type Float = F;
    fn iter(&self) -> u64 {
        self.iter
    }
    fn increment_iter(&mut self) {
        self.iter += 1;
    }
    fn cost_evals(&self) -> u64 {
        self.counts.cost_evals
    }
    fn param(&self) -> &V {
        self.vertices
            .first()
            .expect("simplex storage has been moved out")
    }
    fn cost(&self) -> F {
        self.current().expect("simplex has not been evaluated").1
    }
    fn best_param(&self) -> &V {
        &self
            .best
            .as_ref()
            .expect("no incumbent has been selected")
            .param
    }
    fn best_cost(&self) -> F {
        self.best.as_ref().map_or(F::infinity(), |best| best.cost)
    }
    fn best_iter(&self) -> u64 {
        self.best.as_ref().map_or(0, |best| best.iter)
    }
    fn best_cost_evals(&self) -> u64 {
        self.best.as_ref().map_or(0, |best| best.counts.cost_evals)
    }
    fn update_best(&mut self) {
        if !self.vertices.is_empty() && self.vertices.len() == self.costs.len()
        {
            Incumbent::consider(
                &mut self.best,
                &self.vertices[0],
                self.costs[0],
                self.iter,
                self.counts,
            );
        }
    }
    fn reset_best(&mut self) {
        self.best = None;
    }
}
impl<V: Clone, F: Scalar> CountsMirror for SimplexProgress<V, F> {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.counts = *counts;
    }
}
impl<V: Clone, F: Scalar> SimplexState for SimplexProgress<V, F> {
    fn vertices(&self) -> &[V] {
        &self.vertices
    }
    fn costs(&self) -> &[F] {
        &self.costs
    }
}
impl<V: Clone, F: Scalar> EvaluatedState for SimplexProgress<V, F> {
    fn current_record(&self) -> Option<(&V, F)> {
        self.current()
    }
}
impl<V: Clone, F: Scalar> RawEvaluationState for SimplexProgress<V, F> {
    fn raw_counts(&self) -> &EvalCounts {
        &self.counts
    }
}
impl<V: Clone, F: Scalar> IncumbentState for SimplexProgress<V, F> {
    fn incumbent_record(&self) -> Option<IncumbentRef<'_, V, F>> {
        self.best.as_ref().map(|best| IncumbentRef {
            param: &best.param,
            cost: best.cost,
            iter: best.iter,
            counts: &best.counts,
        })
    }
}
impl<V: Clone, F: Scalar> ObjectiveIncumbentState for SimplexProgress<V, F> {}
