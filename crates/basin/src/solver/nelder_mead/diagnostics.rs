//! Native simplex trials, distinct from published recommendations.

use crate::{EvalCounts, Scalar};

/// Operation that generated a projected or unbounded trial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrialKind {
    /// Reflection of the worst vertex about the centroid.
    Reflection,
    /// Expansion along the reflected direction.
    Expansion,
    /// Contraction toward the reflected vertex.
    OutsideContraction,
    /// Contraction toward the worst vertex.
    InsideContraction,
    /// Shrink of the specified pre-sort vertex toward the best.
    Shrink(usize),
}

/// An attempted cost callback and its native selection decision.
///
/// Missing cost means the callback failed. Missing acceptance means selection
/// was interrupted, including reflection followed by a failed expansion.
/// Acceptance does not promise publication or a returned checkpoint.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Trial<V, F: Scalar = f64> {
    /// Monotone ordinal within a fresh solve.
    pub sequence: u64,
    /// Native operation, before the final simplex sort.
    pub kind: TrialKind,
    /// Actual callback coordinates after projection.
    pub point: V,
    /// Completed callback value, including non-finite values.
    pub cost: Option<F>,
    /// Whether the operation selected this trial for the simplex.
    pub accepted: Option<bool>,
    /// Authoritative counts immediately before the attempted callback.
    pub counts_before: EvalCounts,
    /// Authoritative counts immediately after the attempted callback.
    pub counts_after: EvalCounts,
}

pub(super) struct Trace<V, F: Scalar> {
    pub enabled: bool,
    pub sequence: u64,
    pub trials: Vec<Trial<V, F>>,
}

impl<V: Clone, F: Scalar> Trace<V, F> {
    #[inline]
    pub fn evaluate<P: crate::CostFunction<Param = V, Output = F>>(
        &mut self,
        problem: &mut crate::Problem<P>,
        point: &V,
        kind: TrialKind,
    ) -> Result<F, P::Error> {
        if !self.enabled {
            return problem.cost(point);
        }
        self.evaluate_traced(problem, point, kind)
    }

    // Keep optional record construction out of the ordinary evaluation path,
    // so the solver can inline cost callbacks without copying trace metadata.
    #[cold]
    fn evaluate_traced<P: crate::CostFunction<Param = V, Output = F>>(
        &mut self,
        problem: &mut crate::Problem<P>,
        point: &V,
        kind: TrialKind,
    ) -> Result<F, P::Error> {
        let before = *problem.counts();
        let result = problem.cost(point);
        self.sequence += 1;
        self.trials.push(Trial {
            sequence: self.sequence,
            kind,
            point: point.clone(),
            cost: result.as_ref().ok().copied(),
            accepted: None,
            counts_before: before,
            counts_after: *problem.counts(),
        });
        result
    }

    pub fn accept(&mut self, index: usize, accepted: bool) {
        if self.enabled {
            self.trials[index].accepted = Some(accepted);
        }
    }
}

impl<V, F: Scalar> Default for Trace<V, F> {
    fn default() -> Self {
        Self {
            enabled: false,
            sequence: 0,
            trials: Vec::new(),
        }
    }
}
