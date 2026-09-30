//! Shared population records, independent of distribution and per-member models.

use super::progress::Incumbent;
use super::{
    CountsMirror, EvaluatedState, IncumbentRef, IncumbentState,
    ObjectiveIncumbentState, PopulationState, RawEvaluationState, State,
};
use crate::core::math::{Scalar, VectorLen};
use crate::core::problem::EvalCounts;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Current<V, F> {
    Seed(V),
    Member(usize),
    Representative(V, F),
}

/// Observable population members, matching costs, and historical progress.
///
/// Members live directly in this state; solvers own distributions, per-member
/// models, RNGs, and working buffers. [`replace`](Self::replace) validates the
/// entire replacement before changing anything and preserves member order so
/// solver-owned information indexed by member stays aligned. The current
/// record is the lowest-cost member unless [`publish_representative`](Self::publish_representative)
/// supplies another evaluated point, such as a distribution mean or a swarm's
/// global best. NaNs rank after every other cost;
/// equal costs retain the first member.
///
/// The executor considers both members and the representative for strict objective
/// improvements at publication boundaries. It retains a matching historical
/// point, cost, iteration, and all six evaluation counts even when the entire
/// population changes. NaN and positive infinity never establish an incumbent;
/// negative infinity can be retained without establishing unboundedness.
///
/// [`empty`](Self::empty) lets the solver generate its initial population.
/// [`from_population`](Self::from_population) supplies explicit unevaluated
/// members; [`from_point`](Self::from_point) supplies a representative seed
/// for distribution-based solvers. Construction supplies no current cost or incumbent. Use
/// [`current`](Self::current), [`best`](Self::best), and
/// [`evaluated_members`](Self::evaluated_members) to check availability.
/// Population size and algorithm settings belong to the solver.
///
/// Fresh [`reset`](Self::reset) preserves members and any representative as
/// unevaluated seeds and clears all
/// evaluated records and bookkeeping. Exact continuation requires a checkpoint
/// containing the solver, state, and counts. With `serde`, this state serializes
/// when its parameter and scalar types do; it contains no solver machinery.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`, with
/// `F = f32` or `f64`. Construction from members and replacement require
/// [`VectorLen`]; observation and bookkeeping require only `Clone`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PopulationProgress<V, F: Scalar = f64> {
    pub(crate) candidates: Vec<V>,
    pub(crate) costs: Vec<F>,
    current: Option<Current<V, F>>,
    iter: u64,
    counts: EvalCounts,
    best: Option<Incumbent<V, F>>,
}

/// A proposed population or representative has inconsistent dimensions or availability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PopulationShapeError {
    /// An evaluated population requires at least one member.
    EmptyPopulation,
    /// Members must have at least one coordinate.
    EmptyParameter,
    /// All members and the representative must have the same coordinate count.
    DimensionMismatch {
        /// Member index, or the population length for an inconsistent representative.
        index: usize,
        /// Coordinate count of the first member.
        expected: usize,
        /// Coordinate count of the proposed member or representative.
        actual: usize,
    },
    /// Each member requires one cost.
    CostCountMismatch {
        /// Number of supplied members.
        members: usize,
        /// Number of supplied costs.
        costs: usize,
    },
    /// A representative can only accompany an evaluated population.
    UnevaluatedPopulation,
}
impl std::fmt::Display for PopulationShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPopulation => {
                write!(f, "population requires at least one member")
            }
            Self::EmptyParameter => {
                write!(f, "population members require at least one coordinate")
            }
            Self::DimensionMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "population member or representative {index} has length {actual}, expected {expected}"
            ),
            Self::CostCountMismatch { members, costs } => write!(
                f,
                "{members} population members require {members} costs, got {costs}"
            ),
            Self::UnevaluatedPopulation => {
                write!(f, "population has not been evaluated")
            }
        }
    }
}
impl std::error::Error for PopulationShapeError {}

impl<V: VectorLen, F: Scalar> PopulationProgress<V, F> {
    fn validate(members: &[V]) -> Result<(), PopulationShapeError> {
        let dimension = members
            .first()
            .ok_or(PopulationShapeError::EmptyPopulation)?
            .vec_len();
        if dimension == 0 {
            return Err(PopulationShapeError::EmptyParameter);
        }
        for (index, member) in members.iter().enumerate().skip(1) {
            if member.vec_len() != dimension {
                return Err(PopulationShapeError::DimensionMismatch {
                    index,
                    expected: dimension,
                    actual: member.vec_len(),
                });
            }
        }
        Ok(())
    }

    /// Supply an unevaluated representative seed, such as a CMA distribution mean.
    ///
    /// # Panics
    ///
    /// Panics if the point is empty.
    pub fn from_point(point: V) -> Self {
        assert!(
            point.vec_len() > 0,
            "population seed requires a nonempty point"
        );
        Self {
            current: Some(Current::Seed(point)),
            ..Self::empty()
        }
    }

    /// Supply explicit unevaluated members for the solver's initialization.
    ///
    /// # Panics
    ///
    /// Panics for an empty population or unequal or zero coordinate lengths.
    pub fn from_population(candidates: Vec<V>) -> Self {
        Self::validate(&candidates).expect("invalid initial population");
        Self {
            candidates,
            ..Self::empty()
        }
    }

    /// Replace every member and cost together, retaining member order.
    /// The lowest-cost member becomes current, replacing any published representative.
    /// Errors preserve the entire old state. Valid replacements may change
    /// population size or parameter dimension. Counters and the incumbent
    /// remain unchanged until the executor publishes this record.
    pub fn replace(
        &mut self,
        candidates: Vec<V>,
        costs: Vec<F>,
    ) -> Result<(), PopulationShapeError> {
        Self::validate(&candidates)?;
        if candidates.len() != costs.len() {
            return Err(PopulationShapeError::CostCountMismatch {
                members: candidates.len(),
                costs: costs.len(),
            });
        }
        self.candidates = candidates;
        self.costs = costs;
        self.select_best_member();
        Ok(())
    }

    /// Publish an evaluated representative as the current record without
    /// changing any member. Its dimension must match the evaluated population.
    /// The executor still considers both the representative and members for the incumbent.
    pub fn publish_representative(
        &mut self,
        point: V,
        cost: F,
    ) -> Result<(), PopulationShapeError> {
        self.validate_representative(&point)?;
        self.current = Some(Current::Representative(point, cost));
        Ok(())
    }

    fn validate_representative(
        &self,
        point: &V,
    ) -> Result<(), PopulationShapeError> {
        let (members, _) = self
            .evaluated_members()
            .ok_or(PopulationShapeError::UnevaluatedPopulation)?;
        let expected = members[0].vec_len();
        if point.vec_len() != expected {
            return Err(PopulationShapeError::DimensionMismatch {
                index: members.len(),
                expected,
                actual: point.vec_len(),
            });
        }
        Ok(())
    }

    pub(crate) fn publish_representative_from(
        &mut self,
        point: &V,
        cost: F,
    ) -> Result<(), PopulationShapeError>
    where
        V: Clone,
    {
        self.validate_representative(point)?;
        // Reuse the observable record when the solver retains the underlying model point.
        if let Some(Current::Representative(current, current_cost)) =
            &mut self.current
        {
            current.clone_from(point);
            *current_cost = cost;
        } else {
            self.current = Some(Current::Representative(point.clone(), cost));
        }
        Ok(())
    }
}

impl<V, F: Scalar> PopulationProgress<V, F> {
    /// An unevaluated container whose initial members will be supplied by the solver.
    pub fn empty() -> Self {
        Self {
            candidates: Vec::new(),
            costs: Vec::new(),
            current: None,
            iter: 0,
            counts: EvalCounts::default(),
            best: None,
        }
    }
    /// Borrow authoritative members in their stored order, including unevaluated seeds.
    pub fn candidates(&self) -> &[V] {
        &self.candidates
    }
    /// Borrow matching costs, or an empty slice before evaluation.
    pub fn costs(&self) -> &[F] {
        &self.costs
    }
    /// Borrow complete member/cost arrays, or `None` before evaluation.
    pub fn evaluated_members(&self) -> Option<(&[V], &[F])> {
        (!self.candidates.is_empty()
            && self.candidates.len() == self.costs.len())
        .then_some((&self.candidates, &self.costs))
    }
    /// Borrow the evaluated representative, or `None` before evaluation.
    pub fn current(&self) -> Option<(&V, F)> {
        match self.current.as_ref()? {
            Current::Seed(_) => None,
            Current::Member(index) => {
                Some((&self.candidates[*index], self.costs[*index]))
            }
            Current::Representative(point, cost) => Some((point, *cost)),
        }
    }
    /// Borrow the historical objective incumbent, if one exists.
    pub fn best(&self) -> Option<(&V, F)> {
        self.best.as_ref().map(|best| (&best.param, best.cost))
    }
    /// All raw evaluation categories at the latest publication boundary.
    pub fn counts(&self) -> &EvalCounts {
        &self.counts
    }
    /// All raw evaluation categories at the incumbent's selection boundary.
    pub fn best_counts(&self) -> Option<&EvalCounts> {
        self.best.as_ref().map(|best| &best.counts)
    }
    /// Preserve members and the representative as unevaluated seeds, clearing all
    /// evaluated progress and bookkeeping.
    pub fn reset(&mut self) {
        self.costs.clear();
        self.current = match self.current.take() {
            Some(Current::Seed(point) | Current::Representative(point, _)) => {
                Some(Current::Seed(point))
            }
            _ => None,
        };
        self.iter = 0;
        self.counts = EvalCounts::default();
        self.best = None;
    }
    /// Move member storage out for in-place solver work, retaining historical
    /// bookkeeping. Restore a complete record with `replace` before publication.
    /// Current and evaluated-member readers return `None` in the meantime.
    pub fn take_members(&mut self) -> (Vec<V>, Vec<F>) {
        self.current = None;
        (
            std::mem::take(&mut self.candidates),
            std::mem::take(&mut self.costs),
        )
    }
    pub(crate) fn reset_segment_iter(&mut self) {
        self.iter = 0;
    }
    pub(crate) fn seed(&self) -> Option<&V> {
        match self.current.as_ref() {
            Some(Current::Seed(point) | Current::Representative(point, _)) => {
                Some(point)
            }
            Some(Current::Member(index)) => self.candidates.get(*index),
            None => self.candidates.first(),
        }
    }
    pub(crate) fn select_best_member(&mut self) {
        debug_assert_eq!(self.candidates.len(), self.costs.len());
        self.current = best_index(&self.costs).map(Current::Member);
    }
}
impl<V, F: Scalar> Default for PopulationProgress<V, F> {
    fn default() -> Self {
        Self::empty()
    }
}

fn precedes<F: Scalar>(a: F, b: F) -> bool {
    !a.is_nan() && (b.is_nan() || a < b)
}
fn best_index<F: Scalar>(costs: &[F]) -> Option<usize> {
    (!costs.is_empty()).then(|| {
        (1..costs.len()).fold(0, |best, i| {
            if precedes(costs[i], costs[best]) {
                i
            } else {
                best
            }
        })
    })
}

impl<V: Clone, F: Scalar> State for PopulationProgress<V, F> {
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
        self.current()
            .map(|record| record.0)
            .or_else(|| self.seed())
            .expect("population has no members")
    }
    fn cost(&self) -> F {
        self.current().expect("population has not been evaluated").1
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
        let Some(current) = &self.current else {
            return;
        };
        let index = match current {
            Current::Seed(_) => return,
            Current::Member(index) => *index,
            Current::Representative(_, _) => best_index(&self.costs)
                .expect("representative requires evaluated members"),
        };
        let mut point = &self.candidates[index];
        let mut cost = self.costs[index];
        if let Current::Representative(representative, representative_cost) =
            current
        {
            if precedes(*representative_cost, cost) {
                point = representative;
                cost = *representative_cost;
            }
        }
        Incumbent::consider(
            &mut self.best,
            point,
            cost,
            self.iter,
            self.counts,
        );
    }
    fn reset_best(&mut self) {
        self.best = None;
    }
}
impl<V: Clone, F: Scalar> CountsMirror for PopulationProgress<V, F> {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.counts = *counts;
    }
}
impl<V: Clone, F: Scalar> PopulationState for PopulationProgress<V, F> {
    fn candidates(&self) -> &[V] {
        &self.candidates
    }
    fn costs(&self) -> &[F] {
        &self.costs
    }
}
impl<V: Clone, F: Scalar> EvaluatedState for PopulationProgress<V, F> {
    fn current_record(&self) -> Option<(&V, F)> {
        self.current()
    }
}
impl<V: Clone, F: Scalar> RawEvaluationState for PopulationProgress<V, F> {
    fn raw_counts(&self) -> &EvalCounts {
        &self.counts
    }
}
impl<V: Clone, F: Scalar> IncumbentState for PopulationProgress<V, F> {
    fn incumbent_record(&self) -> Option<IncumbentRef<'_, V, F>> {
        self.best.as_ref().map(|best| IncumbentRef {
            param: &best.param,
            cost: best.cost,
            iter: best.iter,
            counts: &best.counts,
        })
    }
}
impl<V: Clone, F: Scalar> ObjectiveIncumbentState for PopulationProgress<V, F> {}
