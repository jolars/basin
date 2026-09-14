use basin::core::math::{Scalar, VectorLen};
use basin::core::state::{CountsMirror, GradientState};
use basin::{EvalCounts, State};

#[derive(Clone, Debug, PartialEq)]
pub struct Best<V, F: Scalar = f64> {
    pub param: V,
    pub cost: F,
    pub iter: u64,
    pub counts: EvalCounts,
}

#[derive(Clone, Debug)]
struct Core<V, F: Scalar> {
    param: Option<V>,
    cost: Option<F>,
    best: Option<Best<V, F>>,
    counts: EvalCounts,
    iter: u64,
    changed: bool,
}

impl<V: Clone, F: Scalar> Core<V, F> {
    fn new(param: Option<V>) -> Self {
        Self {
            param,
            cost: None,
            best: None,
            counts: EvalCounts::default(),
            iter: 0,
            changed: false,
        }
    }

    fn reset(&mut self) {
        self.cost = None;
        self.best = None;
        self.counts = EvalCounts::default();
        self.iter = 0;
        self.changed = false;
    }

    fn improves(&self, cost: F) -> bool {
        !cost.is_nan()
            && cost != F::infinity()
            && self.best.as_ref().is_none_or(|best| cost < best.cost)
    }

    fn retain(&mut self, param: V, cost: F) {
        self.best = Some(Best {
            param,
            cost,
            iter: self.iter,
            counts: self.counts,
        });
        self.changed = true;
    }

    fn consider_current(&mut self) {
        if let Some(cost) = self.cost {
            if self.improves(cost) {
                self.retain(self.param.as_ref().unwrap().clone(), cost);
            }
        }
    }
}

pub trait Progress: State + CountsMirror {
    fn is_ready(&self) -> bool;
    fn counts(&self) -> &EvalCounts;
    fn new_best(&self) -> bool;
    fn fresh(&mut self);

    fn publish(&mut self, completed: bool, counts: EvalCounts) {
        self.mirror(&counts);
        if completed {
            self.increment_iter();
        }
        self.update_best();
    }
}

/// Only states whose incumbents admit objective-only controls implement this.
pub trait ObjectiveBest: Progress {
    fn objective_best(&self) -> Option<Self::Float>;
}

#[derive(Debug, PartialEq)]
pub enum ShapeError {
    Empty,
    Dimension,
    Length,
}

#[derive(Clone, Debug)]
pub struct PointState<V, F: Scalar = f64> {
    core: Core<V, F>,
}

impl<V: Clone, F: Scalar> PointState<V, F> {
    pub fn new(param: V) -> Self {
        Self {
            core: Core::new(Some(param)),
        }
    }

    pub fn current(&self) -> Option<(&V, F)> {
        Some((self.core.param.as_ref()?, self.core.cost?))
    }

    pub fn best(&self) -> Option<&Best<V, F>> {
        self.core.best.as_ref()
    }

    pub fn replace(&mut self, param: V, cost: F) {
        self.core.param = Some(param);
        self.core.cost = Some(cost);
    }

    fn reset(&mut self) {
        self.core.reset();
    }

    fn ready(&self) -> bool {
        self.current().is_some()
    }

    fn record_best(&mut self) {
        self.core.changed = false;
        self.core.consider_current();
    }
}

#[derive(Clone, Debug)]
pub struct FirstOrderState<V, F: Scalar = f64> {
    core: Core<V, F>,
    gradient: Option<V>,
}

impl<V: Clone + VectorLen, F: Scalar> FirstOrderState<V, F> {
    pub fn new(param: V) -> Self {
        Self {
            core: Core::new(Some(param)),
            gradient: None,
        }
    }

    pub fn current(&self) -> Option<(&V, F, &V)> {
        Some((
            self.core.param.as_ref()?,
            self.core.cost?,
            self.gradient.as_ref()?,
        ))
    }

    pub fn replace(
        &mut self,
        param: V,
        cost: F,
        gradient: V,
    ) -> Result<(), ShapeError> {
        if param.vec_len() != gradient.vec_len() {
            return Err(ShapeError::Dimension);
        }
        self.core.param = Some(param);
        self.core.cost = Some(cost);
        self.gradient = Some(gradient);
        Ok(())
    }

    pub fn take_current(&mut self) -> Option<(V, F, V)> {
        self.current()?;
        Some((
            self.core.param.take()?,
            self.core.cost.take()?,
            self.gradient.take()?,
        ))
    }

    fn reset(&mut self) {
        self.core.reset();
        self.gradient = None;
    }

    fn ready(&self) -> bool {
        self.current().is_some()
    }

    fn record_best(&mut self) {
        self.core.changed = false;
        self.core.consider_current();
    }
}

#[derive(Clone, Debug)]
pub struct Members<V, F: Scalar = f64, const SIMPLEX: bool = false> {
    core: Core<V, F>,
    points: Vec<V>,
    costs: Vec<F>,
}

pub type PopulationState<V, F = f64> = Members<V, F, false>;
pub type SimplexState<V, F = f64> = Members<V, F, true>;

impl<V: Clone + VectorLen, F: Scalar, const SIMPLEX: bool>
    Members<V, F, SIMPLEX>
{
    pub fn new(points: Vec<V>) -> Result<Self, ShapeError> {
        Self::validate(&points)?;
        Ok(Self {
            core: Core::new(points.first().cloned()),
            points,
            costs: Vec::new(),
        })
    }

    fn validate(points: &[V]) -> Result<(), ShapeError> {
        let n = points.first().ok_or(ShapeError::Empty)?.vec_len();
        if points.iter().any(|point| point.vec_len() != n) {
            return Err(ShapeError::Dimension);
        }
        if SIMPLEX && (n == 0 || points.len() != n + 1) {
            return Err(ShapeError::Length);
        }
        Ok(())
    }

    pub fn points(&self) -> &[V] {
        &self.points
    }
    pub fn costs(&self) -> &[F] {
        &self.costs
    }

    pub fn take_members(&mut self) -> (Vec<V>, Vec<F>) {
        self.core.cost = None;
        (
            std::mem::take(&mut self.points),
            std::mem::take(&mut self.costs),
        )
    }

    pub fn replace_members(
        &mut self,
        mut points: Vec<V>,
        mut costs: Vec<F>,
    ) -> Result<(), ShapeError> {
        Self::validate(&points)?;
        if points.len() != costs.len() {
            return Err(ShapeError::Length);
        }
        // Simplexes are ordered by cost. Populations retain member order so
        // solver-owned information indexed by member does not become stale.
        if SIMPLEX {
            for i in 1..costs.len() {
                let mut j = i;
                while j > 0 && precedes(costs[j], costs[j - 1]) {
                    points.swap(j, j - 1);
                    costs.swap(j, j - 1);
                    j -= 1;
                }
            }
        }
        let current = (1..costs.len()).fold(0, |best, i| {
            if precedes(costs[i], costs[best]) {
                i
            } else {
                best
            }
        });
        self.core.param = Some(points[current].clone());
        self.core.cost = Some(costs[current]);
        self.points = points;
        self.costs = costs;
        Ok(())
    }

    fn reset(&mut self) {
        self.core.reset();
        self.costs.clear();
    }

    fn ready(&self) -> bool {
        !self.points.is_empty()
            && self.points.len() == self.costs.len()
            && self.core.cost.is_some()
    }

    fn record_best(&mut self) {
        self.core.changed = false;
        for (point, &cost) in self.points.iter().zip(&self.costs) {
            if self.core.improves(cost) {
                self.core.retain(point.clone(), cost);
            }
        }
        self.core.consider_current();
    }
}

fn precedes<F: Scalar>(a: F, b: F) -> bool {
    !a.is_nan() && (b.is_nan() || a < b)
}

impl<V: Clone + VectorLen, F: Scalar> PopulationState<V, F> {
    pub fn publish_mean(&mut self, mean: V, cost: F) -> Result<(), ShapeError> {
        let first = self.points.first().ok_or(ShapeError::Empty)?;
        if mean.vec_len() != first.vec_len() {
            return Err(ShapeError::Dimension);
        }
        self.core.param = Some(mean);
        self.core.cost = Some(cost);
        Ok(())
    }
}

/// Explicit solver selection, without the objective-ordering capability.
#[derive(Clone, Debug)]
pub struct SelectedState<V, F: Scalar = f64> {
    core: Core<V, F>,
    selection: Option<u64>,
    pending: bool,
    violation: F,
}

impl<V: Clone, F: Scalar> SelectedState<V, F> {
    pub fn new(param: V) -> Self {
        Self {
            core: Core::new(Some(param)),
            selection: None,
            pending: false,
            violation: F::infinity(),
        }
    }

    pub fn select(&mut self, id: u64, param: V, cost: F, violation: F) {
        self.pending |= self.selection != Some(id);
        self.selection = Some(id);
        self.violation = violation;
        self.core.param = Some(param);
        self.core.cost = Some(cost);
    }

    pub fn violation(&self) -> F {
        self.violation
    }

    fn ready(&self) -> bool {
        self.core.param.is_some() && self.core.cost.is_some()
    }

    fn reset(&mut self) {
        self.core.reset();
        self.selection = None;
        self.pending = false;
        self.violation = F::infinity();
    }

    fn record_best(&mut self) {
        self.core.changed = false;
        if std::mem::take(&mut self.pending) {
            self.core.retain(
                self.core.param.as_ref().unwrap().clone(),
                self.core.cost.unwrap(),
            );
        }
    }
}

macro_rules! state_impl {
    ($ty:ty, [$($generics:tt)*]) => {
        impl<$($generics)*> State for $ty where V: Clone + VectorLen, F: Scalar {
            type Param = V;
            type Float = F;
            fn iter(&self) -> u64 { self.core.iter }
            fn increment_iter(&mut self) { self.core.iter += 1; }
            fn cost_evals(&self) -> u64 { self.core.counts.cost_evals }
            fn param(&self) -> &V { self.core.param.as_ref().expect("point unavailable during update") }
            fn cost(&self) -> F { self.core.cost.expect("point has not been evaluated") }
            fn best_param(&self) -> &V { &self.core.best.as_ref().expect("no incumbent").param }
            fn best_cost(&self) -> F { self.core.best.as_ref().map_or(F::infinity(), |b| b.cost) }
            fn best_iter(&self) -> u64 { self.core.best.as_ref().map_or(0, |b| b.iter) }
            fn best_cost_evals(&self) -> u64 { self.core.best.as_ref().map_or(0, |b| b.counts.cost_evals) }
            fn update_best(&mut self) { self.record_best(); }
            fn reset_best(&mut self) { self.core.best = None; self.core.changed = false; }
        }
        impl<$($generics)*> CountsMirror for $ty where V: Clone + VectorLen, F: Scalar {
            fn mirror(&mut self, counts: &EvalCounts) { self.core.counts = *counts; }
        }
        impl<$($generics)*> Progress for $ty where V: Clone + VectorLen, F: Scalar {
            fn is_ready(&self) -> bool { self.ready() }
            fn counts(&self) -> &EvalCounts { &self.core.counts }
            fn new_best(&self) -> bool { self.core.changed }
            fn fresh(&mut self) { self.reset(); }
        }
    };
}

state_impl!(PointState<V, F>, [V, F]);
state_impl!(FirstOrderState<V, F>, [V, F]);
state_impl!(Members<V, F, SIMPLEX>, [V, F, const SIMPLEX: bool]);
state_impl!(SelectedState<V, F>, [V, F]);

macro_rules! objective_impl {
    ($ty:ty, [$($generics:tt)*]) => {
        impl<$($generics)*> ObjectiveBest for $ty where V: Clone + VectorLen, F: Scalar {
            fn objective_best(&self) -> Option<F> { self.core.best.as_ref().map(|b| b.cost) }
        }
    };
}
objective_impl!(PointState<V, F>, [V, F]);
objective_impl!(FirstOrderState<V, F>, [V, F]);
objective_impl!(Members<V, F, SIMPLEX>, [V, F, const SIMPLEX: bool]);

impl<V: Clone + VectorLen, F: Scalar> GradientState for FirstOrderState<V, F> {
    fn gradient(&self) -> Option<&V> {
        self.gradient.as_ref()
    }
    fn gradient_evals(&self) -> u64 {
        self.core.counts.gradient_evals
    }
    fn best_gradient_evals(&self) -> u64 {
        self.core
            .best
            .as_ref()
            .map_or(0, |b| b.counts.gradient_evals)
    }
}

impl<V: Clone + VectorLen, F: Scalar> basin::core::state::SimplexState
    for SimplexState<V, F>
{
    fn vertices(&self) -> &[V] {
        &self.points
    }
    fn costs(&self) -> &[F] {
        &self.costs
    }
}

impl<V: Clone + VectorLen, F: Scalar> basin::core::state::PopulationState
    for PopulationState<V, F>
{
    fn candidates(&self) -> &[V] {
        &self.points
    }
    fn costs(&self) -> &[F] {
        &self.costs
    }
}
