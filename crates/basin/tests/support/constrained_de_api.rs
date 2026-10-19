//! Executable API experiment, outside Basin's published surface.
//!
//! Scripted populations replace mutation and sampling. The retained `De`
//! configuration demonstrates an additive wrapper, not a numerical DE solver.
//! Only finite boxes and one nonlinear inequality block are exercised here.

use std::ops::{Index, IndexMut};

use basin::{
    BoxBounds, ConstraintAxis, ConstraintBlock, ConstraintError,
    ConstraintSolveError, Constraints, CostFunction, CountsMirror, De,
    EvalCounts, EvaluatedState, IncumbentRef, IncumbentState, InnerExecutor,
    MemeticInner, NoLinearConstraints, NoNonlinearBlock, NonlinearBlock,
    NonlinearBlocks, NonlinearInequalityValues, PartialResultPolicy, Problem,
    RawEvaluationState, Scalar, Solver, SolverStep, State, VectorLen,
};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PopulationMember<V, F: Scalar = f64> {
    param: V,
    cost: Option<F>,
    violations: Vec<F>,
}

impl<V, F: Scalar> PopulationMember<V, F> {
    pub fn new(param: V, cost: Option<F>, violations: Vec<F>) -> Self {
        Self {
            param,
            cost,
            violations,
        }
    }
    pub fn param(&self) -> &V {
        &self.param
    }
    pub fn cost(&self) -> Option<F> {
        self.cost
    }
    pub fn violations(&self) -> &[F] {
        &self.violations
    }
    pub fn feasible(&self) -> bool {
        self.violations.iter().all(|&v| v == F::zero())
    }
    fn total_violation(&self) -> F {
        self.violations.iter().copied().sum()
    }
}

/// Population observation that does not promise an objective for every member.
pub trait SelectedPopulationState: State {
    fn members(&self) -> Option<&[PopulationMember<Self::Param, Self::Float>]>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeError {
    Empty,
    Dimension,
    ViolationWidth,
    InvalidViolation,
    SelectedIndex,
    MissingObjective,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Selection<V, F: Scalar> {
    member: PopulationMember<V, F>,
    iter: u64,
    counts: EvalCounts,
}

/// Solvers publish selection events; only the executor stamps their metadata.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// with `F = f32` or `f64`. Replacement needs `VectorLen`; observation needs
/// only `Clone`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SelectedPopulationProgress<V, F: Scalar = f64> {
    members: Vec<PopulationMember<V, F>>,
    current: Option<usize>,
    best: Option<Selection<V, F>>,
    pending: Option<PopulationMember<V, F>>,
    iter: u64,
    counts: EvalCounts,
}

impl<V, F: Scalar> SelectedPopulationProgress<V, F> {
    pub fn empty() -> Self {
        Self {
            members: Vec::new(),
            current: None,
            best: None,
            pending: None,
            iter: 0,
            counts: EvalCounts::default(),
        }
    }
    pub fn current(&self) -> Option<&PopulationMember<V, F>> {
        self.current.map(|i| &self.members[i])
    }
    /// Initialization can borrow every seed without inventing evaluated records.
    pub fn parameters(&self) -> impl ExactSizeIterator<Item = &V> {
        self.members.iter().map(|m| &m.param)
    }
    pub fn best(&self) -> Option<&PopulationMember<V, F>> {
        self.best.as_ref().map(|s| &s.member)
    }
    pub fn reset(&mut self) {
        for member in &mut self.members {
            member.cost = None;
            member.violations.clear();
        }
        self.current = None;
        self.best = None;
        self.pending = None;
        self.iter = 0;
        self.counts = EvalCounts::default();
    }
}

impl<V: VectorLen, F: Scalar> SelectedPopulationProgress<V, F> {
    pub fn from_population(params: Vec<V>) -> Self {
        assert!(!params.is_empty());
        let n = params[0].vec_len();
        assert!(n > 0 && params.iter().all(|p| p.vec_len() == n));
        Self {
            members: params
                .into_iter()
                .map(|p| PopulationMember::new(p, None, vec![]))
                .collect(),
            ..Self::empty()
        }
    }
    /// Validate the complete record before touching a pending or published selection.
    pub fn replace(
        &mut self,
        members: Vec<PopulationMember<V, F>>,
        current: usize,
    ) -> Result<(), ShapeError> {
        let first = members.first().ok_or(ShapeError::Empty)?;
        let n = first.param.vec_len();
        if n == 0 || members.iter().any(|m| m.param.vec_len() != n) {
            return Err(ShapeError::Dimension);
        }
        if members
            .iter()
            .any(|m| m.violations.len() != first.violations.len())
        {
            return Err(ShapeError::ViolationWidth);
        }
        if members
            .iter()
            .any(|m| m.violations.iter().any(|v| v.is_nan() || *v < F::zero()))
        {
            return Err(ShapeError::InvalidViolation);
        }
        let selected = members.get(current).ok_or(ShapeError::SelectedIndex)?;
        if selected.cost.is_none()
            || members.iter().any(|m| m.feasible() && m.cost.is_none())
        {
            return Err(ShapeError::MissingObjective);
        }
        self.members = members;
        self.current = Some(current);
        Ok(())
    }
}

impl<V: Clone, F: Scalar> SelectedPopulationProgress<V, F> {
    pub fn select_current(&mut self) -> bool {
        let Some(member) = self.current() else {
            return false;
        };
        self.pending = Some(member.clone());
        true
    }
}

impl<V: Clone, F: Scalar> State for SelectedPopulationProgress<V, F> {
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
            .or_else(|| self.members.first())
            .expect("population has no seed")
            .param()
    }
    fn cost(&self) -> F {
        self.current()
            .and_then(|m| m.cost)
            .expect("representative has not been evaluated")
    }
    fn best_param(&self) -> &V {
        self.best().expect("no selected incumbent").param()
    }
    fn best_cost(&self) -> F {
        self.best().and_then(|m| m.cost).unwrap_or(F::infinity())
    }
    fn best_iter(&self) -> u64 {
        self.best.as_ref().map_or(0, |s| s.iter)
    }
    fn best_cost_evals(&self) -> u64 {
        self.best.as_ref().map_or(0, |s| s.counts.cost_evals)
    }
    fn update_best(&mut self) {
        if let Some(member) = self.pending.take() {
            self.best = Some(Selection {
                member,
                iter: self.iter,
                counts: self.counts,
            });
        }
    }
    fn reset_best(&mut self) {
        self.best = None;
        self.pending = None;
    }
}
impl<V: Clone, F: Scalar> CountsMirror for SelectedPopulationProgress<V, F> {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.counts = *counts;
    }
}
impl<V: Clone, F: Scalar> RawEvaluationState
    for SelectedPopulationProgress<V, F>
{
    fn raw_counts(&self) -> &EvalCounts {
        &self.counts
    }
}
impl<V: Clone, F: Scalar> EvaluatedState for SelectedPopulationProgress<V, F> {
    fn current_record(&self) -> Option<(&V, F)> {
        let m = self.current()?;
        Some((&m.param, m.cost?))
    }
}
impl<V: Clone, F: Scalar> IncumbentState for SelectedPopulationProgress<V, F> {
    fn incumbent_record(&self) -> Option<IncumbentRef<'_, V, F>> {
        let s = self.best.as_ref()?;
        Some(IncumbentRef::new(
            &s.member.param,
            s.member.cost?,
            s.iter,
            &s.counts,
        ))
    }
}
impl<V: Clone, F: Scalar> SelectedPopulationState
    for SelectedPopulationProgress<V, F>
{
    fn members(&self) -> Option<&[PopulationMember<V, F>]> {
        self.current.map(|_| self.members.as_slice())
    }
}

pub type PrototypeConstraints<V> = Constraints<
    BoxBounds<V>,
    NoLinearConstraints,
    NonlinearBlocks<NoNonlinearBlock, NonlinearBlock>,
>;

pub trait PrototypeProblem<V, F: Scalar>:
    CostFunction<Param = V, Output = F, Constraints = PrototypeConstraints<V>>
    + NonlinearInequalityValues<
        Param = V,
        Values = Vec<F>,
        Error = <Self as CostFunction>::Error,
    >
{
}
impl<P, V, F: Scalar> PrototypeProblem<V, F> for P where
    P: CostFunction<
            Param = V,
            Output = F,
            Constraints = PrototypeConstraints<V>,
        > + NonlinearInequalityValues<
            Param = V,
            Values = Vec<F>,
            Error = <Self as CostFunction>::Error,
        >
{
}

fn invalid<E>(error: ConstraintError) -> ConstraintSolveError<E> {
    ConstraintSolveError::InvalidConstraints(error)
}

fn dimension<E>(
    block: ConstraintBlock,
    expected: usize,
    actual: usize,
) -> Result<(), ConstraintSolveError<E>> {
    if expected != actual {
        return Err(invalid(ConstraintError::DimensionMismatch {
            block,
            axis: ConstraintAxis::Length,
            expected,
            actual,
        }));
    }
    Ok(())
}

fn bounds<P, V, F>(
    p: &P,
    n: usize,
) -> Result<(V, V), ConstraintSolveError<<P as CostFunction>::Error>>
where
    P: PrototypeProblem<V, F>,
    V: VectorLen + Clone + Index<usize, Output = F>,
    F: Scalar,
{
    let set = p.constraint_set();
    let lo = set.bounds().lower().copied().ok_or_else(|| {
        invalid(ConstraintError::MissingBound {
            side: basin::BoundSide::Lower,
        })
    })?;
    let hi = set.bounds().upper().copied().ok_or_else(|| {
        invalid(ConstraintError::MissingBound {
            side: basin::BoundSide::Upper,
        })
    })?;
    for b in [lo, hi] {
        dimension(ConstraintBlock::Bounds, n, b.vec_len())?;
    }
    for j in 0..n {
        if !lo[j].is_finite() || !hi[j].is_finite() {
            return Err(invalid(ConstraintError::NonFiniteInterval {
                coordinate: j,
            }));
        }
        if lo[j] > hi[j] {
            return Err(invalid(ConstraintError::InvertedBounds {
                coordinate: j,
            }));
        }
    }
    Ok((lo.clone(), hi.clone()))
}

/// Constraints use the current callback-request accounting contract.
pub fn evaluate<P, V, F>(
    problem: &mut Problem<P>,
    param: V,
    layout: (usize, usize),
) -> Result<
    PopulationMember<V, F>,
    ConstraintSolveError<<P as CostFunction>::Error>,
>
where
    P: PrototypeProblem<V, F>,
    V: VectorLen,
    F: Scalar,
{
    dimension(ConstraintBlock::Bounds, layout.0, param.vec_len())?;
    if problem
        .inner()
        .constraint_set()
        .nonlinear()
        .inequalities()
        .count()
        != layout.1
    {
        return Err(invalid(ConstraintError::LayoutChanged));
    }
    let values = if layout.1 == 0 {
        vec![]
    } else {
        // Attempted and malformed callbacks still consume one request.
        problem.counts_mut().residual_evals += 1;
        let values = problem.inner().nonlinear_inequalities(&param)?;
        dimension(
            ConstraintBlock::NonlinearInequalities,
            layout.1,
            values.len(),
        )?;
        values
    };
    // A non-finite constraint value cannot establish feasibility.
    let violations = values
        .into_iter()
        .map(|c| {
            if c.is_finite() {
                c.max(F::zero())
            } else {
                F::infinity()
            }
        })
        .collect();
    let mut member = PopulationMember::new(param, None, violations);
    if member.feasible() {
        member.cost = Some(problem.cost(&member.param)?);
    }
    Ok(member)
}

/// Componentwise trial replacement is distinct from representative ranking.
/// The comparison follows SciPy 1.16.2's `_accept_trial`, which implements
/// Lampinen's feasibility rule. The experiment does not reproduce its RNG.
pub fn accepts<V, F: Scalar>(
    trial: &PopulationMember<V, F>,
    original: &PopulationMember<V, F>,
) -> bool {
    assert_eq!(trial.violations.len(), original.violations.len());
    match (trial.feasible(), original.feasible()) {
        (true, true) => trial.cost.unwrap() <= original.cost.unwrap(),
        (true, false) => true,
        (false, true) => false,
        (false, false) => trial
            .violations
            .iter()
            .zip(&original.violations)
            .all(|(a, b)| a <= b),
    }
}

fn better<V, F: Scalar>(
    a: &PopulationMember<V, F>,
    b: &PopulationMember<V, F>,
) -> bool {
    match (a.feasible(), b.feasible()) {
        (true, true) => {
            let (a, b) = (a.cost.unwrap(), b.cost.unwrap());
            !a.is_nan() && (b.is_nan() || a < b)
        }
        (true, false) => true,
        (false, true) => false,
        (false, false) => a.total_violation() < b.total_violation(),
    }
}

fn publish<P, V, F>(
    problem: &mut Problem<P>,
    state: &mut SelectedPopulationProgress<V, F>,
    mut members: Vec<PopulationMember<V, F>>,
) -> Result<(), ConstraintSolveError<<P as CostFunction>::Error>>
where
    P: PrototypeProblem<V, F>,
    V: Clone + VectorLen,
    F: Scalar,
{
    let current = (1..members.len()).fold(0, |best, i| {
        if better(&members[i], &members[best]) {
            i
        } else {
            best
        }
    });
    if members[current].cost.is_none() {
        members[current].cost = Some(problem.cost(&members[current].param)?);
    }
    let cost = members[current].cost.unwrap();
    let select = !cost.is_nan()
        && cost != F::infinity()
        && state
            .best()
            .is_none_or(|old| better(&members[current], old));
    state
        .replace(members, current)
        .expect("prototype published malformed population");
    if select {
        state.select_current();
    }
    Ok(())
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Explicit constrained mode with scripted generations instead of mutation.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// with `F = f32` or `f64`. Requires `Clone`, `VectorLen`, and mutable indexing.
pub struct ConstrainedDePrototype<V, F: Scalar = f64> {
    _configuration: De<F>,
    script: Vec<Vec<V>>,
    position: usize,
    layout: (usize, usize),
}
impl<V, F: Scalar> ConstrainedDePrototype<V, F> {
    /// The first scripted population supplies seeds when the state is empty.
    pub fn new(de: De<F>, scripted_trials: Vec<Vec<V>>) -> Self {
        assert!(!scripted_trials.is_empty() && !scripted_trials[0].is_empty());
        Self {
            _configuration: de,
            script: scripted_trials,
            position: 0,
            layout: (0, 0),
        }
    }
    pub fn position(&self) -> usize {
        self.position
    }
}
impl<P, V, F> Solver<P, SelectedPopulationProgress<V, F>>
    for ConstrainedDePrototype<V, F>
where
    P: PrototypeProblem<V, F>,
    V: VectorLen + Clone + IndexMut<usize, Output = F>,
    F: Scalar,
{
    type Error = ConstraintSolveError<<P as CostFunction>::Error>;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SelectedPopulationProgress<V, F>,
    ) -> Result<SelectedPopulationProgress<V, F>, Self::Error> {
        state.reset();
        let seeds = if state.parameters().len() == 0 {
            self.script[0].clone()
        } else {
            state.parameters().cloned().collect()
        };
        let n = seeds[0].vec_len();
        assert!(n > 0);
        bounds(problem.inner(), n)?;
        self.layout = (
            n,
            problem
                .inner()
                .constraint_set()
                .nonlinear()
                .inequalities()
                .count(),
        );
        self.position = 1;
        let members = self.evaluate_repaired(problem, seeds)?;
        publish(problem, &mut state, members)?;
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SelectedPopulationProgress<V, F>,
    ) -> Result<SolverStep<SelectedPopulationProgress<V, F>>, Self::Error> {
        if self.position == self.script.len() {
            return Ok(SolverStep::stopped(
                state,
                basin::Termination::Application(basin::ApplicationStop::new(
                    "script exhausted",
                )),
            ));
        }
        let params = self.script[self.position].clone();
        assert_eq!(params.len(), state.members().unwrap().len());
        let trials = self.evaluate_repaired(problem, params)?;
        self.position += 1;
        let mut members = state.members().unwrap().to_vec();
        for (trial, original) in trials.into_iter().zip(&mut members) {
            if accepts(&trial, original) {
                *original = trial;
            }
        }
        publish(problem, &mut state, members)?;
        Ok(SolverStep::completed(state))
    }
}
impl<V: VectorLen + Clone + IndexMut<usize, Output = F>, F: Scalar>
    ConstrainedDePrototype<V, F>
{
    fn evaluate_repaired<P: PrototypeProblem<V, F>>(
        &self,
        problem: &mut Problem<P>,
        params: Vec<V>,
    ) -> Result<
        Vec<PopulationMember<V, F>>,
        ConstraintSolveError<<P as CostFunction>::Error>,
    > {
        let (lo, hi) = bounds(problem.inner(), self.layout.0)?;
        params
            .into_iter()
            .map(|mut param| {
                dimension(
                    ConstraintBlock::Bounds,
                    self.layout.0,
                    param.vec_len(),
                )?;
                for j in 0..self.layout.0 {
                    param[j] = param[j].max(lo[j]).min(hi[j]);
                }
                evaluate(problem, param, self.layout)
            })
            .collect()
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "I: serde::Serialize, V: serde::Serialize, F: serde::Serialize",
        deserialize = "I: serde::Deserialize<'de>, V: serde::Deserialize<'de>, F: serde::Deserialize<'de>"
    ))
)]
/// Local refinement that must accept the original constrained problem.
///
/// # Backends
///
/// The outer supports `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and
/// faer `Col<F>`, with `F = f32` or `f64`. The inner must support the same
/// parameter type and canonical constraint declaration.
pub struct ConstrainedDeInjectPrototype<I, V, F: Scalar = f64>
where
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    outer: ConstrainedDePrototype<V, F>,
    inner: InnerExecutor<I::State, I>,
    k: usize,
    refine_every: u64,
}
impl<I, V, F: Scalar> ConstrainedDeInjectPrototype<I, V, F>
where
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    pub fn with_inner_solver(
        outer: ConstrainedDePrototype<V, F>,
        inner: I,
    ) -> Self {
        Self {
            outer,
            inner: InnerExecutor::new(inner).max_iter(1),
            k: 1,
            refine_every: 1,
        }
    }
    pub fn with_k(mut self, k: usize) -> Self {
        assert!(k > 0);
        self.k = k;
        self
    }
    pub fn with_refine_every(mut self, n: u64) -> Self {
        assert!(n > 0);
        self.refine_every = n;
        self
    }
    pub fn with_inner_max_iter(mut self, n: u64) -> Self {
        self.inner = self.inner.max_iter(n);
        self
    }
}
impl<P, I, V, F> Solver<P, SelectedPopulationProgress<V, F>>
    for ConstrainedDeInjectPrototype<I, V, F>
where
    P: PrototypeProblem<V, F>,
    V: VectorLen + Clone + IndexMut<usize, Output = F>,
    F: Scalar,
    I: MemeticInner<V, F>
        + Solver<
            P,
            I::State,
            Error = ConstraintSolveError<<P as CostFunction>::Error>,
        >,
    I::State: State<Param = V, Float = F> + CountsMirror,
{
    type Error = ConstraintSolveError<<P as CostFunction>::Error>;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: SelectedPopulationProgress<V, F>,
    ) -> Result<SelectedPopulationProgress<V, F>, Self::Error> {
        self.outer.init(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: SelectedPopulationProgress<V, F>,
    ) -> Result<SolverStep<SelectedPopulationProgress<V, F>>, Self::Error> {
        let step = self.outer.next_iter(problem, state)?;
        if step.termination.is_some()
            || step.state.iter() % self.refine_every != 0
        {
            return Ok(step);
        }
        let mut state = step.state;
        let members = state.members().unwrap();
        let mut order: Vec<_> = (0..members.len()).collect();
        order.sort_by(|&a, &b| {
            if better(&members[a], &members[b]) {
                std::cmp::Ordering::Less
            } else if better(&members[b], &members[a]) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        for i in order.into_iter().take(self.k) {
            let seed = self
                .inner
                .solver_mut()
                .seed_scaled(state.members().unwrap()[i].param(), F::one());
            let result = self.inner.run(problem, seed)?;
            if let Some(stop) = result
                .report
                .into_outer_termination(PartialResultPolicy::Consume)
            {
                return Ok(SolverStep::stopped(state, stop));
            }
            // Rechecking after box repair prevents an inner result from bypassing selection.
            let refined = self
                .outer
                .evaluate_repaired(problem, vec![result.state.param().clone()])?
                .remove(0);
            if accepts(&refined, &state.members().unwrap()[i]) {
                let mut members = state.members().unwrap().to_vec();
                members[i] = refined;
                publish(problem, &mut state, members)?;
            }
        }
        Ok(SolverStep::completed(state))
    }
}
