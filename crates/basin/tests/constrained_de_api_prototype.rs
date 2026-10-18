//! Constrained-DE API experiment. No prototype types are exported by Basin.
//! Run with `cargo test -p basin --test constrained_de_api_prototype`.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/constrained_de_api.rs"]
mod prototype;

use std::{
    cell::{Cell, RefCell},
    ops::{Index, IndexMut},
    rc::Rc,
};

use basin::{
    BoxBounds, ConstraintDescription, ConstraintSolveError, Constraints,
    CostFunction, CountsMirror, De, EvalCounts, EvaluatedState, EvaluationKind,
    Executor, IncumbentState, InitialState, MemeticInner, NumericalFailure,
    PointState, Problem, ProblemConstraints, RawEvaluationState, Scalar,
    Solver, SolverStep, State, Termination, TerminationStage, VectorLen,
    WarmStart,
};
use prototype::{
    ConstrainedDeInjectPrototype, ConstrainedDePrototype, PopulationMember,
    PrototypeConstraints, PrototypeProblem, SelectedPopulationProgress,
    SelectedPopulationState, ShapeError, accepts, evaluate,
};

type Trace<F> = Rc<RefCell<Vec<(u8, Vec<F>)>>>;

#[derive(Debug, PartialEq, Eq)]
enum CallbackError {
    Cost,
    Constraint,
}

struct Fixture<V, F: Scalar = f64> {
    lower: V,
    upper: V,
    count: Rc<Cell<usize>>,
    trace: Trace<F>,
    fail_cost: Option<F>,
    fail_constraint: Option<F>,
    override_values: Option<Vec<F>>,
}
impl<V, F: Scalar> Fixture<V, F> {
    fn new(make: impl Fn(&[F]) -> V) -> Self {
        Self {
            lower: make(&[num(-5.0), num(-5.0)]),
            upper: make(&[num(5.0), num(5.0)]),
            count: Rc::new(Cell::new(2)),
            trace: Rc::default(),
            fail_cost: None,
            fail_constraint: None,
            override_values: None,
        }
    }
}
fn num<F: Scalar>(v: f64) -> F {
    F::from_f64(v).unwrap()
}
#[cfg(feature = "serde")]
fn serde_round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
) -> T {
    postcard::from_bytes(&postcard::to_allocvec(value).unwrap()).unwrap()
}
impl<V, F: Scalar> ProblemConstraints for Fixture<V, F> {
    type Constraints = PrototypeConstraints<V>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as ConstraintDescription>::Borrowed<'_> {
        Constraints::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
            .with_nonlinear_inequalities(self.count.get())
    }
}
impl<V: VectorLen + Index<usize, Output = F>, F: Scalar> CostFunction
    for Fixture<V, F>
{
    type Param = V;
    type Output = F;
    type Error = CallbackError;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        self.trace
            .borrow_mut()
            .push((0, (0..x.vec_len()).map(|i| x[i]).collect()));
        if self.fail_cost == Some(x[0]) {
            return Err(CallbackError::Cost);
        }
        Ok((0..x.vec_len()).map(|i| x[i] * x[i]).sum())
    }
}
impl<V: VectorLen + Index<usize, Output = F>, F: Scalar>
    basin::NonlinearInequalityValues for Fixture<V, F>
{
    type Param = V;
    type Values = Vec<F>;
    type Error = CallbackError;
    fn nonlinear_inequalities(
        &self,
        x: &V,
    ) -> Result<Vec<F>, ConstraintSolveError<CallbackError>> {
        self.trace
            .borrow_mut()
            .push((1, (0..x.vec_len()).map(|i| x[i]).collect()));
        if self.fail_constraint == Some(x[0]) {
            return Err(CallbackError::Constraint.into());
        }
        Ok(self
            .override_values
            .clone()
            .unwrap_or_else(|| vec![F::one() - x[0], x[1] - F::one()]))
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct ScriptedInner<V, F: Scalar = f64> {
    returned: V,
    stop_on_seed: Option<F>,
    stop: Option<Termination<F>>,
}
impl<V, F: Scalar> ScriptedInner<V, F> {
    fn new(returned: V) -> Self {
        Self {
            returned,
            stop_on_seed: None,
            stop: None,
        }
    }
}
impl<V: Clone, F: Scalar> InitialState<V> for ScriptedInner<V, F> {
    type State = PointState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        PointState::new(x.clone())
    }
}
impl<V: Clone, F: Scalar> WarmStart<V> for ScriptedInner<V, F> {}
impl<V: Clone, F: Scalar> MemeticInner<V, F> for ScriptedInner<V, F> {}
impl<P, V, F> Solver<P, PointState<V, F>> for ScriptedInner<V, F>
where
    P: PrototypeProblem<V, F>,
    V: VectorLen + Clone + Index<usize, Output = F>,
    F: Scalar,
{
    type Error = ConstraintSolveError<<P as CostFunction>::Error>;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        state.reset();
        let n = state.param().vec_len();
        let m = problem
            .inner()
            .constraint_set()
            .nonlinear()
            .inequalities()
            .count();
        let member = evaluate(problem, state.param().clone(), (n, m))?;
        let cost = match member.cost() {
            Some(c) => c,
            None => problem.cost(member.param())?,
        };
        state.replace(member.param().clone(), cost);
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<SolverStep<PointState<V, F>>, Self::Error> {
        let stop = self.stop_on_seed.is_none_or(|v| state.param()[0] == v);
        let m = problem
            .inner()
            .constraint_set()
            .nonlinear()
            .inequalities()
            .count();
        let member = evaluate(
            problem,
            self.returned.clone(),
            (state.param().vec_len(), m),
        )?;
        let cost = match member.cost() {
            Some(c) => c,
            None => problem.cost(member.param())?,
        };
        state.replace(member.param().clone(), cost);
        if stop {
            if let Some(reason) = &self.stop {
                return Ok(SolverStep::stopped(state, reason.clone()));
            }
        }
        Ok(SolverStep::completed(state))
    }
}

fn script<V, F: Scalar>(make: impl Fn(&[F]) -> V) -> Vec<Vec<V>> {
    [
        [[0.0, 0.0], [-2.0, 0.0], [0.0, 3.0], [-1.0, 4.0]],
        [[1.0, 0.0], [-1.0, 0.0], [0.5, 2.0], [-2.0, 0.0]],
        [[2.0, 0.0], [1.0, 0.0], [0.75, 1.5], [0.0, 2.0]],
        [[1.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
    ]
    .into_iter()
    .map(|generation| {
        generation
            .into_iter()
            .map(|point| make(&point.map(num)))
            .collect()
    })
    .collect()
}
fn de() -> ConstrainedDePrototype<Vec<f64>> {
    ConstrainedDePrototype::new(
        De::new(42).with_pop_size(4),
        script(|x| x.to_vec()),
    )
}
fn fixture() -> Fixture<Vec<f64>> {
    Fixture::new(|x| x.to_vec())
}
fn member(
    x: f64,
    cost: Option<f64>,
    violations: &[f64],
) -> PopulationMember<Vec<f64>> {
    PopulationMember::new(vec![x, 0.0], cost, violations.to_vec())
}

fn check_backend<V, F>(make: impl Fn(&[F]) -> V + Copy)
where
    V: VectorLen
        + Clone
        + IndexMut<usize, Output = F>
        + std::fmt::Debug
        + PartialEq,
    F: Scalar,
{
    let p = Fixture::new(make);
    let trace = p.trace.clone();
    let solver = ConstrainedDePrototype::new(De::<F>::new(42), script(make));
    let mut stepper =
        Executor::new(p, solver, SelectedPopulationProgress::empty())
            .require_evaluated_state()
            .max_iter(3)
            .into_stepper()
            .unwrap();
    let initial = stepper.state();
    assert_eq!(initial.cost(), F::zero());
    assert_eq!(
        initial
            .members()
            .unwrap()
            .iter()
            .filter(|m| m.cost().is_some())
            .count(),
        1
    );
    assert_eq!(initial.raw_counts().cost_evals, 1);
    assert_eq!(initial.raw_counts().residual_evals, 4);
    assert!(!initial.best().unwrap().feasible());
    assert_eq!(
        trace.borrow().iter().filter(|(kind, _)| *kind == 0).count(),
        1
    );
    stepper.step().unwrap();
    let selected = stepper.state();
    assert_eq!(selected.cost(), F::one());
    assert_eq!(selected.best_cost(), F::one());
    assert!(selected.best().unwrap().feasible());
    assert_eq!(selected.best_iter(), 1);
    assert_eq!(selected.best_param(), &make(&[F::one(), F::zero()]));
    assert_eq!(
        selected.members().unwrap()[3].param(),
        &make(&[num(-1.0), num(4.0)])
    );
    assert_eq!(selected.raw_counts().cost_evals, 2);
    assert_eq!(selected.raw_counts().residual_evals, 8);
    stepper.step().unwrap();
    let state = stepper.state();
    assert_eq!(state.best_iter(), 1);
    assert_eq!(state.incumbent_record().unwrap().counts.cost_evals, 2);
    assert_eq!(state.current_record().unwrap().1, F::one());
    assert_eq!(state.best().unwrap().violations(), &[F::zero(), F::zero()]);
    let outer = ConstrainedDePrototype::new(De::<F>::new(42), script(make));
    let inner = ScriptedInner::new(make(&[num(0.5), F::zero()]));
    let result = Executor::new(
        Fixture::new(make),
        ConstrainedDeInjectPrototype::with_inner_solver(outer, inner),
        SelectedPopulationProgress::empty(),
    )
    .require_evaluated_state()
    .max_iter(3)
    .run()
    .unwrap();
    assert_eq!(result.best_param(), &make(&[F::one(), F::zero()]));
    assert_eq!(result.best_cost(), F::one());
    assert_eq!(result.state.raw_counts(), &result.counts);
}

#[test]
fn feasibility_selection_and_missing_objectives_work_in_both_precisions() {
    check_backend::<Vec<f64>, f64>(|x| x.to_vec());
    check_backend::<Vec<f32>, f32>(|x| x.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_supports_both_precisions() {
    use backend_aliases::nalgebra::DVector;
    check_backend::<DVector<f64>, f64>(DVector::from_column_slice);
    check_backend::<DVector<f32>, f32>(DVector::from_column_slice);
}
#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_supports_both_precisions() {
    use backend_aliases::ndarray::Array1;
    check_backend::<Array1<f64>, f64>(|x| Array1::from_vec(x.to_vec()));
    check_backend::<Array1<f32>, f32>(|x| Array1::from_vec(x.to_vec()));
}
#[cfg(feature = "faer_all")]
#[test]
fn faer_supports_both_precisions() {
    use backend_aliases::faer::Col;
    check_backend::<Col<f64>, f64>(|x| Col::from_fn(x.len(), |i| x[i]));
    check_backend::<Col<f32>, f32>(|x| Col::from_fn(x.len(), |i| x[i]));
}

#[test]
fn componentwise_replacement_is_not_total_violation_ranking() {
    let original = member(0.0, Some(-100.0), &[1.0, 3.0]);
    assert!(!accepts(&member(0.0, None, &[2.0, 1.0]), &original));
    assert!(accepts(&member(0.0, None, &[1.0, 2.0]), &original));
    assert!(accepts(&member(1.0, Some(100.0), &[0.0, 0.0]), &original));
    let feasible = member(1.0, Some(1.0), &[0.0, 0.0]);
    assert!(!accepts(&original, &feasible));
    assert!(accepts(&member(2.0, Some(1.0), &[0.0, 0.0]), &feasible));
    assert!(!accepts(
        &member(2.0, Some(f64::NAN), &[0.0, 0.0]),
        &feasible
    ));
}

#[test]
fn storage_updates_are_atomic_and_pending_selections_own_their_records() {
    let mut state =
        SelectedPopulationProgress::from_population(vec![vec![0.0, 0.0]]);
    assert!(state.current_record().is_none());
    assert!(state.members().is_none());
    assert!(!state.select_current());
    state
        .replace(vec![member(0.0, Some(0.0), &[1.0, 0.0])], 0)
        .unwrap();
    state.select_current();
    let snapshot = state.clone();
    let invalid = [
        (vec![], 0, ShapeError::Empty),
        (
            vec![PopulationMember::new(vec![], Some(0.0), vec![])],
            0,
            ShapeError::Dimension,
        ),
        (
            vec![
                member(0.0, Some(0.0), &[1.0]),
                member(1.0, Some(1.0), &[0.0, 0.0]),
            ],
            0,
            ShapeError::ViolationWidth,
        ),
        (
            vec![member(0.0, Some(0.0), &[-1.0])],
            0,
            ShapeError::InvalidViolation,
        ),
        (
            vec![member(0.0, Some(0.0), &[f64::NAN])],
            0,
            ShapeError::InvalidViolation,
        ),
        (
            vec![member(0.0, Some(0.0), &[1.0])],
            1,
            ShapeError::SelectedIndex,
        ),
        (
            vec![member(0.0, None, &[1.0])],
            0,
            ShapeError::MissingObjective,
        ),
        (
            vec![member(0.0, Some(0.0), &[1.0]), member(1.0, None, &[0.0])],
            0,
            ShapeError::MissingObjective,
        ),
    ];
    for (members, selected, error) in invalid {
        assert_eq!(state.replace(members, selected), Err(error));
        assert_eq!(state, snapshot);
    }
    state
        .replace(vec![member(2.0, Some(4.0), &[0.0, 0.0])], 0)
        .unwrap();
    let mut counts = EvalCounts::default();
    counts.cost_evals = 2;
    counts.gradient_evals = 3;
    counts.residual_evals = 5;
    counts.jacobian_evals = 7;
    counts.hessian_evals = 11;
    counts.hessian_product_evals = 13;
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.best_param(), &[0.0, 0.0]);
    assert_eq!(state.best().unwrap().violations(), &[1.0, 0.0]);
    assert_eq!(state.incumbent_record().unwrap().counts, &counts);
    state.increment_iter();
    state.update_best();
    assert_eq!(state.best_iter(), 0);
    state.select_current();
    state.update_best();
    assert_eq!(state.best_cost(), 4.0);
    state
        .replace(vec![member(3.0, Some(9.0), &[0.0, 0.0])], 0)
        .unwrap();
    state.update_best();
    assert_eq!(state.best_param(), &[2.0, 0.0]);
    state.reset();
    assert_eq!(state.param(), &[3.0, 0.0]);
    assert!(state.current().is_none() && state.best().is_none());
    assert_eq!(state.raw_counts(), &EvalCounts::default());
}

#[test]
fn evaluated_nonfinite_costs_are_distinct_from_missing_costs() {
    let mut state = SelectedPopulationProgress::empty();
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        state
            .replace(
                vec![
                    member(0.0, Some(value), &[1.0]),
                    member(-1.0, None, &[2.0]),
                ],
                0,
            )
            .unwrap();
        let (_, observed) = state.current_record().unwrap();
        assert_eq!(observed.to_bits(), value.to_bits());
        assert!(state.members().unwrap()[1].cost().is_none());
    }
}

#[test]
fn objective_controls_and_incompatible_solvers_remain_unavailable() {
    trait WithoutObjective<A> {
        fn check() {}
    }
    impl<T: ?Sized> WithoutObjective<()> for T {}
    struct HasObjective;
    impl<T: ?Sized + basin::ObjectiveIncumbentState>
        WithoutObjective<HasObjective> for T
    {
    }
    let _ =
        <SelectedPopulationProgress<Vec<f64>> as WithoutObjective<_>>::check;
    trait WithoutPopulation<A> {
        fn check() {}
    }
    impl<T: ?Sized> WithoutPopulation<()> for T {}
    struct HasPopulation;
    impl<T: ?Sized + basin::PopulationState> WithoutPopulation<HasPopulation>
        for T
    {
    }
    let _ =
        <SelectedPopulationProgress<Vec<f64>> as WithoutPopulation<_>>::check;
    trait WithoutConstrainedSolver<A> {
        fn check() {}
    }
    impl<T: ?Sized> WithoutConstrainedSolver<()> for T {}
    struct HasConstrainedSolver;
    impl<
        T: ?Sized
            + Solver<Fixture<Vec<f64>>, SelectedPopulationProgress<Vec<f64>>>,
    > WithoutConstrainedSolver<HasConstrainedSolver> for T
    {
    }
    type Incompatible = ConstrainedDeInjectPrototype<
        basin::RelaxedInner<basin::NelderMead<Vec<f64>>>,
        Vec<f64>,
    >;
    let _ = <Incompatible as WithoutConstrainedSolver<_>>::check;
    trait WithoutOrdinaryConstrainedDe<A> {
        fn check() {}
    }
    impl<T: ?Sized> WithoutOrdinaryConstrainedDe<()> for T {}
    struct HasOrdinaryConstrainedDe;
    impl<
        T: ?Sized + Solver<Fixture<Vec<f64>>, basin::PopulationProgress<Vec<f64>>>,
    > WithoutOrdinaryConstrainedDe<HasOrdinaryConstrainedDe> for T
    {
    }
    let _ = <De as WithoutOrdinaryConstrainedDe<_>>::check;
    fn objective_capability<S: basin::ObjectiveIncumbentState>() {}
    objective_capability::<basin::PopulationProgress<Vec<f64>>>();
}

#[test]
fn representative_errors_and_constraint_errors_keep_their_charges() {
    let mut p = fixture();
    p.fail_cost = Some(0.0);
    let mut p = Problem::new(p);
    let error = de()
        .init(&mut p, SelectedPopulationProgress::empty())
        .unwrap_err();
    assert_eq!(error, ConstraintSolveError::Evaluation(CallbackError::Cost));
    assert_eq!(p.counts().cost_evals, 1);
    assert_eq!(p.counts().residual_evals, 4);
    let mut p = fixture();
    p.fail_constraint = Some(-2.0);
    let mut p = Problem::new(p);
    let error = de()
        .init(&mut p, SelectedPopulationProgress::empty())
        .unwrap_err();
    assert_eq!(
        error,
        ConstraintSolveError::Evaluation(CallbackError::Constraint)
    );
    assert_eq!(p.counts().residual_evals, 2);
    assert_eq!(p.counts().cost_evals, 0);
    let mut p = fixture();
    p.override_values = Some(vec![0.0]);
    let mut p = Problem::new(p);
    assert!(matches!(
        de().init(&mut p, SelectedPopulationProgress::empty()),
        Err(ConstraintSolveError::InvalidConstraints(
            basin::ConstraintError::DimensionMismatch { .. }
        ))
    ));
    assert_eq!(p.counts().residual_evals, 1);
}

#[test]
fn absent_callbacks_layout_changes_and_nonfinite_constraints_are_explicit() {
    let p = fixture();
    p.count.set(0);
    let result = Executor::new(p, de(), SelectedPopulationProgress::empty())
        .max_iter(0)
        .run()
        .unwrap();
    assert_eq!(result.counts.cost_evals, 4);
    assert_eq!(result.counts.residual_evals, 0);
    let p = fixture();
    let count = p.count.clone();
    let mut stepper =
        Executor::new(p, de(), SelectedPopulationProgress::empty())
            .into_stepper()
            .unwrap();
    count.set(1);
    assert!(matches!(
        stepper.step(),
        Err(ConstraintSolveError::InvalidConstraints(
            basin::ConstraintError::LayoutChanged
        ))
    ));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = fixture();
        p.override_values = Some(vec![value, 0.0]);
        let result =
            Executor::new(p, de(), SelectedPopulationProgress::empty())
                .max_iter(0)
                .run()
                .unwrap();
        assert!(
            result
                .state
                .members()
                .unwrap()
                .iter()
                .all(|m| !m.feasible())
        );
        assert_eq!(result.counts.cost_evals, 1);
    }
}

#[test]
fn constraint_work_can_exhaust_a_budget_before_any_generation() {
    let result =
        Executor::new(fixture(), de(), SelectedPopulationProgress::empty())
            .max_evaluations(EvaluationKind::Residual, 4)
            .run()
            .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(result.counts.cost_evals, 1);
    assert_eq!(result.counts.residual_evals, 4);
    assert!(matches!(result.report.termination, Termination::Limit(_)));
}

fn injection(
    returned: Vec<f64>,
) -> ConstrainedDeInjectPrototype<ScriptedInner<Vec<f64>>, Vec<f64>> {
    ConstrainedDeInjectPrototype::with_inner_solver(
        de(),
        ScriptedInner::new(returned),
    )
    .with_k(2)
    .with_refine_every(2)
    .with_inner_max_iter(1)
}

#[test]
fn local_search_rechecks_feasibility_and_repairs_before_evaluation() {
    // This zero-cost local result must not replace the feasible outer member.
    let result = Executor::new(
        fixture(),
        injection(vec![0.0, 0.0]),
        SelectedPopulationProgress::empty(),
    )
    .require_evaluated_state()
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.best_param(), &[1.0, 0.0]);
    assert!(result.state.best().unwrap().feasible());
    // The repaired result remains infeasible and is rejected; no clipped objective is needed.
    let p = fixture();
    let trace = p.trace.clone();
    let result = Executor::new(
        p,
        injection(vec![8.0, 8.0]),
        SelectedPopulationProgress::empty(),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.best_param(), &[1.0, 0.0]);
    assert!(trace.borrow().contains(&(1, vec![5.0, 5.0])));
    assert!(!trace.borrow().contains(&(0, vec![5.0, 5.0])));
}

#[test]
fn local_search_can_improve_an_infeasible_member_without_objective_ordering() {
    let outer = ConstrainedDePrototype::new(
        De::new(42),
        vec![vec![vec![0.0, 0.0], vec![-1.0, 0.0]]; 2],
    );
    let inner = ScriptedInner::new(vec![0.5, 0.0]);
    let result = Executor::new(
        fixture(),
        ConstrainedDeInjectPrototype::with_inner_solver(outer, inner),
        SelectedPopulationProgress::empty(),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.best_param(), &[0.5, 0.0]);
    assert_eq!(result.best_cost(), 0.25);
    assert_eq!(result.state.best().unwrap().violations(), &[0.5, 0.0]);
    assert!(!result.state.best().unwrap().feasible());
    assert_eq!(result.counts.cost_evals, 5);
    assert_eq!(result.counts.residual_evals, 7);
}

impl basin::Gradient for Fixture<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, CallbackError> {
        Ok(x.iter().map(|v| 2.0 * v).collect())
    }
}
impl basin::Hessian for Fixture<Vec<f64>> {
    type Hessian = basin::DenseMatrix;
    fn hessian(&self, x: &Vec<f64>) -> Result<Self::Hessian, CallbackError> {
        Ok(basin::DenseMatrix::from_fn(x.len(), x.len(), |i, j| {
            if i == j { 2.0 } else { 0.0 }
        }))
    }
}
impl basin::HessianProduct for Fixture<Vec<f64>> {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, CallbackError> {
        Ok(v.iter().map(|v| 2.0 * v).collect())
    }
}
impl basin::Residual for Fixture<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = CallbackError;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, CallbackError> {
        Ok(x.clone())
    }
}
impl basin::Jacobian for Fixture<Vec<f64>> {
    type Jacobian = basin::DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<Self::Jacobian, CallbackError> {
        Ok(basin::DenseMatrix::from_fn(x.len(), x.len(), |i, j| {
            if i == j { 1.0 } else { 0.0 }
        }))
    }
}

struct AuditInner(ScriptedInner<Vec<f64>>);
impl InitialState<Vec<f64>> for AuditInner {
    type State = PointState<Vec<f64>>;
    fn seed(&self, x: &Vec<f64>) -> Self::State {
        self.0.seed(x)
    }
}
impl WarmStart<Vec<f64>> for AuditInner {}
impl MemeticInner<Vec<f64>> for AuditInner {}
impl Solver<Fixture<Vec<f64>>, PointState<Vec<f64>>> for AuditInner {
    type Error = ConstraintSolveError<CallbackError>;
    fn init(
        &mut self,
        p: &mut Problem<Fixture<Vec<f64>>>,
        s: PointState<Vec<f64>>,
    ) -> Result<PointState<Vec<f64>>, Self::Error> {
        self.0.init(p, s)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<Fixture<Vec<f64>>>,
        s: PointState<Vec<f64>>,
    ) -> Result<SolverStep<PointState<Vec<f64>>>, Self::Error> {
        let step = self.0.next_iter(p, s)?;
        let x = step.state.param();
        p.cost_and_gradient_and_hessian(x)?;
        p.residual_and_jacobian(x)?;
        p.hessian_product(x, x)?;
        Ok(step)
    }
}

#[test]
fn local_derivative_work_retains_all_six_raw_categories_once() {
    let solver = ConstrainedDeInjectPrototype::with_inner_solver(
        de(),
        AuditInner(ScriptedInner::new(vec![1.0, 0.0])),
    );
    let result =
        Executor::new(fixture(), solver, SelectedPopulationProgress::empty())
            .max_evaluations(EvaluationKind::Gradient, 1)
            .max_iter(3)
            .run()
            .unwrap();
    let counts = &result.counts;
    assert_eq!(result.iter(), 1);
    assert_eq!(counts.cost_evals, 6);
    assert_eq!(counts.residual_evals, 12);
    assert_eq!(counts.gradient_evals, 1);
    assert_eq!(counts.jacobian_evals, 1);
    assert_eq!(counts.hessian_evals, 1);
    assert_eq!(counts.hessian_product_evals, 1);
    assert_eq!(result.state.raw_counts(), counts);
    assert_eq!(result.state.cost_evals(), counts.cost_evals);
    assert!(matches!(result.report.termination, Termination::Limit(_)));
}

#[test]
fn hard_inner_errors_never_publish_partial_state() {
    let mut p = fixture();
    p.fail_cost = Some(0.5);
    let observed = Rc::new(Cell::new(0));
    let calls = observed.clone();
    let mut stepper = Executor::new(
        p,
        injection(vec![0.5, 0.0]),
        SelectedPopulationProgress::empty(),
    )
    .observe_solver(
        move |_, _, _| calls.set(calls.get() + 1),
        basin::ObserverMode::Always,
    )
    .into_stepper()
    .unwrap();
    assert_eq!(
        stepper.step(),
        Err(ConstraintSolveError::Evaluation(CallbackError::Cost))
    );
    assert_eq!(observed.get(), 1);
}

#[test]
fn local_convergence_permits_outer_work_and_clipped_costs_match_members() {
    let mut inner = ScriptedInner::new(vec![8.0, 0.0]);
    inner.stop = Some(Termination::custom(
        "fixture",
        "scripted inner convergence",
        vec![],
    ));
    let solver =
        ConstrainedDeInjectPrototype::with_inner_solver(de(), inner).with_k(2);
    let result =
        Executor::new(fixture(), solver, SelectedPopulationProgress::empty())
            .max_iter(1)
            .run()
            .unwrap();
    assert_eq!(result.iter(), 1);
    assert!(matches!(result.report.termination, Termination::Limit(_)));
    let repaired = &result.state.members().unwrap()[2];
    assert_eq!(repaired.param(), &[5.0, 0.0]);
    assert_eq!(repaired.cost(), Some(25.0));
    assert!(repaired.feasible());
}

#[test]
fn fresh_explicit_populations_can_change_dimension_and_reset_script_position() {
    let prior =
        Executor::new(fixture(), de(), SelectedPopulationProgress::empty())
            .max_iter(3)
            .run_with_solver()
            .unwrap();
    let mut p = fixture();
    p.lower = vec![-5.0; 3];
    p.upper = vec![5.0; 3];
    let state = SelectedPopulationProgress::from_population(vec![
        vec![8.0, 0.0, 2.0],
        vec![0.0, 0.0, 0.0],
    ]);
    let result = Executor::new(p, prior.solver, state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(result.solver.position(), 1);
    assert_eq!(result.state.iter(), 0);
    assert_eq!(result.state.best_iter(), 0);
    assert_eq!(result.best_param(), &[5.0, 0.0, 2.0]);
    assert_eq!(result.best_cost(), 29.0);
    assert_eq!(result.counts.cost_evals, 1);
    assert_eq!(result.counts.residual_evals, 2);
}

fn failure() -> Termination {
    Termination::Failed(NumericalFailure::Solver {
        message: "scripted failure".into(),
        measurements: vec![],
    })
}
#[test]
fn partial_refinement_failure_publishes_prior_selection_and_all_work() {
    let outer = ConstrainedDePrototype::new(
        De::new(42),
        vec![vec![vec![2.0, 0.0], vec![3.0, 0.0]]; 2],
    );
    let mut inner = ScriptedInner::new(vec![1.0, 0.0]);
    inner.stop_on_seed = Some(3.0);
    inner.stop = Some(failure());
    let solver =
        ConstrainedDeInjectPrototype::with_inner_solver(outer, inner).with_k(2);
    let result =
        Executor::new(fixture(), solver, SelectedPopulationProgress::empty())
            .require_evaluated_state()
            .max_iter(1)
            .run()
            .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(result.best_param(), &[1.0, 0.0]);
    assert_eq!(result.counts.cost_evals, 9);
    assert_eq!(result.counts.residual_evals, 9);
    assert_eq!(result.state.raw_counts(), &result.counts);
    assert_eq!(
        result.state.incumbent_record().unwrap().counts,
        &result.counts
    );
    assert_eq!(
        result.report.stage,
        TerminationStage::Step { completed: false }
    );
    assert!(matches!(
        result.report.termination,
        Termination::Failed(NumericalFailure::Inner { .. })
    ));
}

#[test]
fn inner_cancellation_and_application_stops_propagate() {
    for termination in [
        Termination::Cancelled,
        Termination::Application(basin::ApplicationStop::new("inner stop")),
    ] {
        let mut inner = ScriptedInner::new(vec![1.0, 0.0]);
        inner.stop = Some(termination.clone());
        let solver =
            ConstrainedDeInjectPrototype::with_inner_solver(de(), inner);
        let result = Executor::new(
            fixture(),
            solver,
            SelectedPopulationProgress::empty(),
        )
        .max_iter(1)
        .run()
        .unwrap();
        assert_eq!(result.report.termination, termination);
        assert_eq!(result.iter(), 0);
    }
}

#[test]
fn fresh_runs_and_exact_checkpoints_preserve_the_right_history() {
    let expected = Executor::new(
        fixture(),
        injection(vec![0.5, 0.0]),
        SelectedPopulationProgress::empty(),
    )
    .max_iter(3)
    .run_with_solver()
    .unwrap();
    for split in 0..=3 {
        let checkpoint = Executor::new(
            fixture(),
            injection(vec![0.5, 0.0]),
            SelectedPopulationProgress::empty(),
        )
        .max_iter(split)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
        #[cfg(feature = "serde")]
        let checkpoint = serde_round_trip(&checkpoint);
        let resumed = Executor::resume_from_checkpoint(fixture(), checkpoint)
            .require_evaluated_state()
            .max_iter(3)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
    }
    let restarted = Executor::new(
        fixture(),
        expected.solver,
        SelectedPopulationProgress::empty(),
    )
    .max_iter(3)
    .run_with_solver()
    .unwrap();
    assert_eq!(restarted.state, expected.state);
    assert_eq!(restarted.counts, expected.counts);
    let final_members = restarted
        .state
        .members()
        .unwrap()
        .iter()
        .map(|m| m.param().clone())
        .collect::<Vec<_>>();
    let warm = Executor::new(fixture(), restarted.solver, restarted.state)
        .max_iter(0)
        .run()
        .unwrap();
    assert_eq!(warm.iter(), 0);
    assert_eq!(warm.counts.residual_evals, 4);
    assert_eq!(
        warm.state
            .members()
            .unwrap()
            .iter()
            .map(|m| m.param().clone())
            .collect::<Vec<_>>(),
        final_members
    );
    assert_eq!(warm.state.best_iter(), 0);
}

#[cfg(feature = "serde")]
#[test]
fn f32_injection_checkpoint_round_trips_optional_costs_and_selection() {
    let make = |x: &[f32]| x.to_vec();
    let outer = ConstrainedDePrototype::new(De::<f32>::new(42), script(make));
    let solver = ConstrainedDeInjectPrototype::with_inner_solver(
        outer,
        ScriptedInner::new(vec![0.5_f32, 0.0]),
    );
    let checkpoint = Executor::new(
        Fixture::new(make),
        solver,
        SelectedPopulationProgress::empty(),
    )
    .max_iter(0)
    .run_with_solver()
    .unwrap()
    .into_checkpoint();
    let decoded = serde_round_trip(&checkpoint);
    assert_eq!(decoded.state(), checkpoint.state());
    let resumed = Executor::resume_from_checkpoint(Fixture::new(make), decoded)
        .max_iter(3)
        .run()
        .unwrap();
    assert_eq!(resumed.best_param(), &[1.0, 0.0]);
    assert_eq!(resumed.best_cost(), 1.0);
}

#[test]
fn standalone_checkpoints_resume_the_script_and_observers_do_no_work() {
    let make =
        || Executor::new(fixture(), de(), SelectedPopulationProgress::empty());
    let expected = make().max_iter(3).run_with_solver().unwrap();
    for split in 0..=3 {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(fixture(), checkpoint)
            .max_iter(3)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.solver.position(), 4);
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
    }
    let observed = Rc::new(Cell::new(0));
    let calls = observed.clone();
    let result = make()
        .max_iter(3)
        .observe_solver(
            move |state, _, _| {
                assert!(state.current_record().is_some());
                assert!(state.members().is_some());
                calls.set(calls.get() + 1);
            },
            basin::ObserverMode::Always,
        )
        .run()
        .unwrap();
    assert_eq!(result.counts, expected.counts);
    assert_eq!(observed.get(), 5);
}
