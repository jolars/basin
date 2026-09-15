//! Capability controls exercised through the public owned and borrowed drivers.

use basin::{
    CostFunction, CountsMirror, EvalCounts, EvaluatedGradientState,
    EvaluatedState, EvaluationKind, ExactCheckpoint, Executor, FirstOrderState,
    Gradient, Hessian, HessianProduct, IncumbentRef, IncumbentState,
    InnerExecutor, Jacobian, ObjectiveIncumbentState, Observe, ObserverMode,
    PointState, Problem, RawEvaluationState, Residual, RunControl, Solver,
    State, TerminationReason, run_loop_with_control,
};
use std::cell::Cell;
use std::convert::Infallible;
use std::rc::Rc;

struct Identity;

impl CostFunction for Identity {
    type Param = f64;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &f64) -> Result<f64, Infallible> {
        Ok(*x)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Trajectory {
    points: Vec<f64>,
    next: usize,
}

impl Trajectory {
    fn new(points: &[f64]) -> Self {
        Self {
            points: points.to_vec(),
            next: 0,
        }
    }
}

impl Solver<Identity, PointState<f64>> for Trajectory {
    type Error = Infallible;

    fn init(
        &mut self,
        problem: &mut Problem<Identity>,
        mut state: PointState<f64>,
    ) -> Result<PointState<f64>, Infallible> {
        self.next = 0;
        state.reset();
        let x = *state.param();
        state.replace(x, problem.cost(&x)?);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Identity>,
        mut state: PointState<f64>,
    ) -> Result<(PointState<f64>, Option<TerminationReason>), Infallible> {
        let x = self
            .points
            .get(self.next)
            .copied()
            .unwrap_or(*state.param());
        self.next += 1;
        state.replace(x, problem.cost(&x)?);
        Ok((state, None))
    }
}

#[test]
fn generic_records_distinguish_seeds_rejections_and_incumbents() {
    fn current<S: EvaluatedState<Float = f64, Param = f64>>(
        s: &S,
    ) -> Option<f64> {
        s.current_record().map(|(_, cost)| cost)
    }
    let mut state = PointState::new(5.0);
    assert_eq!(current(&state), None);
    assert!(state.incumbent_record().is_none());
    state.replace(5.0, f64::INFINITY);
    state.update_best();
    assert_eq!(current(&state), Some(f64::INFINITY));
    assert!(state.incumbent_record().is_none());
    state.replace(5.0, 5.0);
    state.mirror(&EvalCounts {
        cost_evals: 2,
        ..EvalCounts::default()
    });
    state.update_best();
    state.increment_iter();
    state.replace(7.0, 7.0);
    state.mirror(&EvalCounts {
        cost_evals: 3,
        ..EvalCounts::default()
    });
    state.update_best();
    let best = state.incumbent_record().unwrap();
    assert_eq!((*best.param, best.cost, best.iter), (5.0, 5.0, 0));
    assert_eq!(best.counts.cost_evals, 2);
    assert_eq!(state.raw_counts().cost_evals, 3);

    let mut first = FirstOrderState::new(vec![2.0]);
    assert!(first.current_record().is_none());
    assert!(first.current_gradient_record().is_none());
    first.replace(vec![2.0], 4.0, vec![4.0]).unwrap();
    let (x, cost, gradient) = first.current_gradient_record().unwrap();
    assert_eq!(
        (x.as_slice(), cost, gradient.as_slice()),
        (&[2.0][..], 4.0, &[4.0][..])
    );
    assert!(first.replace(vec![3.0], 9.0, vec![]).is_err());
    assert_eq!(first.current_record(), Some((&vec![2.0], 4.0)));
}

struct CountObservers(Rc<Cell<usize>>);

impl<S> Observe<S> for CountObservers {
    fn observe_init(&mut self, _: &S) {
        self.0.set(self.0.get() + 1);
    }
    fn observe_iter(&mut self, _: &S) {
        self.0.set(self.0.get() + 1);
    }
    fn observe_final(&mut self, _: &S, _: &TerminationReason) {
        self.0.set(self.0.get() + 1);
    }
}

struct Incomplete {
    init: bool,
    mid_step: bool,
}

impl Solver<(), FirstOrderState<Vec<f64>>> for Incomplete {
    type Error = Infallible;

    fn init(
        &mut self,
        _: &mut Problem<()>,
        mut state: FirstOrderState<Vec<f64>>,
    ) -> Result<FirstOrderState<Vec<f64>>, Infallible> {
        if !self.init {
            state.replace(vec![1.0], 1.0, vec![2.0]).unwrap();
        }
        Ok(state)
    }

    fn next_iter(
        &mut self,
        _: &mut Problem<()>,
        mut state: FirstOrderState<Vec<f64>>,
    ) -> Result<
        (FirstOrderState<Vec<f64>>, Option<TerminationReason>),
        Infallible,
    > {
        state.reset();
        Ok((
            state,
            self.mid_step.then_some(TerminationReason::SolverFailed),
        ))
    }
}

fn assert_incomplete(f: impl FnOnce()) {
    let panic =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_err();
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(
        message.contains("solver published incomplete state"),
        "{message}"
    );
}

#[test]
fn incomplete_publications_never_reach_observers() {
    for (init, mid_step) in [(true, false), (false, false), (false, true)] {
        let calls = Rc::new(Cell::new(0));
        assert_incomplete(|| {
            let _ = Executor::new(
                (),
                Incomplete { init, mid_step },
                FirstOrderState::new(vec![1.0]),
            )
            .require_evaluated_state()
            .observe_with(CountObservers(calls.clone()), ObserverMode::Always)
            .run();
        });
        assert_eq!(calls.get(), usize::from(!init));
    }
}

#[test]
fn restored_checkpoints_and_borrowed_initialization_are_validated() {
    let calls = Rc::new(Cell::new(0));
    assert_incomplete(|| {
        let checkpoint = ExactCheckpoint::from_parts(
            Incomplete {
                init: true,
                mid_step: false,
            },
            FirstOrderState::new(vec![1.0]),
            EvalCounts::default(),
        );
        let _ = Executor::resume_from_checkpoint((), checkpoint)
            .require_evaluated_state()
            .observe_with(CountObservers(calls.clone()), ObserverMode::Always)
            .into_stepper();
    });
    assert_eq!(calls.get(), 0);
    assert_incomplete(|| {
        let _ = run_loop_with_control(
            &mut Problem::new(()),
            FirstOrderState::new(vec![1.0]),
            &mut Incomplete {
                init: true,
                mid_step: false,
            },
            &mut RunControl::new().require_evaluated_state(),
        );
    });
}

#[test]
fn raw_budget_observes_init_and_nested_run_deltas() {
    let result =
        Executor::new(Identity, Trajectory::new(&[]), PointState::new(1.0))
            .max_evaluations(EvaluationKind::Cost, 0)
            .run()
            .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(
        result.reason,
        TerminationReason::MaxEvaluations(EvaluationKind::Cost)
    );
    let mut problem = Problem::new(Identity);
    problem.cost(&10.0).unwrap();
    let mut inner = InnerExecutor::new(Trajectory::new(&[]))
        .require_evaluated_state()
        .max_evaluations(EvaluationKind::Cost, 3);
    for _ in 0..2 {
        let result = inner.run(&mut problem, PointState::new(1.0)).unwrap();
        assert_eq!(result.iter(), 2);
        assert_eq!(result.state.raw_counts().cost_evals, 3);
    }
    assert_eq!(problem.counts().cost_evals, 7);
}

#[test]
fn objective_targets_require_an_available_incumbent() {
    for seed in [f64::NAN, f64::INFINITY] {
        let result = Executor::new(
            Identity,
            Trajectory::new(&[5.0, 1.0]),
            PointState::new(seed),
        )
        .require_evaluated_state()
        .target_objective(2.0)
        .run()
        .unwrap();
        assert_eq!(result.iter(), 2);
        assert_eq!(result.reason, TerminationReason::TargetCost);
    }
    let result = Executor::new(
        Identity,
        Trajectory::new(&[]),
        PointState::new(f64::NEG_INFINITY),
    )
    .require_evaluated_state()
    .target_objective(0.0)
    .run()
    .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(result.reason, TerminationReason::TargetCost);
}

#[test]
fn objective_stalls_use_publication_age_and_threshold_anchors() {
    for (delta, expected) in [(0.0, 7), (1.0, 5)] {
        let result = Executor::new(
            Identity,
            Trajectory::new(&[9.5, 8.5, 11.0, 8.4]),
            PointState::new(10.0),
        )
        .no_objective_improvement(3, delta)
        .run()
        .unwrap();
        assert_eq!(result.iter(), expected);
        assert_eq!(result.reason, TerminationReason::NoImprovement);
    }
    let result = Executor::new(
        Identity,
        Trajectory::new(&[]),
        PointState::new(f64::INFINITY),
    )
    .max_iter(4)
    .no_objective_improvement(1, 0.0)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxIter);
}

#[test]
fn checkpoint_reattachment_preserves_only_zero_delta_stall_age() {
    for (delta, expected) in [(0.0, 3), (1.0, 5)] {
        let checkpoint = Executor::new(
            Identity,
            Trajectory::new(&[]),
            PointState::new(10.0),
        )
        .max_iter(2)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
        let result = Executor::resume_from_checkpoint(Identity, checkpoint)
            .require_evaluated_state()
            .no_objective_improvement(3, delta)
            .run()
            .unwrap();
        assert_eq!(result.iter(), expected);
    }
}

#[test]
fn exact_resume_budgets_cumulative_raw_work() {
    let checkpoint =
        Executor::new(Identity, Trajectory::new(&[]), PointState::new(10.0))
            .max_iter(2)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
    let result = Executor::resume_from_checkpoint(Identity, checkpoint)
        .max_evaluations(EvaluationKind::Cost, 5)
        .run_with_solver()
        .unwrap();
    assert_eq!(result.iter(), 4);
    assert_eq!(result.counts.cost_evals, 5);
    assert_eq!(result.state.raw_counts(), &result.counts);
}

#[test]
fn borrowed_runs_reset_positive_delta_history() {
    let mut problem = Problem::new(Identity);
    let mut control = RunControl::new().no_objective_improvement(4, 1.0);
    let first = run_loop_with_control(
        &mut problem,
        PointState::new(10.0),
        &mut Trajectory::new(&[9.5, 9.0, 8.0]),
        &mut control,
    )
    .unwrap();
    assert_eq!(first.iter(), 7);
    let second = run_loop_with_control(
        &mut problem,
        PointState::new(20.0),
        &mut Trajectory::new(&[]),
        &mut control,
    )
    .unwrap();
    assert_eq!(second.iter(), 4);
}

#[cfg(feature = "serde")]
#[test]
fn capability_controls_are_never_silently_discarded_by_serialization() {
    let controls: [InnerExecutor<PointState<f64>, _>; 4] = [
        InnerExecutor::new(Trajectory::new(&[])).require_evaluated_state(),
        InnerExecutor::new(Trajectory::new(&[]))
            .max_evaluations(EvaluationKind::Cost, 3),
        InnerExecutor::new(Trajectory::new(&[])).target_objective(1.0),
        InnerExecutor::new(Trajectory::new(&[]))
            .no_objective_improvement(3, 0.0),
    ];
    for inner in controls {
        assert!(
            bincode::serde::encode_to_vec(&inner, bincode::config::standard())
                .is_err()
        );
    }
}

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = f64;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
        if *x == 999.0 {
            Err("evaluation failed")
        } else {
            Ok(0.5 * x * x)
        }
    }
}

impl Gradient for Quadratic {
    type Gradient = f64;
    fn gradient(&self, x: &f64) -> Result<f64, Self::Error> {
        Ok(*x)
    }
}

impl Hessian for Quadratic {
    type Hessian = f64;
    fn hessian(&self, _: &f64) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
}

impl HessianProduct for Quadratic {
    fn hessian_product(&self, _: &f64, v: &f64) -> Result<f64, Self::Error> {
        Ok(*v)
    }
}

impl Residual for Quadratic {
    type Param = f64;
    type Output = [f64; 1];
    type Error = &'static str;
    fn residual(&self, x: &f64) -> Result<[f64; 1], Self::Error> {
        Ok([*x])
    }
}

impl Jacobian for Quadratic {
    type Jacobian = [[f64; 1]; 1];
    fn jacobian(
        &self,
        _: &f64,
    ) -> Result<Self::Jacobian, <Self as Residual>::Error> {
        Ok([[1.0]])
    }
}

fn evaluate_all(
    problem: &mut Problem<Quadratic>,
    x: f64,
) -> Result<f64, &'static str> {
    let (cost, _, _) = problem.cost_and_gradient_and_hessian(&x)?;
    problem.residual_and_jacobian(&x)?;
    problem.cost_batch(&[x, x])?;
    problem.gradient(&x)?;
    for _ in 0..2 {
        problem.residual(&x)?;
    }
    for _ in 0..3 {
        problem.jacobian(&x)?;
    }
    for _ in 0..4 {
        problem.hessian(&x)?;
    }
    for _ in 0..6 {
        problem.hessian_product(&x, &1.0)?;
    }
    Ok(cost)
}

const WORK: EvalCounts = EvalCounts {
    cost_evals: 3,
    gradient_evals: 2,
    residual_evals: 3,
    jacobian_evals: 4,
    hessian_evals: 5,
    hessian_product_evals: 6,
};

enum WorkStep {
    Complete,
    Stop,
    Fail,
}

impl Solver<Quadratic, PointState<f64>> for WorkStep {
    type Error = &'static str;
    fn init(
        &mut self,
        problem: &mut Problem<Quadratic>,
        mut state: PointState<f64>,
    ) -> Result<PointState<f64>, Self::Error> {
        state.reset();
        state.replace(1.0, evaluate_all(problem, 1.0)?);
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<Quadratic>,
        mut state: PointState<f64>,
    ) -> Result<(PointState<f64>, Option<TerminationReason>), Self::Error> {
        state.replace(0.0, evaluate_all(problem, 0.0)?);
        if matches!(self, Self::Fail) {
            problem.cost_batch(&[0.0, 999.0, 2.0])?;
        }
        Ok((
            state,
            matches!(self, Self::Stop)
                .then_some(TerminationReason::UserRequested),
        ))
    }
}

#[test]
fn every_raw_budget_counts_fused_and_batch_work_without_folding() {
    for (kind, work) in [
        (EvaluationKind::Cost, 3),
        (EvaluationKind::Gradient, 2),
        (EvaluationKind::Residual, 3),
        (EvaluationKind::Jacobian, 4),
        (EvaluationKind::Hessian, 5),
        (EvaluationKind::HessianProduct, 6),
        (EvaluationKind::TotalWork, 23),
    ] {
        let result =
            Executor::new(Quadratic, WorkStep::Complete, PointState::new(1.0))
                .max_evaluations(kind, work + 1)
                .run_with_solver()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::MaxEvaluations(kind));
        assert_eq!(result.iter(), 1);
        let mut expected = WORK;
        expected.add(&WORK);
        assert_eq!(result.state.raw_counts(), &expected);
        assert_eq!(result.counts, expected);
        assert_eq!(result.state.incumbent_record().unwrap().counts, &expected);
    }
}

#[test]
fn clean_stops_stamp_all_counts_and_errors_keep_charged_batch_work() {
    let result = Executor::new(Quadratic, WorkStep::Stop, PointState::new(1.0))
        .require_evaluated_state()
        .max_evaluations(EvaluationKind::Cost, 100)
        .run_with_solver()
        .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(result.reason, TerminationReason::UserRequested);
    let best = result.state.incumbent_record().unwrap();
    assert_eq!((*best.param, best.cost, best.iter), (0.0, 0.0, 0));
    assert_eq!(best.counts, &result.counts);
    assert_eq!(best.counts.total_work(), 46);

    let calls = Rc::new(Cell::new(0));
    let mut stepper =
        Executor::new(Quadratic, WorkStep::Fail, PointState::new(1.0))
            .require_evaluated_state()
            .observe_with(CountObservers(calls.clone()), ObserverMode::Always)
            .into_stepper()
            .unwrap();
    assert_eq!(stepper.step(), Err("evaluation failed"));
    assert_eq!(stepper.counts().cost_evals, 9);
    assert_eq!(stepper.counts().total_work(), 49);
    assert_eq!(calls.get(), 1);
    assert!(stepper.into_checkpoint().is_none());
}

// Feasibility-first selection belongs to this external solver's state. The
// shared point stores its current record and counters, but not its incumbent.
struct Selected {
    current: PointState<f64>,
    selection: Option<u64>,
    pending: bool,
    best: Option<(f64, f64, u64, EvalCounts)>,
}

impl Selected {
    fn select(&mut self, id: u64, point: f64, cost: f64) {
        self.pending |= self.selection != Some(id);
        self.selection = Some(id);
        self.current.replace(point, cost);
    }
}

impl State for Selected {
    type Param = f64;
    type Float = f64;
    fn iter(&self) -> u64 {
        self.current.iter()
    }
    fn increment_iter(&mut self) {
        self.current.increment_iter();
    }
    fn cost_evals(&self) -> u64 {
        self.current.cost_evals()
    }
    fn param(&self) -> &f64 {
        self.current.param()
    }
    fn cost(&self) -> f64 {
        self.current.cost()
    }
    fn best_param(&self) -> &f64 {
        &self.best.as_ref().unwrap().0
    }
    fn best_cost(&self) -> f64 {
        self.best.as_ref().map_or(f64::INFINITY, |best| best.1)
    }
    fn best_iter(&self) -> u64 {
        self.best.as_ref().map_or(0, |best| best.2)
    }
    fn best_cost_evals(&self) -> u64 {
        self.best.as_ref().map_or(0, |best| best.3.total_work())
    }
    fn update_best(&mut self) {
        if std::mem::take(&mut self.pending) {
            self.best = Some((
                *self.param(),
                self.cost(),
                self.iter(),
                *self.raw_counts(),
            ));
        }
    }
    fn reset_best(&mut self) {
        self.best = None;
        self.selection = None;
        self.pending = false;
    }
}

impl CountsMirror for Selected {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.current.mirror(counts);
    }
}

impl RawEvaluationState for Selected {
    fn raw_counts(&self) -> &EvalCounts {
        self.current.raw_counts()
    }
}

impl EvaluatedState for Selected {
    fn current_record(&self) -> Option<(&f64, f64)> {
        self.current.current_record()
    }
}

impl IncumbentState for Selected {
    fn incumbent_record(&self) -> Option<IncumbentRef<'_, f64>> {
        let (param, cost, iter, counts) = self.best.as_ref()?;
        Some(IncumbentRef {
            param,
            cost: *cost,
            iter: *iter,
            counts,
        })
    }
}

#[test]
fn external_selection_is_explicit_and_does_not_claim_objective_ordering() {
    let mut state = Selected {
        current: PointState::new(-1.0),
        selection: None,
        pending: false,
        best: None,
    };
    state.select(1, -1.0, -100.0);
    state.mirror(&WORK);
    state.update_best();
    // The feasible replacement has a higher cost and is published in the
    // same iteration, so iteration numbers cannot identify this event.
    state.select(2, 1.0, 10.0);
    let mut later = WORK;
    later.add(&WORK);
    state.mirror(&later);
    state.update_best();
    assert_eq!(*state.incumbent_record().unwrap().param, 1.0);
    assert_eq!(state.incumbent_record().unwrap().cost, 10.0);
    let published_counts = later;
    for _ in 0..2 {
        state.increment_iter();
        state.select(2, 1.0, 10.0);
        later.add(&WORK);
        state.mirror(&later);
        state.update_best();
        let best = state.incumbent_record().unwrap();
        assert_eq!(best.iter, 0);
        assert_eq!(best.counts, &published_counts);
    }
    // Inference becomes ambiguous if a blanket implementation accidentally
    // grants objective controls to this feasibility-first state.
    trait WithoutObjective<A> {
        fn check() {}
    }
    impl<T: ?Sized> WithoutObjective<()> for T {}
    struct HasObjective;
    impl<T: ?Sized + ObjectiveIncumbentState> WithoutObjective<HasObjective> for T {}
    let _ = <Selected as WithoutObjective<_>>::check;
}

#[cfg(feature = "serde")]
#[test]
fn existing_binary_layouts_remain_compatible() {
    let config = bincode::config::standard();
    // Basin 1.11's unevaluated scalar seed: parameter, absent record and
    // incumbent, then the iteration and six raw counters.
    let seed_bytes = [0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let (seed, read): (PointState<f64>, _) =
        bincode::serde::decode_from_slice(&seed_bytes, config).unwrap();
    assert_eq!(read, seed_bytes.len());
    assert_eq!(*seed.param(), 1.0);
    assert!(seed.current_record().is_none());
    assert_eq!(
        bincode::serde::encode_to_vec(seed, config).unwrap(),
        seed_bytes
    );

    // Empty trajectory, next index zero, max_iter ten, and three unset limits.
    let inner_bytes = [0, 0, 10, 0, 0, 0];
    let (inner, read): (InnerExecutor<PointState<f64>, Trajectory>, _) =
        bincode::serde::decode_from_slice(&inner_bytes, config).unwrap();
    assert_eq!(read, inner_bytes.len());
    assert_eq!(
        bincode::serde::encode_to_vec(inner, config).unwrap(),
        inner_bytes
    );
    assert_eq!(
        bincode::serde::encode_to_vec(
            TerminationReason::NumericalNoProgress,
            config
        )
        .unwrap(),
        [22]
    );
}
