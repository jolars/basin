use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use basin::{
    Backtracking, BasinHopping, BoundedFiniteDiff, CmaEs, Cobyla, Constrained,
    Constraints, CostFunction, DenseMatrix, EvalCounts, EvaluationBudget,
    EvaluationContext, EvaluationError, EvaluationKind, Executor, FiniteDiff,
    Gradient, GradientDescent, Hessian, HessianProduct, HuberLoss, Jacobian,
    LevenbergMarquardt, NelderMead, NoConstraints, Observe, ObserverMode,
    PointState, PopulationProgress, ProblemConstraints, Residual,
    RobustLeastSquares, Scalar, State, Termination, TerminationReport,
    TerminationStage, VectorLen,
};

#[derive(Default)]
struct Calls {
    costs: AtomicU64,
    gradients: AtomicU64,
    residuals: AtomicU64,
    jacobians: AtomicU64,
}

#[derive(Debug, PartialEq)]
struct CallbackFailure(u64);

struct Probe {
    calls: Arc<Calls>,
    fail_at: Option<u64>,
}

impl Probe {
    fn new(calls: &Arc<Calls>) -> Self {
        Self {
            calls: calls.clone(),
            fail_at: None,
        }
    }
}
impl ProblemConstraints for Probe {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        NoConstraints::EMPTY
    }
}
impl CostFunction for Probe {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = CallbackFailure;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        let call = self.calls.costs.fetch_add(1, Ordering::Relaxed);
        if self.fail_at == Some(call) {
            return Err(CallbackFailure(call));
        }
        Ok(x.iter().map(|v| v * v).sum())
    }
}
impl Gradient for Probe {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.calls.gradients.fetch_add(1, Ordering::Relaxed);
        Ok(x.iter().map(|v| 2.0 * v).collect())
    }
}
impl Residual for Probe {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = CallbackFailure;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.calls.residuals.fetch_add(1, Ordering::Relaxed);
        Ok(x.clone())
    }
}
impl Jacobian for Probe {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        self.calls.jacobians.fetch_add(1, Ordering::Relaxed);
        Ok(DenseMatrix::from_fn(x.len(), x.len(), |i, j| {
            f64::from(i == j)
        }))
    }
}
impl basin::NonlinearEqualityValues for Probe {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = CallbackFailure;
    fn nonlinear_equalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, basin::ConstraintSolveError<Self::Error>> {
        self.calls.residuals.fetch_add(1, Ordering::Relaxed);
        Ok(vec![x[1] - 0.5])
    }
}
impl basin::NonlinearInequalityValues for Probe {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = CallbackFailure;
    fn nonlinear_inequalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, basin::ConstraintSolveError<Self::Error>> {
        self.calls.residuals.fetch_add(1, Ordering::Relaxed);
        Ok(vec![x[0] - 0.5])
    }
}

struct FixedObjective {
    calls: Arc<AtomicU64>,
    value: f64,
}
impl ProblemConstraints for FixedObjective {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        NoConstraints::EMPTY
    }
}
impl CostFunction for FixedObjective {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.value)
    }
}

fn assert_coherent<S: State<Param = Vec<f64>, Float = f64>>(state: &S) {
    let expected: f64 = state.param().iter().map(|v| v * v).sum();
    assert!((state.cost() - expected).abs() < 1e-12);
    let best: f64 = state.best_param().iter().map(|v| v * v).sum();
    assert!((state.best_cost() - best).abs() < 1e-12);
}

#[test]
fn cobyla_enforces_every_small_cap_including_incomplete_initialization() {
    for budget in 0..40 {
        let calls = Arc::new(Calls::default());
        let result = Executor::from_start(
            Probe::new(&calls),
            Cobyla::new().with_final_radius(1e-15),
            vec![1.0, 2.0],
        )
        .max_evaluations(EvaluationKind::Cost, budget)
        .run()
        .unwrap();
        assert_eq!(calls.costs.load(Ordering::Relaxed), budget);
        assert_eq!(result.counts.cost_evals, budget);
        assert!(matches!(result.report.termination, Termination::Limit(_)));
        if budget < 3 {
            assert!(result.initialized_state().is_none());
            assert_eq!(result.report.stage, TerminationStage::Initialization);
        } else {
            let state = result.initialized_state().unwrap();
            assert_coherent(state);
            assert_eq!(state.cost_evals(), budget);
            assert_eq!(state.iter(), result.report.iteration);
        }
    }
    let calls = Arc::new(Calls::default());
    let mut stepper = Executor::from_start(
        Probe::new(&calls),
        Cobyla::new().with_final_radius(1e-15),
        vec![1.0, 2.0],
    )
    .max_iter(200)
    .into_stepper()
    .unwrap();
    let mut sizes = [false; 3];
    for _ in 0..100 {
        let before = stepper.counts().cost_evals;
        let _ = stepper.step().unwrap();
        let delta = stepper.counts().cost_evals - before;
        assert!(delta <= 2);
        sizes[delta as usize] = true;
    }
    let mut flat = Executor::from_start(
        FixedObjective {
            calls: Arc::new(AtomicU64::new(0)),
            value: 0.0,
        },
        Cobyla::new(),
        vec![1.0, 2.0],
    )
    .into_stepper()
    .unwrap();
    for _ in 0..100 {
        let before = flat.counts().cost_evals;
        let outcome = flat.step().unwrap();
        let delta = flat.counts().cost_evals - before;
        assert!(delta <= 2);
        sizes[delta as usize] = true;
        if matches!(outcome, basin::StepOutcome::Stopped(_)) {
            break;
        }
    }
    assert_eq!(sizes, [true; 3]);
}

#[test]
fn population_batches_reserve_before_dispatch_and_retain_published_members() {
    for (budget, used, initialized) in [
        (3, 0, false),
        (4, 4, false),
        (5, 5, true),
        (8, 5, true),
        (9, 5, true),
    ] {
        let calls = Arc::new(Calls::default());
        let result = Executor::new(
            Probe::new(&calls),
            CmaEs::<Vec<f64>, DenseMatrix>::new(4, 0.3).with_lambda(4),
            PopulationProgress::from_point(vec![1.0, 2.0]),
        )
        .max_evaluations(EvaluationKind::Cost, budget)
        .run()
        .unwrap();
        assert_eq!(result.counts.cost_evals, used);
        assert_eq!(calls.costs.load(Ordering::Relaxed), used);
        assert_eq!(result.initialized_state().is_some(), initialized);
        if let Some(state) = result.initialized_state() {
            assert_coherent(state);
            assert_eq!(state.iter(), 0);
        }
        if budget == 8 {
            assert_eq!(
                result.report.stage,
                TerminationStage::Step { completed: false }
            );
        }
    }
}

#[test]
fn fused_reservations_are_atomic_across_categories_and_total_work() {
    for budget in [
        EvaluationBudget::new().with_limit(EvaluationKind::Gradient, 0),
        EvaluationBudget::new().with_limit(EvaluationKind::TotalWork, 1),
    ] {
        let calls = Arc::new(Calls::default());
        let context = EvaluationContext::new(budget);
        let probe = Probe::new(&calls);
        assert!(matches!(
            context.wrap(&probe).cost_and_gradient(&vec![1.0, 2.0]),
            Err(EvaluationError::BudgetExhausted(_))
        ));
        assert_eq!(context.counts(), EvalCounts::default());
        assert_eq!(calls.costs.load(Ordering::Relaxed), 0);
        assert_eq!(calls.gradients.load(Ordering::Relaxed), 0);
    }
    let calls = Arc::new(Calls::default());
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::TotalWork, 2),
    );
    let probe = Probe::new(&calls);
    assert_eq!(
        context
            .wrap(&probe)
            .cost_and_gradient(&vec![1.0, 2.0])
            .unwrap(),
        (5.0, vec![2.0, 4.0])
    );
    assert_eq!(context.counts().cost_evals, 1);
    assert_eq!(context.counts().gradient_evals, 1);
    assert!(matches!(
        context.wrap(&probe).cost(&vec![1.0]),
        Err(EvaluationError::BudgetExhausted(_))
    ));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 1);
}

#[test]
fn parallel_cost_batches_charge_once_and_preserve_order() {
    let calls = Arc::new(Calls::default());
    let probe = Probe::new(&calls);
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::Cost, 3),
    );
    assert_eq!(
        context
            .wrap(&probe)
            .cost_batch(&[vec![3.0], vec![1.0], vec![2.0]])
            .unwrap(),
        vec![9.0, 1.0, 4.0]
    );
    assert_eq!(context.counts().cost_evals, 3);
    assert_eq!(calls.costs.load(Ordering::Relaxed), 3);
}

#[test]
fn callback_failure_at_the_last_allowed_call_remains_an_error() {
    let calls = Arc::new(Calls::default());
    let result = Executor::from_start(
        Probe {
            calls: calls.clone(),
            fail_at: Some(2),
        },
        Cobyla::new(),
        vec![1.0, 2.0],
    )
    .max_evaluations(EvaluationKind::Cost, 3)
    .run();
    assert!(matches!(
        result,
        Err(basin::ConstraintSolveError::Evaluation(CallbackFailure(2)))
    ));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 3);
}

#[test]
fn finite_differences_count_probes_and_interrupt_incomplete_gradients() {
    for budget in [1, 2, 3, 4, 5, 11] {
        let calls = Arc::new(Calls::default());
        let result = Executor::from_start(
            FiniteDiff::new(Probe::new(&calls)),
            GradientDescent::new(0.1),
            vec![1.0, 2.0],
        )
        .max_evaluations(EvaluationKind::Cost, budget)
        .run()
        .unwrap();
        assert_eq!(result.counts.cost_evals, budget);
        assert_eq!(calls.costs.load(Ordering::Relaxed), budget);
        assert_eq!(calls.gradients.load(Ordering::Relaxed), 0);
        if budget < 5 {
            assert!(result.initialized_state().is_none());
        } else {
            assert_coherent(result.initialized_state().unwrap());
        }
    }
    let calls = Arc::new(Calls::default());
    let result = Executor::from_start(
        BoundedFiniteDiff::new(
            Probe::new(&calls),
            vec![0.0, 0.0],
            vec![3.0, 3.0],
        ),
        GradientDescent::new(0.1),
        vec![1.0, 2.0],
    )
    .max_evaluations(EvaluationKind::Cost, 5)
    .run()
    .unwrap();
    assert_eq!(result.counts.cost_evals, 5);
    assert_eq!(calls.costs.load(Ordering::Relaxed), 5);
    assert!(result.initialized_state().is_none());
}

#[test]
fn synthesized_hessians_and_products_share_probe_accounting() {
    let calls = Arc::new(Calls::default());
    let probe = FiniteDiff::new(Probe::new(&calls));
    let context = EvaluationContext::new(EvaluationBudget::new());
    let guarded = context.wrap(&probe);
    let h = guarded.hessian(&vec![1.0, 2.0]).unwrap();
    assert!((h.get(0, 0) - 2.0).abs() < 1e-5);
    assert!((h.get(1, 1) - 2.0).abs() < 1e-5);
    let hv = guarded
        .hessian_product(&vec![1.0, 2.0], &vec![1.0, 0.0])
        .unwrap();
    assert!((hv[0] - 2.0).abs() < 1e-3);
    assert!(hv[1].abs() < 1e-3);
    assert_eq!(context.counts().hessian_evals, 1);
    assert_eq!(context.counts().hessian_product_evals, 1);
    assert_eq!(context.counts().gradient_evals, 2);
    assert_eq!(
        context.counts().cost_evals,
        calls.costs.load(Ordering::Relaxed)
    );
}

#[test]
fn residual_probes_and_robust_nlls_share_the_hard_budget() {
    let calls = Arc::new(Calls::default());
    let result = Executor::from_start(
        FiniteDiff::new(Probe::new(&calls)),
        LevenbergMarquardt::<Vec<f64>, DenseMatrix>::new(),
        vec![1.0, 2.0],
    )
    .max_evaluations(EvaluationKind::Residual, 3)
    .run()
    .unwrap();
    assert_eq!(result.counts.residual_evals, 3);
    assert_eq!(calls.residuals.load(Ordering::Relaxed), 3);
    assert!(result.initialized_state().is_none());

    let calls = Arc::new(Calls::default());
    let result = Executor::from_start(
        RobustLeastSquares::new(Probe::new(&calls), HuberLoss),
        LevenbergMarquardt::<Vec<f64>, DenseMatrix>::new(),
        vec![1.0, 2.0],
    )
    .max_evaluations(EvaluationKind::Residual, 2)
    .run()
    .unwrap();
    assert_eq!(result.counts.residual_evals, 2);
    assert_eq!(calls.residuals.load(Ordering::Relaxed), 2);
    assert!(result.initialized_state().is_some());
    assert_eq!(result.counts.cost_evals, 0);
}

#[test]
fn constraint_blocks_and_their_finite_differences_are_guarded() {
    let calls = Arc::new(Calls::default());
    let constraints = Constraints::new()
        .with_nonlinear_equalities(1)
        .with_nonlinear_inequalities(1);
    let result = Executor::from_start(
        Constrained::new(Probe::new(&calls), constraints),
        Cobyla::new(),
        vec![1.0, 2.0],
    )
    .max_evaluations(EvaluationKind::Residual, 3)
    .run()
    .unwrap();
    assert_eq!(result.counts.residual_evals, 3);
    assert_eq!(calls.residuals.load(Ordering::Relaxed), 3);
    assert!(result.initialized_state().is_none());

    let calls = Arc::new(Calls::default());
    let constraints = Constraints::new()
        .with_nonlinear_equalities(1)
        .with_nonlinear_inequalities(1);
    let problem =
        FiniteDiff::new(Constrained::new(Probe::new(&calls), constraints));
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::Residual, 3),
    );
    let result = basin::ConstraintJacobian::constraint_jacobian(
        &context.wrap(&problem),
        &vec![1.0, 2.0],
    );
    assert!(matches!(
        result,
        Err(basin::ConstraintSolveError::Evaluation(
            EvaluationError::BudgetExhausted(_)
        ))
    ));
    assert_eq!(context.counts().residual_evals, 3);
    assert_eq!(calls.residuals.load(Ordering::Relaxed), 3);
}

#[test]
fn nested_solvers_and_adapter_problems_cannot_reset_the_allowance() {
    let calls = Arc::new(Calls::default());
    let result = Executor::new(
        Probe::new(&calls),
        BasinHopping::new(NelderMead::new(), 42).with_inner_max_iter(1),
        PointState::new(vec![1.0, 2.0]),
    )
    .max_evaluations(EvaluationKind::Cost, 9)
    .run()
    .unwrap();
    assert_eq!(result.counts.cost_evals, 9);
    assert_eq!(calls.costs.load(Ordering::Relaxed), 9);
    assert_coherent(result.initialized_state().unwrap());

    let calls = Arc::new(Calls::default());
    let constraints =
        Constraints::new().with_linear_inequalities(basin::LinearBlock::new(
            DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            vec![3.0],
        ));
    let result = Executor::new(
        Constrained::new(Probe::new(&calls), constraints),
        basin::BarrierMethod::with_inner_solver(NelderMead::new())
            .with_inner_max_iter(50),
        PointState::new(vec![1.0, 1.0]),
    )
    .max_evaluations(EvaluationKind::Cost, 7)
    .run()
    .unwrap();
    assert_eq!(result.counts.cost_evals, 7);
    assert_eq!(calls.costs.load(Ordering::Relaxed), 7);
    assert_coherent(result.initialized_state().unwrap());
}

#[derive(Default)]
struct Events {
    init: u64,
    iter: u64,
    final_calls: u64,
    final_costs: u64,
}
struct Watch(Arc<Mutex<Events>>);
impl<S: State> Observe<S> for Watch {
    fn observe_init(&mut self, _: &S) {
        self.0.lock().unwrap().init += 1;
    }
    fn observe_iter(&mut self, _: &S) {
        self.0.lock().unwrap().iter += 1;
    }
    fn observe_final(
        &mut self,
        state: &S,
        report: &TerminationReport<S::Float>,
    ) {
        assert_eq!(state.iter(), report.iteration);
        let mut events = self.0.lock().unwrap();
        events.final_calls += 1;
        events.final_costs = state.cost_evals();
    }
}

#[test]
fn interrupted_steps_publish_only_coherent_progress_to_observers() {
    let calls = Arc::new(Calls::default());
    let events = Arc::new(Mutex::new(Events::default()));
    let result = Executor::from_start(
        Probe::new(&calls),
        GradientDescent::with_line_search(Backtracking::new()),
        vec![1.0, 2.0],
    )
    .max_evaluations(EvaluationKind::Cost, 2)
    .observe_with(Watch(events.clone()), ObserverMode::Always)
    .run()
    .unwrap();
    assert!(matches!(
        result.report.stage,
        TerminationStage::Interrupted { .. }
    ));
    let state = result.state;
    assert_coherent(&state);
    assert_eq!(state.param(), &vec![1.0, 2.0]);
    let events = events.lock().unwrap();
    assert_eq!(
        (
            events.init,
            events.iter,
            events.final_calls,
            events.final_costs
        ),
        (1, 0, 1, 2)
    );
}

struct GenericSphere<V, F>(PhantomData<(V, F)>);
impl<V, F> ProblemConstraints for GenericSphere<V, F> {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        NoConstraints::EMPTY
    }
}
impl<V: VectorLen + Index<usize, Output = F>, F: Scalar> CostFunction
    for GenericSphere<V, F>
{
    type Param = V;
    type Output = F;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        Ok((0..x.vec_len()).map(|i| x[i] * x[i]).sum())
    }
}
fn backend<V, F>(start: V)
where
    V: Clone + VectorLen + Index<usize, Output = F> + IndexMut<usize>,
    F: Scalar,
{
    let result = Executor::from_start(
        GenericSphere::<V, F>(PhantomData),
        Cobyla::new(),
        start,
    )
    .max_evaluations(EvaluationKind::Cost, 7)
    .run()
    .unwrap();
    assert_eq!(result.counts.cost_evals, 7);
    let state = result.state;
    let cost: F = (0..state.param().vec_len())
        .map(|i| state.param()[i] * state.param()[i])
        .sum();
    assert!(
        (state.cost() - cost).abs()
            <= F::epsilon() * F::from_f64(10.0).unwrap()
    );
}
#[test]
fn vec_f32_and_f64() {
    backend(vec![1.0_f32, 2.0]);
    backend(vec![1.0_f64, 2.0]);
}
#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_f32_and_f64() {
    backend(nalgebra::DVector::from_vec(vec![1.0_f32, 2.0]));
    backend(nalgebra::DVector::from_vec(vec![1.0_f64, 2.0]));
}
#[cfg(feature = "ndarray")]
#[test]
fn ndarray_f32_and_f64() {
    backend(ndarray::Array1::from_vec(vec![1.0_f32, 2.0]));
    backend(ndarray::Array1::from_vec(vec![1.0_f64, 2.0]));
}
#[cfg(feature = "faer")]
#[test]
fn faer_f32_and_f64() {
    backend(faer::Col::from_fn(2, |i| (i + 1) as f32));
    backend(faer::Col::from_fn(2, |i| (i + 1) as f64));
}

#[test]
fn nonfinite_callbacks_and_zero_allowances_never_escape_the_cap() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for budget in [0, 1, 3, 8] {
            let calls = Arc::new(AtomicU64::new(0));
            let result = Executor::from_start(
                FixedObjective {
                    calls: calls.clone(),
                    value,
                },
                Cobyla::new(),
                vec![1.0, 2.0],
            )
            .max_evaluations(EvaluationKind::Cost, budget)
            .run()
            .unwrap();
            assert!(result.counts.cost_evals <= budget);
            assert_eq!(calls.load(Ordering::Relaxed), result.counts.cost_evals);
        }
    }
    let events = Arc::new(Mutex::new(Events::default()));
    let calls = Arc::new(Calls::default());
    let result =
        Executor::from_start(Probe::new(&calls), Cobyla::new(), vec![1.0, 2.0])
            .max_evaluations(EvaluationKind::Cost, 0)
            .observe_with(Watch(events.clone()), ObserverMode::Always)
            .run()
            .unwrap();
    assert!(result.initialized_state().is_none());
    let events = events.lock().unwrap();
    assert_eq!((events.init, events.iter, events.final_calls), (0, 0, 0));
}

#[test]
fn direct_barrier_and_penalty_adapters_forward_finite_difference_contexts() {
    let calls = Arc::new(Calls::default());
    let constraints =
        Constraints::new().with_linear_inequalities(basin::LinearBlock::new(
            DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            vec![3.0],
        ));
    let problem =
        Constrained::new(FiniteDiff::new(Probe::new(&calls)), constraints);
    let barrier = basin::LogBarrier::new(&problem, 0.1);
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::Cost, 3),
    );
    assert!(matches!(
        context.wrap(&barrier).gradient(&vec![1.0, 1.0]),
        Err(EvaluationError::BudgetExhausted(_))
    ));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 3);
    assert_eq!(context.counts().cost_evals, 3);

    let calls = Arc::new(Calls::default());
    let constraints =
        Constraints::new().with_linear_equalities(basin::LinearBlock::new(
            DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            vec![3.0],
        ));
    let problem =
        Constrained::new(FiniteDiff::new(Probe::new(&calls)), constraints);
    let multipliers = vec![0.0];
    let penalty = basin::AugmentedLagrangian::new(&problem, &multipliers, 1.0);
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::Cost, 3),
    );
    assert!(matches!(
        context.wrap(&penalty).gradient(&vec![1.0, 1.0]),
        Err(EvaluationError::BudgetExhausted(_))
    ));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 3);
    assert_eq!(context.counts().cost_evals, 3);
}

#[test]
fn bounded_robust_least_squares_preserves_its_native_evaluation_path() {
    let calls = Arc::new(Calls::default());
    let problem = RobustLeastSquares::new(
        Constrained::new(
            Probe::new(&calls),
            Constraints::new().with_bounds(basin::BoxBounds::new(
                Some(vec![-3.0, -3.0]),
                Some(vec![3.0, 3.0]),
            )),
        ),
        HuberLoss,
    );
    let result = Executor::from_start(
        problem,
        basin::TrustRegionReflective::new(),
        vec![1.0, 2.0],
    )
    .max_evaluations(EvaluationKind::Residual, 2)
    .run()
    .unwrap();
    assert_eq!(calls.residuals.load(Ordering::Relaxed), 2);
    assert_eq!(result.counts.residual_evals, 2);
    assert_eq!(result.counts.cost_evals, 0);
    assert!(result.initialized_state().is_some());
}

#[test]
fn unrepresentable_work_is_declined_without_counter_overflow() {
    let context = EvaluationContext::new(EvaluationBudget::new());
    assert!(
        context
            .reserve(EvalCounts {
                cost_evals: u64::MAX,
                gradient_evals: 1,
                ..EvalCounts::default()
            })
            .is_err()
    );
    assert_eq!(context.counts(), EvalCounts::default());
    let context = EvaluationContext::new(EvaluationBudget::new());
    context
        .reserve(EvalCounts {
            cost_evals: u64::MAX,
            ..EvalCounts::default()
        })
        .unwrap();
    assert!(
        context
            .reserve(EvalCounts {
                gradient_evals: 1,
                ..EvalCounts::default()
            })
            .is_err()
    );
    assert_eq!(context.counts().cost_evals, u64::MAX);
    assert_eq!(context.counts().gradient_evals, 0);
}

#[test]
fn transparent_adapters_preserve_whole_batch_reservations() {
    let calls = Arc::new(Calls::default());
    let probe = Probe::new(&calls);
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::Cost, 2),
    );
    let adapter = FiniteDiff::new(context.wrap(&probe));
    let result = adapter.cost_batch(&[vec![1.0], vec![2.0], vec![3.0]]);
    assert!(matches!(result, Err(EvaluationError::BudgetExhausted(_))));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 0);
    assert_eq!(context.counts(), EvalCounts::default());

    let calls = Arc::new(Calls::default());
    let probe = Probe::new(&calls);
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::Cost, 2),
    );
    let adapter = Constrained::new(context.wrap(&probe), Constraints::new());
    let result = adapter.cost_batch(&[vec![1.0], vec![2.0], vec![3.0]]);
    assert!(matches!(result, Err(EvaluationError::BudgetExhausted(_))));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 0);
    assert_eq!(context.counts(), EvalCounts::default());
}

#[test]
fn batch_reservations_cannot_be_reused_to_exceed_their_allowance() {
    let calls = Arc::new(Calls::default());
    let probe = Probe::new(&calls);
    let context = EvaluationContext::new(
        EvaluationBudget::new().with_limit(EvaluationKind::Cost, 1),
    );
    let guarded = context.wrap(&probe);
    let reservation = guarded.prepare_cost_batch(1).unwrap();
    assert_eq!(
        guarded.cost_in_batch(&vec![2.0], &reservation).unwrap(),
        4.0
    );
    assert!(matches!(
        guarded.cost_in_batch(&vec![2.0], &reservation),
        Err(EvaluationError::BudgetExhausted(_))
    ));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 1);
    assert_eq!(context.counts().cost_evals, 1);

    let other_context = EvaluationContext::new(EvaluationBudget::new());
    let reservation = other_context.wrap(&probe).prepare_cost_batch(1).unwrap();
    assert!(matches!(
        guarded.cost_in_batch(&vec![2.0], &reservation),
        Err(EvaluationError::BudgetExhausted(_))
    ));
    assert_eq!(calls.costs.load(Ordering::Relaxed), 1);
}

impl Hessian for Probe {
    type Hessian = DenseMatrix;
    fn hessian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(DenseMatrix::from_fn(x.len(), x.len(), |i, j| {
            if i == j { 2.0 } else { 0.0 }
        }))
    }
}
impl HessianProduct for Probe {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Self::Error> {
        Ok(v.iter().map(|v| 2.0 * v).collect())
    }
}

fn sweep_caps<P, So, S>(
    problem: impl Fn(&Arc<Calls>) -> P,
    solver: impl Fn() -> So,
    state: impl Fn() -> S,
) where
    P: CostFunction<Param = Vec<f64>, Output = f64>,
    So: basin::Solver<P, S>,
    So::Error: std::fmt::Debug,
    S: State<Param = Vec<f64>, Float = f64>
        + basin::CountsMirror
        + basin::EvaluatedState,
{
    for kind in [EvaluationKind::Cost, EvaluationKind::TotalWork] {
        for cap in 0..32 {
            let calls = Arc::new(Calls::default());
            let result = Executor::new(problem(&calls), solver(), state())
                .max_iter(6)
                .require_evaluated_state()
                .max_evaluations(kind, cap)
                .run()
                .unwrap();
            let used = match kind {
                EvaluationKind::Cost => result.counts.cost_evals,
                EvaluationKind::TotalWork => result.counts.total_work(),
                _ => unreachable!(),
            };
            assert!(used <= cap, "{kind:?}: {used} exceeded {cap}");
            assert_eq!(
                result.counts.cost_evals,
                calls.costs.load(Ordering::Relaxed)
            );
            if let Some(state) = result.initialized_state() {
                let expected: f64 = state.param().iter().map(|v| v * v).sum();
                assert!(
                    (state.cost() - expected).abs() < 1e-12,
                    "{} {kind:?} cap {cap}: cost {} at {:?}",
                    std::any::type_name::<So>(),
                    state.cost(),
                    state.param()
                );
                assert_coherent(state);
                assert!(state.current_record().is_some());
            }
        }
    }
}

#[test]
fn existing_budget_builders_preserve_progress_across_solver_families() {
    let first_order = || basin::FirstOrderState::new(vec![1.0, 2.0]);
    let population = || PopulationProgress::empty();
    let bounded = |calls: &Arc<Calls>| {
        Constrained::new(
            Probe::new(calls),
            Constraints::new().with_bounds(basin::BoxBounds::new(
                Some(vec![-3.0; 2]),
                Some(vec![3.0; 2]),
            )),
        )
    };
    sweep_caps(
        Probe::new,
        || GradientDescent::new(0.1).with_momentum(0.8),
        first_order,
    );
    sweep_caps(Probe::new, basin::Bfgs::new, first_order);
    sweep_caps(Probe::new, || basin::Lbfgs::new().unbounded(), first_order);
    sweep_caps(Probe::new, basin::TrustRegion::new, first_order);
    sweep_caps(Probe::new, basin::TrustRegion::matrix_free, first_order);
    sweep_caps(Probe::new, NelderMead::new, || {
        basin::SimplexProgress::from_simplex(vec![
            vec![1.0, 2.0],
            vec![1.1, 2.0],
            vec![1.0, 2.1],
        ])
    });
    sweep_caps(
        bounded,
        || basin::ProjectedGradientDescent::new(0.1),
        first_order,
    );
    sweep_caps(bounded, basin::Lbfgs::new, first_order);
    sweep_caps(
        bounded,
        || basin::GlobalBestPso::new(42).with_swarm_size(4),
        population,
    );
    sweep_caps(bounded, || basin::De::new(42).with_pop_size(6), population);
    sweep_caps(
        bounded,
        || {
            basin::Ssga::new(42)
                .with_pop_size(4)
                .with_offspring_per_step(3)
        },
        population,
    );
}

#[test]
fn least_squares_caps_retain_the_matching_residual_objective() {
    for kind in [
        EvaluationKind::Residual,
        EvaluationKind::Jacobian,
        EvaluationKind::TotalWork,
    ] {
        for cap in 0..20 {
            let calls = Arc::new(Calls::default());
            let result = Executor::from_start(
                Probe::new(&calls),
                LevenbergMarquardt::<Vec<f64>, DenseMatrix>::new(),
                vec![1.0, 2.0],
            )
            .max_evaluations(kind, cap)
            .max_iter(6)
            .require_evaluated_state()
            .run()
            .unwrap();
            let used = match kind {
                EvaluationKind::Residual => result.counts.residual_evals,
                EvaluationKind::Jacobian => result.counts.jacobian_evals,
                EvaluationKind::TotalWork => result.counts.total_work(),
                _ => unreachable!(),
            };
            assert!(used <= cap);
            assert_eq!(
                result.counts.residual_evals,
                calls.residuals.load(Ordering::Relaxed)
            );
            assert_eq!(
                result.counts.jacobian_evals,
                calls.jacobians.load(Ordering::Relaxed)
            );
            if let Some(state) = result.initialized_state() {
                let expected: f64 =
                    state.param().iter().map(|v| 0.5 * v * v).sum();
                assert!((state.cost() - expected).abs() < 1e-12);
            }
        }
    }
}

struct SwallowsBudget;
impl basin::Solver<Probe, PointState<Vec<f64>>> for SwallowsBudget {
    type Error = CallbackFailure;
    fn init(
        &mut self,
        problem: &mut basin::Problem<Probe>,
        state: &mut PointState<Vec<f64>>,
    ) -> basin::EvaluationResult<(), Self::Error> {
        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        Ok(())
    }
    fn next_iter(
        &mut self,
        problem: &mut basin::Problem<Probe>,
        state: &mut PointState<Vec<f64>>,
    ) -> basin::EvaluationResult<
        basin::SolverStep<PointState<Vec<f64>>>,
        Self::Error,
    > {
        problem.cost(state.param())?;
        let _ = problem.cost(state.param());
        Ok(basin::SolverStep::completed())
    }
}

#[test]
fn custom_error_handling_cannot_erase_a_declined_reservation() {
    let calls = Arc::new(Calls::default());
    let result = Executor::new(
        Probe::new(&calls),
        SwallowsBudget,
        PointState::new(vec![1.0]),
    )
    .max_cost_evals(2)
    .run_with_solver()
    .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(calls.costs.load(Ordering::Relaxed), 2);
    assert!(matches!(
        result.report.stage,
        TerminationStage::Interrupted { .. }
    ));
    assert!(result.into_checkpoint().is_none());
}

impl basin::ConstraintJacobian for Probe {
    type Param = Vec<f64>;
    type Jacobian = DenseMatrix;
    type Error = CallbackFailure;
    fn constraint_jacobian(
        &self,
        _: &Vec<f64>,
    ) -> Result<DenseMatrix, basin::ConstraintSolveError<Self::Error>> {
        self.calls.jacobians.fetch_add(1, Ordering::Relaxed);
        Ok(DenseMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]))
    }
}

#[test]
fn slsqp_guards_nonlinear_values_and_jacobians_during_every_transition() {
    for kind in [
        EvaluationKind::Residual,
        EvaluationKind::Jacobian,
        EvaluationKind::TotalWork,
    ] {
        for cap in 0..32 {
            let calls = Arc::new(Calls::default());
            let problem = Constrained::new(
                Probe::new(&calls),
                Constraints::new()
                    .with_nonlinear_equalities(1)
                    .with_nonlinear_inequalities(1),
            );
            let result = Executor::from_start(
                problem,
                basin::Slsqp::new(),
                vec![1.0, 2.0],
            )
            .max_evaluations(kind, cap)
            .max_iter(6)
            .require_evaluated_state()
            .run()
            .unwrap();
            let used = match kind {
                EvaluationKind::Residual => result.counts.residual_evals,
                EvaluationKind::Jacobian => result.counts.jacobian_evals,
                EvaluationKind::TotalWork => result.counts.total_work(),
                _ => unreachable!(),
            };
            assert!(used <= cap);
            assert_eq!(
                result.counts.residual_evals,
                calls.residuals.load(Ordering::Relaxed)
            );
            assert_eq!(
                result.counts.jacobian_evals,
                calls.jacobians.load(Ordering::Relaxed)
            );
            if let Some(state) = result.initialized_state() {
                assert_coherent(state);
            }
        }
    }
}
