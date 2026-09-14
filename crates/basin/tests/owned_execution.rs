//! Ownership extraction must work without cloneable or serializable machinery.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use basin::{
    CancellationToken, CheckpointSink, CostFunction, CountsMirror, EvalCounts,
    Executor, Gradient, Hessian, HessianProduct, Jacobian, Observe,
    ObserverMode, OptimizationResultWithSolver, Problem, Residual, Solver,
    State, StepOutcome, TerminationReason,
};

#[derive(Default)]
struct Quadratic {
    calls: Cell<u64>,
    fail_at: Option<u64>,
}

impl CostFunction for Quadratic {
    type Param = f64;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        if self.fail_at == Some(self.calls.get()) {
            return Err("evaluation failed");
        }
        Ok(x * x)
    }
}

impl Gradient for Quadratic {
    type Gradient = f64;

    fn gradient(&self, x: &f64) -> Result<f64, Self::Error> {
        Ok(2.0 * x)
    }
}

impl Hessian for Quadratic {
    type Hessian = f64;

    fn hessian(&self, _: &f64) -> Result<f64, Self::Error> {
        Ok(2.0)
    }
}

impl HessianProduct for Quadratic {
    fn hessian_product(&self, _: &f64, v: &f64) -> Result<f64, Self::Error> {
        Ok(2.0 * v)
    }
}

impl Residual for Quadratic {
    type Param = f64;
    type Output = f64;
    type Error = &'static str;

    fn residual(&self, x: &f64) -> Result<f64, Self::Error> {
        Ok(*x)
    }
}

impl Jacobian for Quadratic {
    type Jacobian = f64;

    fn jacobian(&self, _: &f64) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
}

struct OwnedState {
    point: Box<f64>,
    cost: f64,
    iter: u64,
    work: u64,
    best: (f64, f64, u64, u64),
}

impl OwnedState {
    fn new() -> Self {
        Self {
            point: Box::new(8.0),
            cost: f64::INFINITY,
            iter: 0,
            work: 0,
            best: (8.0, f64::INFINITY, 0, 0),
        }
    }
}

impl State for OwnedState {
    type Param = f64;
    type Float = f64;

    fn param(&self) -> &f64 {
        &self.point
    }
    fn cost(&self) -> f64 {
        self.cost
    }
    fn iter(&self) -> u64 {
        self.iter
    }
    fn increment_iter(&mut self) {
        self.iter += 1;
    }
    fn cost_evals(&self) -> u64 {
        self.work
    }
    fn best_param(&self) -> &f64 {
        &self.best.0
    }
    fn best_cost(&self) -> f64 {
        self.best.1
    }
    fn best_iter(&self) -> u64 {
        self.best.2
    }
    fn best_cost_evals(&self) -> u64 {
        self.best.3
    }
    fn update_best(&mut self) {
        if self.cost < self.best.1 {
            self.best = (*self.point, self.cost, self.iter, self.work);
        }
    }
    fn reset_best(&mut self) {
        self.best = (*self.point, f64::INFINITY, 0, 0);
    }
}

impl CountsMirror for OwnedState {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.work = counts.total_work();
    }
}

#[derive(Default)]
struct OwnedSolver {
    init_calls: u64,
    resets: u64,
    steps: u64,
    stop_mid_step: bool,
    workspace: Box<[u8; 32]>,
}

impl Solver<Quadratic, OwnedState> for OwnedSolver {
    type Error = &'static str;

    fn reset_convergence(&mut self) {
        self.resets += 1;
    }

    fn init(
        &mut self,
        problem: &mut Problem<Quadratic>,
        mut state: OwnedState,
    ) -> Result<OwnedState, Self::Error> {
        self.init_calls += 1;
        state.cost = problem.cost(state.param())?;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Quadratic>,
        mut state: OwnedState,
    ) -> Result<(OwnedState, Option<TerminationReason>), Self::Error> {
        self.steps += 1;
        self.workspace[0] += 1;
        *state.point *= 0.5;
        state.cost = problem.cost_and_gradient_and_hessian(state.param())?.0;
        problem.residual_and_jacobian(state.param())?;
        problem.hessian_product(state.param(), &1.0)?;
        Ok((
            state,
            self.stop_mid_step
                .then_some(TerminationReason::UserRequested),
        ))
    }
}

fn counts(steps: u64) -> EvalCounts {
    EvalCounts {
        cost_evals: steps + 1,
        gradient_evals: steps,
        residual_evals: steps,
        jacobian_evals: steps,
        hessian_evals: steps,
        hessian_product_evals: steps,
    }
}

type Log = Rc<RefCell<Vec<(&'static str, u64)>>>;

struct Recorder(Log);

impl Observe<OwnedState> for Recorder {
    fn observe_init(&mut self, state: &OwnedState) {
        self.0.borrow_mut().push(("init", state.iter()));
    }
    fn observe_iter(&mut self, state: &OwnedState) {
        self.0.borrow_mut().push(("iter", state.iter()));
    }
    fn observe_final(&mut self, state: &OwnedState, _: &TerminationReason) {
        self.0.borrow_mut().push(("final", state.iter()));
    }
}

impl CheckpointSink<OwnedSolver, OwnedState> for Recorder {
    fn save(
        &mut self,
        solver: &OwnedSolver,
        state: &OwnedState,
        raw: &EvalCounts,
    ) {
        assert_eq!(*raw, counts(solver.steps));
        assert_eq!(state.cost_evals(), raw.total_work());
        self.0.borrow_mut().push(("checkpoint", state.iter()));
    }
}

fn executor(log: &Log) -> Executor<Quadratic, OwnedState, OwnedSolver> {
    Executor::new(
        Quadratic::default(),
        OwnedSolver::default(),
        OwnedState::new(),
    )
    .max_iter(3)
    .observe_with(Recorder(Rc::clone(log)), ObserverMode::Always)
    .checkpoint_with(Recorder(Rc::clone(log)), ObserverMode::Always)
}

#[test]
fn checkpoint_moves_allocations_and_resumes_without_init_or_reset() {
    for pause in [0, 2] {
        let log = Log::default();
        let state = OwnedState::new();
        let point_address = state.param() as *const f64;
        let solver = OwnedSolver::default();
        let workspace_address = solver.workspace.as_ptr();
        let mut stepper = Executor::new(Quadratic::default(), solver, state)
            .observe_with(Recorder(Rc::clone(&log)), ObserverMode::Always)
            .checkpoint_with(Recorder(Rc::clone(&log)), ObserverMode::Always)
            .into_stepper()
            .unwrap();
        for _ in 0..pause {
            assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
        }
        let before = log.borrow().clone();
        let checkpoint = stepper.into_checkpoint().unwrap();
        assert_eq!(*log.borrow(), before);
        assert_eq!(checkpoint.state().param() as *const f64, point_address);
        assert_eq!(checkpoint.solver().workspace.as_ptr(), workspace_address);
        assert_eq!(*checkpoint.counts(), counts(pause));
        let result =
            Executor::resume_from_checkpoint(Quadratic::default(), checkpoint)
                .max_iter(3)
                .run_with_solver()
                .unwrap();
        assert_eq!(result.solver.init_calls, 1);
        assert_eq!(result.solver.resets, 1);
        assert_eq!(result.solver.steps, 3);
        assert_eq!(result.counts, counts(3));
        assert_eq!(result.iter(), 3);
        assert_eq!(*result.param(), 1.0);
    }
}

#[test]
fn owned_completion_matches_ordinary_results_and_callbacks() {
    let ordinary_log = Log::default();
    let ordinary = executor(&ordinary_log).run().unwrap();
    for via_stepper in [false, true] {
        let log = Log::default();
        let result: OptimizationResultWithSolver<OwnedState, OwnedSolver> =
            if via_stepper {
                let mut stepper = executor(&log).into_stepper().unwrap();
                stepper.step().unwrap();
                stepper.run_to_end_with_solver().unwrap()
            } else {
                executor(&log).run_with_solver().unwrap()
            };
        assert_eq!(*log.borrow(), *ordinary_log.borrow());
        assert_eq!(result.param(), ordinary.param());
        assert_eq!(result.cost(), ordinary.cost());
        assert_eq!(result.iter(), ordinary.iter());
        assert_eq!(result.cost_evals(), ordinary.cost_evals());
        assert_eq!(result.best_param(), ordinary.best_param());
        assert_eq!(result.best_cost(), ordinary.best_cost());
        assert_eq!(result.best_iter(), ordinary.best_iter());
        assert_eq!(result.best_cost_evals(), ordinary.best_cost_evals());
        assert_eq!(result.reason, ordinary.reason);
        assert_eq!(result.counts, counts(3));
        assert_ne!(result.cost_evals(), result.counts.cost_evals);
        assert_eq!(result.solver.workspace[0], 3);
        let state = result.into_result().into_state();
        assert_eq!(state.param(), ordinary.param());
    }
}

#[test]
fn stopped_stepper_extraction_does_not_repeat_final_callbacks() {
    for as_checkpoint in [false, true] {
        let log = Log::default();
        let mut stepper = executor(&log).into_stepper().unwrap();
        while stepper.step().unwrap() == StepOutcome::Continue {}
        let before = log.borrow().clone();
        let checkpoint = if as_checkpoint {
            stepper.into_checkpoint().unwrap()
        } else {
            let result = stepper.run_to_end_with_solver().unwrap();
            assert_eq!(result.reason, TerminationReason::MaxIter);
            result.into_checkpoint()
        };
        assert_eq!(*log.borrow(), before);
        assert_eq!(*checkpoint.counts(), counts(3));
        assert_eq!(checkpoint.state().iter(), 3);
        let result =
            Executor::resume_from_checkpoint(Quadratic::default(), checkpoint)
                .max_iter(4)
                .run_with_solver()
                .unwrap();
        assert_eq!(result.counts, counts(4));
        assert_eq!(result.solver.steps, 4);
        assert_eq!(result.into_state().iter(), 4);
    }
}

#[test]
fn clean_mid_step_stop_extracts_charged_counts_without_incrementing_iteration()
{
    let mut stepper = Executor::new(
        Quadratic::default(),
        OwnedSolver {
            stop_mid_step: true,
            ..OwnedSolver::default()
        },
        OwnedState::new(),
    )
    .into_stepper()
    .unwrap();
    assert_eq!(
        stepper.step().unwrap(),
        StepOutcome::Stopped(TerminationReason::UserRequested)
    );
    let result = stepper.run_to_end_with_solver().unwrap();
    assert_eq!(result.reason, TerminationReason::UserRequested);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.cost(), 16.0);
    assert_eq!(result.counts, counts(1));
    assert_eq!(result.best_cost_evals(), counts(1).total_work());
    let checkpoint = result.into_checkpoint();
    assert_eq!(checkpoint.state().iter(), 0);
    assert_eq!(*checkpoint.counts(), counts(1));
}

#[test]
fn cancelled_run_can_resume_with_new_execution_policy() {
    for pause in [0, 2] {
        let token = CancellationToken::new();
        let mut stepper = executor(&Log::default())
            .with_cancellation_token(token.clone())
            .into_stepper()
            .unwrap();
        for _ in 0..pause {
            stepper.step().unwrap();
        }
        token.cancel();
        let result = stepper.run_to_end_with_solver().unwrap();
        assert_eq!(result.reason, TerminationReason::Cancelled);
        assert_eq!(result.iter(), pause);
        assert_eq!(result.counts, counts(pause));
        let resumed = Executor::resume_from_checkpoint(
            Quadratic::default(),
            result.into_checkpoint(),
        )
        .max_iter(3)
        .run_with_solver()
        .unwrap();
        assert_eq!(resumed.reason, TerminationReason::MaxIter);
        assert_eq!(resumed.counts, counts(3));
    }
}

#[test]
fn hard_errors_do_not_publish_a_checkpoint_or_final_callbacks() {
    let log = Log::default();
    let mut stepper = Executor::new(
        Quadratic {
            fail_at: Some(2),
            ..Quadratic::default()
        },
        OwnedSolver::default(),
        OwnedState::new(),
    )
    .observe_with(Recorder(Rc::clone(&log)), ObserverMode::Always)
    .checkpoint_with(Recorder(Rc::clone(&log)), ObserverMode::Always)
    .into_stepper()
    .unwrap();
    assert_eq!(stepper.step(), Err("evaluation failed"));
    assert_eq!(stepper.counts().cost_evals, 2);
    assert_eq!(stepper.finished(), None);
    assert!(stepper.into_checkpoint().is_none());
    assert_eq!(*log.borrow(), vec![("init", 0)]);

    for fail_at in [1, 2] {
        let log = Log::default();
        let result = Executor::new(
            Quadratic {
                fail_at: Some(fail_at),
                ..Quadratic::default()
            },
            OwnedSolver::default(),
            OwnedState::new(),
        )
        .observe_with(Recorder(Rc::clone(&log)), ObserverMode::Always)
        .checkpoint_with(Recorder(Rc::clone(&log)), ObserverMode::Always)
        .run_with_solver();
        assert!(matches!(result, Err("evaluation failed")));
        assert_eq!(log.borrow().len(), if fail_at == 1 { 0 } else { 1 });
    }
}

fn assert_continuation<P, S, So>(
    make_executor: impl Fn() -> Executor<P, S, So>,
    make_problem: impl Fn() -> P,
) -> OptimizationResultWithSolver<S, So>
where
    S: State + CountsMirror,
    S::Param: std::fmt::Debug + PartialEq,
    S::Float: std::fmt::Debug + PartialEq,
    So: Solver<P, S>,
    So::Error: std::fmt::Debug,
{
    let baseline = make_executor().max_iter(80).run_with_solver().unwrap();
    for pause in [0, 3] {
        let mut reference =
            make_executor().max_iter(80).into_stepper().unwrap();
        let mut split = make_executor().into_stepper().unwrap();
        for _ in 0..pause {
            assert_eq!(reference.step().unwrap(), StepOutcome::Continue);
            assert_eq!(split.step().unwrap(), StepOutcome::Continue);
        }
        let mut resumed = Executor::resume_from_checkpoint(
            make_problem(),
            split.into_checkpoint().unwrap(),
        )
        .max_iter(80)
        .into_stepper()
        .unwrap();
        loop {
            let expected = reference.step().unwrap();
            assert_eq!(resumed.step().unwrap(), expected);
            assert_eq!(resumed.state().param(), reference.state().param());
            assert_eq!(resumed.state().cost(), reference.state().cost());
            assert_eq!(
                resumed.state().best_param(),
                reference.state().best_param()
            );
            assert_eq!(
                resumed.state().best_cost(),
                reference.state().best_cost()
            );
            assert_eq!(
                resumed.state().best_iter(),
                reference.state().best_iter()
            );
            assert_eq!(
                resumed.state().best_cost_evals(),
                reference.state().best_cost_evals()
            );
            assert_eq!(resumed.counts(), reference.counts());
            if let StepOutcome::Stopped(reason) = expected {
                let result = resumed.run_to_end_with_solver().unwrap();
                assert_eq!(reason, baseline.reason);
                assert_eq!(result.iter(), baseline.iter());
                assert_eq!(result.param(), baseline.param());
                assert_eq!(result.counts, baseline.counts);
                break;
            }
        }
    }
    baseline
}

struct Sphere<V, F>(std::marker::PhantomData<fn() -> (V, F)>);

impl<V, F> Sphere<V, F> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<V: basin::Dot<F>, F: basin::Scalar> CostFunction for Sphere<V, F> {
    type Param = V;
    type Output = F;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        Ok(x.dot(x))
    }
}

impl<V, F> Gradient for Sphere<V, F>
where
    V: basin::Dot<F> + basin::ScaleInPlace<F> + Clone,
    F: basin::Scalar,
{
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        let mut gradient = x.clone();
        gradient.scale_in_place(F::one() + F::one());
        Ok(gradient)
    }
}

macro_rules! gradient_continuation {
    ($name:ident, $point:expr, $scalar:ty) => {
        #[test]
        fn $name() {
            let result = assert_continuation(
                || {
                    Executor::from_start(
                        Sphere::new(),
                        basin::GradientDescent::new(0.1 as $scalar)
                            .with_relative_gradient_tolerance(0.25 as $scalar),
                        $point,
                    )
                },
                Sphere::new,
            );
            assert_eq!(
                result.reason,
                TerminationReason::RelativeGradientTolerance
            );
            assert_eq!(result.iter(), 7);
            assert_eq!(result.counts.cost_evals, 8);
            assert_eq!(result.counts.gradient_evals, 8);
        }
    };
}

gradient_continuation!(vec_f64_convergence_history, vec![1.0_f64], f64);
gradient_continuation!(vec_f32_convergence_history, vec![1.0_f32], f32);

#[cfg(feature = "nalgebra_all")]
gradient_continuation!(
    nalgebra_convergence_history,
    backend_aliases::nalgebra::DVector::from_vec(vec![1.0]),
    f64
);
#[cfg(feature = "ndarray_all")]
gradient_continuation!(
    ndarray_convergence_history,
    backend_aliases::ndarray::Array1::from_vec(vec![1.0]),
    f64
);
#[cfg(feature = "faer_all")]
gradient_continuation!(
    faer_convergence_history,
    backend_aliases::faer::Col::from_fn(1, |_| 1.0),
    f64
);

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

#[test]
fn seeded_annealing_preserves_rng_neighbor_and_reannealing_history() {
    use basin::core::rng::{ChaCha8Rng, Rng};
    use basin::{Neighbor, SimulatedAnnealing, TemperatureSchedule};
    use std::convert::Infallible;

    struct RuggedCost;
    impl CostFunction for RuggedCost {
        type Param = i32;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &i32) -> Result<f64, Infallible> {
            let x = f64::from(*x);
            Ok((x - 3.0).powi(2) + (x * 2.3).sin())
        }
    }

    #[derive(Clone)]
    struct StatefulNeighbor {
        calls: u64,
    }

    impl Neighbor<i32> for StatefulNeighbor {
        type Error = Infallible;
        fn propose(
            &mut self,
            current: &i32,
            _: f64,
            rng: &mut ChaCha8Rng,
        ) -> Result<i32, Infallible> {
            self.calls += 1;
            let distance = 1 + (self.calls % 3) as i32;
            Ok(if rng.next_u64() & 1 == 0 {
                current - distance
            } else {
                current + distance
            })
        }
    }

    let result = assert_continuation(
        || {
            Executor::from_start(
                RuggedCost,
                SimulatedAnnealing::new(
                    StatefulNeighbor { calls: 0 },
                    4.0,
                    TemperatureSchedule::geometric(0.97)
                        .with_steps_per_temperature(3),
                    0x5eed,
                )
                .with_reannealing_fixed(17)
                .with_reannealing_accepted(11)
                .with_reannealing_best(13),
                12,
            )
        },
        || RuggedCost,
    );
    assert_eq!(result.counts.cost_evals, 81);
    assert!(result.state.reannealings() > 0);
    assert!(result.state.accepted_moves() > 0);
    assert!(result.state.rejected_moves() > 0);
}
