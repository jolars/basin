//! Barrier progress records the objective only inside the strict feasible domain.

use basin::{
    Backtracking, BarrierMethod, CostFunction, DenseMatrix, Executor, Gradient,
    GradientDescent, InitialState, LinearInequalityConstraints,
    MatrixFromDiagonal, Problem, Solver, State,
};
use std::cell::Cell;

struct Inequality {
    a: DenseMatrix,
    b: Vec<f64>,
    fail_gradient: Cell<bool>,
}
impl Inequality {
    fn new(n: usize) -> Self {
        Self {
            a: DenseMatrix::from_diagonal(&vec![1.0; n]),
            b: vec![1.0; n],
            fail_gradient: Cell::new(false),
        }
    }
}
impl CostFunction for Inequality {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        assert!(
            x.iter().all(|x| *x < 1.0),
            "objective outside strict domain"
        );
        Ok(x.iter().map(|x| (x - 2.0).powi(2)).sum())
    }
}
impl Gradient for Inequality {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        assert!(x.iter().all(|x| *x < 1.0), "gradient outside strict domain");
        if self.fail_gradient.get() {
            return Err("gradient callback aborted");
        }
        Ok(x.iter().map(|x| 2.0 * (x - 2.0)).collect())
    }
}
impl LinearInequalityConstraints for Inequality {
    type Matrix = DenseMatrix;
    fn a(&self) -> &DenseMatrix {
        &self.a
    }
    fn b(&self) -> &Vec<f64> {
        &self.b
    }
}

type Barrier = BarrierMethod<GradientDescent<Backtracking, Vec<f64>>>;
fn solver() -> Barrier {
    BarrierMethod::with_inner_solver(GradientDescent::with_line_search(
        Backtracking::new(),
    ))
    .with_inner_max_iter(20)
}

#[test]
fn fresh_barrier_resets_progress() {
    let prior = Executor::from_start(Inequality::new(1), solver(), vec![0.0])
        .max_iter(3)
        .run_with_solver()
        .unwrap();
    let fresh = Executor::new(Inequality::new(1), prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.state.iter(), 0);
    assert_eq!(fresh.state.best_iter(), 0);
}

#[test]
fn failed_phase_two_work_is_charged_before_propagating_error() {
    let mut problem = Problem::new(Inequality::new(1));
    let mut solver = solver();
    let seed = solver.seed(&vec![0.0]);
    let state = solver.init(&mut problem, seed).unwrap();
    let before = *problem.counts();
    problem.inner().fail_gradient.set(true);
    assert_eq!(
        solver.next_iter(&mut problem, state).err().unwrap(),
        "gradient callback aborted"
    );
    assert_eq!(problem.counts().cost_evals, before.cost_evals + 1);
    assert_eq!(problem.counts().gradient_evals, before.gradient_evals + 1);
}

fn coherent(state: &basin::PointState<Vec<f64>>) {
    let (x, cost) = state.current().unwrap();
    if x.iter().all(|x| *x < 1.0) {
        assert_eq!(cost, x.iter().map(|x| (x - 2.0).powi(2)).sum::<f64>());
    } else {
        assert_eq!(cost, f64::INFINITY);
    }
    if let Some((x, cost)) = state.best() {
        assert!(x.iter().all(|x| *x < 1.0));
        assert_eq!(cost, x.iter().map(|x| (x - 2.0).powi(2)).sum::<f64>());
    }
}

#[test]
fn exact_continuation_preserves_both_phases_and_counts() {
    for start in [0.0, 3.0, 1000.0] {
        let make = || {
            Executor::from_start(Inequality::new(2), solver(), vec![start; 2])
        };
        let expected = make().max_iter(8).run_with_solver().unwrap();
        coherent(&expected.state);
        assert_eq!(expected.state.counts(), &expected.counts);
        for split in [0, 1, 3, 7] {
            let partial = make().max_iter(split).run_with_solver().unwrap();
            coherent(&partial.state);
            let resumed = Executor::resume_from_checkpoint(
                Inequality::new(2),
                partial.into_checkpoint(),
            )
            .require_evaluated_state()
            .max_iter(8)
            .run_with_solver()
            .unwrap();
            assert_eq!(resumed.state, expected.state);
            assert_eq!(resumed.counts, expected.counts);
        }
    }
}

#[test]
fn fresh_run_resets_phase_schedule_and_accepts_a_new_dimension() {
    let prior =
        Executor::from_start(Inequality::new(2), solver(), vec![0.0; 2])
            .max_iter(5)
            .run_with_solver()
            .unwrap();
    let mut state = prior.state;
    state.replace(vec![3.0; 3], -1000.0);
    let fresh = Executor::new(Inequality::new(3), prior.solver, state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    coherent(&fresh.state);
    assert_eq!(fresh.state.iter(), 0);
    assert!(fresh.state.best().is_none());
    assert_eq!(fresh.counts.total_work(), 0);
    let restarted = Executor::resume_from_checkpoint(
        Inequality::new(3),
        fresh.into_checkpoint(),
    )
    .max_iter(8)
    .run_with_solver()
    .unwrap();
    let rebuilt =
        Executor::from_start(Inequality::new(3), solver(), vec![3.0; 3])
            .max_iter(8)
            .run_with_solver()
            .unwrap();
    assert_eq!(restarted.state, rebuilt.state);
    assert_eq!(restarted.counts, rebuilt.counts);
}

#[test]
fn phase_one_does_not_trigger_objective_or_step_stopping() {
    let solver = BarrierMethod::with_inner_solver(GradientDescent::new(0.0))
        .with_inner_max_iter(1)
        .with_absolute_cost_change_tolerance(1e10)
        .with_absolute_step_tolerance(1e10);
    let result = Executor::from_start(Inequality::new(1), solver, vec![3.0])
        .require_evaluated_state()
        .target_objective(0.0)
        .max_iter(4)
        .run()
        .unwrap();
    assert_eq!(result.reason, basin::TerminationReason::MaxIter);
    assert_eq!(result.state.iter(), 4);
    assert_eq!(result.state.current(), Some((&vec![3.0], f64::INFINITY)));
    assert!(result.state.best().is_none());
    assert_eq!(result.state.counts().cost_evals, 8);
    assert_eq!(result.state.counts().gradient_evals, 8);
}

struct AbortInner;
impl InitialState<Vec<f64>> for AbortInner {
    type State = basin::FirstOrderState<Vec<f64>>;
    fn seed(&self, x: &Vec<f64>) -> Self::State {
        Self::State::new(x.clone())
    }
}
impl basin::WarmStart<Vec<f64>> for AbortInner {}
impl<P> Solver<P, basin::FirstOrderState<Vec<f64>>> for AbortInner
where
    P: CostFunction<Param = Vec<f64>, Output = f64, Error = &'static str>
        + Gradient<Gradient = Vec<f64>>,
{
    type Error = &'static str;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: basin::FirstOrderState<Vec<f64>>,
    ) -> Result<basin::FirstOrderState<Vec<f64>>, Self::Error> {
        problem.cost_and_gradient(state.param())?;
        Err("inner init aborted")
    }
    fn next_iter(
        &mut self,
        _: &mut Problem<P>,
        _: basin::FirstOrderState<Vec<f64>>,
    ) -> Result<
        (
            basin::FirstOrderState<Vec<f64>>,
            Option<basin::TerminationReason>,
        ),
        Self::Error,
    > {
        unreachable!()
    }
}

#[test]
fn failed_phase_one_adapter_work_is_charged() {
    let mut problem = Problem::new(Inequality::new(1));
    let mut solver = BarrierMethod::with_inner_solver(AbortInner);
    let state = solver
        .init(&mut problem, basin::PointState::new(vec![3.0]))
        .unwrap();
    assert_eq!(problem.counts().total_work(), 0);
    assert_eq!(
        solver.next_iter(&mut problem, state).err().unwrap(),
        "inner init aborted"
    );
    assert_eq!(problem.counts().cost_evals, 1);
    assert_eq!(problem.counts().gradient_evals, 1);
}

#[cfg(feature = "serde")]
#[test]
fn serialized_barrier_checkpoint_retains_phase_and_schedule() {
    for (start, split) in [(0.0, 3), (1000.0, 1)] {
        let make =
            || Executor::from_start(Inequality::new(1), solver(), vec![start]);
        let expected = make().max_iter(8).run_with_solver().unwrap();
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let bytes = postcard::to_allocvec(&checkpoint).unwrap();
        let restored: basin::ExactCheckpoint<
            Barrier,
            basin::PointState<Vec<f64>>,
        > = postcard::from_bytes(&bytes).unwrap();
        let resumed =
            Executor::resume_from_checkpoint(Inequality::new(1), restored)
                .require_evaluated_state()
                .max_iter(8)
                .run_with_solver()
                .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(
            postcard::to_allocvec(&resumed.solver).unwrap(),
            postcard::to_allocvec(&expected.solver).unwrap()
        );
    }
}

#[test]
fn feasible_initialization_seeds_outer_change_checks() {
    let solver = BarrierMethod::with_inner_solver(GradientDescent::new(0.0))
        .with_inner_max_iter(1)
        .with_absolute_duality_gap_tolerance(None)
        .with_absolute_cost_change_tolerance(0.0);
    let result = Executor::from_start(Inequality::new(1), solver, vec![0.0])
        .max_iter(3)
        .run()
        .unwrap();
    assert_eq!(result.reason, basin::TerminationReason::CostTolerance);
    assert_eq!(result.state.iter(), 1);
}
