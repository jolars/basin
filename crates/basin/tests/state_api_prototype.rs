//! Executable API experiment, deliberately outside Basin's published surface.
//! Run with `cargo test -p basin --test state_api_prototype`.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/state_api.rs"]
mod prototype;

use basin::core::math::{NormSquared, ScaledAdd};
use basin::core::math::{Scalar, VectorLen};
use basin::core::rng::{ChaCha8Rng, RngExt};
use basin::solver::simulated_annealing::{Neighbor, TemperatureSchedule};
use basin::{
    CostFunction, Gradient, Problem, Solver, State, TerminationReason,
};
use prototype::driver::Run;
use prototype::solvers::{Annealing, Bfgs, Lbfgs, NelderMead};
use prototype::state::{FirstOrderState, PointState, Progress};
use prototype::state::{
    ObjectiveBest, PopulationState, SelectedState, SimplexState,
};
use std::convert::Infallible;
use std::{cell::RefCell, rc::Rc};

trait Coordinates<F: Scalar>: Clone + VectorLen {
    fn from_values(values: &[F]) -> Self;
    fn values(&self) -> Vec<F>;
}

impl<F: Scalar> Coordinates<F> for Vec<F> {
    fn from_values(values: &[F]) -> Self {
        values.to_vec()
    }
    fn values(&self) -> Vec<F> {
        self.clone()
    }
}

#[cfg(feature = "nalgebra_all")]
impl<F: Scalar + backend_aliases::nalgebra::Scalar> Coordinates<F>
    for backend_aliases::nalgebra::DVector<F>
{
    fn from_values(values: &[F]) -> Self {
        Self::from_column_slice(values)
    }
    fn values(&self) -> Vec<F> {
        self.as_slice().to_vec()
    }
}
#[cfg(feature = "ndarray_all")]
impl<F: Scalar> Coordinates<F> for backend_aliases::ndarray::Array1<F> {
    fn from_values(values: &[F]) -> Self {
        Self::from_vec(values.to_vec())
    }
    fn values(&self) -> Vec<F> {
        self.to_vec()
    }
}
#[cfg(feature = "faer_all")]
impl<F: Scalar> Coordinates<F> for backend_aliases::faer::Col<F> {
    fn from_values(values: &[F]) -> Self {
        Self::from_fn(values.len(), |i| values[i])
    }
    fn values(&self) -> Vec<F> {
        (0..self.nrows()).map(|i| self[i]).collect()
    }
}

type Evaluations<F> = Rc<RefCell<Vec<(u8, Vec<F>)>>>;

struct Rosenbrock<V, F: Scalar = f64> {
    trace: Evaluations<F>,
    vector: std::marker::PhantomData<V>,
}

impl<V, F: Scalar> Rosenbrock<V, F> {
    fn new() -> Self {
        Self {
            trace: Rc::default(),
            vector: std::marker::PhantomData,
        }
    }
}

impl<V: Coordinates<F>, F: Scalar> CostFunction for Rosenbrock<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, param: &V) -> Result<F, Infallible> {
        let x = param.values();
        self.trace.borrow_mut().push((0, x.clone()));
        Ok((F::one() - x[0]).powi(2)
            + F::from_f64(100.0).unwrap() * (x[1] - x[0] * x[0]).powi(2))
    }
}

impl<V: Coordinates<F>, F: Scalar> Gradient for Rosenbrock<V, F> {
    type Gradient = V;
    fn gradient(&self, param: &V) -> Result<V, Infallible> {
        let x = param.values();
        self.trace.borrow_mut().push((1, x.clone()));
        let two = F::from_f64(2.0).unwrap();
        let two_hundred = F::from_f64(200.0).unwrap();
        Ok(V::from_values(&[
            -two * (F::one() - x[0])
                - two * two_hundred * x[0] * (x[1] - x[0] * x[0]),
            two_hundred * (x[1] - x[0] * x[0]),
        ]))
    }
}

struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Infallible> {
        Ok(x.norm_squared())
    }
}

impl Gradient for Sphere {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}

// This solver only has access to the prototype's public storage operations.
struct ExternalDescent;

impl Solver<Sphere, FirstOrderState<Vec<f64>>> for ExternalDescent {
    type Error = Infallible;

    fn init(
        &mut self,
        problem: &mut Problem<Sphere>,
        mut state: FirstOrderState<Vec<f64>>,
    ) -> Result<FirstOrderState<Vec<f64>>, Infallible> {
        let x = state.param().clone();
        let (cost, gradient) = problem.cost_and_gradient(&x)?;
        state.replace(x, cost, gradient).unwrap();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Sphere>,
        mut state: FirstOrderState<Vec<f64>>,
    ) -> Result<
        (FirstOrderState<Vec<f64>>, Option<TerminationReason>),
        Infallible,
    > {
        let (x, _, gradient) = state.take_current().unwrap();
        let mut next = x;
        next.scaled_add(-0.1, &gradient);
        let (cost, gradient) = problem.cost_and_gradient(&next)?;
        state.replace(next, cost, gradient).unwrap();
        Ok((state, None))
    }
}

#[test]
fn external_solver_reuses_storage_and_executor_bookkeeping() {
    let result =
        Run::new(Sphere, ExternalDescent, FirstOrderState::new(vec![2.0]))
            .unwrap()
            .run(3)
            .unwrap();
    assert_eq!(result.state.iter(), 3);
    assert_eq!(result.counts.cost_evals, 4);
    assert_eq!(result.counts.gradient_evals, 4);
    assert_eq!(result.state.counts(), &result.counts);
    assert!((result.state.cost() - 4.0 * 0.8_f64.powi(6)).abs() < 1e-14);
}

#[test]
fn incomplete_and_rejected_records_are_distinct() {
    let mut state = PointState::new(vec![1.0]);
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    state.replace(vec![1.0], f64::INFINITY);
    state.publish(false, basin::EvalCounts::default());
    assert_eq!(state.current().unwrap().1, f64::INFINITY);
    assert!(state.best().is_none());
    state.replace(vec![2.0], f64::NAN);
    state.publish(true, basin::EvalCounts::default());
    assert!(state.best().is_none());
}

#[test]
fn first_order_update_is_atomic_on_shape_error() {
    let mut state = FirstOrderState::new(vec![1.0, 2.0]);
    state.replace(vec![1.0, 2.0], 5.0, vec![2.0, 4.0]).unwrap();
    assert!(state.replace(vec![3.0, 4.0], 25.0, vec![6.0]).is_err());
    let (x, cost, gradient) = state.current().unwrap();
    assert_eq!(x, &[1.0, 2.0]);
    assert_eq!(cost, 5.0);
    assert_eq!(gradient, &[2.0, 4.0]);
}

#[test]
fn best_records_survive_nonmonotone_and_equal_cost_steps() {
    let mut state = PointState::new(vec![1.0]);
    state.replace(vec![1.0], 1.0);
    state.publish(
        false,
        basin::EvalCounts {
            cost_evals: 1,
            ..Default::default()
        },
    );
    assert!(state.new_best());
    for (x, cost) in [(vec![9.0], 81.0), (vec![-1.0], 1.0)] {
        state.replace(x, cost);
        state.publish(
            true,
            basin::EvalCounts {
                cost_evals: 5,
                ..Default::default()
            },
        );
        assert!(!state.new_best());
        assert_eq!(state.best().unwrap().param, vec![1.0]);
        assert_eq!(state.best_iter(), 0);
        assert_eq!(state.best_cost_evals(), 1);
    }
    state.replace(vec![0.0], 0.0);
    state.publish(
        false,
        basin::EvalCounts {
            cost_evals: 8,
            ..Default::default()
        },
    );
    assert_eq!(state.iter(), 2);
    assert_eq!(state.best_cost_evals(), 8);
    assert!(state.new_best());
    state.publish(
        false,
        basin::EvalCounts {
            cost_evals: 8,
            ..Default::default()
        },
    );
    assert!(!state.new_best());
}

#[test]
fn shared_population_retains_history_and_considers_evaluated_mean() {
    let mut state = PopulationState::new(vec![vec![1.0], vec![2.0]]).unwrap();
    state
        .replace_members(vec![vec![2.0], vec![1.0]], vec![4.0, 1.0])
        .unwrap();
    state.publish(false, basin::EvalCounts::default());
    assert_eq!(state.points(), &[vec![2.0], vec![1.0]]);
    assert_eq!(state.cost(), 1.0);
    state
        .replace_members(vec![vec![3.0], vec![4.0]], vec![9.0, 16.0])
        .unwrap();
    state.publish(true, basin::EvalCounts::default());
    assert_eq!(state.best_param(), &[1.0]);
    assert_eq!(state.best_cost(), 1.0);
    state.publish_mean(vec![0.5], 0.25).unwrap();
    state.publish(true, basin::EvalCounts::default());
    assert_eq!(state.best_param(), &[0.5]);
    assert_eq!(state.costs(), &[9.0, 16.0]);
    let old = state.points().to_vec();
    assert!(state.replace_members(vec![vec![0.0]], vec![]).is_err());
    assert_eq!(state.points(), old);
    state
        .replace_members(
            vec![vec![9.0], vec![2.0], vec![0.0]],
            vec![f64::NAN, 4.0, f64::NEG_INFINITY],
        )
        .unwrap();
    state.publish(true, basin::EvalCounts::default());
    assert!(state.costs()[0].is_nan());
    assert_eq!(state.best_cost(), f64::NEG_INFINITY);
}

#[test]
fn simplex_shape_validation_preserves_the_previous_members() {
    assert!(SimplexState::<Vec<f64>>::new(vec![]).is_err());
    assert!(SimplexState::<Vec<f64>>::new(vec![vec![]]).is_err());
    assert!(
        SimplexState::<Vec<f64>>::new(vec![vec![0.0], vec![1.0, 2.0]]).is_err()
    );
    let mut state = SimplexState::new(vec![vec![0.0], vec![1.0]]).unwrap();
    state
        .replace_members(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0])
        .unwrap();
    assert!(state.replace_members(vec![vec![2.0]], vec![4.0]).is_err());
    assert_eq!(state.points(), &[vec![0.0], vec![1.0]]);
    assert_eq!(state.costs(), &[0.0, 1.0]);
}

#[test]
fn constrained_selection_can_improve_feasibility_with_higher_cost() {
    let mut state = SelectedState::new(vec![-1.0]);
    state.select(1, vec![-1.0], -100.0, 1.0);
    state.publish(false, basin::EvalCounts::default());
    state.select(2, vec![1.0], 10.0, 0.0);
    state.publish(true, basin::EvalCounts::default());
    assert_eq!(state.best_param(), &[1.0]);
    assert_eq!(state.best_cost(), 10.0);
    assert_eq!(state.violation(), 0.0);
    assert!(state.new_best());
    state.select(2, vec![1.0], 10.0, 0.0);
    state.publish(true, basin::EvalCounts::default());
    assert!(!state.new_best());
    assert_eq!(state.best_iter(), 1);

    // These become ambiguous and fail compilation if either absent capability
    // is accidentally implemented. No external compile-test dependency is needed.
    trait WithoutObjective<A> {
        fn check() {}
    }
    impl<T: ?Sized> WithoutObjective<()> for T {}
    struct HasObjective;
    impl<T: ?Sized + ObjectiveBest> WithoutObjective<HasObjective> for T {}
    let _ = <SelectedState<Vec<f64>> as WithoutObjective<_>>::check;
    trait WithoutGradient<A> {
        fn check() {}
    }
    impl<T: ?Sized> WithoutGradient<()> for T {}
    struct HasGradient;
    impl<T: ?Sized + basin::core::state::GradientState>
        WithoutGradient<HasGradient> for T
    {
    }
    let _ = <PointState<Vec<f64>> as WithoutGradient<_>>::check;
}

fn compare_trace(actual: &[(u8, Vec<f64>)], expected: &[(u8, Vec<f64>)]) {
    assert_eq!(actual.len(), expected.len());
    for ((kind, x), (expected_kind, y)) in actual.iter().zip(expected) {
        assert_eq!(kind, expected_kind);
        for (a, b) in x.iter().zip(y) {
            assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "{a} != {b}");
        }
    }
}

#[test]
fn bfgs_and_lbfgs_match_shipped_solver_evaluations() {
    let original = Rosenbrock::<Vec<f64>>::new();
    let expected = original.trace.clone();
    let baseline = basin::Executor::from_start(
        original,
        basin::Bfgs::new(),
        vec![-1.2, 1.0],
    )
    .max_iter(12)
    .run()
    .unwrap();
    let problem = Rosenbrock::<Vec<f64>>::new();
    let actual = problem.trace.clone();
    let result =
        Run::new(problem, Bfgs::new(), FirstOrderState::new(vec![-1.2, 1.0]))
            .unwrap()
            .run(12)
            .unwrap();
    compare_trace(&actual.borrow(), &expected.borrow());
    assert!((result.state.cost() - baseline.cost()).abs() < 1e-10);
    assert_eq!(result.counts.cost_evals, baseline.cost_evals());
    assert_eq!(result.solver.inverse_hessian().unwrap().nrows(), 2);

    let original = Rosenbrock::<Vec<f64>>::new();
    let expected = original.trace.clone();
    let baseline = basin::Executor::from_start(
        original,
        basin::Lbfgs::<basin::solver::lbfgs::Unbounded>::new()
            .with_m_capacity(4),
        vec![-1.2, 1.0],
    )
    .max_iter(12)
    .run()
    .unwrap();
    let problem = Rosenbrock::<Vec<f64>>::new();
    let actual = problem.trace.clone();
    let result = Run::new(
        problem,
        Lbfgs::new(4),
        FirstOrderState::new(vec![-1.2, 1.0]),
    )
    .unwrap()
    .run(12)
    .unwrap();
    compare_trace(&actual.borrow(), &expected.borrow());
    assert!((result.state.cost() - baseline.cost()).abs() < 1e-10);
    assert_eq!(result.counts.cost_evals, baseline.cost_evals());
    assert_eq!(result.solver.history_len(), 4);
}

fn resume_first_order<
    So: Solver<Rosenbrock<Vec<f64>>, FirstOrderState<Vec<f64>>, Error = Infallible>,
>(
    make: impl Fn() -> So,
) {
    let whole = Rosenbrock::<Vec<f64>>::new();
    let expected = whole.trace.clone();
    let baseline =
        Run::new(whole, make(), FirstOrderState::new(vec![-1.2, 1.0]))
            .unwrap()
            .run(15)
            .unwrap();
    for split in [0, 1, 7] {
        let first = Rosenbrock::<Vec<f64>>::new();
        let actual = first.trace.clone();
        let checkpoint =
            Run::new(first, make(), FirstOrderState::new(vec![-1.2, 1.0]))
                .unwrap()
                .run(split)
                .unwrap()
                .into_checkpoint();
        let continuation = Rosenbrock {
            trace: actual.clone(),
            vector: std::marker::PhantomData,
        };
        let result = Run::resume(continuation, checkpoint).run(15).unwrap();
        assert_eq!(*actual.borrow(), *expected.borrow());
        assert_eq!(result.state.param(), baseline.state.param());
        assert_eq!(result.counts, baseline.counts);
        assert_eq!(result.state.best_iter(), baseline.state.best_iter());
    }
}

#[test]
fn checkpoints_preserve_solver_owned_matrices_and_history() {
    resume_first_order(Bfgs::new);
    resume_first_order(|| Lbfgs::new(4));

    let run = Run::new(
        Rosenbrock::<Vec<f64>>::new(),
        Bfgs::new(),
        FirstOrderState::new(vec![-1.2, 1.0]),
    )
    .unwrap();
    let pointer = run.solver().inverse_hessian().unwrap().data().as_ptr();
    let checkpoint = run.into_checkpoint().unwrap();
    assert_eq!(
        pointer,
        checkpoint
            .solver()
            .inverse_hessian()
            .unwrap()
            .data()
            .as_ptr()
    );
    let resumed = Run::resume(Rosenbrock::<Vec<f64>>::new(), checkpoint);
    assert_eq!(
        pointer,
        resumed.solver().inverse_hessian().unwrap().data().as_ptr()
    );
}

#[test]
fn fresh_runs_reset_workspace_after_dimension_changes() {
    let first =
        Run::new(Sphere, Bfgs::new(), FirstOrderState::new(vec![2.0, 3.0]))
            .unwrap()
            .run(1)
            .unwrap();
    let reused = Run::new(
        Sphere,
        first.solver,
        FirstOrderState::new(vec![2.0, 3.0, 4.0]),
    )
    .unwrap()
    .run(1)
    .unwrap();
    let fresh = Run::new(
        Sphere,
        Bfgs::new(),
        FirstOrderState::new(vec![2.0, 3.0, 4.0]),
    )
    .unwrap()
    .run(1)
    .unwrap();
    assert_eq!(reused.state.param(), fresh.state.param());
    assert_eq!(reused.counts, fresh.counts);
    assert_eq!(reused.solver.inverse_hessian().unwrap().nrows(), 3);

    let first = Run::new(
        Rosenbrock::<Vec<f64>>::new(),
        Lbfgs::new(4),
        FirstOrderState::new(vec![-1.2, 1.0]),
    )
    .unwrap()
    .run(7)
    .unwrap();
    assert_eq!(first.solver.history_len(), 4);
    let reused = Run::new(
        Sphere,
        first.solver,
        FirstOrderState::new(vec![2.0, 3.0, 4.0]),
    )
    .unwrap();
    assert_eq!(reused.solver().history_len(), 0);
    assert_eq!(reused.counts().cost_evals, 1);
}

#[test]
fn nelder_mead_preserves_authoritative_simplex_allocations() {
    let points = vec![vec![-1.2, 1.0], vec![-1.1, 1.0], vec![-1.2, 1.1]];
    let original = Rosenbrock::<Vec<f64>>::new();
    let expected = original.trace.clone();
    let baseline = basin::Executor::new(
        original,
        basin::NelderMead::new(),
        basin::BasicSimplexState::from_simplex(points.clone()),
    )
    .max_iter(10)
    .run()
    .unwrap();
    let problem = Rosenbrock::<Vec<f64>>::new();
    let actual = problem.trace.clone();
    let allocation = points.as_ptr();
    let mut run = Run::new(
        problem,
        NelderMead::new(),
        SimplexState::new(points).unwrap(),
    )
    .unwrap();
    let costs = run.state().unwrap().costs().as_ptr();
    for _ in 0..10 {
        run.step().unwrap();
        assert_eq!(allocation, run.state().unwrap().points().as_ptr());
        assert_eq!(costs, run.state().unwrap().costs().as_ptr());
        assert_eq!(run.solver().scratch().len(), 3);
    }
    compare_trace(&actual.borrow(), &expected.borrow());
    assert!((run.state().unwrap().cost() - baseline.cost()).abs() < 1e-12);
}

#[derive(Clone, Default)]
struct RandomNeighbor {
    calls: u64,
}

impl<V: Coordinates<F>, F: Scalar> Neighbor<V, F> for RandomNeighbor {
    type Error = Infallible;

    fn propose(
        &mut self,
        current: &V,
        temperature: F,
        rng: &mut ChaCha8Rng,
    ) -> Result<V, Infallible> {
        self.calls += 1;
        let scale = F::from_f64(0.05 / (self.calls as f64).sqrt()).unwrap();
        let mut values = current.values();
        for value in &mut values {
            *value = *value
                + F::from_f64(rng.random::<f64>() - 0.5).unwrap()
                    * scale
                    * temperature;
        }
        Ok(V::from_values(&values))
    }
}

fn annealing() -> Annealing<RandomNeighbor> {
    Annealing::new(
        RandomNeighbor::default,
        5.0,
        TemperatureSchedule::geometric(0.98),
        42,
    )
}

#[test]
fn annealing_resumes_rng_temperature_and_stateful_neighbor() {
    let whole = Rosenbrock::<Vec<f64>>::new();
    let expected = whole.trace.clone();
    let baseline =
        Run::new(whole, annealing(), PointState::new(vec![-1.2, 1.0]))
            .unwrap()
            .run(40)
            .unwrap();
    for split in [0, 1, 17] {
        let first = Rosenbrock::<Vec<f64>>::new();
        let actual = first.trace.clone();
        let checkpoint =
            Run::new(first, annealing(), PointState::new(vec![-1.2, 1.0]))
                .unwrap()
                .run(split)
                .unwrap()
                .into_checkpoint();
        let continuation = Rosenbrock {
            trace: actual.clone(),
            vector: std::marker::PhantomData,
        };
        let result = Run::resume(continuation, checkpoint).run(40).unwrap();
        assert_eq!(*actual.borrow(), *expected.borrow());
        assert_eq!(result.state.current(), baseline.state.current());
        assert_eq!(result.state.best(), baseline.state.best());
        assert_eq!(result.solver.temperature(), baseline.solver.temperature());
        assert_eq!(result.solver.neighbor().calls, 40);
        assert_eq!(result.counts, baseline.counts);
    }

    let original = Rosenbrock::<Vec<f64>>::new();
    let original_trace = original.trace.clone();
    let solver = basin::SimulatedAnnealing::new(
        RandomNeighbor::default(),
        5.0,
        TemperatureSchedule::geometric(0.98),
        42,
    );
    let original =
        basin::Executor::from_start(original, solver, vec![-1.2, 1.0])
            .max_iter(40)
            .run()
            .unwrap();
    compare_trace(&expected.borrow(), &original_trace.borrow());
    assert_eq!(baseline.state.param(), original.param());
}

#[test]
fn fresh_annealing_resets_machinery_and_chains_ignore_live_rng() {
    let first = Run::new(Sphere, annealing(), PointState::new(vec![2.0, 3.0]))
        .unwrap()
        .run(10)
        .unwrap();
    let chain_a = first.solver.seed_chain(99);
    let chain_b = annealing().seed_chain(99);
    let a = Run::new(Sphere, chain_a, PointState::new(vec![2.0, 3.0]))
        .unwrap()
        .run(20)
        .unwrap();
    let b = Run::new(Sphere, chain_b, PointState::new(vec![2.0, 3.0]))
        .unwrap()
        .run(20)
        .unwrap();
    assert_eq!(a.state.current(), b.state.current());

    // Seeding chains must also leave the prototype's existing stream intact.
    let continued = Run::resume(Sphere, first.into_checkpoint())
        .run(20)
        .unwrap();
    let uninterrupted =
        Run::new(Sphere, annealing(), PointState::new(vec![2.0, 3.0]))
            .unwrap()
            .run(20)
            .unwrap();
    assert_eq!(continued.state.current(), uninterrupted.state.current());

    let reused = Run::new(
        Sphere,
        continued.solver,
        PointState::new(vec![2.0, 3.0, 4.0]),
    )
    .unwrap()
    .run(20)
    .unwrap();
    let fresh =
        Run::new(Sphere, annealing(), PointState::new(vec![2.0, 3.0, 4.0]))
            .unwrap()
            .run(20)
            .unwrap();
    assert_eq!(reused.state.current(), fresh.state.current());
    assert_eq!(reused.solver.neighbor().calls, 20);
    assert_eq!(reused.counts, fresh.counts);
}

// These are concrete call sites so they test type inference as well as math.
macro_rules! backend_test {
    ($name:ident, $vector:ty, $float:ty) => {
        #[test]
        fn $name() {
            type V = $vector;
            type F = $float;
            let x = V::from_values(&[-1.2, 1.0]);
            let first = Run::from_start(
                Rosenbrock::<V, F>::new(),
                Bfgs::new(),
                x.clone(),
            )
            .unwrap()
            .run(8)
            .unwrap();
            let _: F = first.state.cost();
            let _: &V = first.state.param();
            assert!(first.state.cost() < 10.0);
            let second =
                Run::new(Rosenbrock::<V, F>::new(), Lbfgs::new(4), first.state)
                    .unwrap()
                    .run(8)
                    .unwrap();
            assert!(second.state.cost() < 10.0);
            assert_eq!(second.state.counts(), &second.counts);
            assert!(second.solver.history_len() > 0);
            let points = vec![
                x.clone(),
                V::from_values(&[-1.1, 1.0]),
                V::from_values(&[-1.2, 1.1]),
            ];
            let simplex = Run::new(
                Rosenbrock::<V, F>::new(),
                NelderMead::new(),
                SimplexState::new(points).unwrap(),
            )
            .unwrap()
            .run(8)
            .unwrap();
            assert!(simplex.state.cost() < 10.0);
            let solver = Annealing::new(
                RandomNeighbor::default,
                5.0 as F,
                TemperatureSchedule::geometric(0.98),
                42,
            );
            let chain =
                Run::new(Rosenbrock::<V, F>::new(), solver, PointState::new(x))
                    .unwrap()
                    .run(8)
                    .unwrap();
            assert!(chain.state.best_cost().is_finite());
            assert_eq!(chain.counts.cost_evals, 9);
        }
    };
}
backend_test!(vec_f64, Vec<f64>, f64);
backend_test!(vec_f32, Vec<f32>, f32);
#[cfg(feature = "nalgebra_all")]
backend_test!(nalgebra_f64, backend_aliases::nalgebra::DVector<f64>, f64);
#[cfg(feature = "nalgebra_all")]
backend_test!(nalgebra_f32, backend_aliases::nalgebra::DVector<f32>, f32);
#[cfg(feature = "ndarray_all")]
backend_test!(ndarray_f64, backend_aliases::ndarray::Array1<f64>, f64);
#[cfg(feature = "ndarray_all")]
backend_test!(ndarray_f32, backend_aliases::ndarray::Array1<f32>, f32);
#[cfg(feature = "faer_all")]
backend_test!(faer_f64, backend_aliases::faer::Col<f64>, f64);
#[cfg(feature = "faer_all")]
backend_test!(faer_f32, backend_aliases::faer::Col<f32>, f32);

#[test]
#[should_panic(expected = "solver published incomplete state")]
fn a_first_order_solver_must_initialize_the_gradient() {
    struct MissingGradient;
    impl Solver<Sphere, FirstOrderState<Vec<f64>>> for MissingGradient {
        type Error = Infallible;
        fn next_iter(
            &mut self,
            _: &mut Problem<Sphere>,
            _: FirstOrderState<Vec<f64>>,
        ) -> Result<
            (FirstOrderState<Vec<f64>>, Option<TerminationReason>),
            Infallible,
        > {
            unreachable!()
        }
    }
    let _ = Run::new(Sphere, MissingGradient, FirstOrderState::new(vec![2.0]));
}

#[test]
fn objective_targets_require_an_eligible_incumbent() {
    let mut run =
        Run::new(Sphere, ExternalDescent, FirstOrderState::new(vec![2.0]))
            .unwrap();
    assert!(!run.target_reached(3.0));
    run.step().unwrap();
    assert!(run.target_reached(3.0));
    assert!(run.cost_budget_reached(2));
    assert!(!run.cost_budget_reached(3));
    let result = run.run(1).unwrap();
    assert_eq!(result.reason, TerminationReason::MaxIter);
}

struct HistoryStop<So> {
    solver: So,
    checks: u64,
    last: Option<(u64, basin::EvalCounts)>,
}

impl<So> HistoryStop<So> {
    fn new(solver: So) -> Self {
        Self {
            solver,
            checks: 0,
            last: None,
        }
    }
}

impl<P, S: Progress, So: Solver<P, S>> Solver<P, S> for HistoryStop<So> {
    type Error = So::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<S, Self::Error> {
        self.solver.init(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<(S, Option<TerminationReason>), Self::Error> {
        self.solver.next_iter(problem, state)
    }
    fn reset_convergence(&mut self) {
        self.checks = 0;
        self.last = None;
        self.solver.reset_convergence();
    }
    fn check_convergence(
        &mut self,
        problem: &Problem<P>,
        state: &S,
    ) -> Option<TerminationReason> {
        let boundary = (state.iter(), *state.counts());
        if self.last != Some(boundary) {
            self.checks += 1;
            self.last = Some(boundary);
        }
        if self.checks >= 8 {
            Some(TerminationReason::SolverConverged)
        } else {
            self.solver.check_convergence(problem, state)
        }
    }
}

#[test]
fn exact_resume_preserves_convergence_history_and_fresh_run_resets_it() {
    let baseline = Run::new(
        Sphere,
        HistoryStop::new(ExternalDescent),
        FirstOrderState::new(vec![2.0]),
    )
    .unwrap()
    .run(100)
    .unwrap();
    assert_eq!(baseline.reason, TerminationReason::SolverConverged);
    assert_eq!(baseline.state.iter(), 7);
    let checkpoint = Run::new(
        Sphere,
        HistoryStop::new(ExternalDescent),
        FirstOrderState::new(vec![2.0]),
    )
    .unwrap()
    .run(3)
    .unwrap()
    .into_checkpoint();
    let (mut solver, state, counts) = checkpoint.into_parts();
    let mut problem = Problem::new(Sphere);
    *problem.counts_mut() = counts;
    assert!(solver.check_convergence(&problem, &state).is_none());
    assert!(solver.check_convergence(&problem, &state).is_none());
    let checkpoint = basin::ExactCheckpoint::from_parts(solver, state, counts);
    let result = Run::resume(Sphere, checkpoint).run(100).unwrap();
    assert_eq!(result.reason, baseline.reason);
    assert_eq!(result.state.param(), baseline.state.param());
    assert_eq!(result.state.iter(), baseline.state.iter());
    assert_eq!(result.counts, baseline.counts);
    let x = result.state.param().clone();
    let warm = Run::new(Sphere, result.solver, result.state)
        .unwrap()
        .run(1)
        .unwrap();
    let fresh = Run::new(
        Sphere,
        HistoryStop::new(ExternalDescent),
        FirstOrderState::new(x),
    )
    .unwrap()
    .run(1)
    .unwrap();
    assert_eq!(warm.state.param(), fresh.state.param());
    assert_eq!(warm.solver.checks, 1);
    assert_eq!(warm.counts, fresh.counts);
}

struct FallibleSphere;

impl CostFunction for FallibleSphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        if x[0] < 0.0 {
            Err("negative input")
        } else {
            Ok(x.norm_squared())
        }
    }
}
impl Gradient for FallibleSphere {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}
impl basin::Residual for FallibleSphere {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(x.clone())
    }
}
impl basin::Jacobian for FallibleSphere {
    type Jacobian = Vec<Vec<f64>>;
    fn jacobian(&self, x: &Vec<f64>) -> Result<Self::Jacobian, Self::Error> {
        Ok((0..x.len())
            .map(|i| (0..x.len()).map(|j| f64::from(i == j)).collect())
            .collect())
    }
}

impl basin::Hessian for FallibleSphere {
    type Hessian = Vec<Vec<f64>>;
    fn hessian(
        &self,
        x: &Vec<f64>,
    ) -> Result<Self::Hessian, <Self as CostFunction>::Error> {
        Ok((0..x.len())
            .map(|i| (0..x.len()).map(|j| 2.0 * f64::from(i == j)).collect())
            .collect())
    }
}

impl basin::HessianProduct for FallibleSphere {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, <Self as CostFunction>::Error> {
        self.gradient(v)
    }
}

struct EvaluationScript {
    fail: bool,
}
impl Solver<FallibleSphere, PointState<Vec<f64>>> for EvaluationScript {
    type Error = &'static str;
    fn init(
        &mut self,
        problem: &mut Problem<FallibleSphere>,
        mut state: PointState<Vec<f64>>,
    ) -> Result<PointState<Vec<f64>>, Self::Error> {
        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<FallibleSphere>,
        mut state: PointState<Vec<f64>>,
    ) -> Result<(PointState<Vec<f64>>, Option<TerminationReason>), Self::Error>
    {
        let x = vec![0.5];
        let (cost, _) = problem.cost_and_gradient(&x)?;
        problem.residual_and_jacobian(&x)?;
        problem.hessian(&x)?;
        problem.hessian_product(&x, &x)?;
        state.replace(x, cost);
        problem.cost_batch(&[vec![1.0], vec![2.0], vec![3.0]])?;
        if self.fail {
            // A consumed candidate must not escape through observation on Err.
            problem.cost_and_gradient(&vec![-1.0])?;
        }
        Ok((state, Some(TerminationReason::SolverConverged)))
    }
}

#[test]
fn counts_keep_categories_at_clean_stops_and_hard_failures() {
    let mut run = Run::new(
        FallibleSphere,
        EvaluationScript { fail: false },
        PointState::new(vec![2.0]),
    )
    .unwrap();
    assert_eq!(
        run.step().unwrap(),
        Some(TerminationReason::SolverConverged)
    );
    let state = run.state().unwrap();
    assert_eq!(state.iter(), 0);
    assert_eq!(state.cost_evals(), 5);
    assert_eq!(state.best().unwrap().counts, *run.counts());
    assert_eq!(run.counts().gradient_evals, 1);
    assert_eq!(run.counts().residual_evals, 1);
    assert_eq!(run.counts().jacobian_evals, 1);
    assert_eq!(run.counts().hessian_evals, 1);
    assert_eq!(run.counts().hessian_product_evals, 1);
    assert_eq!(run.counts().total_work(), 10);
    assert!(!run.cost_budget_reached(6));
    let result = run.run(0).unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);

    let mut failed = Run::new(
        FallibleSphere,
        EvaluationScript { fail: true },
        PointState::new(vec![2.0]),
    )
    .unwrap();
    assert_eq!(failed.step(), Err("negative input"));
    assert!(failed.state().is_none());
    assert_eq!(failed.counts().cost_evals, 6);
    assert_eq!(failed.counts().gradient_evals, 2);
    assert!(failed.into_checkpoint().is_none());

    let mut problem = Problem::new(FallibleSphere);
    assert!(
        problem
            .cost_batch(&[vec![-1.0], vec![2.0], vec![3.0]])
            .is_err()
    );
    assert_eq!(problem.counts().cost_evals, 3);
}

#[test]
fn shared_inner_runs_use_deltas_without_adding_counts_twice() {
    let mut shared = Problem::new(Sphere);
    shared.cost(&vec![2.0]).unwrap();
    let base = *shared.counts();
    let mut inner = basin::InnerExecutor::new(ExternalDescent).max_iter(2);
    let result = inner
        .run(&mut shared, FirstOrderState::new(vec![2.0]))
        .unwrap();
    let delta = shared.counts().delta_since(&base);
    assert_eq!(delta.cost_evals, 3);
    assert_eq!(delta.gradient_evals, 3);
    assert_eq!(result.state.counts(), &delta);

    // A separately counted adapter contributes its work once at the boundary.
    let adapted = Run::new(
        Sphere,
        ExternalDescent,
        FirstOrderState::new(result.state.param().clone()),
    )
    .unwrap()
    .run(1)
    .unwrap();
    shared.counts_mut().add(&adapted.counts);
    let mut published = adapted.state;
    published.publish(false, *shared.counts());
    assert_eq!(published.cost_evals(), 6);
    assert_eq!(published.counts().gradient_evals, 5);
}

#[test]
fn bfgs_accepts_an_explicit_nonclone_matrix_override() {
    use basin::core::math::{MatVec, MatrixIdentity, ScaleInPlace};
    use prototype::solvers::{Dense, RankUpdate};
    struct Custom(Dense<f64>);
    impl MatrixIdentity for Custom {
        fn identity(n: usize) -> Self {
            Self(Dense::identity(n))
        }
    }
    impl MatVec<Vec<f64>> for Custom {
        fn matvec(&self, x: &Vec<f64>) -> Vec<f64> {
            self.0.matvec(x)
        }
    }
    impl ScaleInPlace<f64> for Custom {
        fn scale_in_place(&mut self, alpha: f64) {
            self.0.scale_in_place(alpha);
        }
    }
    impl RankUpdate<Vec<f64>, f64> for Custom {
        fn general_rank_one_update(
            &mut self,
            alpha: f64,
            u: &Vec<f64>,
            v: &Vec<f64>,
        ) {
            self.0.general_rank_one_update(alpha, u, v);
        }
    }
    let solver = Bfgs::<Vec<f64>, f64, Custom>::with_search(
        basin::line_search::Wolfe::new,
    );
    let result =
        Run::from_start(Rosenbrock::<Vec<f64>>::new(), solver, vec![-1.2, 1.0])
            .unwrap()
            .run(8)
            .unwrap();
    let _: &Custom = result.solver.inverse_hessian().unwrap();
    assert!(result.state.cost() < 10.0);
}

#[test]
fn production_observers_borrow_the_same_simplex_storage() {
    use basin::core::observer::{Observe, ObserverMode};
    struct Observer {
        allocation: *const Vec<f64>,
        calls: Rc<std::cell::Cell<u64>>,
    }
    impl<S: basin::core::state::SimplexState<Param = Vec<f64>, Float = f64>>
        Observe<S> for Observer
    {
        fn observe_iter(&mut self, state: &S) {
            assert_eq!(state.vertices().as_ptr(), self.allocation);
            assert_eq!(state.vertices().len(), state.costs().len());
            self.calls.set(self.calls.get() + 1);
        }
    }
    let points = vec![vec![-1.2, 1.0], vec![-1.1, 1.0], vec![-1.2, 1.1]];
    let calls = Rc::new(std::cell::Cell::new(0));
    let observer = Observer {
        allocation: points.as_ptr(),
        calls: calls.clone(),
    };
    basin::Executor::new(
        Rosenbrock::<Vec<f64>>::new(),
        NelderMead::new(),
        SimplexState::new(points).unwrap(),
    )
    .observe_with(observer, ObserverMode::Always)
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(calls.get(), 10);
}
