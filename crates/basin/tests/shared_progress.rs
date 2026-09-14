//! External solvers use shared storage through Basin's public API.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

use basin::core::executor::run_loop_with_control;
use basin::core::math::{
    NormSquared, Scalar, ScaleInPlace, ScaledAdd, VectorLen,
};
use basin::{
    CostFunction, CountsMirror, EvalCounts, Executor, FirstOrderState,
    Gradient, GradientDimensionMismatch, GradientState, PointState, Problem,
    RunControl, Solver, State, TerminationReason,
};
use std::convert::Infallible;
use std::marker::PhantomData;

struct Sphere<V, F>(PhantomData<(V, F)>);

impl<V: NormSquared<F>, F: Scalar> CostFunction for Sphere<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        Ok(x.norm_squared())
    }
}

impl<V: Clone + NormSquared<F> + ScaleInPlace<F>, F: Scalar> Gradient
    for Sphere<V, F>
{
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        let mut gradient = x.clone();
        gradient.scale_in_place(F::one() + F::one());
        Ok(gradient)
    }
}

// Deliberately not Clone: owned continuation must retain solver machinery.
#[derive(Default)]
struct Descent {
    steps: u64,
    init_calls: u64,
}

impl<P, V, F> Solver<P, FirstOrderState<V, F>> for Descent
where
    P: Gradient<Param = V, Output = F, Gradient = V>,
    V: Clone + VectorLen + ScaledAdd<F>,
    F: Scalar,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        self.steps = 0;
        self.init_calls += 1;
        state.reset();
        let x = state.param().clone();
        let (cost, gradient) = problem.cost_and_gradient(&x)?;
        state.replace(x, cost, gradient).unwrap();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<(FirstOrderState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let (x, _, gradient) = state.current().unwrap();
        let mut next = x.clone();
        next.scaled_add(-F::from_f64(0.25).unwrap(), gradient);
        let (cost, gradient) = problem.cost_and_gradient(&next)?;
        state.replace(next, cost, gradient).unwrap();
        self.steps += 1;
        Ok((state, None))
    }
}

fn backend_round_trip<V, F>(seed: V)
where
    V: Clone + VectorLen + NormSquared<F> + ScaleInPlace<F> + ScaledAdd<F>,
    F: Scalar,
{
    let initial_cost = seed.norm_squared();
    let result = Executor::new(
        Sphere(PhantomData),
        Descent::default(),
        FirstOrderState::new(seed.clone()),
    )
    .max_iter(3)
    .run_with_solver()
    .unwrap();
    let state = result.state;
    assert_eq!(state.iter(), 3);
    assert_eq!(result.solver.steps, 3);
    assert_eq!(result.counts.cost_evals, 4);
    assert_eq!(result.counts.gradient_evals, 4);
    assert_eq!(state.counts(), &result.counts);
    let best_evals = if initial_cost == F::zero() { 1 } else { 4 };
    assert_eq!(state.best_counts().unwrap().cost_evals, best_evals);
    assert_eq!(state.best_counts().unwrap().gradient_evals, best_evals);
    assert_eq!(state.cost(), initial_cost / F::from_f64(64.0).unwrap());
    assert_eq!(state.best().unwrap().1, state.cost());
    assert_eq!(state.gradient().unwrap().vec_len(), state.param().vec_len());

    let point_result =
        Executor::new(Sphere(PhantomData), Halve, PointState::new(seed))
            .max_iter(3)
            .run_with_solver()
            .unwrap();
    assert_eq!(point_result.state.cost(), state.cost());
    assert_eq!(point_result.state.iter(), 3);
    assert_eq!(point_result.state.counts(), &point_result.counts);
    assert_eq!(point_result.counts.cost_evals, 4);
    assert_eq!(point_result.counts.gradient_evals, 0);
    assert_eq!(point_result.state.best_cost_evals(), best_evals);
}

struct Halve;

impl<P, V, F> Solver<P, PointState<V, F>> for Halve
where
    P: CostFunction<Param = V, Output = F>,
    V: Clone + ScaleInPlace<F>,
    F: Scalar,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        state.reset();
        let x = state.param().clone();
        let cost = problem.cost(&x)?;
        state.replace(x, cost);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<(PointState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let mut x = state.param().clone();
        x.scale_in_place(F::from_f64(0.5).unwrap());
        let cost = problem.cost(&x)?;
        state.replace(x, cost);
        Ok((state, None))
    }
}

#[test]
fn vec_backends() {
    backend_round_trip::<_, f64>(vec![2.0, 4.0]);
    backend_round_trip::<_, f32>(vec![2.0, 4.0]);
    backend_round_trip::<_, f64>(vec![]);
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_backends() {
    use backend_aliases::nalgebra::DVector;
    backend_round_trip::<_, f64>(DVector::from_vec(vec![2.0, 4.0]));
    backend_round_trip::<_, f32>(DVector::from_vec(vec![2.0, 4.0]));
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_backends() {
    use backend_aliases::ndarray::Array1;
    backend_round_trip::<_, f64>(Array1::from_vec(vec![2.0, 4.0]));
    backend_round_trip::<_, f32>(Array1::from_vec(vec![2.0, 4.0]));
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_backends() {
    use backend_aliases::faer::Col;
    backend_round_trip::<_, f64>(Col::from_fn(2, |i| [2.0, 4.0][i]));
    backend_round_trip::<_, f32>(Col::from_fn(2, |i| [2.0, 4.0][i]));
}

#[test]
fn seed_and_invalid_costs_do_not_establish_an_incumbent() {
    let mut state = PointState::new(1.0);
    assert_eq!(state.param(), &1.0);
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    assert!(state.best_counts().is_none());
    for cost in [f64::INFINITY, f64::NAN] {
        state.replace(2.0, cost);
        state.update_best();
        assert!(state.current().is_some());
        assert!(state.best().is_none());
    }
    state.replace(3.0, f64::NEG_INFINITY);
    state.update_best();
    assert_eq!(state.best(), Some((&3.0, f64::NEG_INFINITY)));
}

#[test]
fn gradient_replacement_is_atomic_and_reset_retains_only_the_seed() {
    let mut state = FirstOrderState::new(vec![1.0, 2.0]);
    assert!(state.current().is_none());
    assert!(state.gradient().is_none());
    state.replace(vec![1.0, 2.0], 5.0, vec![2.0, 4.0]).unwrap();
    state.mirror(&all_counts());
    state.increment_iter();
    state.update_best();
    let before = state.clone();
    assert_eq!(
        state.replace(vec![3.0, 4.0], 25.0, vec![6.0]),
        Err(GradientDimensionMismatch {
            param_len: 2,
            gradient_len: 1
        })
    );
    assert_eq!(state, before);
    state.reset();
    assert_eq!(state.param(), &[1.0, 2.0]);
    assert!(state.current().is_none());
    assert!(state.gradient().is_none());
    assert!(state.best().is_none());
    assert_eq!(state.iter(), 0);
    assert_eq!(state.counts(), &EvalCounts::default());
    // A fresh solve can use a different dimension.
    state.replace(vec![3.0], 9.0, vec![6.0]).unwrap();
}

fn all_counts() -> EvalCounts {
    EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 5,
        jacobian_evals: 7,
        hessian_evals: 11,
        hessian_product_evals: 13,
    }
}

#[test]
fn counts_keep_all_categories_and_legacy_folds() {
    let counts = all_counts();
    let mut point = PointState::new(vec![1.0]);
    point.replace(vec![1.0], 1.0);
    point.mirror(&counts);
    point.update_best();
    assert_eq!(point.counts(), &counts);
    assert_eq!(point.best_counts(), Some(&counts));
    assert_eq!(point.cost_evals(), 41);
    assert_eq!(point.best_cost_evals(), 41);

    let mut first = FirstOrderState::new(vec![1.0]);
    first.replace(vec![1.0], 1.0, vec![2.0]).unwrap();
    first.mirror(&counts);
    first.update_best();
    assert_eq!(first.counts(), &counts);
    assert_eq!(first.best_counts(), Some(&counts));
    assert_eq!(first.cost_evals(), 7);
    assert_eq!(first.gradient_evals(), 34);
    assert_eq!(first.best_cost_evals(), 7);
    assert_eq!(first.best_gradient_evals(), 34);
}

#[test]
fn nonmonotone_and_equal_cost_publications_retain_matching_history() {
    let mut state = PointState::new(vec![1.0]);
    state.replace(vec![1.0], 1.0);
    state.mirror(&all_counts());
    state.update_best();
    for (x, cost) in [(2.0, 4.0), (-1.0, 1.0), (3.0, f64::NAN)] {
        state.replace(vec![x], cost);
        state.increment_iter();
        state.mirror(&EvalCounts {
            cost_evals: 100,
            ..all_counts()
        });
        state.update_best();
        state.update_best();
        assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
        assert_eq!(state.best_iter(), 0);
        assert_eq!(state.best_counts(), Some(&all_counts()));
    }
    state.reset_best();
    assert!(state.best().is_none());
    assert_eq!(state.iter(), 3);
    assert!(state.current().unwrap().1.is_nan());
}

#[test]
fn fresh_reuse_resets_progress_and_checkpoint_resume_preserves_it() {
    for pause in [0, 2] {
        let mut stepper = Executor::new(
            Sphere(PhantomData),
            Descent::default(),
            FirstOrderState::new(vec![8.0]),
        )
        .into_stepper()
        .unwrap();
        for _ in 0..pause {
            stepper.step().unwrap();
        }
        let checkpoint = stepper.into_checkpoint().unwrap();
        let resumed =
            Executor::resume_from_checkpoint(Sphere(PhantomData), checkpoint)
                .max_iter(4)
                .run_with_solver()
                .unwrap();
        let direct = Executor::new(
            Sphere(PhantomData),
            Descent::default(),
            FirstOrderState::new(vec![8.0]),
        )
        .max_iter(4)
        .run_with_solver()
        .unwrap();
        assert_eq!(resumed.state, direct.state);
        assert_eq!(resumed.counts, direct.counts);
        assert_eq!(resumed.solver.init_calls, 1);
        assert_eq!(resumed.solver.steps, 4);

        let seed = resumed.state.param().clone();
        let fresh =
            Executor::new(Sphere(PhantomData), resumed.solver, resumed.state)
                .max_iter(1)
                .run_with_solver()
                .unwrap();
        let expected = Executor::new(
            Sphere(PhantomData),
            Descent::default(),
            FirstOrderState::new(seed),
        )
        .max_iter(1)
        .run_with_solver()
        .unwrap();
        assert_eq!(fresh.state, expected.state);
        assert_eq!(fresh.solver.steps, 1);
        assert_eq!(fresh.solver.init_calls, 2);
    }
}

#[test]
fn nested_runs_mirror_only_their_own_work() {
    let mut problem = Problem::new(Sphere(PhantomData));
    problem.cost(&vec![10.0]).unwrap();
    let mut solver = Descent::default();
    let result = run_loop_with_control(
        &mut problem,
        FirstOrderState::new(vec![2.0]),
        &mut solver,
        &mut RunControl::new().max_iter(2),
    )
    .unwrap();
    assert_eq!(problem.counts().cost_evals, 4);
    assert_eq!(result.state.counts().cost_evals, 3);
    assert_eq!(result.state.counts().gradient_evals, 3);
}

struct FallibleIdentity;

impl CostFunction for FallibleIdentity {
    type Param = f64;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
        if *x == 99.0 {
            Err("failed probe")
        } else {
            Ok(*x)
        }
    }
}

struct Probe {
    hard_error: bool,
}

impl Solver<FallibleIdentity, PointState<f64>> for Probe {
    type Error = &'static str;

    fn init(
        &mut self,
        problem: &mut Problem<FallibleIdentity>,
        mut state: PointState<f64>,
    ) -> Result<PointState<f64>, Self::Error> {
        state.reset();
        let x = *state.param();
        state.replace(x, problem.cost(&x)?);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<FallibleIdentity>,
        mut state: PointState<f64>,
    ) -> Result<(PointState<f64>, Option<TerminationReason>), Self::Error> {
        state.replace(-2.0, problem.cost(&-2.0)?);
        // Unpublished probes still count, including the failing one.
        problem.cost(&-10.0)?;
        let error = problem.cost(&99.0).unwrap_err();
        if self.hard_error {
            return Err(error);
        }
        Ok((state, Some(TerminationReason::SolverFailed)))
    }
}

#[test]
fn mid_step_stop_stamps_final_counts_without_completing_an_iteration() {
    let result = Executor::new(
        FallibleIdentity,
        Probe { hard_error: false },
        PointState::new(2.0),
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.state.iter(), 0);
    assert_eq!(result.state.best(), Some((&-2.0, -2.0)));
    assert_eq!(result.state.best_iter(), 0);
    assert_eq!(result.state.best_cost_evals(), 4);
    assert_eq!(result.state.best_counts(), Some(&result.counts));
    assert_eq!(result.counts.cost_evals, 4);
}

#[test]
fn hard_error_cannot_yield_a_checkpoint_of_partial_progress() {
    let mut stepper = Executor::new(
        FallibleIdentity,
        Probe { hard_error: true },
        PointState::new(2.0),
    )
    .into_stepper()
    .unwrap();
    assert_eq!(stepper.step(), Err("failed probe"));
    assert_eq!(stepper.counts().cost_evals, 4);
    assert!(stepper.into_checkpoint().is_none());
}

#[test]
fn nonfinite_cost_does_not_choose_a_termination_reason() {
    for seed in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let result = Executor::new(
            FallibleIdentity,
            Probe { hard_error: false },
            PointState::new(seed),
        )
        .max_iter(0)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::MaxIter);
        assert_eq!(result.state.best().is_some(), seed == f64::NEG_INFINITY);
    }
}

#[cfg(feature = "serde")]
#[test]
fn serde_preserves_seeds_records_and_raw_history_for_both_scalars() {
    fn round_trip<S>(state: &S)
    where
        S: serde::Serialize
            + serde::de::DeserializeOwned
            + PartialEq
            + std::fmt::Debug,
    {
        let bytes =
            bincode::serde::encode_to_vec(state, bincode::config::standard())
                .unwrap();
        let (decoded, read): (S, _) = bincode::serde::decode_from_slice(
            &bytes,
            bincode::config::standard(),
        )
        .unwrap();
        assert_eq!(read, bytes.len());
        assert_eq!(&decoded, state);
    }

    fn check<F: Scalar + serde::Serialize + serde::de::DeserializeOwned>() {
        let mut point = PointState::new(vec![F::one()]);
        round_trip(&point);
        point.replace(vec![F::one()], F::one());
        point.mirror(&all_counts());
        point.update_best();
        point.replace(vec![F::zero()], F::infinity());
        point.increment_iter();
        round_trip(&point);

        let mut first = FirstOrderState::new(vec![F::one()]);
        round_trip(&first);
        first
            .replace(vec![F::one()], F::one(), vec![F::one()])
            .unwrap();
        first.mirror(&all_counts());
        first.update_best();
        first
            .replace(vec![F::zero()], F::infinity(), vec![F::zero()])
            .unwrap();
        first.increment_iter();
        round_trip(&first);
    }
    check::<f64>();
    check::<f32>();
}
