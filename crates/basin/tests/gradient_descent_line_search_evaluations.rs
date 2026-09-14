use std::cell::{Cell, RefCell};
use std::fmt::Debug;

use basin::{
    BasicState, CostFunction, Dot, Executor, Gradient, GradientDescent,
    GradientState, LineSearch, LineSearchOutcome, MoreThuente, NegInPlace,
    NormSquared, Problem, Scalar, ScaleInPlace, ScaledAdd, TerminationReason,
};

#[derive(Debug, PartialEq)]
enum Stop {
    TargetReached,
    GradientFailed,
}

struct Calls<V> {
    costs: RefCell<Vec<V>>,
    gradients: Cell<u64>,
}

impl<V> Default for Calls<V> {
    fn default() -> Self {
        Self {
            costs: RefCell::new(Vec::new()),
            gradients: Cell::new(0),
        }
    }
}

struct Sphere<'a, V, F = f64> {
    calls: &'a Calls<V>,
    target: Option<F>,
    fail_gradient_at: Option<u64>,
}

impl<'a, V, F> Sphere<'a, V, F> {
    fn new(calls: &'a Calls<V>) -> Self {
        Self {
            calls,
            target: None,
            fail_gradient_at: None,
        }
    }
}

impl<V: NormSquared<F> + Clone, F: Scalar> CostFunction for Sphere<'_, V, F> {
    type Param = V;
    type Output = F;
    type Error = Stop;

    fn cost(&self, x: &V) -> Result<F, Stop> {
        self.calls.costs.borrow_mut().push(x.clone());
        let cost = x.norm_squared();
        if self.target.is_some_and(|target| cost <= target) {
            return Err(Stop::TargetReached);
        }
        Ok(cost)
    }
}

impl<V: NormSquared<F> + ScaleInPlace<F> + Clone, F: Scalar> Gradient
    for Sphere<'_, V, F>
{
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, Stop> {
        let calls = self.calls.gradients.get() + 1;
        self.calls.gradients.set(calls);
        if self.fail_gradient_at == Some(calls) {
            return Err(Stop::GradientFailed);
        }
        let mut gradient = x.clone();
        gradient.scale_in_place(F::one() + F::one());
        Ok(gradient)
    }
}

// Exercise the compatibility fallback and reproduce the previous evaluation
// behavior while using the same production line search for trial selection.
struct StepOnly<L>(L);

impl<P, V, F, L: LineSearch<P, V, F>> LineSearch<P, V, F> for StepOnly<L> {
    type Error = L::Error;

    fn next(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<F, Self::Error> {
        self.0.next(problem, param, cost, gradient, direction)
    }
}

fn check_reuse<V, F>(start: V)
where
    F: Scalar,
    V: Clone
        + Debug
        + PartialEq
        + NormSquared<F>
        + Dot<F>
        + ScaledAdd<F>
        + NegInPlace
        + ScaleInPlace<F>,
{
    let calls = Calls::default();
    let result = Executor::new(
        Sphere::<_, F>::new(&calls),
        GradientDescent::with_line_search(MoreThuente::new()),
        BasicState::new(start.clone()),
    )
    .max_iter(2)
    .run()
    .unwrap();
    let old_calls = Calls::default();
    let old = Executor::new(
        Sphere::<_, F>::new(&old_calls),
        GradientDescent::with_line_search(StepOnly(MoreThuente::new())),
        BasicState::new(start.clone()),
    )
    .max_iter(2)
    .run()
    .unwrap();

    assert_eq!(result.iter(), 2);
    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.param(), old.param());
    assert_eq!(result.cost(), old.cost());
    assert_eq!(result.state.gradient(), old.state.gradient());
    assert_eq!(result.cost_evals(), 5);
    assert_eq!(result.state.gradient_evals(), 5);
    assert_eq!(calls.costs.borrow().len(), 5);
    assert_eq!(calls.gradients.get(), 5);
    assert_eq!(old.cost_evals(), 7);
    assert_eq!(old.state.gradient_evals(), 7);
    let mut distinct = old_calls.costs.into_inner();
    distinct.dedup();
    assert_eq!(*calls.costs.borrow(), distinct);

    let expected_cost = start.norm_squared() * F::from_f64(1e-12).unwrap();
    let tolerance = F::from_f64(1e-3).unwrap() * expected_cost;
    assert!((result.cost() - expected_cost).abs() <= tolerance);
    assert_eq!(result.cost(), result.param().norm_squared());
    let mut gradient = result.param().clone();
    gradient.scale_in_place(F::one() + F::one());
    assert_eq!(result.state.gradient(), Some(&gradient));
}

#[test]
fn retained_evaluations_vec() {
    check_reuse::<_, f64>(vec![1.0; 20]);
}

#[test]
fn retained_evaluations_f32() {
    check_reuse::<_, f32>(vec![1.0, 2.0]);
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn retained_evaluations_nalgebra() {
    check_reuse::<_, f64>(backend_aliases::nalgebra::DVector::from_vec(vec![
        1.0, 2.0,
    ]));
}

#[cfg(feature = "ndarray_all")]
#[test]
fn retained_evaluations_ndarray() {
    check_reuse::<_, f64>(backend_aliases::ndarray::Array1::from_vec(vec![
        1.0, 2.0,
    ]));
}

#[cfg(feature = "faer_all")]
#[test]
fn retained_evaluations_faer() {
    check_reuse::<_, f64>(backend_aliases::faer::Col::from_fn(2, |i| {
        (i + 1) as f64
    }));
}

#[test]
fn sphere_target_interrupt_uses_five_cost_and_four_gradient_calls() {
    let calls = Calls::default();
    let mut problem = Sphere::new(&calls);
    problem.target = Some(1e-6);
    let result = Executor::new(
        problem,
        GradientDescent::with_line_search(MoreThuente::new()),
        BasicState::new(vec![1.0; 20]),
    )
    .max_iter(10)
    .run();

    assert!(matches!(result, Err(Stop::TargetReached)));
    assert_eq!(calls.costs.borrow().len(), 5);
    assert_eq!(calls.gradients.get(), 4);
    assert!(calls.costs.borrow().last().unwrap().norm_squared() <= 1e-6);
}

#[test]
fn momentum_evaluates_the_actual_point() {
    let calls = Calls::default();
    let result = Executor::new(
        Sphere::new(&calls),
        GradientDescent::with_line_search(MoreThuente::new().alpha_init(0.25))
            .with_momentum(0.5),
        BasicState::new(vec![1.0, 2.0]),
    )
    .max_iter(2)
    .run()
    .unwrap();

    // The second search accepts x0/4, but accumulated velocity lands at zero.
    assert_eq!(result.param(), &[0.0, 0.0]);
    assert_eq!(result.cost(), 0.0);
    assert_eq!(result.state.gradient().unwrap(), &[0.0, 0.0]);
    assert_eq!(calls.costs.borrow()[3], vec![0.25, 0.5]);
    assert_eq!(calls.costs.borrow()[4], vec![0.0, 0.0]);
    assert_eq!(result.cost_evals(), 5);
    assert_eq!(result.state.gradient_evals(), 5);
}

#[test]
fn zero_step_without_an_evaluation_keeps_a_complete_state() {
    let calls = Calls::default();
    let result = Executor::new(
        Sphere::new(&calls),
        GradientDescent::with_line_search(MoreThuente::new()),
        BasicState::new(vec![0.0, 0.0]),
    )
    .max_iter(1)
    .run()
    .unwrap();

    assert_eq!(result.param(), &[0.0, 0.0]);
    assert_eq!(result.cost(), 0.0);
    assert_eq!(result.state.gradient().unwrap(), &[0.0, 0.0]);
    assert_eq!(result.cost_evals(), 2);
    assert_eq!(result.state.gradient_evals(), 2);
}

struct FailedSearch;

impl<P: CostFunction, V> LineSearch<P, V> for FailedSearch {
    type Error = P::Error;

    fn next(
        &mut self,
        _: &mut Problem<P>,
        _: &V,
        _: f64,
        _: &V,
        _: &V,
    ) -> Result<f64, Self::Error> {
        panic!("the distinguishable failure must be preserved");
    }

    fn next_with_outcome(
        &mut self,
        _: &mut Problem<P>,
        _: &V,
        _: f64,
        _: &V,
        _: &V,
    ) -> Result<LineSearchOutcome<f64>, Self::Error> {
        Ok(LineSearchOutcome::Failed)
    }
}

#[test]
fn legacy_soft_failure_restores_cost_and_gradient() {
    for beta in [0.0, 0.5] {
        let calls = Calls::default();
        let result = Executor::new(
            Sphere::new(&calls),
            GradientDescent::with_line_search(FailedSearch).with_momentum(beta),
            BasicState::new(vec![1.0, 2.0]),
        )
        .max_iter(1)
        .run()
        .unwrap();

        assert_eq!(result.reason, TerminationReason::SolverFailed);
        assert_eq!(result.param(), &[1.0, 2.0]);
        assert_eq!(result.cost(), 5.0);
        assert_eq!(result.state.gradient().unwrap(), &[2.0, 4.0]);
        assert_eq!(result.cost_evals(), 1);
        assert_eq!(result.state.gradient_evals(), 1);
    }
}

#[test]
fn gradient_error_in_line_search_propagates() {
    let calls = Calls::default();
    let mut problem = Sphere::new(&calls);
    problem.fail_gradient_at = Some(2);
    let result = Executor::new(
        problem,
        GradientDescent::with_line_search(MoreThuente::new()),
        BasicState::new(vec![1.0, 2.0]),
    )
    .max_iter(1)
    .run();
    assert!(matches!(result, Err(Stop::GradientFailed)));
    assert_eq!(calls.gradients.get(), 2);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
