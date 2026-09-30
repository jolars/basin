//! Explicit constrained selections preserve feasibility ordering and publication metadata.

use basin::{
    AugmentedLagrangianMethod, CostFunction, DenseMatrix, Executor, Gradient,
    GradientDescent, LinearEqualityConstraints, MatrixFromDiagonal, State,
};
use std::convert::Infallible;

struct Equality {
    a: DenseMatrix,
    b: Vec<f64>,
    cost_override: Option<f64>,
}
impl Equality {
    fn new(n: usize) -> Self {
        Self {
            a: DenseMatrix::from_diagonal(&vec![1.0; n]),
            b: vec![1.0; n],
            cost_override: None,
        }
    }
}
impl CostFunction for Equality {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(self
            .cost_override
            .unwrap_or_else(|| x.iter().map(|x| x * x).sum()))
    }
}
impl Gradient for Equality {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}
impl LinearEqualityConstraints for Equality {
    type Matrix = DenseMatrix;
    fn a(&self) -> &DenseMatrix {
        &self.a
    }
    fn b(&self) -> &Vec<f64> {
        &self.b
    }
}

#[test]
fn selected_incumbent_can_increase_objective_to_improve_feasibility() {
    let result = Executor::from_start(
        Equality::new(1),
        AugmentedLagrangianMethod::with_inner_solver(GradientDescent::new(
            0.05,
        ))
        .with_inner_max_iter(100),
        vec![0.0],
    )
    .max_iter(4)
    .run()
    .unwrap();
    assert!(
        result.best_param()[0] > 0.99,
        "infeasible zero-cost seed remained selected"
    );
    assert!(result.best_cost() > 0.98);
}

type EqualitySolver = AugmentedLagrangianMethod<
    GradientDescent<basin::Constant, Vec<f64>>,
    Vec<f64>,
>;

fn solver() -> EqualitySolver {
    AugmentedLagrangianMethod::with_inner_solver(GradientDescent::new(0.05))
        .with_inner_max_iter(100)
}

fn coherent(state: &basin::SelectedState<Vec<f64>>) {
    let (x, cost, violation) = state.current().unwrap();
    assert_eq!(cost, x.iter().map(|x| x * x).sum::<f64>());
    assert_eq!(
        violation,
        x.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>().sqrt()
    );
    if let Some((x, cost, violation)) = state.best() {
        assert_eq!(cost, x.iter().map(|x| x * x).sum::<f64>());
        assert_eq!(
            violation,
            x.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>().sqrt()
        );
    }
}

#[test]
fn augmented_lagrangian_fresh_and_exact_lifecycles() {
    let make =
        || Executor::from_start(Equality::new(2), solver(), vec![0.0; 2]);
    let expected = make().max_iter(6).run_with_solver().unwrap();
    coherent(&expected.state);
    assert_eq!(expected.state.counts(), &expected.counts);
    for split in [0, 1, 3, 5] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed =
            Executor::resume_from_checkpoint(Equality::new(2), checkpoint)
                .require_evaluated_state()
                .max_iter(6)
                .run_with_solver()
                .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
    }
    let mut seed = expected.state;
    seed.replace(vec![2.0; 3], -1000.0, 0.0);
    let initialized = Executor::new(Equality::new(3), expected.solver, seed)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    coherent(&initialized.state);
    assert_eq!(initialized.state.iter(), 0);
    assert_eq!(initialized.state.best_iter(), 0);
    assert_eq!(initialized.counts.cost_evals, 1);
    assert_eq!(initialized.counts.total_work(), 1);
    let restarted = Executor::resume_from_checkpoint(
        Equality::new(3),
        initialized.into_checkpoint(),
    )
    .max_iter(6)
    .run_with_solver()
    .unwrap();
    let rebuilt =
        Executor::from_start(Equality::new(3), solver(), vec![2.0; 3])
            .max_iter(6)
            .run_with_solver()
            .unwrap();
    assert_eq!(restarted.state, rebuilt.state);
    assert_eq!(restarted.counts, rebuilt.counts);
}

#[test]
fn explicit_selection_stamps_once_and_keeps_its_own_record() {
    use basin::{CountsMirror, EvalCounts, IncumbentState, SelectedState};
    let mut state = SelectedState::new(vec![0.0]);
    assert!(state.current().is_none());
    assert!(!state.select_current());
    state.replace(vec![0.0], 0.0, 1.0);
    assert!(state.select_current());
    let first = EvalCounts {
        cost_evals: 1,
        gradient_evals: 2,
        residual_evals: 3,
        jacobian_evals: 4,
        hessian_evals: 5,
        hessian_product_evals: 6,
    };
    state.mirror(&first);
    state.update_best();
    assert_eq!(state.best(), Some((&vec![0.0], 0.0, 1.0)));
    assert_eq!(state.best_counts(), Some(&first));
    // A higher-cost feasible selection wins at the same iteration number.
    state.replace(vec![1.0], 1.0, 0.0);
    assert!(state.select_current());
    // Later work may publish a different current point without changing that choice.
    state.replace(vec![2.0], 4.0, 1.0);
    let second = EvalCounts {
        cost_evals: 7,
        ..first
    };
    state.mirror(&second);
    state.update_best();
    assert_eq!(state.current(), Some((&vec![2.0], 4.0, 1.0)));
    assert_eq!(state.best(), Some((&vec![1.0], 1.0, 0.0)));
    assert_eq!(state.best_counts(), Some(&second));
    state.increment_iter();
    state.replace(vec![3.0], 9.0, 2.0);
    state.mirror(&EvalCounts {
        cost_evals: 8,
        ..second
    });
    state.update_best();
    state.update_best();
    assert_eq!(state.best(), Some((&vec![1.0], 1.0, 0.0)));
    assert_eq!(state.best_iter(), 0);
    assert_eq!(state.best_counts(), Some(&second));
    let best = state.incumbent_record().unwrap();
    assert_eq!(best.param, &vec![1.0]);
    assert_eq!(best.cost, 1.0);
    assert_eq!(best.counts, &second);
    state.reset();
    assert_eq!(state.param(), &vec![3.0]);
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    assert_eq!(state.iter(), 0);
    assert_eq!(state.counts(), &EvalCounts::default());
}

struct FailingEquality(Equality);
impl CostFunction for FailingEquality {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(self.0.cost(x).unwrap())
    }
}
impl Gradient for FailingEquality {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Err("gradient callback aborted")
    }
}
impl LinearEqualityConstraints for FailingEquality {
    type Matrix = DenseMatrix;
    fn a(&self) -> &DenseMatrix {
        &self.0.a
    }
    fn b(&self) -> &Vec<f64> {
        &self.0.b
    }
}

#[test]
fn failed_adapter_work_is_charged_before_propagating_error() {
    use basin::{Problem, SelectedState, Solver};
    let mut problem = Problem::new(FailingEquality(Equality::new(1)));
    let mut solver = solver();
    let state = solver
        .init(&mut problem, SelectedState::new(vec![0.0]))
        .unwrap();
    assert_eq!(problem.counts().total_work(), 1);
    let err = solver.next_iter(&mut problem, state).unwrap_err();
    assert_eq!(err, "gradient callback aborted");
    assert_eq!(problem.counts().cost_evals, 2);
    assert_eq!(problem.counts().gradient_evals, 1);
    assert_eq!(problem.counts().total_work(), 3);
}

#[cfg(feature = "serde")]
#[test]
fn serialized_augmented_lagrangian_checkpoint_retains_multipliers() {
    let make =
        || Executor::from_start(Equality::new(2), solver(), vec![0.0; 2]);
    let expected = make().max_iter(6).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(3)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: basin::ExactCheckpoint<
        EqualitySolver,
        basin::SelectedState<Vec<f64>>,
    > = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Equality::new(2), restored)
        .require_evaluated_state()
        .max_iter(6)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
}

#[test]
fn nonfinite_objectives_do_not_create_invalid_incumbents() {
    for cost in [f64::NAN, f64::INFINITY] {
        let mut problem = Equality::new(1);
        problem.cost_override = Some(cost);
        let result = Executor::from_start(problem, solver(), vec![0.0])
            .require_evaluated_state()
            .max_iter(0)
            .run()
            .unwrap();
        assert!(result.state.current().is_some());
        assert!(result.state.best().is_none());
        assert_eq!(result.state.counts().cost_evals, 1);
    }
    let mut problem = Equality::new(1);
    problem.cost_override = Some(f64::NEG_INFINITY);
    let result = Executor::from_start(problem, solver(), vec![0.0])
        .require_evaluated_state()
        .max_iter(0)
        .run()
        .unwrap();
    assert_eq!(result.report.code(), basin::TerminationCode::MaxIter);
    assert_eq!(result.state.best().unwrap().1, f64::NEG_INFINITY);
}

#[test]
fn constrained_target_hook_requires_feasibility_as_well_as_cost() {
    let result = Executor::from_start(Equality::new(1), solver(), vec![0.0])
        .stop_when(|state| {
            state.best().and_then(|(_, cost, violation)| {
                (violation <= 1e-3 && cost <= 1.0)
                    .then(|| basin::ApplicationStop::new("feasible_target"))
            })
        })
        .max_iter(10)
        .run()
        .unwrap();
    assert_eq!(
        result.report.termination,
        basin::Termination::Application(basin::ApplicationStop::new(
            "feasible_target"
        ))
    );
    assert!(result.state.iter() > 0);
    assert!(result.state.best().unwrap().2 <= 1e-3);
}
