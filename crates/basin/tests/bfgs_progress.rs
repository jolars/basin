//! BFGS owns its model; progress and exact continuation share the public state API.

use basin::{
    Bfgs, CostFunction, DenseMatrix, Executor, FirstOrderState, Gradient,
    State, Wolfe,
};
use std::convert::Infallible;

struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}
impl Gradient for Sphere {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|v| 2.0 * v).collect())
    }
}

#[test]
fn model_lives_on_solver_and_fresh_reuse_resets_dimension_and_progress() {
    let result = Executor::from_start(Sphere, Bfgs::new(), vec![3.0, 2.0])
        .max_iter(1)
        .run_with_solver()
        .unwrap();
    assert!(result.solver.inverse_hessian().is_some());
    assert_eq!(result.state.counts(), &result.counts);
    let mut state = result.state;
    state.replace(vec![2.0; 3], 12.0, vec![4.0; 3]).unwrap();
    let fresh = Executor::new(Sphere, result.solver, state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.state.iter(), 0);
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 1);
    assert_eq!(fresh.counts.gradient_evals, 1);
    assert_eq!(fresh.state.counts(), &fresh.counts);
    assert_eq!(
        fresh.solver.inverse_hessian().unwrap(),
        &DenseMatrix::identity(3)
    );
}

use basin::MatrixIdentity;

#[test]
fn checkpoint_at_initialization_preserves_model_counts_and_records() {
    let make = || Executor::from_start(Sphere, Bfgs::new(), vec![3.0, 2.0]);
    let expected = make().max_iter(1).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(0)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
        .require_evaluated_state()
        .max_iter(1)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        resumed.solver.inverse_hessian(),
        expected.solver.inverse_hessian()
    );
}

struct CustomMatrix(DenseMatrix);

impl basin::MatrixIdentity for CustomMatrix {
    fn identity(n: usize) -> Self {
        Self(DenseMatrix::identity(n))
    }
}
impl basin::MatVec<Vec<f64>> for CustomMatrix {
    fn matvec(&self, x: &Vec<f64>) -> Vec<f64> {
        self.0.matvec(x)
    }
}
impl basin::ScaleInPlace for CustomMatrix {
    fn scale_in_place(&mut self, alpha: f64) {
        self.0.scale_in_place(alpha);
    }
}
impl basin::GeneralRankOneUpdate<Vec<f64>> for CustomMatrix {
    fn general_rank_one_update(
        &mut self,
        alpha: f64,
        u: &Vec<f64>,
        v: &Vec<f64>,
    ) {
        self.0.general_rank_one_update(alpha, u, v);
    }
}

#[test]
fn explicit_matrix_and_line_search_remain_available_without_clone() {
    let solver =
        Bfgs::<Vec<f64>, f64, CustomMatrix>::with_matrix_and_line_search(
            Wolfe::new(),
        );
    let result = Executor::new(Sphere, solver, FirstOrderState::new(vec![1.0]))
        .max_iter(1)
        .run_with_solver()
        .unwrap();
    assert_eq!(result.state.cost(), 0.0);
    assert!(result.solver.inverse_hessian().is_some());
    let _checkpoint = result.into_checkpoint();
}

// This line search is deliberately not Clone. Fresh initialization must reset
// its history through the component contract; checkpoints retain it by ownership.
struct StatefulSearch {
    calls: u64,
}

impl basin::LineSearch<Sphere, Vec<f64>> for StatefulSearch {
    type Error = Infallible;
    fn reset(&mut self) {
        self.calls = 0;
    }
    fn next(
        &mut self,
        _: &mut basin::Problem<Sphere>,
        _: &Vec<f64>,
        _: f64,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<f64, Infallible> {
        self.calls += 1;
        Ok(0.1 / self.calls as f64)
    }
}

#[test]
fn fresh_search_history_resets_but_exact_continuation_retains_it() {
    let make = || {
        Executor::from_start(
            Sphere,
            Bfgs::with_line_search(StatefulSearch { calls: 99 }),
            vec![3.0, 2.0],
        )
    };
    let expected = make().max_iter(4).run_with_solver().unwrap();
    let first = make().max_iter(2).run_with_solver().unwrap();
    let resumed =
        Executor::resume_from_checkpoint(Sphere, first.into_checkpoint())
            .max_iter(4)
            .run_with_solver()
            .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    let seed = resumed.state.param().clone();
    let fresh = Executor::new(Sphere, resumed.solver, resumed.state)
        .max_iter(2)
        .run_with_solver()
        .unwrap();
    let new = Executor::from_start(
        Sphere,
        Bfgs::with_line_search(StatefulSearch { calls: 99 }),
        seed,
    )
    .max_iter(2)
    .run_with_solver()
    .unwrap();
    assert_eq!(fresh.state, new.state);
    assert_eq!(fresh.counts, new.counts);
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoint_preserves_model_and_trajectory() {
    let make = || {
        Executor::from_start(
            Sphere,
            Bfgs::with_line_search(basin::Constant(0.1)),
            vec![3.0, 2.0],
        )
    };
    let expected = make().max_iter(4).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(2)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    type Checkpoint = basin::ExactCheckpoint<
        Bfgs<Vec<f64>, f64, DenseMatrix, basin::Constant>,
        FirstOrderState<Vec<f64>>,
    >;
    let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Sphere, restored)
        .max_iter(4)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        resumed.solver.inverse_hessian(),
        expected.solver.inverse_hessian()
    );
}
