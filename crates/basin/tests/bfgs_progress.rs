//! BFGS owns its model; progress and exact continuation share the public state API.

use basin::{
    Bfgs, CostFunction, DenseMatrix, Executor, FirstOrderState, Gradient,
    State, Wolfe,
};
use std::convert::Infallible;

struct Sphere;

impl basin::Constraints for Sphere {
    type Set = basin::NoConstraints;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new()
    }
}
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
fn executor_results_expose_checked_incumbents() {
    let run =
        |x| Executor::from_start(Sphere, Bfgs::new(), vec![x]).max_iter(0);
    let expected_limit =
        basin::Termination::Limit(vec![basin::ExecutionLimit::Iterations {
            observed: 0,
            limit: 0,
        }]);

    // The squared objective overflows while the gradient stays finite.
    let result = run(1e200).run().unwrap();
    assert_eq!(result.report.termination, expected_limit);
    assert_eq!(
        result.current_record().unwrap().cost.unwrap(),
        f64::INFINITY
    );
    assert!(result.selected_record().is_none());

    let result = run(1e200).run_with_solver().unwrap();
    assert_eq!(result.report.termination, expected_limit);
    assert_eq!(
        result.current_record().unwrap().cost.unwrap(),
        f64::INFINITY
    );
    assert!(result.selected_record().is_none());

    let result = run(2.0).run().unwrap();
    let with_solver = run(2.0).run_with_solver().unwrap();
    for incumbent in [result.selected_record(), with_solver.selected_record()] {
        let incumbent = incumbent.unwrap();
        assert_eq!(incumbent.param.as_slice(), &[2.0]);
        assert_eq!(incumbent.cost.unwrap(), 4.0);
        assert_eq!(incumbent.iter, 0);
        assert_eq!(incumbent.counts, &result.counts);
        assert_eq!(incumbent.counts.cost_evals, 1);
        assert_eq!(incumbent.counts.gradient_evals, 1);
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
    assert_eq!(fresh.state.selected_record().unwrap().iter, 0);
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
    assert_eq!(result.state.current_record().unwrap().cost.unwrap(), 0.0);
    assert!(result.solver.inverse_hessian().is_some());
    let _checkpoint = result.into_checkpoint();
}

// Named initializers can use the original matrix contract without adding
// entry access, cloning, or backend factorization requirements.
struct HalfIdentity;

impl basin::solver::bfgs::InitialInverseHessian<Vec<f64>, CustomMatrix>
    for HalfIdentity
{
    fn initialize(&self, x: &Vec<f64>, _: &Vec<f64>) -> CustomMatrix {
        let mut h = CustomMatrix::identity(x.len());
        basin::ScaleInPlace::scale_in_place(&mut h, 0.5);
        h
    }
}

#[test]
fn named_seed_supports_nonclone_matrix_without_entry_access() {
    let solver =
        Bfgs::<Vec<f64>, f64, CustomMatrix>::with_matrix_and_line_search(
            Wolfe::new(),
        )
        .with_initial_inverse_hessian_initializer(HalfIdentity);
    let result = Executor::from_start(Sphere, solver, vec![3.0, 2.0])
        .max_iter(1)
        .run_with_solver()
        .unwrap();
    assert_eq!(result.state.current_record().unwrap().cost.unwrap(), 0.0);
    assert_eq!(
        result.solver.inverse_hessian().unwrap().0,
        DenseMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]),
    );
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
    type Checkpoint = basin::Checkpoint<
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

struct ScaledQuadratic;

impl CostFunction for ScaledQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(0.5 * 1e-19 * x[0] * x[0])
    }
}

impl Gradient for ScaledQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![1e-19 * x[0]])
    }
}

#[test]
fn more_thuente_expands_rounded_away_bfgs_step() {
    use basin::{MoreThuente, TerminationCode};

    let target = 1e-8;
    let result = Executor::from_start(
        ScaledQuadratic,
        Bfgs::with_line_search(MoreThuente::new())
            .with_absolute_cost_change_tolerance(target),
        vec![1e12],
    )
    .max_iter(20)
    .target_objective(target)
    .run()
    .unwrap();

    assert_eq!(result.report.code(), TerminationCode::TargetCost);
    assert!(result.current_record().unwrap().cost.unwrap() <= target);
}

#[test]
fn more_thuente_rounded_trials_respect_step_and_evaluation_limits() {
    use basin::line_search::more_thuente::diagnostics::SearchExit;
    use basin::{LineSearch, LineSearchBounds, LineSearchOutcome, MoreThuente};

    let x = vec![1e12];
    let f = ScaledQuadratic.cost(&x).unwrap();
    let g = ScaledQuadratic.gradient(&x).unwrap();
    let d = vec![-g[0]];
    for (cap, maxfev, evaluations, exit, step) in [
        (1024.0, 1, 1, SearchExit::Warning, 1024.0),
        (512.0, 1, 0, SearchExit::Warning, 0.0),
        (1024.0, 0, 0, SearchExit::Exhausted, 0.0),
    ] {
        for configured_cap in [false, true] {
            let mut p = basin::Problem::new(ScaledQuadratic);
            let mut search = MoreThuente::new()
                .stpmax(if configured_cap { cap } else { 1e10 })
                .maxfev(maxfev)
                .with_trial_diagnostics(true);
            let result = search
                .next_with_bounds(
                    &mut p,
                    &x,
                    f,
                    &g,
                    &d,
                    LineSearchBounds::new(
                        1.0,
                        if configured_cap { 1e10 } else { cap },
                    ),
                )
                .unwrap();

            assert_eq!(result.outcome, LineSearchOutcome::Step(step));
            assert_eq!(search.search_exit(), Some(exit));
            assert_eq!(search.selected_step(), Some(step));
            assert_eq!(p.counts().cost_evals, evaluations);
            assert_eq!(p.counts().gradient_evals, evaluations);
            assert_eq!(search.trial_observations().len() as u64, evaluations);
            if let Some(evaluation) = result.evaluation {
                assert_eq!(evaluations, 1);
                assert_ne!(evaluation.param, x);
                assert_eq!(search.trial_observations()[0].step, cap);
            } else {
                assert_eq!(evaluations, 0);
            }
        }
    }
}

#[derive(Default)]
struct UnitQuadratic {
    cost_points: std::cell::RefCell<Vec<f64>>,
    gradient_points: std::cell::RefCell<Vec<f64>>,
}

impl CostFunction for &UnitQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        self.cost_points.borrow_mut().push(x[0]);
        Ok(0.5 * x[0] * x[0])
    }
}

impl Gradient for &UnitQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        self.gradient_points.borrow_mut().push(x[0]);
        Ok(x.clone())
    }
}

fn assert_single_bfgs_evaluation_per_point(
    search: impl for<'a> basin::LineSearch<
        &'a UnitQuadratic,
        Vec<f64>,
        f64,
        Error = Infallible,
    >,
) {
    let problem = UnitQuadratic::default();
    let result = Executor::from_start(
        &problem,
        Bfgs::with_line_search(search).with_absolute_gradient_tolerance(1e-12),
        vec![1.0],
    )
    .max_iter(2)
    .run()
    .unwrap();

    assert_eq!(result.state.param(), &vec![0.0]);
    assert_eq!(result.current_record().unwrap().cost.unwrap(), 0.0);
    assert_eq!(*problem.cost_points.borrow(), [1.0, 0.0]);
    assert_eq!(*problem.gradient_points.borrow(), [1.0, 0.0]);
}

#[test]
fn bfgs_reuses_the_accepted_line_search_evaluation() {
    assert_single_bfgs_evaluation_per_point(Wolfe::new());
    assert_single_bfgs_evaluation_per_point(basin::MoreThuente::new());
}

#[test]
fn bfgs_evaluates_the_accepted_point_for_step_only_searches() {
    assert_single_bfgs_evaluation_per_point(basin::Constant(1.0));
}
