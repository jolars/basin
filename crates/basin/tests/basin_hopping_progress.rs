//! Basin-hopping owns adaptation and publishes point-only progress.

use basin::{
    BasinHopping, CostFunction, Executor, Gradient, GradientDescent,
    PointState, State,
};
use std::convert::Infallible;

struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}
impl Gradient for Sphere {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}
type Hopping =
    BasinHopping<GradientDescent<basin::Constant, Vec<f64>>, Vec<f64>>;

fn solver() -> Hopping {
    BasinHopping::new(GradientDescent::new(0.1), 73)
        .with_inner_max_iter(2)
        .with_adaptive_interval(2)
        .with_stepsize(0.4)
}

#[test]
fn fresh_run_resets_progress_and_restarts_random_walk() {
    let old = Executor::new(Sphere, solver(), PointState::new(vec![3.0, -2.0]))
        .max_iter(5)
        .run_with_solver()
        .unwrap();
    let seed = old.state.param().clone();
    let initialized = Executor::new(Sphere, old.solver, old.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(initialized.state.iter(), 0);
    let restarted =
        Executor::resume_from_checkpoint(Sphere, initialized.into_checkpoint())
            .max_iter(8)
            .run_with_solver()
            .unwrap();
    let rebuilt = Executor::new(Sphere, solver(), PointState::new(seed))
        .max_iter(8)
        .run_with_solver()
        .unwrap();
    assert_eq!(restarted.state.param(), rebuilt.state.param());
    assert_eq!(restarted.state.cost(), rebuilt.state.cost());
    assert_eq!(restarted.counts, rebuilt.counts);
}

#[test]
fn exact_resume_retains_adaptation_and_all_progress() {
    let make = || Executor::from_start(Sphere, solver(), vec![3.0, -2.0]);
    let expected = make().max_iter(11).run_with_solver().unwrap();
    for split in [0, 1, 2, 5, 8] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
            .require_evaluated_state()
            .max_iter(11)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(
            resumed.solver.step_taker().stepsize(),
            expected.solver.step_taker().stepsize()
        );
        let (x, cost) = resumed.state.current().unwrap();
        assert_eq!(cost, Sphere.cost(x).unwrap());
        let (x, cost) = resumed.state.best().unwrap();
        assert_eq!(cost, Sphere.cost(x).unwrap());
        assert_eq!(resumed.state.counts(), &resumed.counts);
        assert_eq!(resumed.state.cost_evals(), resumed.counts.cost_evals);
        assert_eq!(resumed.counts.gradient_evals, resumed.counts.cost_evals);
    }
}

struct StatefulStep {
    calls: u64,
}
impl basin::StepTaker<Vec<f64>> for StatefulStep {
    fn reset(&mut self) {
        self.calls = 0;
    }
    fn take_step<R: basin::core::rng::Rng + ?Sized>(
        &mut self,
        x: &Vec<f64>,
        _: &mut R,
    ) -> Vec<f64> {
        self.calls += 1;
        x.iter().map(|x| x + 0.1 * self.calls as f64).collect()
    }
}
struct StatefulAcceptance {
    calls: std::cell::Cell<u64>,
}
impl basin::AcceptanceTest for StatefulAcceptance {
    fn reset(&mut self) {
        self.calls.set(0);
    }
    fn accept<R: basin::core::rng::Rng + ?Sized>(
        &self,
        _: f64,
        _: f64,
        _: &mut R,
    ) -> bool {
        let calls = self.calls.get() + 1;
        self.calls.set(calls);
        calls % 2 == 1
    }
}

#[test]
fn custom_strategies_reset_only_for_fresh_runs() {
    let make_solver = || {
        solver()
            .with_step_taker(StatefulStep { calls: 99 })
            .with_acceptance_test(StatefulAcceptance {
                calls: std::cell::Cell::new(99),
            })
    };
    let make = || Executor::from_start(Sphere, make_solver(), vec![3.0, -2.0]);
    let expected = make().max_iter(7).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(3)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
        .max_iter(7)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    let mut state = resumed.state;
    state.replace(vec![2.0; 3], -1000.0);
    let fresh = Executor::new(Sphere, resumed.solver, state)
        .require_evaluated_state()
        .max_iter(7)
        .run_with_solver()
        .unwrap();
    let rebuilt = Executor::from_start(Sphere, make_solver(), vec![2.0; 3])
        .max_iter(7)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.state, rebuilt.state);
    assert_eq!(fresh.counts, rebuilt.counts);
    assert_eq!(fresh.solver.step_taker().calls, 7);
}

struct Reject;
impl basin::AcceptanceTest for Reject {
    fn accept<R: basin::core::rng::Rng + ?Sized>(
        &self,
        _: f64,
        _: f64,
        _: &mut R,
    ) -> bool {
        false
    }
}
#[test]
fn rejected_hops_do_not_trigger_change_convergence() {
    let result = Executor::from_start(
        Sphere,
        solver()
            .with_acceptance_test(Reject)
            .with_absolute_cost_change_tolerance(0.0)
            .with_absolute_step_tolerance(0.0),
        vec![3.0, -2.0],
    )
    .require_evaluated_state()
    .max_iter(5)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), basin::TerminationCode::MaxIter);
    assert_eq!(result.state.iter(), 5);
    assert_eq!(result.state.best_iter(), 0);
    assert_eq!(result.state.best_counts().unwrap().cost_evals, 3);
    assert_eq!(result.state.counts().cost_evals, 18);
    assert_eq!(result.state.counts().gradient_evals, 18);
}

impl basin::Residual for Sphere {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.clone())
    }
}
impl basin::Jacobian for Sphere {
    type Jacobian = basin::DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<basin::DenseMatrix, Infallible> {
        use basin::MatrixIdentity;
        Ok(basin::DenseMatrix::identity(x.len()))
    }
}
impl basin::Hessian for Sphere {
    type Hessian = basin::DenseMatrix;
    fn hessian(&self, x: &Vec<f64>) -> Result<basin::DenseMatrix, Infallible> {
        use basin::{MatrixIdentity, ScaleInPlace};
        let mut h = basin::DenseMatrix::identity(x.len());
        h.scale_in_place(2.0);
        Ok(h)
    }
}
impl basin::HessianProduct for Sphere {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Infallible> {
        self.gradient(v)
    }
}
struct EveryCategory;
impl basin::InitialState<Vec<f64>> for EveryCategory {
    type State = PointState<Vec<f64>>;
    fn seed(&self, x: &Vec<f64>) -> Self::State {
        PointState::new(x.clone())
    }
}
impl basin::WarmStart<Vec<f64>> for EveryCategory {}
impl basin::Solver<Sphere, PointState<Vec<f64>>> for EveryCategory {
    type Error = Infallible;
    fn init(
        &mut self,
        p: &mut basin::Problem<Sphere>,
        mut s: PointState<Vec<f64>>,
    ) -> Result<PointState<Vec<f64>>, Infallible> {
        s.reset();
        let x = s.param();
        let (cost, _, _) = p.cost_and_gradient_and_hessian(x)?;
        p.residual_and_jacobian(x)?;
        p.hessian_product(x, x)?;
        s.replace(x.clone(), cost);
        Ok(s)
    }
    fn next_iter(
        &mut self,
        _: &mut basin::Problem<Sphere>,
        s: PointState<Vec<f64>>,
    ) -> Result<basin::SolverStep<PointState<Vec<f64>>>, Infallible> {
        Ok(basin::SolverStep::completed(s))
    }
}

#[test]
fn every_inner_evaluation_category_survives_outer_publication() {
    let result = Executor::from_start(
        Sphere,
        BasinHopping::new(EveryCategory, 73).with_inner_max_iter(0),
        vec![3.0, -2.0],
    )
    .require_evaluated_state()
    .max_evaluations(basin::EvaluationKind::HessianProduct, 3)
    .max_iter(10)
    .run_with_solver()
    .unwrap();
    assert_eq!(result.state.iter(), 2);
    assert_eq!(
        result.counts,
        basin::EvalCounts {
            cost_evals: 3,
            gradient_evals: 3,
            residual_evals: 3,
            jacobian_evals: 3,
            hessian_evals: 3,
            hessian_product_evals: 3,
        }
    );
    assert_eq!(result.state.counts(), &result.counts);
    assert_eq!(result.state.cost_evals(), 3);
    let best = result.state.best_counts().unwrap();
    assert_eq!(best.cost_evals, best.gradient_evals);
    assert_eq!(best.cost_evals, best.residual_evals);
    assert_eq!(best.cost_evals, best.jacobian_evals);
    assert_eq!(best.cost_evals, best.hessian_evals);
    assert_eq!(best.cost_evals, best.hessian_product_evals);
}

#[cfg(feature = "serde")]
#[test]
fn serialization_preserves_inner_solver_and_adaptive_walk() {
    let make = || Executor::from_start(Sphere, solver(), vec![3.0, -2.0]);
    let expected = make().max_iter(11).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(5)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: basin::ExactCheckpoint<Hopping, PointState<Vec<f64>>> =
        postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Sphere, restored)
        .max_iter(11)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
}
