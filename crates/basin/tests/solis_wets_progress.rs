//! Solis-Wets keeps its adaptive model across exact resumes and local-search segments.

use basin::{
    BoxConstraints, CostFunction, Executor, MaLsCh, PointState, Problem,
    ResumableInner, SolisWets, Solver, State, TerminationCode,
};
use std::cell::{Cell, RefCell};
use std::convert::Infallible;
use std::rc::Rc;

struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}
fn solver() -> SolisWets<Vec<f64>> {
    SolisWets::new(73).with_initial_step_size(0.4)
}

#[test]
fn exact_resume_preserves_rng_bias_radius_and_streaks() {
    let make = || Executor::from_start(Sphere, solver(), vec![3.0, -2.0]);
    let expected = make().max_iter(35).run_with_solver().unwrap();
    for split in [0, 1, 4, 17, 29] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
            .require_evaluated_state()
            .max_iter(35)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(resumed.state.counts(), &resumed.counts);
        assert_eq!(resumed.state.cost_evals(), resumed.counts.total_work());
        assert_eq!(resumed.solver.bias(), expected.solver.bias());
        assert_eq!(resumed.solver.step_size(), expected.solver.step_size());
        assert_eq!(
            resumed.solver.success_count(),
            expected.solver.success_count()
        );
        assert_eq!(
            resumed.solver.failure_count(),
            expected.solver.failure_count()
        );
    }
}

#[test]
fn fresh_run_resets_progress_and_rebuilds_model_for_new_dimension() {
    let old = Executor::from_start(Sphere, solver(), vec![3.0, -2.0])
        .max_iter(17)
        .run_with_solver()
        .unwrap();
    let mut seed = old.state;
    seed.replace(vec![4.0; 3], -1000.0);
    let initialized = Executor::new(Sphere, old.solver, seed)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(initialized.state.cost(), 48.0);
    assert_eq!(initialized.state.iter(), 0);
    assert_eq!(initialized.state.best_iter(), 0);
    assert_eq!(initialized.counts.total_work(), 1);
    assert_eq!(initialized.solver.step_size(), 0.4);
    assert_eq!(initialized.solver.bias().unwrap(), &[0.0; 3]);
    assert_eq!(initialized.solver.success_count(), 0);
    assert_eq!(initialized.solver.failure_count(), 0);
    let resumed =
        Executor::resume_from_checkpoint(Sphere, initialized.into_checkpoint())
            .max_iter(20)
            .run_with_solver()
            .unwrap();
    let rebuilt = Executor::from_start(Sphere, solver(), vec![4.0; 3])
        .max_iter(20)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, rebuilt.state);
    assert_eq!(resumed.counts, rebuilt.counts);
    assert_eq!(resumed.solver.bias(), rebuilt.solver.bias());
}

struct Rejected(f64);
impl CostFunction for Rejected {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(if x == &[0.0, 0.0] { 0.0 } else { self.0 })
    }
}

#[test]
fn rejected_trials_do_not_create_cost_or_step_convergence() {
    for cost in [f64::NAN, f64::INFINITY, 0.0, 1.0] {
        let result = Executor::from_start(
            Rejected(cost),
            solver()
                .with_absolute_cost_change_tolerance(0.0)
                .with_absolute_step_tolerance(0.0),
            vec![0.0, 0.0],
        )
        .require_evaluated_state()
        .max_iter(5)
        .run_with_solver()
        .unwrap();
        assert_eq!(result.report.code(), TerminationCode::MaxIter);
        assert_eq!(result.state.iter(), 5);
        assert_eq!(result.state.cost(), 0.0);
        assert_eq!(result.state.best_iter(), 0);
        assert_eq!(result.state.best_counts().unwrap().cost_evals, 1);
        assert_eq!(result.state.counts().cost_evals, 11);
    }
    let result = Executor::from_start(
        Rejected(f64::INFINITY),
        solver().with_absolute_step_size_tolerance(0.2),
        vec![0.0, 0.0],
    )
    .max_iter(10)
    .run_with_solver()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::RhoTolerance);
    assert_eq!(result.state.iter(), 3);
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoint_retains_complete_model() {
    let make = || Executor::from_start(Sphere, solver(), vec![3.0, -2.0]);
    let expected = make().max_iter(35).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(17)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: basin::ExactCheckpoint<
        SolisWets<Vec<f64>>,
        PointState<Vec<f64>>,
    > = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Sphere, restored)
        .max_iter(35)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
}

struct FlatBox {
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl CostFunction for FlatBox {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(1.0)
    }
}
impl BoxConstraints for FlatBox {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

// A custom chain with deliberately destructive fresh initialization makes
// repeated initialization observable.
#[derive(Clone)]
struct Chain {
    initialized_seed: bool,
    init_calls: Rc<Cell<usize>>,
    seen: Rc<RefCell<Vec<u64>>>,
    model_steps: u64,
}
impl ResumableInner<Vec<f64>> for Chain {
    type State = PointState<Vec<f64>>;
    fn seed_chain(
        &self,
        x: &Vec<f64>,
        fx: f64,
        _: f64,
        _: u64,
    ) -> (Self, Self::State) {
        let mut state = PointState::new(x.clone());
        state.replace(x.clone(), fx);
        (self.clone(), state)
    }
    fn seeded_chain_is_initialized(&self) -> bool {
        self.initialized_seed
    }
    fn prepare_resume(&self, state: &mut Self::State) {
        state.reset_progress();
    }
}
impl Solver<FlatBox, PointState<Vec<f64>>> for Chain {
    type Error = Infallible;
    fn init(
        &mut self,
        _: &mut Problem<FlatBox>,
        state: PointState<Vec<f64>>,
    ) -> Result<PointState<Vec<f64>>, Infallible> {
        self.init_calls.set(self.init_calls.get() + 1);
        self.model_steps = 0;
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<FlatBox>,
        state: PointState<Vec<f64>>,
    ) -> Result<basin::SolverStep<PointState<Vec<f64>>>, Infallible> {
        self.model_steps += 1;
        self.seen.borrow_mut().push(self.model_steps);
        problem.cost(state.param())?;
        Ok(basin::SolverStep::from((state, None)))
    }
}

#[test]
fn local_search_segments_skip_reinitialization_and_restart_budgets() {
    for initialized_seed in [false, true] {
        let init_calls = Rc::new(Cell::new(0));
        let seen = Rc::new(RefCell::new(Vec::new()));
        let chain = Chain {
            initialized_seed,
            init_calls: init_calls.clone(),
            seen: seen.clone(),
            model_steps: 0,
        };
        let solver = MaLsCh::with_inner(73, chain)
            .with_pop_size(4)
            .with_nam_pool(4)
            .with_ls_intensity(2)
            .with_nfrec(1)
            .with_ls_improvement_threshold(0.0);
        let result = Executor::new(
            FlatBox {
                lower: vec![-5.0; 2],
                upper: vec![5.0; 2],
            },
            solver,
            basin::PopulationProgress::empty(),
        )
        .max_iter(3)
        .run_with_solver()
        .unwrap();
        assert_eq!(init_calls.get(), usize::from(!initialized_seed));
        assert_eq!(*seen.borrow(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(result.state.cost_evals(), 13);
        assert_eq!(result.solver.ls_application_count(0), 3);
    }
}

#[test]
fn configured_solis_wets_chain_reuses_supplied_cost() {
    let solver = MaLsCh::with_inner(
        73,
        solver().with_absolute_cost_change_tolerance(None),
    )
    .with_pop_size(4)
    .with_nam_pool(4)
    .with_ls_intensity(2)
    .with_nfrec(1)
    .with_ls_improvement_threshold(0.0);
    let result = Executor::new(
        FlatBox {
            lower: vec![-5.0; 2],
            upper: vec![5.0; 2],
        },
        solver,
        basin::PopulationProgress::empty(),
    )
    .max_iter(3)
    .run_with_solver()
    .unwrap();
    // Four initial points, then one offspring and two SW proposals per segment.
    assert_eq!(result.state.cost_evals(), 13);
    assert_eq!(result.solver.ls_application_count(0), 3);
}
