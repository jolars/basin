use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
    rc::Rc,
    sync::mpsc::{self, Sender},
    thread::{self, ThreadId},
};

use basin::{
    ApplicationStop, CheckpointSink, CostFunction, EvalCounts, Executor,
    FirstOrderState, Gradient, GradientDescent, ObservationEvent, Observe,
    ObserverMode, OptimizationResult, State, Termination,
    line_search::Constant,
};
use basin_evaluation_application::executor_prototypes::InferredExecutor;
use basin_evaluation_local::Local;

type Progress = FirstOrderState<Vec<f64>>;
type Descent = GradientDescent<Constant<f64>, Vec<f64>>;

struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, point: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(point.iter().map(|x| x * x / 2.0).sum())
    }
}

impl Gradient for Sphere {
    type Gradient = Vec<f64>;

    fn gradient(&self, point: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(point.clone())
    }
}

#[derive(Debug, PartialEq)]
enum Event {
    Init,
    SolverIter(u64),
    StateIter(u64),
    StopCheck(u64),
    Save(u64, EvalCounts),
    Final(u64),
}

struct StateObserver<F>(F);

impl<F: FnMut(Event)> Observe<Progress> for StateObserver<F> {
    fn observe_iter(&mut self, state: &Progress) {
        (self.0)(Event::StateIter(state.iter()));
    }
}

struct Sink<F>(F);

impl<F: FnMut(Event)> CheckpointSink<Descent, Progress> for Sink<F> {
    fn save(&mut self, _: &Descent, state: &Progress, counts: &EvalCounts) {
        (self.0)(Event::Save(state.iter(), *counts));
    }
}

fn observe<F: FnMut(Event)>(
    mut record: F,
) -> impl FnMut(&Progress, &Descent, ObservationEvent<'_, f64>) {
    move |state, _, event| match event {
        ObservationEvent::Init => record(Event::Init),
        ObservationEvent::Iter => record(Event::SolverIter(state.iter())),
        ObservationEvent::Final(_) => record(Event::Final(state.iter())),
        _ => {}
    }
}

fn stop_after_one<F: FnMut(Event)>(
    mut record: F,
) -> impl FnMut(&Progress) -> Option<ApplicationStop> {
    move |state| {
        record(Event::StopCheck(state.iter()));
        (state.iter() == 1).then(|| ApplicationStop::new("one step"))
    }
}

fn worker_trace(sender: Sender<Event>) -> impl FnMut(Event) + Send {
    let caller = thread::current().id();
    // This capture is Send and !Sync. The callbacks only need to move.
    let owner = Cell::<Option<ThreadId>>::new(None);
    move |event| {
        let current = thread::current().id();
        assert_ne!(current, caller);
        assert_eq!(owner.replace(Some(current)).unwrap_or(current), current);
        sender.send(event).unwrap();
    }
}

fn local_trace(
    history: &Rc<RefCell<Vec<Event>>>,
) -> impl FnMut(Event) + 'static {
    let caller = thread::current().id();
    let history = Rc::clone(history);
    move |event| {
        assert_eq!(thread::current().id(), caller);
        history.borrow_mut().push(event);
    }
}

fn assert_run(result: OptimizationResult<Progress>, trace: &[Event]) {
    assert_eq!(result.iter(), 1);
    assert_eq!(result.param(), &[1.5]);
    assert_eq!(result.counts.cost_evals, 2);
    assert_eq!(result.counts.gradient_evals, 2);
    assert_eq!(
        result.report.termination,
        Termination::Application(ApplicationStop::new("one step")),
    );
    assert_eq!(
        trace,
        [
            Event::Init,
            Event::StopCheck(0),
            Event::Save(1, result.counts),
            Event::SolverIter(1),
            Event::StateIter(1),
            Event::StopCheck(1),
            Event::Save(1, result.counts),
            Event::Final(1),
        ],
    );
}

#[test]
fn explicit_executor_moves_with_all_hooks_already_registered() {
    let (sender, receiver) = mpsc::channel();
    let run = Executor::from_start(Sphere, Descent::new(0.25), vec![2.0])
        .max_iter(5)
        .observe_solver(
            observe(worker_trace(sender.clone())),
            ObserverMode::Always,
        )
        .observe_with(
            StateObserver(worker_trace(sender.clone())),
            ObserverMode::Always,
        )
        .stop_when(stop_after_one(worker_trace(sender.clone())))
        .checkpoint_with(Sink(worker_trace(sender)), ObserverMode::Always);

    let result = thread::spawn(move || run.run()).join().unwrap().unwrap();
    assert_run(result, &receiver.try_iter().collect::<Vec<_>>());
}

#[test]
fn initialized_stepper_moves_later_observations_to_the_worker() {
    let (sender, receiver) = mpsc::channel();
    let caller = thread::current().id();
    let stepper = Executor::from_start(Sphere, Descent::new(0.25), vec![2.0])
        .max_iter(1)
        .observe_solver(
            observe(move |event| {
                sender.send((event, thread::current().id())).unwrap();
            }),
            ObserverMode::Always,
        )
        .into_stepper()
        .unwrap();

    assert_eq!(receiver.try_recv().unwrap(), (Event::Init, caller));
    let worker = thread::spawn(move || {
        let result = stepper.run_to_end().unwrap();
        assert_eq!(result.iter(), 1);
        thread::current().id()
    })
    .join()
    .unwrap();

    assert_ne!(worker, caller);
    assert_eq!(
        receiver.try_iter().collect::<Vec<_>>(),
        [(Event::SolverIter(1), worker), (Event::Final(1), worker)]
    );
}

#[test]
fn inferred_executor_moves_with_all_hooks_already_registered() {
    let (sender, receiver) = mpsc::channel();
    let run =
        InferredExecutor::from_start(Sphere, Descent::new(0.25), vec![2.0])
            .max_iter(5)
            .observe_solver(
                observe(worker_trace(sender.clone())),
                ObserverMode::Always,
            )
            .observe_with(
                StateObserver(worker_trace(sender.clone())),
                ObserverMode::Always,
            )
            .stop_when(stop_after_one(worker_trace(sender.clone())))
            .checkpoint_with(Sink(worker_trace(sender)), ObserverMode::Always);

    let result = thread::spawn(move || run.run()).join().unwrap().unwrap();
    assert_run(result, &receiver.try_iter().collect::<Vec<_>>());
}

#[test]
fn explicit_executor_can_add_hooks_in_a_loop() {
    let (sender, receiver) = mpsc::channel();
    let mut run =
        Executor::from_start(Sphere, Descent::new(0.25), vec![2.0]).max_iter(0);
    for label in ["first", "second"] {
        let sender = sender.clone();
        run = run.observe_solver(
            move |_, _, event| {
                if matches!(event, ObservationEvent::Init) {
                    sender.send(label).unwrap();
                }
            },
            ObserverMode::Always,
        );
    }

    let result = thread::spawn(move || run.run()).join().unwrap().unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(receiver.try_iter().collect::<Vec<_>>(), ["first", "second"]);
}

#[test]
fn explicit_switch_keeps_existing_hooks_and_accepts_local_hooks() {
    let history = Rc::new(RefCell::new(Vec::new()));
    let (sender, receiver) = mpsc::channel();
    let mut record = local_trace(&history);
    let result = Executor::from_start(Sphere, Descent::new(0.25), vec![2.0])
        .max_iter(5)
        .observe_solver(
            observe(move |event| sender.send(event).unwrap()),
            ObserverMode::Always,
        )
        .into_local()
        .observe_solver(
            observe(move |event| {
                assert_eq!(receiver.try_recv().unwrap(), event);
                record(event);
            }),
            ObserverMode::Always,
        )
        .observe_with(
            StateObserver(local_trace(&history)),
            ObserverMode::Always,
        )
        .stop_when(stop_after_one(local_trace(&history)))
        .checkpoint_with(Sink(local_trace(&history)), ObserverMode::Always)
        .run()
        .unwrap();

    assert_run(result, &history.borrow());
}

#[test]
fn inferred_executor_accepts_local_hooks_with_the_ordinary_methods() {
    let history = Rc::new(RefCell::new(Vec::new()));
    let result =
        InferredExecutor::from_start(Sphere, Descent::new(0.25), vec![2.0])
            .max_iter(5)
            .observe_solver(
                observe(local_trace(&history)),
                ObserverMode::Always,
            )
            .observe_with(
                StateObserver(local_trace(&history)),
                ObserverMode::Always,
            )
            .stop_when(stop_after_one(local_trace(&history)))
            .checkpoint_with(Sink(local_trace(&history)), ObserverMode::Always)
            .run()
            .unwrap();

    assert_run(result, &history.borrow());
}

#[test]
fn both_can_run_a_local_problem_and_return_its_typed_error() {
    let expected: Rc<str> = "local failure".into();
    let problem = || Local {
        error: Some(Rc::clone(&expected)),
        ..Local::default()
    };

    let explicit =
        Executor::from_start(problem(), Descent::new(0.25), vec![2.0]).run();
    let inferred =
        InferredExecutor::from_start(problem(), Descent::new(0.25), vec![2.0])
            .run();

    for result in [explicit, inferred] {
        assert!(Rc::ptr_eq(&result.err().unwrap(), &expected));
    }
}

#[test]
fn configured_memetic_solver_moves_and_resets_its_inner_stop() {
    use basin::BasinHopping;

    let caller = thread::current().id();
    let (send, receive) = mpsc::channel();
    let solver = BasinHopping::new(Descent::new(0.25), 7)
        .with_inner_max_iter(5)
        .inner_stop_when_factory(move || {
            assert_ne!(thread::current().id(), caller);
            let send = send.clone();
            let mut checks = 0;
            move |state| {
                checks += 1;
                if state.iter() == 1 {
                    send.send(checks).unwrap();
                }
                None
            }
        });
    let run = Executor::from_start(Sphere, solver, vec![2.0]).max_iter(2);
    let result = thread::spawn(move || run.run()).join().unwrap().unwrap();
    assert_eq!(result.iter(), 2);
    assert_eq!(receive.try_iter().collect::<Vec<_>>(), [2, 2, 2]);
}

#[test]
fn local_memetic_hook_works_with_outer_convergence_settings() {
    use basin::BasinHopping;

    let calls = Rc::new(Cell::new(0));
    let seen = Rc::clone(&calls);
    let solver = BasinHopping::new(Descent::new(0.25), 7)
        .with_absolute_cost_change_tolerance(0.0)
        .into_local()
        .with_inner_max_iter(5)
        .inner_stop_when_factory(move || {
            seen.set(seen.get() + 1);
            |_| None
        });
    let result = Executor::from_start(Sphere, solver, vec![2.0])
        .max_iter(1)
        .run()
        .unwrap();
    assert_eq!(result.iter(), 1);
    assert_eq!(calls.get(), 2);
}

#[test]
fn configured_population_refinement_solvers_are_send() {
    use basin::{
        BoundedCmaEs, BoundedCmaInject, CmaEs, CmaInject, De, DeInject,
        DenseMatrix, NelderMead,
    };

    fn assert_send<T: Send>(_: T) {}

    let inner = NelderMead::<Vec<f64>>::adaptive;
    assert_send(
        CmaInject::with_inner_solver(
            CmaEs::<_, DenseMatrix>::new(7, 0.5),
            inner(),
        )
        .with_inner_max_iter(2)
        .inner_stop_when_factory(|| |_| None),
    );
    assert_send(
        BoundedCmaInject::with_inner_solver(
            BoundedCmaEs::<_, DenseMatrix>::new(7, 0.5),
            inner(),
        )
        .with_inner_max_iter(2)
        .inner_stop_when_factory(|| |_| None),
    );
    assert_send(
        DeInject::with_inner_solver(De::new(7), inner())
            .with_inner_max_iter(2)
            .inner_stop_when_factory(|| |_| None),
    );
}

#[test]
fn local_memetic_solvers_keep_cost_and_step_builders() {
    use basin::{
        BasinHopping, BoundedCmaEs, BoundedCmaInject, CmaEs, CmaInject, De,
        DeInject, DenseMatrix, NelderMead,
    };

    // Start each chain with a different builder to cover both the solver
    // itself and the ConfiguredSolver returned by the first builder.
    macro_rules! check_builders {
        ($solver:expr) => {{
            let _ = $solver
                .into_local()
                .with_absolute_cost_change_tolerance(0.0_f64)
                .with_relative_step_tolerance::<Vec<f64>, f64>(0.0);
            let _ = $solver
                .into_local()
                .with_absolute_step_tolerance::<Vec<f64>, f64>(0.0)
                .with_relative_cost_change_tolerance(0.0_f64);
        }};
    }

    let inner = NelderMead::<Vec<f64>>::adaptive;
    check_builders!(BasinHopping::<_, Vec<f64>, f64>::new(inner(), 7));
    check_builders!(CmaInject::with_inner_solver(
        CmaEs::<_, DenseMatrix>::new(7, 0.5),
        inner()
    ));
    check_builders!(BoundedCmaInject::with_inner_solver(
        BoundedCmaEs::<_, DenseMatrix>::new(7, 0.5),
        inner()
    ));
    check_builders!(DeInject::with_inner_solver(De::new(7), inner()));
}
