#[path = "support/solver_observer.rs"]
mod support;
use basin::ObserveSolver;
use basin::{
    CancellationToken, Executor, Observe, ObserverMode, PointState, State,
    StepOutcome, TerminationCode,
};
#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    Init,
    Iter,
    Final(TerminationCode),
}
impl From<basin::ObservationEvent<'_>> for Event {
    fn from(event: basin::ObservationEvent<'_>) -> Self {
        match event {
            basin::ObservationEvent::Init => Self::Init,
            basin::ObservationEvent::Iter => Self::Iter,
            basin::ObservationEvent::Final(report) => {
                Self::Final(report.code())
            }
            _ => unreachable!("unknown observation event"),
        }
    }
}
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use support::*;

#[derive(Clone, Debug, PartialEq)]
struct Record {
    event: Event,
    iter: u64,
    cost: f64,
    evals: u64,
    diagnostic: f64,
}
type Log = Rc<RefCell<Vec<Record>>>;
fn record(log: &Log, s: &PointState<f64>, value: f64, event: Event) {
    log.borrow_mut().push(Record {
        event,
        iter: s.iter(),
        cost: s.cost(),
        evals: s.cost_evals(),
        diagnostic: value,
    });
}
struct Recorder(Log);
impl ObserveSolver<PointState<f64>, Probe> for Recorder {
    fn observe_init(&mut self, state: &PointState<f64>, solver: &Probe) {
        record(&self.0, state, solver.diagnostic, Event::Init);
    }
    fn observe_iter(&mut self, state: &PointState<f64>, solver: &Probe) {
        record(&self.0, state, solver.diagnostic, Event::Iter);
    }
    fn observe_final(
        &mut self,
        state: &PointState<f64>,
        solver: &Probe,
        reason: &basin::TerminationReport,
    ) {
        record(
            &self.0,
            state,
            solver.diagnostic,
            Event::Final(reason.code()),
        );
    }
}
fn attach(run: Run, kind: u8, log: Log, mode: ObserverMode) -> Run {
    match kind {
        0 => run.observe_solver(
            move |s, so, event| record(&log, s, so.diagnostic, event.into()),
            mode,
        ),
        1 => run.observe_solver_with(Recorder(log), mode),
        _ => unreachable!(),
    }
}
fn manual(scenario: Scenario) -> (Vec<Record>, Result<(), &'static str>) {
    let log = Log::default();
    let mut stepper = match executor(scenario).into_stepper() {
        Ok(s) => s,
        Err(e) => return (Vec::new(), Err(e)),
    };
    record(
        &log,
        stepper.state(),
        stepper.solver().diagnostic,
        Event::Init,
    );
    let result = loop {
        match stepper.step() {
            Ok(StepOutcome::Continue) => record(
                &log,
                stepper.state(),
                stepper.solver().diagnostic,
                Event::Iter,
            ),
            Ok(StepOutcome::Stopped(reason)) => {
                record(
                    &log,
                    stepper.state(),
                    stepper.solver().diagnostic,
                    Event::Final(reason.code()),
                );
                break Ok(());
            }
            Err(e) => break Err(e),
        }
    };
    (Rc::try_unwrap(log).unwrap().into_inner(), result)
}

#[test]
fn trait_and_closure_match_stepper_on_all_exit_paths() {
    for scenario in [
        Scenario::Normal,
        Scenario::InitError,
        Scenario::StepError,
        Scenario::MidStop,
        Scenario::CheckStop,
    ] {
        let (expected, error) = manual(scenario);
        for kind in 0..2 {
            let log = Log::default();
            let actual = attach(
                executor(scenario),
                kind,
                log.clone(),
                ObserverMode::Always,
            )
            .run()
            .map(|_| ());
            assert_eq!(actual, error);
            assert_eq!(*log.borrow(), expected);
            if scenario == Scenario::MidStop {
                let last = log.borrow().last().unwrap().clone();
                assert_eq!(
                    (last.iter, last.evals, last.diagnostic),
                    (0, 2, 1.0)
                );
            }
            if scenario == Scenario::CheckStop {
                assert_eq!(log.borrow().last().unwrap().diagnostic, 99.0);
            }
        }
    }
}

#[test]
fn modes_gate_iterations_and_final_fires_once() {
    for (mode, expected) in [
        (ObserverMode::Always, vec![0, 1, 2, 3, 4, 4]),
        (ObserverMode::Every(2), vec![0, 2, 4, 4]),
        (ObserverMode::Never, vec![0, 4]),
        (ObserverMode::Every(0), vec![0, 4]),
        (ObserverMode::NewBest, vec![0, 1, 2, 3, 4, 4]),
    ] {
        for kind in 0..2 {
            let log = Log::default();
            let mut stepper =
                attach(executor(Scenario::Normal), kind, log.clone(), mode)
                    .into_stepper()
                    .unwrap();
            while stepper.step().unwrap() == StepOutcome::Continue {}
            stepper.step().unwrap();
            assert_eq!(
                log.borrow().iter().map(|r| r.iter).collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn cancellation_and_exact_continuation_keep_boundary_semantics() {
    for kind in 0..2 {
        let token = CancellationToken::new();
        token.cancel();
        let log = Log::default();
        attach(
            executor(Scenario::Normal).with_cancellation_token(token),
            kind,
            log.clone(),
            ObserverMode::Always,
        )
        .run()
        .unwrap();
        assert_eq!(
            log.borrow().iter().map(|r| r.event).collect::<Vec<_>>(),
            vec![Event::Init, Event::Final(TerminationCode::Cancelled)]
        );

        let mut stepper = executor(Scenario::Normal).into_stepper().unwrap();
        stepper.step().unwrap();
        stepper.step().unwrap();
        let checkpoint = stepper.into_checkpoint().unwrap();
        let log = Log::default();
        let run =
            Executor::resume_from_checkpoint(Linear, checkpoint).max_iter(3);
        let result = attach(run, kind, log.clone(), ObserverMode::Always)
            .run_with_solver()
            .unwrap();
        assert_eq!(
            (
                log.borrow()[0].iter,
                log.borrow()[0].diagnostic,
                log.borrow()[0].evals
            ),
            (2, 2.0, 3)
        );
        assert_eq!(
            (
                result.iter(),
                result.counts.cost_evals,
                result.solver.diagnostic
            ),
            (3, 4, 3.0)
        );
    }
}

struct StateTag(Rc<RefCell<Vec<u8>>>);
impl Observe<PointState<f64>> for StateTag {
    fn observe_init(&mut self, _: &PointState<f64>) {
        self.0.borrow_mut().push(0);
    }
    fn observe_iter(&mut self, _: &PointState<f64>) {
        self.0.borrow_mut().push(0);
    }
    fn observe_final(
        &mut self,
        _: &PointState<f64>,
        _: &basin::TerminationReport,
    ) {
        self.0.borrow_mut().push(0);
    }
}
#[test]
fn mixed_observers_preserve_registration_order_and_borrow_workspace() {
    let tags = Rc::new(RefCell::new(Vec::new()));
    let solver_tags = tags.clone();
    let probe = Probe::new(Scenario::Normal);
    let pointer = probe.workspace.as_ptr() as usize;
    let result = Executor::new(Linear, probe, PointState::new(10.0))
        .max_iter(1)
        .observe_with(StateTag(tags.clone()), ObserverMode::Always)
        .observe_solver(
            move |_, so, _| {
                assert_eq!(so.workspace.as_ptr() as usize, pointer);
                solver_tags.borrow_mut().push(1);
            },
            ObserverMode::Always,
        )
        .observe_with(StateTag(tags.clone()), ObserverMode::Always)
        .run_with_solver()
        .unwrap();
    assert_eq!(*tags.borrow(), vec![0, 1, 0, 0, 1, 0, 0, 1, 0]);
    assert_eq!(result.counts.cost_evals, 2);
}

#[test]
fn annealing_diagnostics_describe_next_proposal_including_restarts() {
    use basin::core::rng::ChaCha8Rng;
    use basin::{SimulatedAnnealing, TemperatureSchedule};
    struct Cost;
    impl basin::CostFunction for Cost {
        type Param = f64;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
            Ok(*x)
        }
    }
    let temperatures = Rc::new(RefCell::new(Vec::new()));
    let output = temperatures.clone();
    let solver = SimulatedAnnealing::new(
        |x: &f64, _: f64, _: &mut ChaCha8Rng| x - 1.0,
        8.0,
        TemperatureSchedule::geometric(0.5),
        7,
    )
    .with_reannealing_fixed(3);
    let result = Executor::from_start(Cost, solver, 10.0)
        .max_iter(4)
        .observe_solver(
            move |_, so, _| output.borrow_mut().push(so.temperature()),
            ObserverMode::Always,
        )
        .run_with_solver()
        .unwrap();
    assert_eq!(*temperatures.borrow(), vec![8.0, 4.0, 2.0, 8.0, 4.0, 4.0]);
    assert_eq!(result.counts.cost_evals, 5);
}

#[test]
fn slsqp_observer_reads_optional_borrowed_multipliers() {
    use basin::{Gradient, Slsqp};
    struct Quadratic;
    impl basin::CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(x[0] * x[0])
        }
    }
    impl Gradient for Quadratic {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![2.0 * x[0]])
        }
    }
    impl basin::NonlinearConstraints for Quadratic {
        type Matrix = basin::DenseMatrix;
        fn num_nonlinear_constraints(&self) -> usize {
            0
        }
        fn nonlinear_constraints(
            &self,
            _: &Vec<f64>,
        ) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![])
        }
        fn num_nonlinear_equalities(&self) -> usize {
            1
        }
        fn nonlinear_equalities(
            &self,
            x: &Vec<f64>,
        ) -> Result<Option<Vec<f64>>, Self::Error> {
            Ok(Some(vec![x[0] - 1.0]))
        }
    }
    impl basin::ConstraintJacobian for Quadratic {
        fn constraint_jacobian(
            &self,
            _: &Vec<f64>,
        ) -> Result<basin::DenseMatrix, Self::Error> {
            Ok(basin::DenseMatrix::from_row_slice(1, 1, &[1.0]))
        }
    }
    let solver = Slsqp::new();
    assert!(solver.stationarity().is_none());
    let seen = Rc::new(Cell::new(false));
    let flag = seen.clone();
    Executor::from_start(Quadratic, solver, vec![2.0])
        .observe_solver(
            move |_, so, _| {
                if let Some(values) = so.equality_multipliers() {
                    assert_eq!(values.len(), 1);
                    assert!(so.stationarity().is_some());
                    flag.set(true);
                }
            },
            ObserverMode::Always,
        )
        .run()
        .unwrap();
    assert!(seen.get());
}

#[test]
fn scalar_f32_and_default_trait_hooks_need_no_extra_capabilities() {
    use basin::core::rng::ChaCha8Rng;
    use basin::{ProposalState, SimulatedAnnealing, TemperatureSchedule};
    struct Cost;
    impl basin::CostFunction for Cost {
        type Param = f32;
        type Output = f32;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &f32) -> Result<f32, Self::Error> {
            Ok(*x)
        }
    }
    struct TemperatureLogger(Rc<Cell<f32>>);
    impl<N, R>
        ObserveSolver<ProposalState<f32, f32>, SimulatedAnnealing<N, f32, R>>
        for TemperatureLogger
    {
        fn observe_iter(
            &mut self,
            _: &ProposalState<f32, f32>,
            solver: &SimulatedAnnealing<N, f32, R>,
        ) {
            self.0.set(solver.temperature());
        }
    }
    let last = Rc::new(Cell::new(-1.0));
    let solver = SimulatedAnnealing::new(
        |x: &f32, _: f32, _: &mut ChaCha8Rng| x - 1.0,
        8.0_f32,
        TemperatureSchedule::geometric(0.5),
        7,
    );
    Executor::from_start(Cost, solver, 10.0)
        .max_iter(2)
        .observe_solver_with(
            TemperatureLogger(last.clone()),
            ObserverMode::Always,
        )
        .run()
        .unwrap();
    assert_eq!(last.get(), 2.0);
}

#[test]
fn direct_callback_accepts_borrowed_neighbor() {
    use basin::core::rng::ChaCha8Rng;
    use basin::{CostFunction, SimulatedAnnealing, TemperatureSchedule};
    struct Cost;
    impl CostFunction for Cost {
        type Param = f64;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
            Ok(*x)
        }
    }
    let amount = 1.0;
    let reference = &amount;
    let solver = SimulatedAnnealing::new(
        move |x: &f64, _: f64, _: &mut ChaCha8Rng| x - reference,
        8.0,
        TemperatureSchedule::geometric(0.5),
        7,
    );
    Executor::from_start(Cost, solver, 10.0)
        .max_iter(2)
        .observe_solver(
            |_, so, _| {
                std::hint::black_box(so.temperature());
            },
            ObserverMode::Always,
        )
        .run()
        .unwrap();
}

#[test]
fn init_and_final_fire_without_iterations_or_new_incumbents() {
    for kind in 0..2 {
        for run in [
            executor(Scenario::Normal).max_iter(0),
            executor(Scenario::Flat),
        ] {
            let log = Log::default();
            let result = attach(run, kind, log.clone(), ObserverMode::NewBest)
                .run()
                .unwrap();
            let log = log.borrow();
            assert_eq!(log.len(), 2);
            assert_eq!((log[0].event, log[0].iter), (Event::Init, 0));
            assert_eq!(
                (log[1].event, log[1].iter),
                (Event::Final(TerminationCode::MaxIter), result.iter())
            );
        }
    }
}

#[test]
fn solver_observer_can_cancel_after_a_completed_iteration() {
    let token = CancellationToken::new();
    let request = token.clone();
    let log = Log::default();
    let output = log.clone();
    let result = executor(Scenario::Normal)
        .with_cancellation_token(token)
        .observe_solver(
            move |state, solver, event| {
                record(&output, state, solver.diagnostic, event.into());
                if event == basin::ObservationEvent::Iter {
                    request.cancel();
                }
            },
            ObserverMode::Always,
        )
        .run()
        .unwrap();
    assert_eq!(result.report.code(), TerminationCode::Cancelled);
    assert_eq!(result.iter(), 1);
    assert_eq!(result.cost_evals(), 2);
    assert_eq!(
        log.borrow().iter().map(|r| r.event).collect::<Vec<_>>(),
        vec![
            Event::Init,
            Event::Iter,
            Event::Final(TerminationCode::Cancelled)
        ]
    );
}

#[test]
fn state_and_solver_may_borrow_local_data_with_mixed_observers() {
    use basin::{CostFunction, Problem, Solver};
    use std::convert::Infallible;
    struct Cost<'a>(&'a f64);
    impl<'a> CostFunction for Cost<'a> {
        type Param = &'a f64;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &&'a f64) -> Result<f64, Infallible> {
            Ok(**x * self.0)
        }
    }
    struct BorrowedSolver<'a>(&'a f64);
    impl<'a> Solver<Cost<'a>, PointState<&'a f64>> for BorrowedSolver<'a> {
        type Error = Infallible;
        fn init(
            &mut self,
            problem: &mut Problem<Cost<'a>>,
            mut state: PointState<&'a f64>,
        ) -> Result<PointState<&'a f64>, Infallible> {
            state.replace(*state.param(), problem.cost(state.param())?);
            Ok(state)
        }
        fn next_iter(
            &mut self,
            _: &mut Problem<Cost<'a>>,
            state: PointState<&'a f64>,
        ) -> Result<basin::SolverStep<PointState<&'a f64>>, Infallible>
        {
            Ok(basin::SolverStep::completed(state))
        }
    }
    struct Progress;
    impl<S: State> Observe<S> for Progress {}
    let scale = 2.0;
    let start = 3.0;
    let count = Rc::new(Cell::new(0));
    let output = count.clone();
    Executor::new(
        Cost(&scale),
        BorrowedSolver(&scale),
        PointState::new(&start),
    )
    .max_iter(1)
    .observe_with(Progress, ObserverMode::Always)
    .observe_solver(
        move |state, solver, _| {
            assert_eq!(*solver.0, 2.0);
            assert_eq!(**state.param(), 3.0);
            assert_eq!(state.cost(), 6.0);
            output.set(output.get() + 1);
        },
        ObserverMode::Always,
    )
    .run()
    .unwrap();
    assert_eq!(count.get(), 3);
}
