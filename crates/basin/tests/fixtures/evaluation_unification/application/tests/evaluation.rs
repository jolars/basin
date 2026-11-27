use std::{
    cell::Cell,
    rc::Rc,
    sync::{
        Condvar, Mutex,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
    thread,
    time::Duration,
};

use basin::{
    ApplicationStop, CostFunction, Executor, FiniteDiff, Gradient,
    GradientDescent, ObservationEvent, ObserverMode, State, Termination,
    central_difference_hessian_product, forward_difference_hessian_product,
};
use basin_evaluation_local::{
    Difference, Local, STEP, Serial, Stencil, check_local_derivatives,
    check_local_runs, product, quadratic,
};
use basin_evaluation_threaded::{on_two_workers, two_workers};

#[derive(Default)]
struct Shared {
    costs: AtomicUsize,
    gradients: AtomicUsize,
    caller: Option<thread::ThreadId>,
    cost_rendezvous: Option<(Mutex<usize>, Condvar)>,
    rendezvous: Option<(Mutex<usize>, Condvar)>,
    fail: bool,
}

impl CostFunction for Shared {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, point: &Vec<f64>) -> Result<f64, Self::Error> {
        if let Some(caller) = self.caller {
            assert_eq!(thread::current().id(), caller);
        }
        self.costs.fetch_add(1, Relaxed);
        if let Some((arrived, ready)) = &self.cost_rendezvous {
            let mut arrived = arrived.lock().unwrap();
            *arrived += 1;
            ready.notify_all();
            let (_guard, status) = ready
                .wait_timeout_while(arrived, Duration::from_secs(5), |n| *n < 2)
                .unwrap();
            assert!(!status.timed_out(), "cost calls did not overlap");
        }
        if self.fail {
            return Err("cost failed");
        }
        Ok(quadratic(point))
    }
}

impl Gradient for Shared {
    type Gradient = Vec<f64>;

    fn gradient(&self, point: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        let mut output = point.clone();
        self.gradient_into(point, &mut output)?;
        Ok(output)
    }

    fn gradient_into(
        &self,
        point: &Vec<f64>,
        out: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        self.gradients.fetch_add(1, Relaxed);
        if let Some((arrived, ready)) = &self.rendezvous {
            let mut arrived = arrived.lock().unwrap();
            *arrived += 1;
            ready.notify_all();
            let (_guard, status) = ready
                .wait_timeout_while(arrived, Duration::from_secs(5), |n| *n < 2)
                .unwrap();
            assert!(!status.timed_out(), "gradient calls did not overlap");
        }
        if self.fail {
            return Err("gradient failed");
        }
        for (j, (&x, y)) in point.iter().zip(out).enumerate() {
            *y = (j + 1) as f64 * x;
        }
        Ok(())
    }
}

#[test]
fn local_dependency_runs_with_unified_features() {
    check_local_runs();
    check_local_derivatives();
    let error: Rc<str> = "local failure".into();
    let problem = FiniteDiff::new(Local {
        error: Some(Rc::clone(&error)),
        ..Local::default()
    });
    let actual = problem.gradient(&vec![1.0]).unwrap_err();
    assert!(Rc::ptr_eq(&error, &actual));
}

#[test]
fn explicit_modes_agree_through_executor() {
    let x = vec![2.0, 1.0];
    let (serial, threaded) = on_two_workers(|| {
        let serial = FiniteDiff::new(Shared {
            caller: Some(thread::current().id()),
            ..Shared::default()
        })
        .with_step(STEP);
        let threaded = FiniteDiff::new(Shared {
            cost_rendezvous: Some((Mutex::new(0), Condvar::new())),
            ..Shared::default()
        })
        .with_step(STEP)
        .with_parallel_evaluation();
        // This initial gradient has no base cost, so both worker threads
        // must reach the rendezvous before either probe can finish.
        assert_eq!(serial.gradient(&x), threaded.gradient(&x));
        let serial =
            Executor::from_start(serial, GradientDescent::new(0.25), x.clone())
                .max_iter(2)
                .run()
                .unwrap();
        let threaded =
            Executor::from_start(threaded, GradientDescent::new(0.25), x)
                .max_iter(2)
                .run()
                .unwrap();
        (serial, threaded)
    });
    assert_eq!(serial.param(), threaded.param());
    assert_eq!(
        serial.current_record().unwrap().cost.unwrap().to_bits(),
        threaded.current_record().unwrap().cost.unwrap().to_bits()
    );
    assert_eq!(serial.counts, threaded.counts);
    assert_eq!(threaded.cost_evals(), 3);
    assert_eq!(threaded.counts.gradient_evals, 3);

    let mut problem = basin::Problem::new(Shared {
        fail: true,
        ..Shared::default()
    });
    let error = problem.cost_batch_parallel(&[vec![1.0], vec![2.0]]);
    assert_eq!(error, Err("cost failed"));
    assert_eq!(problem.counts().cost_evals, 2);
}

#[test]
fn configured_run_moves_before_local_hooks_are_created() {
    let caller = thread::current().id();
    let problem = Difference {
        problem: Shared::default(),
        mode: two_workers(),
    };
    let solver = GradientDescent::new(0.25);
    let initial = vec![2.0, 1.0];
    let configured = move || {
        let worker = thread::current().id();
        assert_ne!(caller, worker);
        let observed = Rc::new(Cell::new(0));
        let hook_state = Rc::clone(&observed);
        let result = Executor::from_start(problem, solver, initial)
            .max_iter(5)
            .into_local()
            .observe_solver(
                move |_, _, event| {
                    assert_eq!(thread::current().id(), worker);
                    if matches!(
                        event,
                        ObservationEvent::Init | ObservationEvent::Iter
                    ) {
                        hook_state.set(hook_state.get() + 1);
                    }
                },
                ObserverMode::Always,
            )
            .stop_when(move |state| {
                assert_eq!(thread::current().id(), worker);
                (observed.get() == 2 && state.iter() == 1)
                    .then(|| ApplicationStop::new("worker stop"))
            })
            .run()
            .unwrap();
        assert_eq!(
            result.report.termination,
            Termination::Application(ApplicationStop::new("worker stop"))
        );
        result.iter()
    };
    assert_eq!(thread::spawn(configured).join().unwrap(), 1);
}

#[test]
fn hessian_products_match_and_threaded_calls_overlap() {
    let mode = two_workers();
    let x = vec![2.0, -1.0];
    let v = vec![3.0, 4.0];
    for stencil in [Stencil::Forward, Stencil::Central] {
        let reference = match stencil {
            Stencil::Forward => forward_difference_hessian_product,
            Stencil::Central => central_difference_hessian_product,
        };
        let analytic = Shared::default();
        let expected =
            reference(&analytic, &x, &v, f64::EPSILON, Some(STEP)).unwrap();
        assert_eq!(expected, [3.0, 8.0]);
        assert_eq!(
            product(&Serial, &analytic, &x, &v, stencil).unwrap(),
            expected
        );

        let overlapping = Shared {
            rendezvous: Some((Mutex::new(0), Condvar::new())),
            ..Shared::default()
        };
        assert_eq!(
            product(&mode, &overlapping, &x, &v, stencil).unwrap(),
            expected
        );
        assert_eq!(
            product(&mode, &overlapping, &x, &[0.0, 0.0], stencil).unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(overlapping.gradients.load(Relaxed), 2);

        let synthesized = Difference {
            problem: Shared::default(),
            mode: mode.clone(),
        };
        assert_eq!(
            product(&mode, &synthesized, &x, &v, stencil).unwrap(),
            expected
        );
        assert_eq!(synthesized.problem.costs.load(Relaxed), 8);

        let failing = Shared {
            fail: true,
            ..Shared::default()
        };
        assert_eq!(
            product(&mode, &failing, &x, &v, stencil),
            Err("gradient failed")
        );
        assert_eq!(failing.gradients.load(Relaxed), 2);
    }
}

#[test]
fn built_in_population_modes_agree() {
    use basin::{CmaEs, DenseMatrix, PopulationProgress};

    let start = || PopulationProgress::from_point(vec![2.0, 1.0]);
    let solver = || CmaEs::<Vec<f64>, DenseMatrix>::new(7, 0.25).with_lambda(4);
    let serial = Executor::new(Shared::default(), solver(), start())
        .max_iter(2)
        .run()
        .unwrap();
    let threaded = basin_evaluation_threaded::on_two_workers(|| {
        Executor::new(
            Shared {
                cost_rendezvous: Some((Mutex::new(0), Condvar::new())),
                ..Shared::default()
            },
            solver().with_parallel_evaluation(),
            start(),
        )
        .max_iter(2)
        .run()
        .unwrap()
    });
    assert_eq!(
        serial.state.evaluated_members(),
        threaded.state.evaluated_members()
    );
    assert_eq!(serial.param(), threaded.param());
    assert_eq!(serial.counts, threaded.counts);
}
